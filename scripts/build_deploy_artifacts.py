"""
Builds the committed deploy/ copies of the dashboard's artifacts, so app.py
runs from a clean checkout (Streamlit Community Cloud) without the
gitignored full pipeline outputs:

    deploy/transactions_dashboard.parquet    all rows of data/processed/transactions_clean.parquet,
                                             only the dashboard's columns, compact dtypes, zstd
    deploy/scored_test_transactions.parquet  results/fraud_detection/scored_test_transactions.parquet,
                                             recompressed with zstd (same rows and columns)
    deploy/best_model.joblib                 results/models/best_model.joblib, copied unchanged

results/metrics/model_comparison.json is already committed, so it is not
copied. Every output is checked against its source (rows, fraud count,
values) and must stay under 95 MB, below GitHub's 100 MB file limit.

Usage (from the project root, after the pipeline has run):
    venv\\Scripts\\python scripts\\build_deploy_artifacts.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import config as cfg
from src.dashboard.data import DASHBOARD_COLUMNS, apply_dashboard_dtypes

MAX_BYTES = 95 * 1024 ** 2
OUT_TRANSACTIONS = cfg.DEPLOY_DIR / "transactions_dashboard.parquet"
OUT_SCORED = cfg.DEPLOY_DIR / "scored_test_transactions.parquet"
OUT_MODEL = cfg.DEPLOY_DIR / "best_model.joblib"


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024 ** 2:.1f} MB"


def _same_values(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Equal row by row, ignoring dtype differences (category vs string,
    int8 vs int64)."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    for col in a.columns:
        x, y = a[col], b[col]
        if isinstance(x.dtype, pd.CategoricalDtype) or isinstance(y.dtype, pd.CategoricalDtype):
            x, y = x.astype(str), y.astype(str)
        if not x.reset_index(drop=True).equals(y.reset_index(drop=True).astype(x.dtype)):
            print(f"  column {col} differs")
            return False
    return True


def build_transactions(source: Path = cfg.CLEANED_TRANSACTIONS_FILE, out: Path = OUT_TRANSACTIONS) -> pd.DataFrame:
    src = pd.read_parquet(source, columns=DASHBOARD_COLUMNS)
    for col in ("txn_hour", cfg.CITY_POP_COL):
        if src[col].min() < -2 ** 31 or src[col].max() > 2 ** 31 - 1:
            raise ValueError(f"{col} does not fit the downcast integer type")
    slim = apply_dashboard_dtypes(src.copy())
    out.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(out, index=False, compression="zstd", compression_level=9)

    back = pd.read_parquet(out)
    src_sorted = src.sort_values(cfg.SPLIT_COL, key=lambda s: s.astype(str).map({"train": 0, "test": 1}),
                                 kind="stable", ignore_index=True)
    assert _same_values(src_sorted, back), "deploy transactions differ from the source"
    for split in ("train", "test"):
        a = src[src[cfg.SPLIT_COL].astype(str) == split]
        b = back[back[cfg.SPLIT_COL].astype(str) == split]
        assert len(a) == len(b) and a[cfg.LABEL_COL].sum() == b[cfg.LABEL_COL].sum(), split
    return back


def build_scored(source: Path = cfg.SCORED_TEST_FILE, out: Path = OUT_SCORED) -> pd.DataFrame:
    src = pd.read_parquet(source)
    src.to_parquet(out, index=False, compression="zstd", compression_level=9)
    back = pd.read_parquet(out)
    assert _same_values(src, back), "deploy test-set scores differ from the source"
    return back


def main():
    for path in (cfg.CLEANED_TRANSACTIONS_FILE, cfg.SCORED_TEST_FILE, cfg.MODEL_FILE):
        if not path.exists():
            raise SystemExit(f"{path} not found - run the pipeline first (.\\scripts\\run_pipeline.ps1).")

    tx = build_transactions()
    print(f"{OUT_TRANSACTIONS.relative_to(cfg.PROJECT_ROOT)}: {len(tx):,} rows "
          f"({int(tx[cfg.LABEL_COL].sum()):,} fraud), {tx.shape[1]} columns, {_mb(OUT_TRANSACTIONS)} "
          f"(source {_mb(cfg.CLEANED_TRANSACTIONS_FILE)})")
    for split, g in tx.groupby(cfg.SPLIT_COL, observed=True):
        print(f"  {split}: {len(g):,} rows, {int(g[cfg.LABEL_COL].sum()):,} fraud")

    scored = build_scored()
    print(f"{OUT_SCORED.relative_to(cfg.PROJECT_ROOT)}: {len(scored):,} rows "
          f"({int(scored[cfg.LABEL_COL].sum()):,} fraud), {_mb(OUT_SCORED)} (source {_mb(cfg.SCORED_TEST_FILE)})")

    shutil.copy2(cfg.MODEL_FILE, OUT_MODEL)
    assert OUT_MODEL.read_bytes() == cfg.MODEL_FILE.read_bytes()
    print(f"{OUT_MODEL.relative_to(cfg.PROJECT_ROOT)}: {_mb(OUT_MODEL)} (identical copy)")

    for path in (OUT_TRANSACTIONS, OUT_SCORED, OUT_MODEL):
        assert path.stat().st_size < MAX_BYTES, f"{path} is {_mb(path)}, over the 95 MB limit"
    print("All deploy artifacts match their sources and are under 95 MB.")


if __name__ == "__main__":
    main()
