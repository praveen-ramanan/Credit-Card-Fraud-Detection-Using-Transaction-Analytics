"""
Measures the dashboard's memory use, the way Streamlit Community Cloud would
see it: drives app.py headlessly with Streamlit's AppTest through all six
pages (and all three dataset choices), in this one process, then reports
    - the transactions DataFrame(s) the dashboard holds (memory_usage(deep=True)),
    - process RSS after each page and the peak (psutil).

Usage (from the project root):
    venv\\Scripts\\python scripts\\measure_memory.py            # artifacts as resolved by config
    venv\\Scripts\\python scripts\\measure_memory.py --deploy   # force the deploy/ copies
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import psutil

PAGES = ["Overview", "Transaction analytics", "Fraud detection", "Model performance", "Fraud alerts", "CSV analysis"]
DATASETS = ["Training (fraudTrain.csv)", "Test (fraudTest.csv)", "Both files"]
MB = 1024 ** 2


def peak_rss_mb(proc) -> float:
    mem = proc.memory_info()
    if hasattr(mem, "peak_wset"):  # Windows
        return mem.peak_wset / MB
    import resource  # Linux/macOS: ru_maxrss is KiB on Linux, bytes on macOS
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (MB if sys.platform == "darwin" else 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--deploy", action="store_true", help="use the deploy/ artifacts instead of the full ones")
    args = parser.parse_args()
    if args.deploy:
        os.environ["BDA_DASHBOARD_DATA"] = str(ROOT / "deploy" / "transactions_dashboard.parquet")
        os.environ["BDA_DASHBOARD_MODEL"] = str(ROOT / "deploy" / "best_model.joblib")
        os.environ["BDA_DASHBOARD_SCORED"] = str(ROOT / "deploy" / "scored_test_transactions.parquet")
        os.environ["BDA_DASHBOARD_ALERTS"] = str(ROOT / "deploy" / "fraud_alerts.parquet")

    proc = psutil.Process()
    rss = lambda: proc.memory_info().rss / MB  # noqa: E731

    from streamlit.testing.v1 import AppTest
    from config import config as cfg
    from src.dashboard import data

    print(f"Transactions file: {getattr(cfg, 'DASHBOARD_TRANSACTIONS_FILE', cfg.CLEANED_TRANSACTIONS_FILE)}")
    print(f"RSS after imports: {rss():.0f} MB")

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    t = time.perf_counter()
    at.run()
    rows = [("(first load: Overview)", "", time.perf_counter() - t, rss(), peak_rss_mb(proc))]
    for page in PAGES:
        t = time.perf_counter()
        at.sidebar.radio[0].set_value(page).run()
        if at.exception:
            raise SystemExit(f"{page}: {at.exception}")
        rows.append((page, "", time.perf_counter() - t, rss(), peak_rss_mb(proc)))
        if page == "Overview":
            for ds in DATASETS:
                t = time.perf_counter()
                at.selectbox(key="dataset").set_value(ds).run()
                rows.append((page, ds, time.perf_counter() - t, rss(), peak_rss_mb(proc)))
        if page == "Fraud detection":
            at.button[1].click().run()  # load a fraud example
            at.button(key="FormSubmitter:predict-Assess transaction").click().run()
            rows.append((page, "scored fraud example", 0.0, rss(), peak_rss_mb(proc)))

    print(f"\n{'page':<24}{'action':<28}{'seconds':>8}{'RSS MB':>9}{'peak MB':>9}")
    for page, action, secs, mb, peak in rows:
        print(f"{page:<24}{action:<28}{secs:>8.1f}{mb:>9.0f}{peak:>9.0f}")

    # The frames the dashboard holds (served from the same process-wide caches).
    full = data.load_transactions(None)
    print(f"\nTransactions frame: {len(full):,} rows x {full.shape[1]} cols, "
          f"memory_usage(deep=True) = {full.memory_usage(deep=True).sum() / MB:.1f} MB")
    for split in ("train", "test"):
        part = data.load_transactions(split)
        # amt is a plain float64 column, so this tells whether the split is a view or a copy
        shares = np.shares_memory(full[cfg.AMOUNT_COL].to_numpy(), part[cfg.AMOUNT_COL].to_numpy())
        print(f"  split {split}: {len(part):,} rows, {part.memory_usage(deep=True).sum() / MB:.1f} MB "
              f"({'view of the full frame, no extra memory' if shares else 'separate copy'})")
    print(f"\nRSS now: {rss():.0f} MB, peak RSS: {peak_rss_mb(proc):.0f} MB")


if __name__ == "__main__":
    main()
