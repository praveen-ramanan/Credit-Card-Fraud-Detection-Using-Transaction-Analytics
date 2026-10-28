"""Checks on the real Kaggle data and the train/validation/test protocol:
the CSVs load with the expected schema, the cleaned Parquet keeps the two
files apart, validation is strictly earlier than test, and the stored test
metrics come from the full fraudTest.csv. Skipped where the dataset or the
pipeline outputs are not available."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq
import pytest

from config import config as cfg
from src.fraud_detection.train_model import chronological_split
from src.preprocessing.clean_data import EXPECTED_COLUMNS

needs_csvs = pytest.mark.skipif(not (cfg.TRAIN_CSV.is_file() and cfg.TEST_CSV.is_file()),
                                reason="Kaggle CSVs not available (set BDA_TRAIN_CSV / BDA_TEST_CSV)")
needs_clean = pytest.mark.skipif(not cfg.CLEANED_TRANSACTIONS_FILE.exists(), reason="clean_data.py not run yet")
METRICS = cfg.METRICS_DIR / "model_comparison.json"
needs_metrics = pytest.mark.skipif(not METRICS.exists(), reason="train_model.py not run yet")


@needs_csvs
@pytest.mark.parametrize("path", [cfg.TRAIN_CSV, cfg.TEST_CSV], ids=["train", "test"])
def test_kaggle_csv_header_and_labels(path):
    head = pd.read_csv(path, nrows=5000)
    assert head.columns[0].startswith("Unnamed")  # the leading row-index column
    assert EXPECTED_COLUMNS <= set(head.columns)
    assert "street" in head.columns
    assert set(head[cfg.LABEL_COL].unique()) <= {0, 1}


@needs_csvs
def test_train_and_test_share_a_schema():
    assert list(pd.read_csv(cfg.TRAIN_CSV, nrows=0).columns) == list(pd.read_csv(cfg.TEST_CSV, nrows=0).columns)


@needs_clean
def test_cleaned_parquet_keeps_splits_apart():
    t = pq.read_table(cfg.CLEANED_TRANSACTIONS_FILE, columns=[cfg.SPLIT_COL, cfg.TIMESTAMP_COL,
                                                             cfg.TRANSACTION_ID_COL, cfg.LABEL_COL]).to_pandas()
    assert set(t[cfg.SPLIT_COL].astype(str)) == {"train", "test"}
    assert set(t[cfg.LABEL_COL].unique()) == {0, 1}
    assert not t[cfg.TRANSACTION_ID_COL].duplicated().any()
    assert not [c for c in pq.read_schema(cfg.CLEANED_TRANSACTIONS_FILE).names if c == "" or c.startswith("Unnamed")]
    train_ts = t.loc[t[cfg.SPLIT_COL] == "train", cfg.TIMESTAMP_COL]
    test_ts = t.loc[t[cfg.SPLIT_COL] == "test", cfg.TIMESTAMP_COL]
    assert train_ts.max() < test_ts.min()  # the Kaggle test file is strictly later in time


def test_chronological_split_puts_latest_rows_in_validation():
    ts = pd.date_range("2020-01-01", periods=10, freq="h")
    df = pd.DataFrame({cfg.TIMESTAMP_COL: ts[::-1], cfg.LABEL_COL: [0] * 10})  # deliberately unsorted
    fit, val = chronological_split(df)
    assert len(val) == round(10 * cfg.VALIDATION_FRACTION)
    assert fit[cfg.TIMESTAMP_COL].max() < val[cfg.TIMESTAMP_COL].min()


@needs_metrics
@needs_clean
def test_metrics_separate_validation_and_final_test():
    m = json.loads(METRICS.read_text())
    clean = pq.read_table(cfg.CLEANED_TRANSACTIONS_FILE, columns=[cfg.SPLIT_COL, cfg.LABEL_COL]).to_pandas()
    rows = clean[cfg.SPLIT_COL].astype(str).value_counts()
    test_fraud = int(clean.loc[clean[cfg.SPLIT_COL] == "test", cfg.LABEL_COL].sum())
    split = m["split"]
    assert split["fit_rows"] + split["validation_rows"] == rows["train"]
    assert split["fit_period"][1] < split["validation_period"][0]
    assert m["best_model"] == max(m["models"], key=lambda r: r["pr_auc"])["model"]  # chosen on validation
    test = m["test"]
    assert test["rows"] == rows["test"] and test["fraud_rows"] == test_fraud  # the whole fraudTest.csv
    cm = test["confusion_matrix_at_threshold"]
    assert sum(cm.values()) == test["rows"] and cm["tp"] + cm["fn"] == test["fraud_rows"]
    # the threshold applied to the test set is the one picked on validation
    val_best = next(r for r in m["models"] if r["model"] == m["best_model"])
    assert test["decision_threshold"] == val_best["decision_threshold"] == m["decision_threshold"]
