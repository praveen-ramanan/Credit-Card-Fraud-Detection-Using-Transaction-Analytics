"""
Data cleaning & preprocessing for the Kaggle credit-card transaction dataset.

This is the "Data Cleaning & Preprocessing" stage of the project pipeline
(Transaction Dataset -> Ingestion -> HDFS/Storage -> Cleaning -> Hive/Spark
-> ...). It is implemented in pandas (with the pyarrow CSV reader, which
loads the 1.3M-row training file in about a second); the cleaned Parquet
output feeds the Spark / Spark SQL stages, the model, the charts and the
dashboard.

Input: fraudTrain.csv and fraudTest.csv, read in place from the paths in
config/config.py (TRAIN_CSV / TEST_CSV, overridable via BDA_TRAIN_CSV /
BDA_TEST_CSV). There is no fallback dataset - a missing file is an error.

Steps (applied to each file independently, row by row - nothing is learned
from the data here, so cleaning the test file cannot leak into training):
    1. Load the CSV, drop the unnamed leading row-index column, validate the
       expected columns are present.
    2. Drop duplicate transactions (same trans_num).
    3. Parse timestamp / date-of-birth columns.
    4. Handle missing values (report + drop/impute).
    5. Strip the "fraud_" prefix the Sparkov generator puts on EVERY merchant
       name (it is not a label - legitimate and fraudulent rows both have it).
    6. Derive analytics/model features (hour, day of week, customer age,
       customer-merchant distance, high-value flag).
    7. Tag each row with its split ("train"/"test") and save both as one
       Parquet file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import config as cfg

EXPECTED_COLUMNS = {
    cfg.TIMESTAMP_COL, cfg.UNIX_TIME_COL, cfg.CARD_NUM_COL, cfg.MERCHANT_COL,
    cfg.CATEGORY_COL, cfg.AMOUNT_COL, cfg.GENDER_COL, cfg.CITY_COL, cfg.STATE_COL, cfg.CITY_POP_COL,
    cfg.LAT_COL, cfg.LONG_COL, cfg.JOB_COL, cfg.DOB_COL,
    cfg.TRANSACTION_ID_COL, cfg.MERCH_LAT_COL, cfg.MERCH_LONG_COL,
    cfg.LABEL_COL,
}


def check_raw_file(path: Path) -> None:
    """Raises FileNotFoundError / PermissionError with a useful message if a
    Kaggle CSV is missing or unreadable."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Kaggle dataset file not found: {path}. Download fraudTrain.csv / "
            f"fraudTest.csv (see docs/DATASET.md) and set BDA_TRAIN_CSV / "
            f"BDA_TEST_CSV if they are not at the default location."
        )
    with open(path, "rb") as f:
        f.read(1)


def load_raw(path: Path) -> pd.DataFrame:
    check_raw_file(path)
    df = pd.read_csv(path, engine="pyarrow")
    # The Kaggle files start with an unnamed row-index column ("" or
    # "Unnamed: 0" depending on the reader). It is a file artifact, not data.
    df = df.drop(columns=[c for c in df.columns if c == "" or str(c).startswith("Unnamed")])
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing expected columns: {sorted(missing)}. "
            f"See docs/DATASET.md for the required schema."
        )
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/long points, in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=[cfg.TRANSACTION_ID_COL]).reset_index(drop=True)
    duplicates_dropped = before - len(df)

    # Parse timestamps
    df[cfg.TIMESTAMP_COL] = pd.to_datetime(df[cfg.TIMESTAMP_COL], errors="coerce")
    df[cfg.DOB_COL] = pd.to_datetime(df[cfg.DOB_COL], errors="coerce")

    # Report and handle missing values. For this dataset, rows with a
    # missing amount or missing label can't be used downstream, so they are
    # dropped; missing categorical fields are filled with "unknown" rather
    # than dropped, to avoid losing otherwise-valid transactions.
    missing_report = df.isna().sum()
    missing_report = missing_report[missing_report > 0]

    df = df.dropna(subset=[cfg.AMOUNT_COL, cfg.LABEL_COL, cfg.TIMESTAMP_COL])
    df = df[df[cfg.LABEL_COL].isin([0, 1])]
    df[cfg.LABEL_COL] = df[cfg.LABEL_COL].astype("int8")
    for col in [cfg.MERCHANT_COL, cfg.CATEGORY_COL, cfg.CITY_COL, cfg.STATE_COL, cfg.JOB_COL]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    df[cfg.MERCHANT_COL] = df[cfg.MERCHANT_COL].astype(str).str.removeprefix("fraud_")

    # Drop invalid amounts (non-positive transaction amounts are not
    # meaningful for fraud analytics).
    df = df[df[cfg.AMOUNT_COL] > 0].reset_index(drop=True)

    df = add_derived_features(df)

    df.attrs["duplicates_dropped"] = duplicates_dropped
    df.attrs["missing_value_report"] = missing_report.to_dict()
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the analytics/model features. Expects parsed TIMESTAMP_COL and
    DOB_COL datetimes. Shared by clean() and the inference path
    (src/fraud_detection/inference.py) so training and prediction derive
    features identically."""
    df["txn_hour"] = df[cfg.TIMESTAMP_COL].dt.hour
    df["txn_day_of_week"] = df[cfg.TIMESTAMP_COL].dt.day_name()
    df["txn_month"] = df[cfg.TIMESTAMP_COL].dt.month
    df["is_weekend"] = df[cfg.TIMESTAMP_COL].dt.dayofweek >= 5
    df["is_night_txn"] = df["txn_hour"].between(0, 4)

    df["customer_age"] = (
        (df[cfg.TIMESTAMP_COL] - df[cfg.DOB_COL]).dt.days // 365
    ).clip(lower=0, upper=110)

    df["customer_merchant_distance_km"] = _haversine_km(
        df[cfg.LAT_COL], df[cfg.LONG_COL], df[cfg.MERCH_LAT_COL], df[cfg.MERCH_LONG_COL]
    ).round(2)

    df["is_high_value"] = df[cfg.AMOUNT_COL] >= cfg.HIGH_VALUE_THRESHOLD
    return df


def save(df: pd.DataFrame) -> None:
    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.CLEANED_TRANSACTIONS_FILE, index=False)


def run() -> pd.DataFrame:
    for path in cfg.RAW_SPLITS.values():
        check_raw_file(path)

    parts = []
    for split, path in cfg.RAW_SPLITS.items():
        raw = load_raw(path)
        print(f"[{split}] Loaded {len(raw):,} raw rows x {raw.shape[1]} columns from {path}")
        cleaned = clean(raw)
        del raw
        print(f"[{split}] Dropped {cleaned.attrs.get('duplicates_dropped', 0):,} duplicate transactions")
        for col, n in cleaned.attrs.get("missing_value_report", {}).items():
            print(f"[{split}]   missing {col}: {n:,}")
        print(f"[{split}] Cleaned: {len(cleaned):,} rows, fraud rate "
              f"{100 * cleaned[cfg.LABEL_COL].mean():.3f}% ({int(cleaned[cfg.LABEL_COL].sum()):,} fraud)")
        cleaned[cfg.SPLIT_COL] = split
        parts.append(cleaned)

    overlap = set(parts[0][cfg.TRANSACTION_ID_COL]) & set(parts[1][cfg.TRANSACTION_ID_COL])
    if overlap:
        raise ValueError(f"{len(overlap):,} trans_num values appear in both the train and test files.")

    df = pd.concat(parts, ignore_index=True)
    del parts
    for col in [cfg.SPLIT_COL, cfg.CATEGORY_COL, cfg.STATE_COL, cfg.GENDER_COL, "txn_day_of_week"]:
        df[col] = df[col].astype("category")
    save(df)
    print(f"Saved {len(df):,} cleaned rows ({df.shape[1]} columns) -> {cfg.CLEANED_TRANSACTIONS_FILE}")
    return df


if __name__ == "__main__":
    run()
