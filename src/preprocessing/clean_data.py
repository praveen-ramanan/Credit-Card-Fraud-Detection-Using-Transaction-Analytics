"""
Data cleaning & preprocessing for the credit-card transaction dataset.

This is the "Data Cleaning & Preprocessing" stage of the project pipeline
(Transaction Dataset -> Ingestion -> HDFS/Storage -> Cleaning -> Hive/Spark
-> ...). It is deliberately implemented in pandas rather than Spark: at the
scale this project runs at locally (hundreds of thousands of rows), pandas
is simpler to read/debug for a viva, and the same cleaned output feeds both
the Hive/MapReduce demos (as CSV) and the Spark stage (which re-reads it as
a DataFrame to demonstrate distributed processing).

Steps:
    1. Load raw CSV, validate expected columns are present.
    2. Drop exact duplicate rows.
    3. Parse timestamp / date-of-birth columns.
    4. Handle missing values (report + drop/impute).
    5. Derive analytics-friendly features (hour, day of week, customer age,
       customer-merchant distance, high-value flag).
    6. Save cleaned data as both Parquet (for Spark) and CSV (for
       Hive/MapReduce, which expect plain text input).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd

from config import config as cfg

EXPECTED_COLUMNS = {
    cfg.TIMESTAMP_COL, cfg.UNIX_TIME_COL, cfg.CARD_NUM_COL, cfg.MERCHANT_COL,
    cfg.CATEGORY_COL, cfg.AMOUNT_COL, cfg.CITY_COL, cfg.STATE_COL,
    cfg.LAT_COL, cfg.LONG_COL, cfg.JOB_COL, cfg.DOB_COL,
    cfg.TRANSACTION_ID_COL, cfg.MERCH_LAT_COL, cfg.MERCH_LONG_COL,
    cfg.LABEL_COL,
}


def load_raw(path: Path = cfg.RAW_TRANSACTIONS_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No raw dataset found at {path}. Run "
            f"'python scripts/generate_synthetic_data.py' to create a demo "
            f"dataset, or place a real dataset there (see docs/DATASET.md)."
        )
    df = pd.read_csv(path)
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
    for col in [cfg.MERCHANT_COL, cfg.CATEGORY_COL, cfg.CITY_COL, cfg.STATE_COL, cfg.JOB_COL]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Drop invalid amounts (non-positive transaction amounts are not
    # meaningful for fraud analytics).
    df = df[df[cfg.AMOUNT_COL] > 0].reset_index(drop=True)

    # --- Derived features -------------------------------------------------
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

    df.attrs["duplicates_dropped"] = duplicates_dropped
    df.attrs["missing_value_report"] = missing_report.to_dict()
    return df


def save(df: pd.DataFrame) -> None:
    cfg.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.CLEANED_TRANSACTIONS_FILE, index=False)
    df.to_csv(cfg.CLEANED_TRANSACTIONS_CSV, index=False)


def run() -> pd.DataFrame:
    raw = load_raw()
    print(f"Loaded {len(raw):,} raw rows from {cfg.RAW_TRANSACTIONS_FILE}")

    cleaned = clean(raw)
    print(f"Dropped {cleaned.attrs.get('duplicates_dropped', 0):,} duplicate transactions")
    if cleaned.attrs.get("missing_value_report"):
        print("Missing values found (before handling):")
        for col, n in cleaned.attrs["missing_value_report"].items():
            print(f"  {col}: {n:,}")
    print(f"Cleaned dataset: {len(cleaned):,} rows, {cleaned.shape[1]} columns")

    save(cleaned)
    print(f"Saved cleaned data -> {cfg.CLEANED_TRANSACTIONS_FILE}")
    print(f"Saved cleaned data -> {cfg.CLEANED_TRANSACTIONS_CSV}")
    return cleaned


if __name__ == "__main__":
    run()
