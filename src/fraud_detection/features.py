"""
Feature engineering for the fraud-detection classifier.

Turns the cleaned transaction DataFrame (src/preprocessing/clean_data.py
output) into a numeric feature matrix + label vector.

Column-by-column decisions for the Kaggle schema:
    used          amt, category, gender, state, city_pop, and features derived
                  from the timestamp (hour, weekend, night), dob (customer age
                  at transaction time) and the customer/merchant coordinates
                  (haversine distance). All are known when a card is swiped.
    excluded      is_fraud (the label), trans_num / unix_time (unique per row,
                  pure identifiers), cc_num / first / last / street / zip
                  (identify a specific customer - the model would memorise
                  ~1,000 individual cards instead of learning fraud patterns),
                  merchant (693 names; memorisation risk, and any merchant
                  not seen in training would be unscoreable), city / job
                  (high-cardinality customer attributes, same reason),
                  raw lat/long (only meaningful as the distance feature),
                  txn_month (fraudTrain covers Jan 2019 - Jun 2020 and
                  fraudTest Jun - Dec 2020, so month would mostly encode
                  "which year" rather than a repeatable pattern).
    kept for      trans_num, cc_num (masked), merchant etc. stay in the data
    reporting     for record identification in alerts/outputs - just never as
                  model inputs.

One-hot encoding uses the categories present in the TRAINING rows only; at
validation/test/inference time align_to_training_columns() reindexes to the
saved training columns, so no vocabulary is learned from evaluation data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config import config as cfg

NUMERIC_FEATURES = [
    cfg.AMOUNT_COL,
    "txn_hour",
    "customer_age",
    "customer_merchant_distance_km",
    cfg.CITY_POP_COL,
]
BOOLEAN_FEATURES = ["is_weekend", "is_night_txn", "is_high_value"]
CATEGORICAL_FEATURES = [cfg.CATEGORY_COL, cfg.STATE_COL, cfg.GENDER_COL]

# Columns that must never reach the model (label + identifiers).
FORBIDDEN_FEATURES = {cfg.LABEL_COL, cfg.TRANSACTION_ID_COL, cfg.CARD_NUM_COL, cfg.UNIX_TIME_COL, cfg.SPLIT_COL}


def build_feature_matrix(df: pd.DataFrame):
    """Returns (X, y, feature_names)."""
    X = build_features(df)
    y = df[cfg.LABEL_COL].astype(int).reset_index(drop=True)
    return X, y, list(X.columns)


def align_to_training_columns(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Reorders X to the training column order. One-hot columns for
    categories absent from X (always the case when scoring a single
    transaction) are added as 0, exactly as get_dummies would have encoded
    them in the full training matrix."""
    return X.reindex(columns=feature_names, fill_value=0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix only - no label required, so usable at inference time."""
    numeric = df[NUMERIC_FEATURES + BOOLEAN_FEATURES].astype("float32").reset_index(drop=True)

    # One-hot encode the low-cardinality categoricals (14 categories, 51
    # states incl. DC, 2 genders). get_dummies keeps the encoding readable
    # for a viva versus a hashing trick. Categories are cast to plain strings
    # first so unused pandas category levels do not create empty columns.
    encoded = pd.get_dummies(df[CATEGORICAL_FEATURES].astype(str), prefix=CATEGORICAL_FEATURES, dtype="float32")

    X = pd.concat([numeric, encoded.reset_index(drop=True)], axis=1).fillna(0)
    leaked = FORBIDDEN_FEATURES & set(X.columns)
    assert not leaked, f"label/identifier columns in the feature matrix: {leaked}"
    return X
