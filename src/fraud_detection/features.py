"""
Feature engineering for the fraud-detection classifier.

Turns the cleaned transaction DataFrame (src/preprocessing/clean_data.py
output) into a numeric feature matrix + label vector. Columns that would
leak identity or the label itself (trans_num, cc_num, names, raw
timestamps, dob, is_fraud) are excluded from the feature set - they either
carry no generalizable signal (a specific card number) or would let the
model "cheat".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config import config as cfg

NUMERIC_FEATURES = [
    cfg.AMOUNT_COL,
    "txn_hour",
    "txn_month",
    "customer_age",
    "customer_merchant_distance_km",
    cfg.CITY_POP_COL,
]
BOOLEAN_FEATURES = ["is_weekend", "is_night_txn", "is_high_value"]
CATEGORICAL_FEATURES = [cfg.CATEGORY_COL, cfg.STATE_COL, cfg.GENDER_COL]


def build_feature_matrix(df: pd.DataFrame):
    """Returns (X, y, feature_names)."""
    work = df.copy()

    for col in BOOLEAN_FEATURES:
        work[col] = work[col].astype(int)

    # One-hot encode low-cardinality categoricals. `category` and `gender`
    # are low cardinality; `state` is also small here. get_dummies keeps
    # this readable/explainable for a viva versus a hashing trick.
    encoded = pd.get_dummies(work[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)

    X = pd.concat(
        [work[NUMERIC_FEATURES + BOOLEAN_FEATURES].reset_index(drop=True),
         encoded.reset_index(drop=True)],
        axis=1,
    )
    X = X.fillna(0)
    y = work[cfg.LABEL_COL].astype(int).reset_index(drop=True)

    return X, y, list(X.columns)
