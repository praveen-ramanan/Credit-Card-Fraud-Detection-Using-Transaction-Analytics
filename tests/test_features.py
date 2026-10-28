"""Unit tests for fraud-detection feature engineering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.fraud_detection.features import build_feature_matrix


def _cleaned_like_df(n=10):
    return pd.DataFrame({
        "amt": [10.0 * i for i in range(1, n + 1)],
        "txn_hour": [i % 24 for i in range(n)],
        "customer_age": [30 + i for i in range(n)],
        "customer_merchant_distance_km": [i * 1.5 for i in range(n)],
        "city_pop": [10000 * (i + 1) for i in range(n)],
        "is_weekend": [i % 2 == 0 for i in range(n)],
        "is_night_txn": [i % 3 == 0 for i in range(n)],
        "is_high_value": [i > 5 for i in range(n)],
        "category": ["grocery_pos", "travel"] * (n // 2),
        "state": ["NY", "TX"] * (n // 2),
        "gender": ["M", "F"] * (n // 2),
        "is_fraud": [0, 1] * (n // 2),
    })


def test_build_feature_matrix_shapes_and_types():
    df = _cleaned_like_df()
    X, y, names = build_feature_matrix(df)

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert set(names) == set(X.columns)
    # one-hot columns for category/state/gender must exist
    assert any(c.startswith("category_") for c in names)
    assert any(c.startswith("state_") for c in names)
    # no missing values after feature engineering
    assert X.isna().sum().sum() == 0
    # the label never leaks into the features
    assert "is_fraud" not in names


def test_build_feature_matrix_label_is_binary():
    df = _cleaned_like_df()
    _, y, _ = build_feature_matrix(df)
    assert set(y.unique()).issubset({0, 1})
