"""Unit tests for the cleaning/preprocessing stage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.preprocessing.clean_data import clean, _haversine_km


def _minimal_raw_df():
    return pd.DataFrame({
        "trans_date_trans_time": ["2024-01-01 10:00:00", "2024-01-01 11:00:00",
                                    "2024-01-01 11:00:00", "2024-01-02 02:00:00"],
        "unix_time": [1, 2, 2, 3],
        "cc_num": [111, 222, 222, 333],
        "merchant": ["fraud_A", "fraud_B", "fraud_B", None],
        "category": ["grocery_pos", "travel", "travel", "misc_net"],
        "amt": [50.0, -5.0, 200.0, 700.0],
        "first": ["A", "B", "B", "C"], "last": ["X", "Y", "Y", "Z"],
        "gender": ["M", "F", "F", "M"],
        "city": ["Columbus", "Houston", "Houston", "Albany"],
        "state": ["OH", "TX", "TX", "NY"],
        "zip": [1, 2, 2, 3],
        "lat": [39.96, 29.76, 29.76, 42.65], "long": [-83.0, -95.37, -95.37, -73.76],
        "city_pop": [100000, 200000, 200000, 500000],
        "job": ["Eng", "Doc", "Doc", "Teacher"],
        "dob": ["1990-01-01", "1985-05-05", "1985-05-05", "2000-01-01"],
        "trans_num": ["t1", "t2", "t2", "t4"],
        "merch_lat": [40.0, 29.8, 29.8, 42.7], "merch_long": [-83.1, -95.4, -95.4, -73.8],
        "is_fraud": [0, 0, 0, 1],
    })


def test_clean_drops_duplicates_and_invalid_amounts():
    df = _minimal_raw_df()
    result = clean(df)
    # duplicate trans_num "t2" collapses to one row, which then gets dropped
    # for its negative amount; only t1 and t4 remain valid.
    assert len(result) == 2
    assert (result["amt"] > 0).all()


def test_clean_fills_missing_categoricals():
    df = _minimal_raw_df()
    result = clean(df)
    assert "unknown" in result["merchant"].values


def test_clean_strips_sparkov_merchant_prefix():
    result = clean(_minimal_raw_df())
    assert "A" in result["merchant"].values
    assert not result["merchant"].str.startswith("fraud_").any()


def test_load_raw_drops_unnamed_index_column(tmp_path):
    from src.preprocessing.clean_data import load_raw
    path = tmp_path / "mini.csv"
    _minimal_raw_df().to_csv(path)  # writes the unnamed leading index column, like the Kaggle files
    df = load_raw(path)
    assert not [c for c in df.columns if c == "" or str(c).startswith("Unnamed")]
    assert len(df) == 4


def test_load_raw_missing_file_raises(tmp_path):
    import pytest
    from src.preprocessing.clean_data import load_raw
    with pytest.raises(FileNotFoundError, match="BDA_TRAIN_CSV"):
        load_raw(tmp_path / "nope.csv")


def test_clean_derives_expected_columns():
    df = _minimal_raw_df()
    result = clean(df)
    for col in ["txn_hour", "txn_day_of_week", "is_weekend", "is_night_txn",
                "customer_age", "customer_merchant_distance_km", "is_high_value"]:
        assert col in result.columns


def test_haversine_zero_distance_for_same_point():
    d = _haversine_km(np.array([18.5]), np.array([73.8]), np.array([18.5]), np.array([73.8]))
    assert d[0] < 1e-6


def test_haversine_known_distance_order_of_magnitude():
    # New York (~40.71, -74.01) to Philadelphia (~39.95, -75.17) is roughly 130 km
    d = _haversine_km(np.array([40.71]), np.array([-74.01]), np.array([39.95]), np.array([-75.17]))
    assert 100 < d[0] < 160
