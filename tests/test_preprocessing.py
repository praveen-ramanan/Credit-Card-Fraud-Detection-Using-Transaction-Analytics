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
        "merchant": ["A", "B", "B", None],
        "category": ["grocery_pos", "travel", "travel", "misc_net"],
        "amt": [50.0, -5.0, 200.0, 700.0],
        "first": ["A", "B", "B", "C"], "last": ["X", "Y", "Y", "Z"],
        "gender": ["M", "F", "F", "M"],
        "city": ["Pune", "Chennai", "Chennai", "Mumbai"],
        "state": ["MH", "TN", "TN", "MH"],
        "zip": [1, 2, 2, 3],
        "lat": [18.5, 13.0, 13.0, 19.0], "long": [73.8, 80.2, 80.2, 72.8],
        "city_pop": [100000, 200000, 200000, 500000],
        "job": ["Eng", "Doc", "Doc", "Teacher"],
        "dob": ["1990-01-01", "1985-05-05", "1985-05-05", "2000-01-01"],
        "trans_num": ["t1", "t2", "t2", "t4"],
        "merch_lat": [18.6, 13.1, 13.1, 19.1], "merch_long": [73.9, 80.3, 80.3, 72.9],
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
    # Mumbai (~19.07, 72.87) to Pune (~18.52, 73.85) is roughly 120-150 km
    d = _haversine_km(np.array([19.07]), np.array([72.87]), np.array([18.52]), np.array([73.85]))
    assert 80 < d[0] < 200
