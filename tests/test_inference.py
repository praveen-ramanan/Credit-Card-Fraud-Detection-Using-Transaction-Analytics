"""Tests for the inference path: validation, encoding alignment, and scoring
with the saved model (skipped if the model has not been trained yet)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.fraud_detection import inference as inf
from src.fraud_detection.features import align_to_training_columns, build_features

pytestmark = pytest.mark.skipif(not inf.MODEL_PATH.exists(), reason="model not trained yet")


@pytest.fixture(scope="module")
def bundle():
    return inf.load_model_bundle()


def _txn(**overrides):
    row = {
        "trans_date_trans_time": "2020-08-15 14:10:00", "amt": 42.50,
        "category": "grocery_pos", "state": "NY", "gender": "M",
        "dob": "1990-01-01", "city_pop": 250000,
        "lat": 40.71, "long": -74.01, "merch_lat": 40.80, "merch_long": -73.95,
    }
    row.update(overrides)
    return row


def test_valid_transaction_is_scored(bundle):
    scored, errors = inf.predict(pd.DataFrame([_txn()]), bundle, threshold=0.9)
    assert errors == []
    assert len(scored) == 1
    p = scored["fraud_probability"].iloc[0]
    assert 0.0 <= p <= 1.0
    assert scored["risk_level"].iloc[0] in {"Low", "Elevated", "High"}


def test_single_row_encoding_matches_training_layout(bundle):
    valid, _ = inf.validate_transactions(pd.DataFrame([_txn(category="travel", state="TX")]), bundle)
    X = inf.prepare_features(valid, bundle)
    assert list(X.columns) == bundle["feature_names"]
    assert X["category_travel"].iloc[0] == 1 and X["state_TX"].iloc[0] == 1
    assert X[[c for c in X.columns if c.startswith("category_")]].sum(axis=1).iloc[0] == 1


@pytest.mark.parametrize("override, fragment", [
    ({"amt": -5}, "amt must be >"),
    ({"amt": "abc"}, "amt is missing or not numeric"),
    ({"lat": 123}, "lat must be <="),
    ({"category": "casino"}, "category must be one of"),
    ({"state": "ZZ"}, "state must be one of"),
    ({"trans_date_trans_time": "not a date"}, "not a valid date"),
    ({"dob": "2030-01-01"}, "date of birth must be before"),
])
def test_invalid_inputs_are_rejected(bundle, override, fragment):
    scored, errors = inf.predict(pd.DataFrame([_txn(**override)]), bundle)
    assert scored.empty
    assert len(errors) == 1 and fragment in errors[0]["error"]


def test_mixed_batch_keeps_valid_rows(bundle):
    df = pd.DataFrame([_txn(), _txn(amt=0), _txn(state="CA")])
    scored, errors = inf.predict(df, bundle)
    assert len(scored) == 2
    assert [e["row"] for e in errors] == [1]


def test_bundle_matches_kaggle_schema(bundle):
    from src.fraud_detection.features import FORBIDDEN_FEATURES
    assert bundle["schema_version"] == inf.SCHEMA_VERSION
    assert not FORBIDDEN_FEATURES & set(bundle["feature_names"])
    assert set(inf.known_categorical_values(bundle)["state"]) >= {"NY", "TX", "CA", "DC"}
    assert 0.0 < inf.load_decision_threshold(bundle) < 1.0


def test_stale_bundle_is_rejected(tmp_path):
    import joblib
    path = tmp_path / "old.joblib"
    joblib.dump({"model": None, "feature_names": ["amt"], "model_name": "xgboost"}, path)  # pre-migration layout
    with pytest.raises(inf.StaleModelError, match="Retrain"):
        inf.load_model_bundle(path)


def test_missing_columns_raise(bundle):
    with pytest.raises(ValueError, match="Missing required columns"):
        inf.validate_transactions(pd.DataFrame([{"amt": 10}]), bundle)


def test_align_adds_missing_dummies_as_zero():
    X = pd.DataFrame({"amt": [1.0], "category_travel": [1]})
    aligned = align_to_training_columns(X, ["category_home", "amt", "category_travel"])
    assert list(aligned.columns) == ["category_home", "amt", "category_travel"]
    assert aligned.iloc[0].tolist() == [0, 1.0, 1]
