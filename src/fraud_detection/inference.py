"""
Inference for the trained fraud-detection model.

Scores new transactions with results/models/best_model.joblib (written by
train_model.py) using exactly the training-time preprocessing:
    raw fields -> clean_data.add_derived_features -> features.build_features
    -> aligned to the saved training column order (feature_names).

The aligning step matters: training one-hot encodes category/state/gender
with get_dummies over the whole dataset, so a single transaction on its own
would produce only its own dummy columns, in a different order. Reindexing
to the saved feature_names reproduces the training encoding.

Inputs are validated first - required columns, parseable dates, numeric
ranges, and categorical values the model actually saw during training (an
unseen state or category would silently become an all-zero one-hot row,
which the model never learned from, so such rows are rejected instead).

The bundle records the schema version it was trained with; a bundle from an
older schema (e.g. the former synthetic-data model) is refused with a
"retrain" message rather than silently used.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd

from config import config as cfg
from src.fraud_detection.features import (
    CATEGORICAL_FEATURES, align_to_training_columns, build_features,
)
from src.fraud_detection.train_model import SCHEMA_VERSION
from src.preprocessing.clean_data import add_derived_features

MODEL_PATH = cfg.MODELS_DIR / "best_model.joblib"
METRICS_PATH = cfg.METRICS_DIR / "model_comparison.json"

# Raw transaction fields the model's features are derived from.
REQUIRED_INPUT_COLUMNS = [
    cfg.TIMESTAMP_COL, cfg.AMOUNT_COL, cfg.CATEGORY_COL, cfg.STATE_COL,
    cfg.GENDER_COL, cfg.DOB_COL, cfg.CITY_POP_COL,
    cfg.LAT_COL, cfg.LONG_COL, cfg.MERCH_LAT_COL, cfg.MERCH_LONG_COL,
]

NUMERIC_RANGES = {
    cfg.AMOUNT_COL: (0.0, None),        # must be > 0 (checked strictly below)
    cfg.CITY_POP_COL: (0, None),
    cfg.LAT_COL: (-90.0, 90.0),
    cfg.LONG_COL: (-180.0, 180.0),
    cfg.MERCH_LAT_COL: (-90.0, 90.0),
    cfg.MERCH_LONG_COL: (-180.0, 180.0),
}


class StaleModelError(ValueError):
    """The saved model was trained on a different dataset/feature schema."""


def load_model_bundle(path: Path = MODEL_PATH) -> dict:
    """Returns the bundle saved by train_model.py ("model", "feature_names",
    "model_name", "decision_threshold", "schema_version", ...)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Train it with "
            f"'python src/fraud_detection/train_model.py' (or "
            f"notebooks/train_model_colab.ipynb)."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or bundle.get("schema_version") != SCHEMA_VERSION:
        found = bundle.get("schema_version", "none") if isinstance(bundle, dict) else type(bundle).__name__
        raise StaleModelError(
            f"{path} was trained with feature schema '{found}', but this code expects "
            f"'{SCHEMA_VERSION}' (Kaggle fraudTrain.csv). Retrain it with "
            f"'python src/fraud_detection/train_model.py'."
        )
    missing = {"model", "feature_names", "model_name", "decision_threshold"} - set(bundle)
    if missing:
        raise ValueError(f"{path} is not a valid model bundle (missing keys: {sorted(missing)}).")
    return bundle


def load_decision_threshold(bundle: dict) -> float:
    """Decision threshold chosen on the validation set at training time
    (F1-optimal on the latest 20% of fraudTrain.csv - never on fraudTest)."""
    return float(bundle["decision_threshold"])


def known_categorical_values(bundle: dict) -> dict:
    """Category/state/gender values the model was trained on, recovered
    from the one-hot column names (e.g. 'state_NY' -> 'NY')."""
    values = {}
    for col in CATEGORICAL_FEATURES:
        prefix = f"{col}_"
        values[col] = sorted(n[len(prefix):] for n in bundle["feature_names"] if n.startswith(prefix))
    return values


def validate_transactions(df: pd.DataFrame, bundle: dict):
    """Returns (valid_df, errors). valid_df has parsed dates; errors is a
    list of {"row": original index, "error": message} for rejected rows.
    Raises ValueError if required columns are missing entirely."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Required: {REQUIRED_INPUT_COLUMNS}")

    work = df.copy()
    problems = pd.Series([[] for _ in range(len(work))], index=work.index)

    def flag(mask, message):
        for idx in work.index[mask]:
            problems[idx].append(message)

    work[cfg.TIMESTAMP_COL] = pd.to_datetime(work[cfg.TIMESTAMP_COL], errors="coerce")
    work[cfg.DOB_COL] = pd.to_datetime(work[cfg.DOB_COL], errors="coerce")
    flag(work[cfg.TIMESTAMP_COL].isna(), f"{cfg.TIMESTAMP_COL} is missing or not a valid date/time")
    flag(work[cfg.DOB_COL].isna(), f"{cfg.DOB_COL} is missing or not a valid date")
    flag(work[cfg.DOB_COL] >= work[cfg.TIMESTAMP_COL], "date of birth must be before the transaction time")

    for col, (lo, hi) in NUMERIC_RANGES.items():
        work[col] = pd.to_numeric(work[col], errors="coerce")
        flag(work[col].isna(), f"{col} is missing or not numeric")
        if lo is not None:
            bad = work[col] <= lo if col == cfg.AMOUNT_COL else work[col] < lo
            flag(bad, f"{col} must be {'>' if col == cfg.AMOUNT_COL else '>='} {lo}")
        if hi is not None:
            flag(work[col] > hi, f"{col} must be <= {hi}")

    for col, allowed in known_categorical_values(bundle).items():
        work[col] = work[col].astype("string").str.strip()
        flag(~work[col].isin(allowed), f"{col} must be one of the values seen in training: {allowed}")

    errors = [{"row": idx, "error": "; ".join(p)} for idx, p in problems.items() if p]
    valid = work[problems.map(len) == 0].copy()
    if len(valid):
        valid[cfg.CITY_POP_COL] = valid[cfg.CITY_POP_COL].astype(int)
    return valid, errors


def prepare_features(valid_df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Applies training-time feature derivation + encoding to validated rows."""
    derived = add_derived_features(valid_df.copy())
    X = build_features(derived)
    return align_to_training_columns(X, bundle["feature_names"])


def elevated_cutoff(threshold: float) -> float:
    """Lower edge of the "Elevated" band: 0.5, or half the decision threshold
    if the threshold itself is at or below 0.5."""
    return 0.5 if threshold > 0.5 else threshold / 2


def risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "High"
    if probability >= elevated_cutoff(threshold):
        return "Elevated"
    return "Low"


def predict(df: pd.DataFrame, bundle: dict, threshold: float = None):
    """Validates and scores transactions. Returns (scored_df, errors).
    scored_df is the validated input plus derived features and
    fraud_probability / predicted_fraud / risk_level columns."""
    if threshold is None:
        threshold = load_decision_threshold(bundle)
    valid, errors = validate_transactions(df, bundle)
    if valid.empty:
        return valid, errors

    X = prepare_features(valid, bundle)
    proba = bundle["model"].predict_proba(X)[:, 1]

    scored = add_derived_features(valid.copy())
    scored["fraud_probability"] = proba
    scored["predicted_fraud"] = (proba >= threshold).astype(int)
    scored["risk_level"] = [risk_level(p, threshold) for p in proba]
    return scored, errors


def score_test_set(bundle: dict = None):
    """(y_test, y_proba): the saved model re-scored on the full fraudTest.csv
    (cleaned), with the training-time feature pipeline."""
    from src.fraud_detection.features import build_feature_matrix
    from src.fraud_detection.train_model import load_cleaned_data, model_input_columns

    bundle = bundle or load_model_bundle()
    X, y, _ = build_feature_matrix(load_cleaned_data("test", model_input_columns()))
    X = align_to_training_columns(X, bundle["feature_names"])
    return y.to_numpy(), bundle["model"].predict_proba(X)[:, 1]


def evaluate_saved_model(bundle: dict = None) -> dict:
    """Final-test metrics of the saved model on fraudTest.csv, at its stored
    (validation-chosen) threshold."""
    from src.fraud_detection.train_model import evaluate

    bundle = bundle or load_model_bundle()
    y, proba = score_test_set(bundle)
    return evaluate(y, proba, bundle["model_name"], load_decision_threshold(bundle))


if __name__ == "__main__":
    b = load_model_bundle()
    sample = pd.DataFrame([{
        cfg.TIMESTAMP_COL: "2020-08-15 23:30:00", cfg.AMOUNT_COL: 899.99,
        cfg.CATEGORY_COL: "shopping_net", cfg.STATE_COL: "TX", cfg.GENDER_COL: "F",
        cfg.DOB_COL: "1985-03-12", cfg.CITY_POP_COL: 120000,
        cfg.LAT_COL: 29.76, cfg.LONG_COL: -95.37, cfg.MERCH_LAT_COL: 30.27, cfg.MERCH_LONG_COL: -97.74,
    }])
    scored, errs = predict(sample, b)
    print(f"Model: {b['model_name']} (threshold {load_decision_threshold(b):.4f})")
    print(scored[["fraud_probability", "predicted_fraud", "risk_level"]] if not scored.empty else errs)
