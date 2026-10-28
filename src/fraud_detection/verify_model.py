"""
Verifies the saved model without retraining it:
    1. results/models/best_model.joblib loads, has the expected bundle keys
       and the current schema version (a stale synthetic-data model fails).
    2. Its feature names/order match what the current feature pipeline
       builds from the fraudTrain.csv rows, and contain no label/identifier.
    3. Re-scoring the full fraudTest.csv reproduces the final test metrics
       stored in results/metrics/model_comparison.json.
    4. A sample transaction can be scored through the inference path.

Exit code 0 = all checks passed, 1 = a check failed.
Usage: python src/fraud_detection/verify_model.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config import config as cfg
from src.fraud_detection import inference as inf
from src.fraud_detection.features import FORBIDDEN_FEATURES, build_feature_matrix
from src.fraud_detection.train_model import chronological_split, load_cleaned_data, model_input_columns

METRIC_KEYS = ["roc_auc", "pr_auc", "precision_at_threshold", "recall_at_threshold", "f1_at_threshold",
               "precision_at_0.5", "recall_at_0.5", "f1_at_0.5"]


def run() -> bool:
    ok = True
    bundle = inf.load_model_bundle()
    print(f"[ok] Loaded {inf.MODEL_PATH} ({bundle['model_name']}, {len(bundle['feature_names'])} features, "
          f"schema {bundle['schema_version']}, threshold {inf.load_decision_threshold(bundle):.4f})")

    fit_df, _ = chronological_split(load_cleaned_data("train", model_input_columns()))
    _, _, names = build_feature_matrix(fit_df)
    del fit_df
    if names == bundle["feature_names"] and not FORBIDDEN_FEATURES & set(names):
        print("[ok] Feature names and order match the current feature pipeline (no label/identifier columns)")
    else:
        ok = False
        print("[FAIL] Feature names differ from the current pipeline - retrain the model")

    recomputed = inf.evaluate_saved_model(bundle)
    with open(inf.METRICS_PATH) as f:
        stored = json.load(f)["test"]
    for k in METRIC_KEYS:
        match = abs(recomputed[k] - stored[k]) < 1e-6
        ok &= match
        print(f"[{'ok' if match else 'FAIL'}] test {k}: recomputed={recomputed[k]:.4f} stored={stored[k]:.4f}")
    match = recomputed["confusion_matrix_at_threshold"] == stored["confusion_matrix_at_threshold"]
    ok &= match
    print(f"[{'ok' if match else 'FAIL'}] test confusion matrix: {recomputed['confusion_matrix_at_threshold']}")

    sample = pd.DataFrame([{
        cfg.TIMESTAMP_COL: "2020-08-15 14:10:00", cfg.AMOUNT_COL: 42.50,
        cfg.CATEGORY_COL: "grocery_pos", cfg.STATE_COL: "NY", cfg.GENDER_COL: "M",
        cfg.DOB_COL: "1990-01-01", cfg.CITY_POP_COL: 250000,
        cfg.LAT_COL: 40.71, cfg.LONG_COL: -74.01, cfg.MERCH_LAT_COL: 40.80, cfg.MERCH_LONG_COL: -73.95,
    }])
    scored, errors = inf.predict(sample, bundle)
    if errors or scored.empty:
        ok = False
        print(f"[FAIL] Sample prediction rejected: {errors}")
    else:
        r = scored.iloc[0]
        print(f"[ok] Sample prediction: p_fraud={r['fraud_probability']:.3f} risk={r['risk_level']}")

    print("\nAll checks passed." if ok else "\nSome checks FAILED.")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
