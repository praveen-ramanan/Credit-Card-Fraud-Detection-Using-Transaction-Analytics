"""
Fraud detection model: training, validation, final test evaluation, scoring.

WHY THIS IS A CLASSIFICATION / IMBALANCED-LEARNING PROBLEM
------------------------------------------------------------
Fraud detection is binary classification (fraud vs legitimate) on a
heavily skewed target - 0.58% of fraudTrain.csv and 0.39% of fraudTest.csv
are fraudulent. A model that always predicts "legitimate" would score
~99.5% accuracy while catching zero fraud, which is why accuracy is NOT
used as the headline metric here. Instead:
    - Precision (of transactions flagged fraud, how many really were)
      matters because false positives block real customer purchases and
      generate manual-review cost.
    - Recall (of actual fraud, how much was caught) matters because missed
      fraud is direct financial loss.
    - F1 balances the two; ROC-AUC and PR-AUC summarize ranking quality
      across all thresholds. PR-AUC is reported alongside ROC-AUC because
      ROC-AUC can look deceptively good on highly imbalanced data (it is
      dominated by the large true-negative class), while PR-AUC focuses on
      how well the model ranks the rare positive class.

Class imbalance is handled via `class_weight="balanced"` for Logistic
Regression and Random Forest (penalizes mistakes on the minority class
more heavily) and via `scale_pos_weight` for XGBoost (the library's
native equivalent). This avoids the data-duplication cost of oversampling
while still correcting the models' bias toward the majority class.

EVALUATION PROTOCOL (no test-set leakage)
-----------------------------------------
    fraudTrain.csv  -> chronological split: the earliest 80% of transactions
                       (by timestamp) fit the models, the latest 20% are the
                       VALIDATION set. A time-based holdout mirrors deployment
                       (train on the past, score the future) and avoids the
                       optimistic bias of shuffling transactions of the same
                       card/period across both sides.
                       Validation is used for (a) choosing the model (best
                       PR-AUC) and (b) choosing the decision threshold (best
                       F1 on validation).
    fraudTest.csv   -> scored exactly once by the selected model, at the
                       validation-chosen threshold. Nothing is fitted, tuned
                       or selected on it; these are the final reported
                       metrics.
The one-hot vocabulary and the logistic-regression scaler are fitted on
the training portion only. The selected model is not refitted on the
validation rows, so the stored threshold matches the stored model exactly.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import config as cfg
from src.fraud_detection.features import (
    BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    align_to_training_columns, build_feature_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Bumped whenever the feature schema / dataset changes, so inference can
# refuse a stale artifact (e.g. the old synthetic-data model) instead of
# silently scoring with it.
SCHEMA_VERSION = "kaggle-sparkov-v1"

SCORED_COLUMNS = [
    cfg.TRANSACTION_ID_COL, cfg.TIMESTAMP_COL, cfg.CARD_NUM_COL, cfg.MERCHANT_COL, cfg.CATEGORY_COL,
    cfg.AMOUNT_COL, cfg.CITY_COL, cfg.STATE_COL, "customer_age", "customer_merchant_distance_km", cfg.LABEL_COL,
]


def load_cleaned_data(split: str = None, columns: list = None) -> pd.DataFrame:
    """Cleaned rows of one split ("train"/"test") or of both (split=None)."""
    if not cfg.CLEANED_TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.CLEANED_TRANSACTIONS_FILE} not found. Run "
            f"'python src/preprocessing/clean_data.py' first."
        )
    filters = [(cfg.SPLIT_COL, "==", split)] if split else None
    df = pd.read_parquet(cfg.CLEANED_TRANSACTIONS_FILE, columns=columns, filters=filters)
    if split and df.empty:
        raise ValueError(f"No '{split}' rows in {cfg.CLEANED_TRANSACTIONS_FILE}; rerun clean_data.py.")
    return df


def model_input_columns() -> list:
    return NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES + [cfg.LABEL_COL, cfg.TIMESTAMP_COL]


def chronological_split(train_df: pd.DataFrame):
    """(fit_df, validation_df): the latest VALIDATION_FRACTION of the
    training file by timestamp is the validation set."""
    train_df = train_df.sort_values(cfg.TIMESTAMP_COL, kind="stable").reset_index(drop=True)
    cut = int(len(train_df) * (1 - cfg.VALIDATION_FRACTION))
    return train_df.iloc[:cut].reset_index(drop=True), train_df.iloc[cut:].reset_index(drop=True)


def f1_optimal_threshold(y_true, y_proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    best_idx = int(np.argmax(f1_scores[:-1]))  # last PR point has no threshold
    return float(thresholds[best_idx])


def threshold_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(y_true, y_proba, name: str, threshold: float) -> dict:
    """Metrics for one model on one labelled set. `threshold` must come from
    validation data - it is applied here, never re-tuned on this set."""
    at_05 = threshold_metrics(y_true, y_proba, 0.5)
    at_t = threshold_metrics(y_true, y_proba, threshold)
    return {
        "model": name,
        "rows": int(len(y_true)),
        "fraud_rows": int(np.sum(y_true)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision_at_0.5": at_05["precision"],
        "recall_at_0.5": at_05["recall"],
        "f1_at_0.5": at_05["f1"],
        "confusion_matrix_at_0.5": at_05["confusion_matrix"],
        "decision_threshold": at_t["threshold"],
        "precision_at_threshold": at_t["precision"],
        "recall_at_threshold": at_t["recall"],
        "f1_at_threshold": at_t["f1"],
        "confusion_matrix_at_threshold": at_t["confusion_matrix"],
    }


def candidate_models(scale_pos_weight: float) -> dict:
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=14, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=cfg.RANDOM_STATE, n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist",
            random_state=cfg.RANDOM_STATE, n_jobs=-1,
        )
    else:
        print("xgboost not available - skipping (logistic_regression / random_forest only)")
    return candidates


def run():
    t0 = time.time()
    train_df = load_cleaned_data("train", model_input_columns())
    fit_df, val_df = chronological_split(train_df)
    del train_df

    X_fit, y_fit, feature_names = build_feature_matrix(fit_df)
    X_val, y_val, _ = build_feature_matrix(val_df)
    X_val = align_to_training_columns(X_val, feature_names)
    split_info = {
        "train_file": str(cfg.TRAIN_CSV), "test_file": str(cfg.TEST_CSV),
        "fit_rows": int(len(y_fit)), "fit_fraud": int(y_fit.sum()),
        "fit_period": [str(fit_df[cfg.TIMESTAMP_COL].min()), str(fit_df[cfg.TIMESTAMP_COL].max())],
        "validation_rows": int(len(y_val)), "validation_fraud": int(y_val.sum()),
        "validation_period": [str(val_df[cfg.TIMESTAMP_COL].min()), str(val_df[cfg.TIMESTAMP_COL].max())],
    }
    del fit_df, val_df
    print(f"Feature matrix: {X_fit.shape[1]} features")
    print(f"Fit (earliest {1 - cfg.VALIDATION_FRACTION:.0%} of fraudTrain.csv): {len(y_fit):,} rows, "
          f"{int(y_fit.sum()):,} fraud ({100 * y_fit.mean():.3f}%)  {split_info['fit_period'][0]} -> {split_info['fit_period'][1]}")
    print(f"Validation (latest {cfg.VALIDATION_FRACTION:.0%} of fraudTrain.csv): {len(y_val):,} rows, "
          f"{int(y_val.sum()):,} fraud ({100 * y_val.mean():.3f}%)  {split_info['validation_period'][0]} -> {split_info['validation_period'][1]}")

    scale_pos_weight = (len(y_fit) - y_fit.sum()) / max(y_fit.sum(), 1)
    val_metrics, models = [], {}
    for name, model in candidate_models(scale_pos_weight).items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_val)[:, 1]
        m = evaluate(y_val, proba, name, f1_optimal_threshold(y_val, proba))
        m["train_seconds"] = round(time.time() - start, 1)
        val_metrics.append(m)
        models[name] = model
        print(f"  [validation] roc_auc={m['roc_auc']:.4f}  pr_auc={m['pr_auc']:.4f}  "
              f"@0.5 p={m['precision_at_0.5']:.3f} r={m['recall_at_0.5']:.3f}  "
              f"@{m['decision_threshold']:.3f} p={m['precision_at_threshold']:.3f} r={m['recall_at_threshold']:.3f} "
              f"f1={m['f1_at_threshold']:.3f}  ({m['train_seconds']}s)")
    del X_fit, X_val

    best = max(val_metrics, key=lambda m: m["pr_auc"])
    best_name, threshold = best["model"], best["decision_threshold"]
    best_model = models[best_name]
    del models
    print(f"\nSelected by validation PR-AUC: {best_name} (PR-AUC={best['pr_auc']:.4f}, threshold={threshold:.4f})")

    # ---- Final evaluation on fraudTest.csv (touched once, nothing tuned) ----
    test_df = load_cleaned_data("test")
    X_test, y_test, _ = build_feature_matrix(test_df)
    X_test = align_to_training_columns(X_test, feature_names)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_proba, best_name, threshold)
    del X_test
    cm = test_metrics["confusion_matrix_at_threshold"]
    print(f"[TEST fraudTest.csv] rows={len(y_test):,} fraud={int(y_test.sum()):,}  "
          f"roc_auc={test_metrics['roc_auc']:.4f}  pr_auc={test_metrics['pr_auc']:.4f}  "
          f"@{threshold:.3f} precision={test_metrics['precision_at_threshold']:.4f} "
          f"recall={test_metrics['recall_at_threshold']:.4f} f1={test_metrics['f1_at_threshold']:.4f}  "
          f"TP={cm['tp']:,} FP={cm['fp']:,} FN={cm['fn']:,} TN={cm['tn']:,}")

    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.METRICS_DIR / "model_comparison.json", "w") as f:
        json.dump({
            "schema_version": SCHEMA_VERSION,
            "validation_strategy": (f"chronological holdout: latest {cfg.VALIDATION_FRACTION:.0%} of fraudTrain.csv "
                                    f"by timestamp; model chosen by validation PR-AUC, threshold = validation F1 optimum"),
            "split": split_info,
            "models": val_metrics,  # validation-set metrics, one entry per candidate
            "best_model": best_name,
            "decision_threshold": threshold,
            "test": test_metrics,  # final metrics of best_model on fraudTest.csv
        }, f, indent=2)
    print(f"Saved metrics -> {cfg.METRICS_DIR / 'model_comparison.json'}")

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_model, "feature_names": feature_names, "model_name": best_name,
        "decision_threshold": threshold, "schema_version": SCHEMA_VERSION,
        "numeric_features": NUMERIC_FEATURES, "boolean_features": BOOLEAN_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "trained_on": str(cfg.TRAIN_CSV), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }, cfg.MODELS_DIR / "best_model.joblib")
    print(f"Saved best model -> {cfg.MODELS_DIR / 'best_model.joblib'}")

    # Out-of-sample scores for every fraudTest.csv transaction: used by the
    # NoSQL alerts stage, the charts and the dashboard's test-set curves.
    scored = test_df[SCORED_COLUMNS].copy()
    scored["fraud_probability"] = test_proba.astype("float32")
    scored["predicted_fraud"] = (test_proba >= threshold).astype("int8")
    cfg.SCORED_TEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(cfg.SCORED_TEST_FILE, index=False)
    print(f"Saved scored test transactions -> {cfg.SCORED_TEST_FILE}")
    print(f"Total time: {time.time() - t0:.0f}s")
    return val_metrics, best_name


if __name__ == "__main__":
    run()
