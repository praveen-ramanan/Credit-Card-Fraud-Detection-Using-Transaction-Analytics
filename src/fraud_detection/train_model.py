"""
Fraud detection model: training, evaluation, and scoring.

WHY THIS IS A CLASSIFICATION / IMBALANCED-LEARNING PROBLEM
------------------------------------------------------------
Fraud detection is binary classification (fraud vs legitimate) on a
heavily skewed target - in this dataset, well under 1% of transactions are
fraudulent. A model that always predicts "legitimate" would score ~99%
accuracy while catching zero fraud, which is why accuracy is NOT used as
the headline metric here. Instead:
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

Three models are trained and compared on the held-out test set; the one
with the best PR-AUC (the more informative metric for this imbalance
level) is kept as the "best" model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config import config as cfg
from src.fraud_detection.features import build_feature_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_cleaned_data() -> pd.DataFrame:
    if not cfg.CLEANED_TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.CLEANED_TRANSACTIONS_FILE} not found. Run "
            f"'python src/preprocessing/clean_data.py' first."
        )
    return pd.read_parquet(cfg.CLEANED_TRANSACTIONS_FILE)


def evaluate(model, X_test, y_test, name: str) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_best = (y_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "model": name,
        "precision_at_0.5": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_at_0.5": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_at_0.5": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix_at_0.5": {
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        },
        "best_f1_threshold": best_threshold,
        "precision_at_best_threshold": float(precision_score(y_test, y_pred_best, zero_division=0)),
        "recall_at_best_threshold": float(recall_score(y_test, y_pred_best, zero_division=0)),
        "f1_at_best_threshold": float(f1_score(y_test, y_pred_best, zero_division=0)),
    }
    return metrics, y_proba


def run():
    df = load_cleaned_data()
    X, y, feature_names = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    print(f"Fraud rate: {y.mean() * 100:.3f}% ({y.sum():,} / {len(y):,})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TRAIN_TEST_SPLIT_RATIO, random_state=cfg.RANDOM_STATE, stratify=y,
    )
    print(f"Train: {len(X_train):,} rows ({y_train.sum():,} fraud) | "
          f"Test: {len(X_test):,} rows ({y_test.sum():,} fraud)")

    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=cfg.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            random_state=cfg.RANDOM_STATE, n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        candidates["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=cfg.RANDOM_STATE, n_jobs=-1,
        )
    else:
        print("xgboost not available - skipping (logistic_regression / random_forest only)")

    all_metrics = []
    best_name, best_model, best_pr_auc = None, None, -1.0

    for name, model in candidates.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        metrics, _ = evaluate(model, X_test, y_test, name)
        all_metrics.append(metrics)
        print(f"  precision={metrics['precision_at_0.5']:.3f}  recall={metrics['recall_at_0.5']:.3f}  "
              f"f1={metrics['f1_at_0.5']:.3f}  roc_auc={metrics['roc_auc']:.3f}  pr_auc={metrics['pr_auc']:.3f}")

        if metrics["pr_auc"] > best_pr_auc:
            best_name, best_model, best_pr_auc = name, model, metrics["pr_auc"]

    print(f"\nBest model by PR-AUC: {best_name} (PR-AUC={best_pr_auc:.3f})")

    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.METRICS_DIR / "model_comparison.json", "w") as f:
        json.dump({"models": all_metrics, "best_model": best_name}, f, indent=2)
    print(f"Saved metrics -> {cfg.METRICS_DIR / 'model_comparison.json'}")

    cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": best_model, "feature_names": feature_names, "model_name": best_name},
        cfg.MODELS_DIR / "best_model.joblib",
    )
    print(f"Saved best model -> {cfg.MODELS_DIR / 'best_model.joblib'}")

    # Score the full dataset for downstream NoSQL / visualization stages.
    # NOTE: this includes rows the best model was trained on, so these
    # scores are a demo artifact for the alerting/storage pipeline, not a
    # leak-free evaluation - the evaluation metrics above (computed on the
    # held-out test set only) are what should be quoted as the model's
    # actual performance.
    full_proba = best_model.predict_proba(X)[:, 1]
    scored = df.copy()
    scored["fraud_probability"] = full_proba
    scored["flagged_reason"] = "model_score"

    scored_dir = cfg.RESULTS_DIR / "fraud_detection"
    scored_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(scored_dir / "scored_transactions.csv", index=False)
    print(f"Saved scored transactions -> {scored_dir / 'scored_transactions.csv'}")

    return all_metrics, best_name


if __name__ == "__main__":
    run()
