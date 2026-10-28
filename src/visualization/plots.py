"""
Generates the project's core analytics charts as PNG files under
results/figures/. Reads the cleaned Kaggle dataset (fraudTrain.csv +
fraudTest.csv rows) and the model metrics produced by earlier pipeline
stages - run src/preprocessing/clean_data.py and
src/fraud_detection/train_model.py first. Charts 01-07 describe all labelled
transactions; 08 is the final fraudTest.csv evaluation and 09 the
validation-set model comparison.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import config as cfg

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, name):
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved -> {path}")


def load_data():
    if not cfg.CLEANED_TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.CLEANED_TRANSACTIONS_FILE} not found. Run "
            f"'python src/preprocessing/clean_data.py' first."
        )
    columns = [cfg.TIMESTAMP_COL, cfg.AMOUNT_COL, cfg.CATEGORY_COL, cfg.MERCHANT_COL, cfg.LABEL_COL,
               "txn_hour", "is_high_value"]
    return pd.read_parquet(cfg.CLEANED_TRANSACTIONS_FILE, columns=columns)


def plot_fraud_vs_legit_count(df):
    counts = df[cfg.LABEL_COL].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Transaction count")
    ax.set_title("Fraud vs Legitimate Transactions")
    for b in bars:
        ax.annotate(f"{int(b.get_height()):,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom")
    ax.set_yscale("log")
    _save(fig, "01_fraud_vs_legit_count.png")


def plot_fraud_amount_distribution(df):
    fraud_amt = df.loc[df[cfg.LABEL_COL] == 1, cfg.AMOUNT_COL]
    legit_amt = df.loc[df[cfg.LABEL_COL] == 0, cfg.AMOUNT_COL]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(0, np.log10(max(df[cfg.AMOUNT_COL].max(), 10)), 40)
    ax.hist(legit_amt, bins=bins, alpha=0.6, label="Legitimate", color="#4C72B0")
    ax.hist(fraud_amt, bins=bins, alpha=0.6, label="Fraud", color="#C44E52")
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Count")
    ax.set_title("Transaction Amount Distribution: Fraud vs Legitimate")
    ax.legend()
    _save(fig, "02_fraud_amount_distribution.png")


def plot_transactions_over_time(df):
    daily = df.set_index(cfg.TIMESTAMP_COL).resample("D").agg(
        txn_count=(cfg.LABEL_COL, "count"), fraud_count=(cfg.LABEL_COL, "sum")
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(daily.index, daily["txn_count"], color="#4C72B0")
    ax1.set_ylabel("Total transactions/day")
    ax1.set_title("Transactions Over Time")

    ax2.plot(daily.index, daily["fraud_count"], color="#C44E52")
    ax2.set_ylabel("Fraud transactions/day")
    ax2.set_xlabel("Date")
    _save(fig, "03_transactions_over_time.png")


def plot_fraud_rate_by_hour(df):
    hourly = df.groupby("txn_hour")[cfg.LABEL_COL].agg(["count", "sum"])
    hourly["fraud_rate_pct"] = hourly["sum"] / hourly["count"] * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(hourly.index, hourly["fraud_rate_pct"], color="#C44E52")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud Rate by Hour of Day")
    ax.set_xticks(range(0, 24, 2))
    _save(fig, "04_fraud_rate_by_hour.png")


def plot_top_suspicious_categories(df):
    cat = df.groupby(cfg.CATEGORY_COL)[cfg.LABEL_COL].agg(["count", "sum"])
    cat["fraud_rate_pct"] = cat["sum"] / cat["count"] * 100
    cat = cat.sort_values("fraud_rate_pct", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(cat.index, cat["fraud_rate_pct"], color="#DD8452")
    ax.set_xlabel("Fraud rate (%)")
    ax.set_title("Top Suspicious Categories (by fraud rate)")
    _save(fig, "05_top_suspicious_categories.png")


def plot_top_suspicious_merchants(df):
    merch = df.groupby(cfg.MERCHANT_COL)[cfg.LABEL_COL].agg(["count", "sum"])
    merch = merch[merch["count"] >= 20]  # ignore merchants with too few transactions to be meaningful
    merch["fraud_rate_pct"] = merch["sum"] / merch["count"] * 100
    merch = merch.sort_values("fraud_rate_pct", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(merch.index, merch["fraud_rate_pct"], color="#55A868")
    ax.set_xlabel("Fraud rate (%)")
    ax.set_title("Top Suspicious Merchants (by fraud rate, min 20 txns)")
    _save(fig, "06_top_suspicious_merchants.png")


def plot_high_value_analysis(df):
    grp = df.groupby("is_high_value")[cfg.LABEL_COL].agg(["count", "sum"])
    grp["fraud_rate_pct"] = grp["sum"] / grp["count"] * 100
    labels = ["Normal value", "High value (>= {:.0f})".format(cfg.HIGH_VALUE_THRESHOLD)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, grp["fraud_rate_pct"], color=["#4C72B0", "#C44E52"])
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud Rate: High-Value vs Normal Transactions")
    _save(fig, "07_high_value_fraud_rate.png")


def plot_model_evaluation():
    """Confusion matrix + PR/ROC curve for the best saved model, if available."""
    metrics_path = cfg.METRICS_DIR / "model_comparison.json"
    if not metrics_path.exists():
        print("No model metrics found - run src/fraud_detection/train_model.py first. Skipping model plots.")
        return

    import json
    with open(metrics_path) as f:
        data = json.load(f)

    best_name = data["best_model"]
    test = data["test"]
    cm = test["confusion_matrix_at_threshold"]

    fig, ax = plt.subplots(figsize=(4.5, 4))
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Legit", "Pred Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Legit", "Actual Fraud"])
    ax.set_title(f"fraudTest.csv - {best_name}\n(threshold {test['decision_threshold']:.3f}, chosen on validation)",
                 fontsize=10)
    _save(fig, "08_confusion_matrix.png")

    # Bar chart comparing models on precision/recall/f1/roc_auc/pr_auc
    models = data["models"]
    names = [m["model"] for m in models]
    metric_keys = ["precision_at_0.5", "recall_at_0.5", "f1_at_0.5", "roc_auc", "pr_auc"]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, key in enumerate(metric_keys):
        vals = [m[key] for m in models]
        ax.bar(x + i * width, vals, width, label=key)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison (validation: latest 20% of fraudTrain.csv)")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "09_model_comparison.png")


def run():
    df = load_data()
    plot_fraud_vs_legit_count(df)
    plot_fraud_amount_distribution(df)
    plot_transactions_over_time(df)
    plot_fraud_rate_by_hour(df)
    plot_top_suspicious_categories(df)
    plot_top_suspicious_merchants(df)
    plot_high_value_analysis(df)
    plot_model_evaluation()
    print(f"\nAll figures saved under {cfg.FIGURES_DIR}")


if __name__ == "__main__":
    run()
