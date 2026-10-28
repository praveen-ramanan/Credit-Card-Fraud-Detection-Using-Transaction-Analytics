"""
NoSQL stage: load transaction-level fraud alerts and aggregated analytics
into MongoDB (or its local mongomock fallback) as documents, and run a few
example queries including MongoDB's own aggregation pipeline.

WHY NOSQL HERE
--------------
Relational/Hive-style tables are a good fit for the fixed, uniform schema
of raw transactions. Fraud-analysis *output*, however, is naturally
semi-structured and varies by use case: an alert document might carry a
nested customer profile, a list of matched risk rules, and a model score
all at once, and different alert types (rule-based vs model-based) may not
share identical fields. A document store lets each alert be inserted as-is
without a rigid predefined schema or costly ALTER TABLE migrations, which
matters for a system that will keep evolving its fraud rules. MongoDB also
scales writes horizontally, which suits high-volume transaction streams
better than a single relational instance - relevant even though this
project itself runs at a small demo scale.

This script stores two kinds of documents:
    1. fraud_alerts       - one flexible document per highest-scored
                             fraudTest.csv transaction (out-of-sample scores
                             from src/fraud_detection/train_model.py), with a
                             nested customer/merchant sub-document and both
                             the model's prediction and the actual label.
    2. category_analytics - aggregated category-level analytics (reusing
                             the output already computed by the Spark/Hive
                             stage), for fast dashboard-style lookups.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from config import config as cfg
from src.nosql.mongo_client import get_mongo_db


def _row_to_alert_doc(row: pd.Series) -> dict:
    return {
        "trans_num": row.get(cfg.TRANSACTION_ID_COL),
        "timestamp": str(row.get(cfg.TIMESTAMP_COL)),
        "amount": float(row.get(cfg.AMOUNT_COL, 0.0)),
        "category": row.get(cfg.CATEGORY_COL),
        "customer": {
            "cc_num_last4": str(row.get(cfg.CARD_NUM_COL))[-4:],
            "city": row.get(cfg.CITY_COL),
            "state": row.get(cfg.STATE_COL),
            "age": int(row["customer_age"]) if pd.notna(row.get("customer_age")) else None,
        },
        "merchant": {
            "name": row.get(cfg.MERCHANT_COL),
            "distance_km": float(row.get("customer_merchant_distance_km", 0.0)),
        },
        "actual_label": int(row.get(cfg.LABEL_COL, 0)),
        "predicted_fraud_probability": float(row.get("fraud_probability", 0.0)),
        "predicted_fraud": int(row.get("predicted_fraud", 0)),
        "flagged_reason": "above_decision_threshold" if row.get("predicted_fraud", 0) else "high_model_score",
    }


def load_fraud_alerts(db, scored_file: Path = cfg.SCORED_TEST_FILE, top_n: int = 500):
    """Load the highest fraud-probability test transactions as alert documents."""
    if not scored_file.exists():
        print(f"[nosql] {scored_file} not found - run fraud detection scoring first. Skipping alerts load.")
        return 0

    df = pd.read_parquet(scored_file).nlargest(top_n, "fraud_probability")

    docs = [_row_to_alert_doc(row) for _, row in df.iterrows()]
    collection = db["fraud_alerts"]
    collection.delete_many({})
    if docs:
        collection.insert_many(docs)
    return len(docs)


def load_category_analytics(db, category_csv: Path):
    if not category_csv.exists():
        print(f"[nosql] {category_csv} not found - run the Spark/Hive analytics stage first. Skipping.")
        return 0

    df = pd.read_csv(category_csv)
    docs = df.to_dict(orient="records")
    collection = db["category_analytics"]
    collection.delete_many({})
    if docs:
        collection.insert_many(docs)
    return len(docs)


def run_example_queries(db):
    print("\n=== Example NoSQL queries ===")

    alerts = db["fraud_alerts"]

    top5 = list(alerts.find().sort("predicted_fraud_probability", -1).limit(5))
    print(f"\nTop 5 highest-risk alerts (find + sort + limit): {len(top5)} returned")
    for doc in top5:
        print(f"  {doc['trans_num']}  amt={doc['amount']:.2f}  "
              f"p_fraud={doc['predicted_fraud_probability']:.3f}  "
              f"state={doc['customer']['state']}")

    # MongoDB aggregation pipeline: count + average amount of alerts, grouped
    # by category - demonstrates MongoDB's own analytics engine, distinct
    # from the Spark/Hive aggregation already performed upstream.
    pipeline = [
        {"$group": {
            "_id": "$category",
            "alert_count": {"$sum": 1},
            "avg_amount": {"$avg": "$amount"},
        }},
        {"$sort": {"alert_count": -1}},
    ]
    agg_result = list(alerts.aggregate(pipeline))
    print(f"\nAggregation pipeline - alerts by category ({len(agg_result)} groups):")
    for row in agg_result[:10]:
        print(f"  {row['_id']:<20} alerts={row['alert_count']:<5} avg_amount={row['avg_amount']:.2f}")


def run():
    db, backend = get_mongo_db()
    print(f"[nosql] Using backend: {backend}")

    category_csv = cfg.RESULTS_DIR / "spark_output" / "category_summary.csv"

    n_alerts = load_fraud_alerts(db)
    print(f"[nosql] Loaded {n_alerts:,} documents into 'fraud_alerts'")

    n_cat = load_category_analytics(db, category_csv)
    print(f"[nosql] Loaded {n_cat:,} documents into 'category_analytics'")

    if n_alerts:
        run_example_queries(db)


if __name__ == "__main__":
    run()
