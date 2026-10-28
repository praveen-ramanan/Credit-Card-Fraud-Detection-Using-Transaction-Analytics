"""
Runs the same analytics as src/hive/hql/analytics_queries.hql, but via
Spark SQL against a local SparkSession.

WHY THIS FILE EXISTS
---------------------
The project's Hive stage is implemented as real HiveQL in
src/hive/hql/*.hql, meant to run against an actual Hadoop + Hive
installation (see docs/HIVE_SETUP.md) - that is the authoritative
implementation for the "Hive" syllabus unit. Since a Hive cluster is not
assumed to be available on a student's Windows laptop, this script executes
functionally equivalent queries through Spark SQL (which is HiveQL-
compatible for standard SELECT/GROUP BY/window-function syntax) so the same
analytical questions can be answered, and actually run, in local demo mode.
Every query below is commented with which .hql query it corresponds to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import config as cfg
from src.spark.spark_session import get_spark


QUERIES = {
    "01_totals": """
        SELECT COUNT(*) AS total_transactions,
               ROUND(SUM(amt), 2) AS total_transaction_value,
               ROUND(AVG(amt), 2) AS avg_transaction_value
        FROM transactions
    """,
    "02_fraud_vs_legit": """
        SELECT is_fraud,
               COUNT(*) AS txn_count,
               ROUND(SUM(amt), 2) AS total_amount
        FROM transactions
        GROUP BY is_fraud
    """,
    "03_fraud_amount_share": """
        SELECT
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS total_fraud_amount,
            ROUND(SUM(CASE WHEN is_fraud = 0 THEN amt ELSE 0 END), 2) AS total_legit_amount,
            ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END) / SUM(amt), 3) AS fraud_amount_pct
        FROM transactions
    """,
    "04_category_fraud_rate": """
        SELECT category,
               COUNT(*) AS txn_count,
               ROUND(SUM(amt), 2) AS total_amount,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
        FROM transactions
        GROUP BY category
        ORDER BY fraud_rate_pct DESC
    """,
    "05_highest_value_transactions": """
        SELECT trans_num, cc_num, merchant, category, amt, is_fraud, trans_date_trans_time
        FROM transactions
        ORDER BY amt DESC
        LIMIT 20
    """,
    "06_high_value_fraud_rate": """
        SELECT is_high_value,
               COUNT(*) AS txn_count,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
        FROM transactions
        GROUP BY is_high_value
    """,
    "07_hourly_pattern": """
        SELECT txn_hour,
               COUNT(*) AS txn_count,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct
        FROM transactions
        GROUP BY txn_hour
        ORDER BY txn_hour
    """,
    "08_state_distribution": """
        SELECT state,
               COUNT(*) AS txn_count,
               ROUND(SUM(amt), 2) AS total_amount,
               SUM(is_fraud) AS fraud_count
        FROM transactions
        GROUP BY state
        ORDER BY txn_count DESC
    """,
    "09_top_customers_by_spend": """
        SELECT cc_num,
               COUNT(*) AS txn_count,
               ROUND(SUM(amt), 2) AS total_spend,
               SUM(is_fraud) AS fraud_count
        FROM transactions
        GROUP BY cc_num
        ORDER BY total_spend DESC
        LIMIT 20
    """,
    "10_top_merchants_by_fraud_amount": """
        SELECT merchant,
               COUNT(*) AS txn_count,
               SUM(is_fraud) AS fraud_count,
               ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS fraud_amount
        FROM transactions
        GROUP BY merchant
        ORDER BY fraud_amount DESC
        LIMIT 20
    """,
    "11_split_summary": """
        SELECT split,
               COUNT(*) AS txn_count,
               SUM(is_fraud) AS fraud_count,
               ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
               MIN(trans_date_trans_time) AS first_txn,
               MAX(trans_date_trans_time) AS last_txn
        FROM transactions
        GROUP BY split
        ORDER BY split DESC
    """,
}


def run():
    spark = get_spark(app_name="HiveStyleAnalytics")

    if not cfg.CLEANED_TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.CLEANED_TRANSACTIONS_FILE} not found. Run "
            f"'python src/preprocessing/clean_data.py' first."
        )
    df = spark.read.parquet(str(cfg.CLEANED_TRANSACTIONS_FILE))
    df.createOrReplaceTempView("transactions")

    out_dir = cfg.RESULTS_DIR / "hive_analytics"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, sql in QUERIES.items():
        print(f"\n=== Query: {name} ===")
        result = spark.sql(sql)
        result.show(20, truncate=False)
        result.toPandas().to_csv(out_dir / f"{name}.csv", index=False)

    print(f"\nAll query results saved -> {out_dir}")
    spark.stop()


if __name__ == "__main__":
    run()
