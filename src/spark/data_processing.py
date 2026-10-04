"""
Spark DataFrame processing stage of the pipeline.

Demonstrates the syllabus's "Spark DataFrames and SQL" unit end to end:
    - reading data into a Spark DataFrame
    - schema inspection
    - filtering / cleaning transformations
    - groupBy + aggregation actions
    - Spark SQL over a registered temp view

This reads the ALREADY-CLEANED data from src/preprocessing/clean_data.py
(Parquet, for efficient columnar reads) - Spark is used here to demonstrate
distributed-style DataFrame processing and SQL, not to redo cleaning that
pandas already did.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import config as cfg
from src.spark.spark_session import get_spark


def load_transactions(spark):
    if not cfg.CLEANED_TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"{cfg.CLEANED_TRANSACTIONS_FILE} not found. Run "
            f"'python src/preprocessing/clean_data.py' first."
        )
    df = spark.read.parquet(str(cfg.CLEANED_TRANSACTIONS_FILE))
    return df


def run():
    spark = get_spark(app_name="TransactionDataProcessing")
    df = load_transactions(spark)

    print("=== Schema ===")
    df.printSchema()

    total_rows = df.count()
    print(f"\nTotal rows loaded into Spark: {total_rows:,}")

    # --- Filtering (DataFrame API) -----------------------------------
    high_value_df = df.filter(df[cfg.AMOUNT_COL] >= cfg.HIGH_VALUE_THRESHOLD)
    print(f"High-value transactions (>= {cfg.HIGH_VALUE_THRESHOLD}): {high_value_df.count():,}")

    # --- GroupBy + aggregation (DataFrame API) ------------------------
    from pyspark.sql import functions as F

    category_summary = (
        df.groupBy(cfg.CATEGORY_COL)
        .agg(
            F.count("*").alias("txn_count"),
            F.round(F.sum(cfg.AMOUNT_COL), 2).alias("total_amount"),
            F.sum(cfg.LABEL_COL).alias("fraud_count"),
            F.round(F.avg(cfg.AMOUNT_COL), 2).alias("avg_amount"),
        )
        .orderBy(F.desc("fraud_count"))
    )
    print("\n=== Transaction summary by category (Spark DataFrame API) ===")
    category_summary.show(20, truncate=False)

    # --- Spark SQL over a registered temp view ------------------------
    df.createOrReplaceTempView("transactions")
    state_summary = spark.sql(f"""
        SELECT {cfg.STATE_COL} AS state,
               COUNT(*) AS txn_count,
               ROUND(SUM({cfg.AMOUNT_COL}), 2) AS total_amount,
               SUM({cfg.LABEL_COL}) AS fraud_count
        FROM transactions
        GROUP BY {cfg.STATE_COL}
        ORDER BY txn_count DESC
    """)
    print("=== Transaction summary by state (Spark SQL) ===")
    state_summary.show(20, truncate=False)

    # Persist a couple of the aggregates as evidence of the run.
    out_dir = cfg.RESULTS_DIR / "spark_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    category_summary.toPandas().to_csv(out_dir / "category_summary.csv", index=False)
    state_summary.toPandas().to_csv(out_dir / "state_summary.csv", index=False)
    print(f"\nSaved Spark aggregation outputs -> {out_dir}")

    spark.stop()


if __name__ == "__main__":
    run()
