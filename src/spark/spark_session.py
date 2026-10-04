"""
Shared SparkSession factory.

Centralized here so every Spark script in this project configures the
session the same way, and so the local-vs-cluster master is controlled from
config/config.py (BDA_SPARK_MASTER env var) rather than hardcoded per
script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import config as cfg


def get_spark(app_name: str | None = None, enable_hive: bool = True):
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name or cfg.SPARK_APP_NAME)
        .master(cfg.SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", "8")   # small local dataset -> few partitions
        .config("spark.ui.showConsoleProgress", "false")
    )

    if enable_hive:
        try:
            spark = builder.enableHiveSupport().getOrCreate()
            return spark
        except Exception:
            # No Hive metastore configured on this machine - fall back to
            # plain Spark SQL (temp views instead of a persistent metastore).
            # Functionally equivalent for the read-only analytics queries
            # this project runs.
            pass

    return builder.getOrCreate()
