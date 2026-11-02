"""
Cached data access for the Streamlit dashboard (app.py).

Everything shown in the dashboard is read from artifacts the pipeline
already produced - nothing is recomputed from scratch or hardcoded:
    cleaned dataset      data/processed/transactions_clean.parquet (Kaggle
                         fraudTrain.csv + fraudTest.csv rows, `split` column),
                         or its slim copy deploy/transactions_dashboard.parquet
    Hive/Spark-SQL       results/hive_analytics/*.csv
    Spark                results/spark_output/*.csv
    MapReduce            results/mapreduce_output/*.tsv
    model + metrics      results/models/best_model.joblib, results/metrics/model_comparison.json
    test-set scores      results/fraud_detection/scored_test_transactions.parquet
    fraud alerts         MongoDB 'fraud_alerts' (or the mongomock fallback), loaded
                         by src/nosql/store_analytics.py from the test-set scores

The model and test-set scores also fall back to their deploy/ copies (see the
"Dashboard artifacts" section of config/config.py).

Loaders raise MissingArtifact with the command that produces the file, so
the dashboard can show a useful message instead of a stack trace.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import pyarrow as pa
import streamlit as st

from config import config as cfg
from src.fraud_detection import inference as inf

SPLIT_LABELS = {"train": "Training (fraudTrain.csv)", "test": "Test (fraudTest.csv)"}

# Only the columns the dashboard uses are loaded (1.85M rows).
DASHBOARD_COLUMNS = [
    cfg.TIMESTAMP_COL, cfg.CARD_NUM_COL, cfg.MERCHANT_COL, cfg.CATEGORY_COL, cfg.AMOUNT_COL, cfg.GENDER_COL,
    cfg.STATE_COL, cfg.LAT_COL, cfg.LONG_COL, cfg.CITY_POP_COL, cfg.DOB_COL, cfg.TRANSACTION_ID_COL,
    cfg.MERCH_LAT_COL, cfg.MERCH_LONG_COL, cfg.LABEL_COL, "txn_hour", "is_high_value", cfg.SPLIT_COL,
]
# Low-cardinality text columns are held as pandas categoricals (1-2 byte codes
# instead of one string per row); integer columns are downcast losslessly.
# Floats stay float64 so every figure matches the full pipeline output exactly.
CATEGORICAL_COLUMNS = [cfg.MERCHANT_COL, cfg.CATEGORY_COL, cfg.GENDER_COL, cfg.STATE_COL, cfg.SPLIT_COL]
INTEGER_DTYPES = {cfg.LABEL_COL: "int8", "txn_hour": "int8", cfg.CITY_POP_COL: "int32"}
SPLIT_ORDER = ["train", "test"]

TRANSACTIONS_FILE = cfg.DASHBOARD_TRANSACTIONS_FILE
MODEL_FILE = cfg.DASHBOARD_MODEL_FILE
SCORED_FILE = cfg.DASHBOARD_SCORED_FILE
USING_DEPLOY_DATA = TRANSACTIONS_FILE != cfg.CLEANED_TRANSACTIONS_FILE

HIVE_DIR = cfg.RESULTS_DIR / "hive_analytics"
SPARK_DIR = cfg.RESULTS_DIR / "spark_output"
MAPREDUCE_DIR = cfg.RESULTS_DIR / "mapreduce_output"

PRODUCERS = {
    cfg.CLEANED_TRANSACTIONS_FILE: "python src/preprocessing/clean_data.py",
    inf.MODEL_PATH: "python src/fraud_detection/train_model.py",
    inf.METRICS_PATH: "python src/fraud_detection/train_model.py",
    cfg.SCORED_TEST_FILE: "python src/fraud_detection/train_model.py",
    cfg.DEPLOY_DIR / "transactions_dashboard.parquet": "python scripts/build_deploy_artifacts.py",
    cfg.DEPLOY_DIR / "best_model.joblib": "python scripts/build_deploy_artifacts.py",
    cfg.DEPLOY_DIR / "scored_test_transactions.parquet": "python scripts/build_deploy_artifacts.py",
    HIVE_DIR: "python src/spark/hive_style_analytics.py",
    SPARK_DIR: "python src/spark/data_processing.py",
    MAPREDUCE_DIR: "python src/mapreduce/local_driver.py <job>",
}


class MissingArtifact(FileNotFoundError):
    pass


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(cfg.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _require(path: Path, producer_key: Path = None):
    if not path.exists():
        cmd = PRODUCERS.get(producer_key or path, ".\\scripts\\run_pipeline.ps1")
        raise MissingArtifact(f"`{display_path(path)}` not found. Generate it with `{cmd}`.")


def _release_arrow_memory():
    """Return the buffers pyarrow freed after a Parquet read to the OS. Its
    allocator otherwise keeps them, which roughly doubled the RSS of loading
    the transactions file (measured with scripts/measure_memory.py)."""
    pa.default_memory_pool().release_unused()


def apply_dashboard_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Compact dtypes for the dashboard columns, with rows ordered train
    then test (stable) so each split is one contiguous block."""
    for col in CATEGORICAL_COLUMNS:
        if not isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("category")
    for col, dtype in INTEGER_DTYPES.items():
        df[col] = df[col].astype(dtype)
    df[cfg.SPLIT_COL] = df[cfg.SPLIT_COL].cat.set_categories(SPLIT_ORDER)
    if not df[cfg.SPLIT_COL].cat.codes.is_monotonic_increasing:
        df = df.sort_values(cfg.SPLIT_COL, kind="stable", ignore_index=True)
    return df


# cache_resource (not cache_data): the frame is shared read-only instead of
# being copied on every rerun, which matters at 1.85M rows.
@st.cache_resource(show_spinner="Loading cleaned transactions...")
def _load_all_transactions():
    """(frame, {split: (start, stop)}): the transactions, loaded once, and
    the row range of each split."""
    _require(TRANSACTIONS_FILE)
    df = apply_dashboard_dtypes(pd.read_parquet(TRANSACTIONS_FILE, columns=DASHBOARD_COLUMNS))
    codes = df[cfg.SPLIT_COL].cat.codes.to_numpy()
    bounds = {s: tuple(int(i) for i in codes.searchsorted([k, k + 1])) for k, s in enumerate(SPLIT_ORDER)}
    _release_arrow_memory()
    return df, bounds


def load_transactions(split: str = None) -> pd.DataFrame:
    """Cleaned transactions of one split ("train" / "test"), or both (None).
    A split is a row slice of the single cached frame (a view, not a copy).
    Treat the result as read-only."""
    df, bounds = _load_all_transactions()
    if split is None:
        return df
    start, stop = bounds[split]
    return df.iloc[start:stop]


@st.cache_data
def dataset_summary() -> pd.DataFrame:
    """One row per Kaggle file: rows, fraud, fraud rate, period, amounts."""
    df = load_transactions()
    g = df.groupby(cfg.SPLIT_COL, observed=True).agg(
        transactions=(cfg.LABEL_COL, "size"), fraud=(cfg.LABEL_COL, "sum"),
        first_transaction=(cfg.TIMESTAMP_COL, "min"), last_transaction=(cfg.TIMESTAMP_COL, "max"),
        cards=(cfg.CARD_NUM_COL, "nunique"), merchants=(cfg.MERCHANT_COL, "nunique"),
        total_amount=(cfg.AMOUNT_COL, "sum"),
    ).reindex(SPLIT_ORDER)
    g["fraud_rate_pct"] = 100 * g["fraud"] / g["transactions"]
    g.insert(0, "file", [cfg.RAW_SPLITS[s].name for s in g.index])
    g.index = [SPLIT_LABELS[s] for s in g.index]
    return g.rename_axis("dataset").reset_index()


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    _require(MODEL_FILE)
    bundle = inf.load_model_bundle(MODEL_FILE)
    return bundle, inf.load_decision_threshold(bundle)


@st.cache_data
def load_metrics() -> dict:
    _require(inf.METRICS_PATH)
    with open(inf.METRICS_PATH) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading test-set scores...")
def load_test_set_scores():
    """(y_test, y_proba): actual labels and the saved model's scores for every
    fraudTest.csv transaction, as written by train_model.py."""
    _require(SCORED_FILE)
    scored = pd.read_parquet(SCORED_FILE, columns=[cfg.LABEL_COL, "fraud_probability"])
    _release_arrow_memory()
    return scored[cfg.LABEL_COL].to_numpy(), scored["fraud_probability"].to_numpy()


@st.cache_data
def load_pipeline_output(directory: Path, filename: str) -> pd.DataFrame:
    path = directory / filename
    _require(path, directory)
    return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")


def list_pipeline_outputs(directory: Path) -> list:
    return sorted(p.name for p in directory.glob("*.*")) if directory.exists() else []


@st.cache_resource(show_spinner="Connecting to MongoDB...")
def load_alerts_collection():
    """Returns (collection, backend). Uses the real MongoDB 'fraud_alerts'
    collection if one is reachable and populated; otherwise populates the
    mongomock fallback with the same loader the NoSQL stage uses."""
    from src.nosql.mongo_client import get_mongo_db
    from src.nosql.store_analytics import load_fraud_alerts

    db, backend = get_mongo_db()
    if db["fraud_alerts"].count_documents({}) == 0:
        _require(SCORED_FILE)
        load_fraud_alerts(db, SCORED_FILE)
        _release_arrow_memory()
    return db["fraud_alerts"], backend


def query_alerts(collection, categories, states, min_probability, actual_label, sort_field, descending, limit):
    """Builds and runs a MongoDB find() query; returns (DataFrame, query dict)."""
    query = {"predicted_fraud_probability": {"$gte": float(min_probability)}}
    if categories:
        query["category"] = {"$in": list(categories)}
    if states:
        query["customer.state"] = {"$in": list(states)}
    if actual_label is not None:
        query["actual_label"] = int(actual_label)

    cursor = collection.find(query, {"_id": 0}).sort(sort_field, -1 if descending else 1).limit(int(limit))
    docs = list(cursor)
    if not docs:
        return pd.DataFrame(), query
    return pd.json_normalize(docs), query
