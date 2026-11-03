"""
Central configuration for the Credit Card Fraud Detection Big Data project.

All paths are computed relative to the project root (the parent of this
config/ directory), so the project runs the same regardless of where it is
checked out. Settings that a student might want to change for their machine
(dataset paths, Mongo URI, Spark master) are grouped here instead of being
scattered/hardcoded across scripts.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# Real Kaggle "Credit Card Transactions Fraud Detection Dataset" (Sparkov
# simulation, kartik2112). The two CSVs are read in place - they are never
# copied into the project. fraudTrain.csv (Jan 2019 - Jun 2020) is used for
# training and validation only; fraudTest.csv (Jun 2020 - Dec 2020) is the
# untouched final evaluation set. Override the locations with the
# BDA_TRAIN_CSV / BDA_TEST_CSV environment variables.
_KAGGLE_DIR = Path.home() / "Downloads" / "archive (5)"
TRAIN_CSV = Path(os.environ.get("BDA_TRAIN_CSV") or _KAGGLE_DIR / "fraudTrain.csv")
TEST_CSV = Path(os.environ.get("BDA_TEST_CSV") or _KAGGLE_DIR / "fraudTest.csv")
RAW_SPLITS = {"train": TRAIN_CSV, "test": TEST_CSV}

# Cleaned train + test rows in one Parquet file, tagged by SPLIT_COL. Cleaning
# is row-level only (nothing is learned from the data), so storing both
# splits together cannot leak test information; every learned step (one-hot
# vocabulary, scaling, model fit, threshold) uses split == "train" only.
CLEANED_TRANSACTIONS_FILE = PROCESSED_DATA_DIR / "transactions_clean.parquet"
SCORED_TEST_FILE = RESULTS_DIR / "fraud_detection" / "scored_test_transactions.parquet"
MODEL_FILE = MODELS_DIR / "best_model.joblib"

# ---------------------------------------------------------------------------
# Dashboard artifacts (app.py)
# ---------------------------------------------------------------------------
# The full pipeline outputs above are gitignored. scripts/build_deploy_artifacts.py
# writes slim, committed copies into deploy/ (all rows, only the dashboard's
# columns) so the dashboard also runs from a clean checkout, e.g. on
# Streamlit Community Cloud. The dashboard uses the full local artifact when
# it exists and the deploy/ copy otherwise; an environment variable forces a
# specific file.
DEPLOY_DIR = PROJECT_ROOT / "deploy"


def _dashboard_artifact(env_var: str, full: Path, deploy_name: str) -> Path:
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    return full if full.exists() else DEPLOY_DIR / deploy_name


DASHBOARD_TRANSACTIONS_FILE = _dashboard_artifact("BDA_DASHBOARD_DATA", CLEANED_TRANSACTIONS_FILE,
                                                  "transactions_dashboard.parquet")
DASHBOARD_MODEL_FILE = _dashboard_artifact("BDA_DASHBOARD_MODEL", MODEL_FILE, "best_model.joblib")
DASHBOARD_SCORED_FILE = _dashboard_artifact("BDA_DASHBOARD_SCORED", SCORED_TEST_FILE,
                                            "scored_test_transactions.parquet")
# Precomputed fraud_alerts source: the scored test rows with fraud_probability
# >= ALERT_MIN_PROBABILITY (written by build_deploy_artifacts.py). The pipeline
# has no full-size equivalent, so only the deploy/ copy (or the env var) exists;
# the dashboard falls back to DASHBOARD_SCORED_FILE when it is missing or was
# built with a different floor.
DASHBOARD_ALERTS_FILE = _dashboard_artifact("BDA_DASHBOARD_ALERTS", DEPLOY_DIR / "fraud_alerts.parquet",
                                            "fraud_alerts.parquet")

# ---------------------------------------------------------------------------
# Canonical transaction schema
# ---------------------------------------------------------------------------
# Every downstream module (preprocessing, MapReduce, Hive, Spark, NoSQL,
# fraud detection) is written against these column names - the column names
# of the Kaggle fraudTrain.csv / fraudTest.csv files. (Both files also have an
# unnamed leading row-index column, which load_raw() drops.)
TRANSACTION_ID_COL = "trans_num"
TIMESTAMP_COL = "trans_date_trans_time"
UNIX_TIME_COL = "unix_time"
CARD_NUM_COL = "cc_num"
MERCHANT_COL = "merchant"
CATEGORY_COL = "category"
AMOUNT_COL = "amt"
CUSTOMER_FIRST_COL = "first"
CUSTOMER_LAST_COL = "last"
GENDER_COL = "gender"
STREET_COL = "street"
CITY_COL = "city"
STATE_COL = "state"
ZIP_COL = "zip"
LAT_COL = "lat"
LONG_COL = "long"
CITY_POP_COL = "city_pop"
JOB_COL = "job"
DOB_COL = "dob"
MERCH_LAT_COL = "merch_lat"
MERCH_LONG_COL = "merch_long"
LABEL_COL = "is_fraud"
SPLIT_COL = "split"  # "train" (fraudTrain.csv) or "test" (fraudTest.csv)

# ---------------------------------------------------------------------------
# MongoDB settings
# ---------------------------------------------------------------------------
# If BDA_MONGO_URI is unset and nothing answers at the default localhost
# URI, the nosql module falls back to mongomock (an in-memory
# MongoDB-compatible store) and reports backend "mongomock". If
# BDA_MONGO_URI IS set (local install or MongoDB Atlas), an unreachable
# server is an error - a configured database is never replaced by the mock.
# Set BDA_MONGO_FALLBACK=0 to forbid the mock fallback entirely.
MONGO_URI_CONFIGURED = bool(os.environ.get("BDA_MONGO_URI"))
MONGO_URI = os.environ.get("BDA_MONGO_URI") or "mongodb://localhost:27017"
MONGO_DB_NAME = os.environ.get("BDA_MONGO_DB", "fraud_analytics")
MONGO_ALLOW_MOCK_FALLBACK = os.environ.get("BDA_MONGO_FALLBACK", "1").strip().lower() not in ("0", "false", "no")

# Fraud alerts: every fraudTest.csv transaction scoring at least this
# probability is stored in the 'fraud_alerts' collection. 0.5 covers the
# Elevated (0.5 to threshold) and High (>= threshold) risk bands. Storing all
# of them, not just the top-N scores, is what keeps the alert precision shown
# on the dashboard consistent with model_comparison.json.
ALERT_MIN_PROBABILITY = float(os.environ.get("BDA_ALERT_MIN_PROB", "0.5"))

# ---------------------------------------------------------------------------
# Spark settings
# ---------------------------------------------------------------------------
SPARK_MASTER = os.environ.get("BDA_SPARK_MASTER", "local[*]")
SPARK_APP_NAME = "CreditCardFraudAnalytics"

# ---------------------------------------------------------------------------
# ML settings
# ---------------------------------------------------------------------------
# Model selection / threshold tuning uses a chronological holdout inside
# fraudTrain.csv: the latest VALIDATION_FRACTION of training transactions
# (by timestamp). fraudTest.csv is only scored once, by the selected model.
VALIDATION_FRACTION = 0.2
RANDOM_STATE = 42
HIGH_VALUE_THRESHOLD = 500.0  # amount (in dataset currency units) considered a "high-value" transaction

for _d in ( PROCESSED_DATA_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
