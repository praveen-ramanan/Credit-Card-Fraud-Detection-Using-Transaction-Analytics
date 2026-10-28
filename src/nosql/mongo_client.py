"""
MongoDB connection helper with an explicit, opt-out local fallback.

Connects to a real MongoDB instance (local install, or a free-tier MongoDB
Atlas cluster - configure BDA_MONGO_URI). The in-memory `mongomock` fallback
is used ONLY when all of the following hold:
    - BDA_MONGO_URI is not set (i.e. the default localhost URI is in use),
    - no server answered at that default URI,
    - BDA_MONGO_FALLBACK is not disabled ("0"/"false"/"no").
The backend label returned by get_mongo_db() always says which store is in
use, so callers (the NoSQL stage, the dashboard) can show it.

Anything else - pymongo not installed, a configured server that is
unreachable, an authentication failure - raises MongoUnavailable with the
reason, instead of silently swapping a real database for a mock.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import config as cfg


class MongoUnavailable(RuntimeError):
    """No usable MongoDB backend (missing driver, or connection refused)."""


def _install_hint(package: str) -> str:
    return (f"Python package '{package}' is not installed in the interpreter running this code "
            f"({sys.executable}). Install the project requirements into that interpreter "
            f"(`\"{sys.executable}\" -m pip install -r requirements.txt`), or launch the app with "
            f"the project venv: `.\\venv\\Scripts\\python.exe -m streamlit run app.py`.")


def _mock_db(db_name: str):
    try:
        import mongomock
    except ImportError as e:
        raise MongoUnavailable(_install_hint("mongomock")) from e
    return mongomock.MongoClient()[db_name], "mongomock"


def get_mongo_db(db_name: str = cfg.MONGO_DB_NAME):
    """Return a (db, backend_label) tuple. backend_label is 'mongodb' or 'mongomock'."""
    try:
        import pymongo
        from pymongo.errors import ConnectionFailure, PyMongoError
    except ImportError as e:
        raise MongoUnavailable(_install_hint("pymongo")) from e

    try:
        client = pymongo.MongoClient(cfg.MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")  # forces a connection attempt now
        return client[db_name], "mongodb"
    except ConnectionFailure as e:  # includes ServerSelectionTimeoutError
        if cfg.MONGO_URI_CONFIGURED:
            raise MongoUnavailable(
                f"Could not reach the MongoDB server configured in BDA_MONGO_URI: {e}. "
                "Start the server / check the URI, or unset BDA_MONGO_URI to use the local default."
            ) from e
        if not cfg.MONGO_ALLOW_MOCK_FALLBACK:
            raise MongoUnavailable(
                f"No MongoDB server reachable at {cfg.MONGO_URI} and the mongomock fallback is "
                "disabled (BDA_MONGO_FALLBACK=0)."
            ) from e
        return _mock_db(db_name)
    except PyMongoError as e:  # e.g. authentication failure, bad URI options
        raise MongoUnavailable(f"MongoDB connection failed: {e}") from e


if __name__ == "__main__":
    db, backend = get_mongo_db()
    print(f"Connected using backend: {backend}")
    print(f"Database: {db.name}")
