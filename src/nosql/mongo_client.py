"""
MongoDB connection helper with a zero-install local fallback.

Tries to connect to a real MongoDB instance (local install, or a free-tier
MongoDB Atlas cluster - configure BDA_MONGO_URI). If no server is
reachable within a short timeout, transparently falls back to `mongomock`,
an in-memory library that implements the same pymongo API. This means the
NoSQL stage of the project always runs, even with nothing installed, while
still demonstrating the real pymongo API a production integration would
use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import config as cfg


def get_mongo_db(db_name: str = cfg.MONGO_DB_NAME):
    """Return a (db, backend_label) tuple. backend_label is 'mongodb' or 'mongomock'."""
    try:
        import pymongo
        client = pymongo.MongoClient(cfg.MONGO_URI, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")  # forces a connection attempt now
        return client[db_name], "mongodb"
    except Exception:
        import mongomock
        client = mongomock.MongoClient()
        return client[db_name], "mongomock"


if __name__ == "__main__":
    db, backend = get_mongo_db()
    print(f"Connected using backend: {backend}")
    print(f"Database: {db.name}")
