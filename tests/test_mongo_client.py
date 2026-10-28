"""Tests for the MongoDB connection policy in src/nosql/mongo_client.py:
the mongomock fallback is only used for the unconfigured local default, and
every other failure raises MongoUnavailable instead of silently mocking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

pytest.importorskip("pymongo")
pytest.importorskip("mongomock")

from config import config as cfg
from src.nosql import mongo_client
from src.nosql.mongo_client import MongoUnavailable, get_mongo_db

UNREACHABLE_URI = "mongodb://127.0.0.1:1/?connectTimeoutMS=300"  # nothing listens on port 1


@pytest.fixture
def unreachable(monkeypatch):
    monkeypatch.setattr(cfg, "MONGO_URI", UNREACHABLE_URI)
    monkeypatch.setattr(cfg, "MONGO_URI_CONFIGURED", False)
    monkeypatch.setattr(cfg, "MONGO_ALLOW_MOCK_FALLBACK", True)


def test_default_uri_unreachable_uses_labelled_mongomock(unreachable):
    db, backend = get_mongo_db("bda_test")
    assert backend == "mongomock"
    assert db["fraud_alerts"].count_documents({}) == 0  # empty, nothing fabricated


def test_configured_uri_unreachable_raises(unreachable, monkeypatch):
    monkeypatch.setattr(cfg, "MONGO_URI_CONFIGURED", True)
    with pytest.raises(MongoUnavailable, match="BDA_MONGO_URI"):
        get_mongo_db("bda_test")


def test_fallback_disabled_raises(unreachable, monkeypatch):
    monkeypatch.setattr(cfg, "MONGO_ALLOW_MOCK_FALLBACK", False)
    with pytest.raises(MongoUnavailable, match="BDA_MONGO_FALLBACK"):
        get_mongo_db("bda_test")


def test_missing_pymongo_raises_with_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymongo", None)  # makes `import pymongo` fail
    with pytest.raises(MongoUnavailable, match="pymongo.*not installed"):
        get_mongo_db("bda_test")


def test_missing_mongomock_raises_with_install_hint(unreachable, monkeypatch):
    monkeypatch.setitem(sys.modules, "mongomock", None)
    with pytest.raises(MongoUnavailable, match="mongomock.*not installed"):
        mongo_client.get_mongo_db("bda_test")
