"""Tests for the fraud_alerts collection (src/nosql/store_analytics.py) and
the alert statistics on the dashboard (src/dashboard/data.py).

The key property: the alerts page must agree with model_comparison.json.
The collection used to hold only the 500 highest scores, so every filter
showed ~100% precision. It now holds every test transaction scoring at least
cfg.ALERT_MIN_PROBABILITY, and the predicted_fraud == 1 subset must reproduce
the test confusion matrix exactly."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

mongomock = pytest.importorskip("mongomock")
pytest.importorskip("pyarrow")

from config import config as cfg
from src.dashboard import data
from src.fraud_detection import inference as inf
from src.nosql import store_analytics as store
from src.nosql.store_analytics import load_fraud_alerts

THRESHOLD = 0.949


def _scored_frame():
    probs = [0.20, 0.4999, 0.50, 0.70, 0.90, 0.95, 0.99, 0.999]
    labels = [0, 0, 1, 0, 0, 1, 0, 1]
    n = len(probs)
    return pd.DataFrame({
        cfg.TRANSACTION_ID_COL: [f"t{i}" for i in range(n)],
        cfg.TIMESTAMP_COL: pd.date_range("2020-07-01", periods=n, freq="h"),
        cfg.AMOUNT_COL: [10.0 * (i + 1) for i in range(n)],
        cfg.CATEGORY_COL: ["shopping_net", "grocery_pos"] * (n // 2),
        cfg.CARD_NUM_COL: [4000000000000000 + i for i in range(n)],
        cfg.CITY_COL: ["Austin"] * n,
        cfg.STATE_COL: ["TX", "NY"] * (n // 2),
        "customer_age": [40] * n,
        cfg.MERCHANT_COL: ["m"] * n,
        "customer_merchant_distance_km": [12.5] * n,
        cfg.LABEL_COL: labels,
        "fraud_probability": pd.Series(probs, dtype="float32"),
        "predicted_fraud": [int(p >= THRESHOLD) for p in probs],
    })


@pytest.fixture
def loaded(tmp_path):
    path = tmp_path / "scored.parquet"
    _scored_frame().to_parquet(path, index=False)
    coll = mongomock.MongoClient().db.fraud_alerts
    n = load_fraud_alerts(coll.database, path, min_probability=0.5)
    return coll, n


def test_floor_is_inclusive_and_drops_lower_scores(loaded):
    coll, n = loaded
    assert n == 6 == coll.count_documents({})          # 0.50 kept, 0.4999 and 0.20 dropped
    assert coll.count_documents({"predicted_fraud_probability": {"$lt": 0.5}}) == 0
    assert coll.count_documents({"predicted_fraud": 1}) == 3


def test_reload_replaces_instead_of_appending(loaded, tmp_path):
    coll, _ = loaded
    load_fraud_alerts(coll.database, tmp_path / "scored.parquet", min_probability=0.9)
    assert coll.count_documents({}) == 4


def test_invalid_floor_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_fraud_alerts(mongomock.MongoClient().db, tmp_path / "x.parquet", min_probability=1.5)


def test_flag_reason_matches_decision(loaded):
    coll, _ = loaded
    assert {d["flagged_reason"] for d in coll.find({"predicted_fraud": 1})} == {"above_decision_threshold"}
    assert {d["flagged_reason"] for d in coll.find({"predicted_fraud": 0})} == {"elevated_score"}


def test_stats_count_all_matches_not_just_table_rows(loaded):
    coll, _ = loaded
    high = data.build_alert_query(predicted_fraud_only=True)
    assert data.alert_stats(coll, high) == {"matched": 3, "fraud": 2, "precision": 2 / 3}
    everything = data.build_alert_query(min_probability=0.5)
    assert data.alert_stats(coll, everything)["precision"] == pytest.approx(3 / 6)
    assert len(data.query_alerts(coll, everything, "amount", True, limit=2)) == 2   # table capped
    assert data.alert_stats(coll, everything)["matched"] == 6                       # stats are not


def test_stats_with_label_filter_and_no_match(loaded):
    coll, _ = loaded
    assert data.alert_stats(coll, data.build_alert_query(actual_label=0))["precision"] == 0.0
    assert data.alert_stats(coll, data.build_alert_query(actual_label=1))["precision"] == 1.0
    none = data.alert_stats(coll, data.build_alert_query(categories=["travel"]))
    assert none == {"matched": 0, "fraud": 0, "precision": None}


def test_category_aggregation_matches_counts(loaded):
    coll, _ = loaded
    by_cat = data.alerts_by_category(coll, data.build_alert_query())
    assert by_cat["alerts"].sum() == 6
    assert by_cat["actual_fraud"].sum() == 3


def test_stale_top_n_collection_detected():
    coll = mongomock.MongoClient().db.fraud_alerts
    coll.insert_many([{"predicted_fraud_probability": p} for p in (0.999, 0.9999)])
    assert data._alerts_collection_is_stale(coll)
    coll.insert_one({"predicted_fraud_probability": cfg.ALERT_MIN_PROBABILITY})
    assert not data._alerts_collection_is_stale(coll)


# --- Regression test against the real model outputs ------------------------
@pytest.mark.skipif(not (cfg.DASHBOARD_SCORED_FILE.exists() and inf.METRICS_PATH.exists()),
                    reason="scored test file / metrics not generated")
def test_alert_page_reproduces_model_test_metrics():
    test = json.loads(inf.METRICS_PATH.read_text())["test"]
    coll = mongomock.MongoClient().db.fraud_alerts
    load_fraud_alerts(coll.database, cfg.DASHBOARD_SCORED_FILE, min_probability=0.5)

    cm = test["confusion_matrix_at_threshold"]
    high = data.alert_stats(coll, data.build_alert_query(predicted_fraud_only=True))
    assert high["matched"] == cm["tp"] + cm["fp"]
    assert high["fraud"] == cm["tp"]
    assert high["precision"] == pytest.approx(test["precision_at_threshold"], abs=1e-9)

    # The 0.5 floor reproduces the at-0.5 confusion matrix. Scores are stored
    # as float32, so a mismatch here would point at rounding at the boundary.
    cm05 = test["confusion_matrix_at_0.5"]
    both = data.alert_stats(coll, data.build_alert_query(min_probability=0.5))
    assert both["matched"] == cm05["tp"] + cm05["fp"]
    assert both["fraud"] == cm05["tp"]


# --- Precomputed deploy/fraud_alerts.parquet ---------------------------------
def _docs(coll):
    return sorted(coll.find({}, {"_id": 0}), key=lambda d: d["trans_num"])


def test_built_alerts_file_records_floor_and_loads_like_scored_file(tmp_path):
    from scripts.build_deploy_artifacts import build_alerts
    scored, out = tmp_path / "scored.parquet", tmp_path / "fraud_alerts.parquet"
    _scored_frame().to_parquet(scored, index=False)
    build_alerts(scored, out, floor=0.5)
    assert store.alerts_file_floor(out) == 0.5 and store.alerts_file_floor(scored) is None
    a, b = mongomock.MongoClient().db, mongomock.MongoClient().db
    assert load_fraud_alerts(a, out, min_probability=0.5) == load_fraud_alerts(b, scored, min_probability=0.5) == 6
    assert _docs(a.fraud_alerts) == _docs(b.fraud_alerts)
    with pytest.raises(ValueError):     # a 0.5-floor file cannot serve a lower floor
        load_fraud_alerts(a, out, min_probability=0.4)


def test_floor_mismatch_falls_back_to_scored_file(tmp_path, monkeypatch):
    from scripts.build_deploy_artifacts import build_alerts
    scored, out = tmp_path / "scored.parquet", tmp_path / "fraud_alerts.parquet"
    _scored_frame().to_parquet(scored, index=False)
    monkeypatch.setattr(data, "SCORED_FILE", scored)
    monkeypatch.setattr(data, "ALERTS_FILE", out)
    assert data.alerts_source_file() == scored                    # file missing
    build_alerts(scored, out, floor=0.5)
    monkeypatch.setattr(cfg, "ALERT_MIN_PROBABILITY", 0.5)
    assert data.alerts_source_file() == out
    monkeypatch.setattr(cfg, "ALERT_MIN_PROBABILITY", 0.6)
    assert data.alerts_source_file() == scored                    # built with another floor


DEPLOY_ALERTS = cfg.DEPLOY_DIR / "fraud_alerts.parquet"


@pytest.mark.skipif(not (DEPLOY_ALERTS.exists() and cfg.DASHBOARD_SCORED_FILE.exists() and inf.METRICS_PATH.exists()),
                    reason="deploy/fraud_alerts.parquet, scored test file or metrics not generated")
def test_precomputed_alerts_reproduce_model_test_metrics_and_scored_load():
    test = json.loads(inf.METRICS_PATH.read_text())["test"]
    cm, cm05 = test["confusion_matrix_at_threshold"], test["confusion_matrix_at_0.5"]
    assert store.alerts_file_floor(DEPLOY_ALERTS) == cfg.ALERT_MIN_PROBABILITY == 0.5
    pre, full = mongomock.MongoClient().db, mongomock.MongoClient().db
    load_fraud_alerts(pre, DEPLOY_ALERTS)
    load_fraud_alerts(full, cfg.DASHBOARD_SCORED_FILE)

    high = data.alert_stats(pre.fraud_alerts, data.build_alert_query(predicted_fraud_only=True))
    assert (high["matched"], high["fraud"]) == (cm["tp"] + cm["fp"], cm["tp"]) == (2050, 1666)
    both = data.alert_stats(pre.fraud_alerts, data.build_alert_query(min_probability=0.5))
    assert (both["matched"], both["fraud"]) == (cm05["tp"] + cm05["fp"], cm05["tp"]) == (5321, 1963)
    assert _docs(pre.fraud_alerts) == _docs(full.fraud_alerts)
