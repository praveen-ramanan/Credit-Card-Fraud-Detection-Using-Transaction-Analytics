"""Smoke tests for the Streamlit dashboard: every page renders without an
exception against the real pipeline outputs, and the prediction form scores
a transaction. Skipped if Streamlit or the pipeline outputs are missing."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from config import config as cfg
from src.fraud_detection import inference as inf

pytestmark = pytest.mark.skipif(
    not (inf.MODEL_PATH.exists() and cfg.CLEANED_TRANSACTIONS_FILE.exists()),
    reason="pipeline outputs / model not generated yet",
)

PAGES = ["Overview", "Transaction analytics", "Fraud detection", "Model performance", "Fraud alerts", "CSV analysis"]


def _app(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = _app(page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.header[0].value


def test_prediction_form_scores_transaction():
    at = _app("Fraud detection")
    at.button[0].click().run()  # load a legitimate example from the dataset
    at.button(key="FormSubmitter:predict-Assess transaction").click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert any(m.label == "Fraud probability" for m in at.metric)


def test_prediction_form_rejects_invalid_dates():
    at = _app("Fraud detection")
    txn_date = at.date_input(key="f_date").value
    at.date_input(key="f_dob").set_value(txn_date.replace(year=txn_date.year + 1))
    at.button(key="FormSubmitter:predict-Assess transaction").click().run()
    assert at.error and "date of birth must be before" in at.error[0].value


def test_alert_filters_query_mongo():
    at = _app("Fraud alerts")
    at.selectbox[0].set_value("Fraud").run()
    assert not at.exception, at.exception
    precision = next(m for m in at.metric if m.label == "Alert precision")
    assert precision.value == "100.0%"


@pytest.mark.parametrize("dataset, split", [("Training (fraudTrain.csv)", "train"), ("Test (fraudTest.csv)", "test")])
def test_overview_dataset_selector_switches_split(dataset, split):
    import pyarrow.parquet as pq
    at = _app("Overview")
    at.selectbox(key="dataset").set_value(dataset).run()
    assert not at.exception, at.exception
    splits = pq.read_table(cfg.CLEANED_TRANSACTIONS_FILE, columns=[cfg.SPLIT_COL]).to_pandas()[cfg.SPLIT_COL]
    expected = int((splits == split).sum())
    total = next(m for m in at.metric if m.label == "Total transactions")
    assert total.value == f"{expected:,}"


def test_no_synthetic_data_references_in_app():
    for page in PAGES:
        at = _app(page)
        text = " ".join(str(el.value) for el in [*at.markdown, *at.caption, *at.info, *at.sidebar.caption])
        assert "synthetic" not in text.lower(), page
