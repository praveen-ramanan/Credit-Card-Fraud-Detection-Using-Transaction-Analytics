"""
Streamlit dashboard for the Credit Card Fraud Detection project.

Run from the project root:
    streamlit run app.py

Reads the outputs of the existing pipeline (scripts/run_pipeline.ps1), built
from the Kaggle fraudTrain.csv / fraudTest.csv files, and the saved model -
it never retrains. Page code lives in src/dashboard/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.dashboard import data, views
from src.fraud_detection.inference import StaleModelError

st.set_page_config(page_title="Fraud Analytics", page_icon=":credit_card:", layout="wide")

PAGES = {
    "Overview": views.render_overview,
    "Transaction analytics": views.render_analytics,
    "Fraud detection": views.render_prediction,
    "Model performance": views.render_performance,
    "Fraud alerts": views.render_alerts,
    "CSV analysis": views.render_csv_analysis,
}

st.sidebar.title("Credit Card Fraud Detection")
st.sidebar.caption("Transaction analytics on a Big Data pipeline")
page = st.sidebar.radio("Section", list(PAGES))
st.sidebar.divider()
st.sidebar.caption(views.DATA_NOTE)

try:
    PAGES[page]()
except data.MissingArtifact as e:
    st.error(f"Required pipeline output is missing. {e}")
    st.caption("Run the full pipeline with `.\\scripts\\run_pipeline.ps1` to generate all outputs.")
except StaleModelError as e:
    st.error(f"The saved model does not match the current dataset/feature schema. {e}")
