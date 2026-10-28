"""
Dashboard sections. Each render_* function draws one page of app.py.
All figures are computed from the project's own data/artifacts (see data.py).
"""

import json
import sys
from datetime import date, datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from config import config as cfg
from src.dashboard import data
from src.fraud_detection import inference as inf
from src.nosql.mongo_client import MongoUnavailable

LEGIT_COLOR, FRAUD_COLOR = "#4C72B0", "#C44E52"
CLASS_COLORS = {"Legitimate": LEGIT_COLOR, "Fraud": FRAUD_COLOR}
RISK_COLORS = {"Low": "#2E7D32", "Elevated": "#ED6C02", "High": "#C62828"}
MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest", "xgboost": "XGBoost"}

DATA_NOTE = (
    "Data: Kaggle *Credit Card Transactions Fraud Detection* dataset - `fraudTrain.csv` "
    "(Jan 2019 - Jun 2020) for training/validation and `fraudTest.csv` (Jun - Dec 2020) for the "
    "final test. The transactions are simulated US cardholders (Sparkov generator), so results "
    "describe this dataset, not a live bank's traffic."
)
PREDICTION_DISCLAIMER = (
    "A model score is a statistical risk estimate, **not proof of fraud**. It should only "
    "prioritise transactions for review by an analyst, never block or accuse a customer on its own."
)


def _chart(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    st.plotly_chart(fig, width="stretch")


def _with_class_label(df):
    return df.assign(label=np.where(df[cfg.LABEL_COL] == 1, "Fraud", "Legitimate"))


def _rate_table(df, key):
    g = df.groupby(key, observed=True).agg(
        transactions=(cfg.LABEL_COL, "size"),
        fraud_count=(cfg.LABEL_COL, "sum"),
        total_amount=(cfg.AMOUNT_COL, "sum"),
    ).reset_index()
    g["fraud_rate_pct"] = 100 * g["fraud_count"] / g["transactions"]
    return g


def _mask_card(cc):
    return "**** " + str(cc)[-4:]


def _dataset_selector():
    """Sidebar choice of Kaggle file(s); returns the split key (None = both)."""
    options = {"Training (fraudTrain.csv)": "train", "Test (fraudTest.csv)": "test", "Both files": None}
    choice = st.sidebar.selectbox("Dataset", list(options), key="dataset")
    return options[choice], choice


# --------------------------------------------------------------------------
# A. Overview
# --------------------------------------------------------------------------
def render_overview():
    st.header("Overview")
    st.subheader("Datasets")
    st.dataframe(data.dataset_summary(), hide_index=True, width="stretch",
                 column_config={"fraud_rate_pct": st.column_config.NumberColumn("fraud rate", format="%.3f%%"),
                                "total_amount": st.column_config.NumberColumn(format="$%.0f")})
    split, label = _dataset_selector()
    st.subheader(label)
    df = data.load_transactions(split)
    n, n_fraud = len(df), int(df[cfg.LABEL_COL].sum())
    amt = df[cfg.AMOUNT_COL]

    c = st.columns(4)
    c[0].metric("Total transactions", f"{n:,}")
    c[1].metric("Fraudulent", f"{n_fraud:,}")
    c[2].metric("Legitimate", f"{n - n_fraud:,}")
    c[3].metric("Fraud rate", f"{100 * n_fraud / n:.3f}%")
    c = st.columns(4)
    c[0].metric("Total value", f"${amt.sum():,.0f}")
    c[1].metric("Mean amount", f"${amt.mean():,.2f}")
    c[2].metric("Median amount", f"${amt.median():,.2f}")
    fraud_share = 100 * amt[df[cfg.LABEL_COL] == 1].sum() / amt.sum()
    c[3].metric("Fraud share of value", f"{fraud_share:.2f}%")

    labelled = _with_class_label(df)
    left, right = st.columns(2)
    with left:
        counts = labelled["label"].value_counts().reindex(["Legitimate", "Fraud"]).reset_index()
        fig = px.bar(counts, x="label", y="count", color="label", color_discrete_map=CLASS_COLORS,
                     log_y=True, text="count", title="Fraud vs legitimate (log scale)")
        fig.update_traces(texttemplate="%{text:,}")
        _chart(fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Transactions"))
    with right:
        # Quartiles are precomputed so ~1M raw values aren't shipped to the browser.
        fig = go.Figure()
        for name, color in CLASS_COLORS.items():
            a = labelled.loc[labelled["label"] == name, cfg.AMOUNT_COL]
            fig.add_trace(go.Box(name=name, marker_color=color, q1=[a.quantile(0.25)], median=[a.median()],
                                 q3=[a.quantile(0.75)], lowerfence=[a.min()], upperfence=[a.max()], mean=[a.mean()]))
        fig.update_layout(title="Transaction amount by class (min / quartiles / max, log scale)", yaxis_type="log",
                          showlegend=False, yaxis_title="Amount (USD)")
        _chart(fig)

    daily = df.set_index(cfg.TIMESTAMP_COL).resample("D").agg(
        transactions=(cfg.LABEL_COL, "size"), fraud=(cfg.LABEL_COL, "sum"), value=(cfg.AMOUNT_COL, "sum"))
    daily["fraud_rate_7d_pct"] = 100 * daily["fraud"].rolling(7).sum() / daily["transactions"].rolling(7).sum()
    daily = daily.reset_index()
    left, right = st.columns(2)
    with left:
        _chart(px.line(daily, x=cfg.TIMESTAMP_COL, y="transactions", title="Daily transaction volume")
               .update_layout(xaxis_title="", yaxis_title="Transactions"))
    with right:
        fig = px.bar(daily, x=cfg.TIMESTAMP_COL, y="fraud", title="Daily fraud count and 7-day fraud rate")
        fig.update_traces(marker_color=FRAUD_COLOR, name="Fraud count", showlegend=True)
        fig.add_trace(go.Scatter(x=daily[cfg.TIMESTAMP_COL], y=daily["fraud_rate_7d_pct"], yaxis="y2",
                                 name="7-day fraud rate %", line=dict(color="#333")))
        fig.update_layout(yaxis=dict(title="Fraud transactions"),
                          yaxis2=dict(title="Fraud rate %", overlaying="y", side="right", showgrid=False),
                          xaxis_title="", legend=dict(orientation="h", y=-0.2))
        _chart(fig)


# --------------------------------------------------------------------------
# B. Transaction analytics
# --------------------------------------------------------------------------
def _analytics_filters(df):
    st.sidebar.subheader("Filters")
    dmin, dmax = df[cfg.TIMESTAMP_COL].min().date(), df[cfg.TIMESTAMP_COL].max().date()
    dates = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax,
                                  key=f"dates_{dmin}_{dmax}")
    categories = st.sidebar.multiselect("Merchant category", sorted(df[cfg.CATEGORY_COL].unique()))
    states = st.sidebar.multiselect("State", sorted(df[cfg.STATE_COL].unique()))
    genders = st.sidebar.multiselect("Gender", sorted(df[cfg.GENDER_COL].unique()))
    amax = float(np.ceil(df[cfg.AMOUNT_COL].max()))
    amount = st.sidebar.slider("Amount range (USD)", 0.0, amax, (0.0, amax), key=f"amount_{amax}")

    mask = df[cfg.AMOUNT_COL].between(*amount)
    if isinstance(dates, tuple) and len(dates) == 2:
        start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1)
        mask &= (df[cfg.TIMESTAMP_COL] >= start) & (df[cfg.TIMESTAMP_COL] < end)
    if categories:
        mask &= df[cfg.CATEGORY_COL].isin(categories)
    if states:
        mask &= df[cfg.STATE_COL].isin(states)
    if genders:
        mask &= df[cfg.GENDER_COL].isin(genders)
    return df[mask]


def render_analytics():
    st.header("Transaction analytics")
    split, label = _dataset_selector()
    st.caption(f"Dataset: **{label}**. Fraud figures use the actual `{cfg.LABEL_COL}` labels.")
    df = _analytics_filters(data.load_transactions(split))
    if df.empty:
        st.warning("No transactions match the current filters.")
        return

    n_fraud = int(df[cfg.LABEL_COL].sum())
    c = st.columns(3)
    c[0].metric("Filtered transactions", f"{len(df):,}")
    c[1].metric("Fraudulent", f"{n_fraud:,}")
    c[2].metric("Fraud rate", f"{100 * n_fraud / len(df):.3f}%")

    left, right = st.columns(2)
    with left:
        cat = _rate_table(df, cfg.CATEGORY_COL).sort_values("fraud_rate_pct")
        fig = px.bar(cat, x="fraud_rate_pct", y=cfg.CATEGORY_COL, orientation="h", hover_data=["transactions", "fraud_count"],
                     title="Fraud rate by merchant category")
        _chart(fig.update_traces(marker_color=FRAUD_COLOR).update_layout(xaxis_title="Fraud rate %", yaxis_title=""), 440)
    with right:
        hourly = _rate_table(df, "txn_hour")
        fig = go.Figure()
        fig.add_bar(x=hourly["txn_hour"], y=hourly["transactions"], name="Transactions", marker_color=LEGIT_COLOR, opacity=0.5)
        fig.add_trace(go.Scatter(x=hourly["txn_hour"], y=hourly["fraud_rate_pct"], name="Fraud rate %",
                                 yaxis="y2", line=dict(color=FRAUD_COLOR, width=3)))
        fig.update_layout(title="Volume and fraud rate by hour of day", xaxis=dict(title="Hour", dtick=2),
                          yaxis=dict(title="Transactions"),
                          yaxis2=dict(title="Fraud rate %", overlaying="y", side="right", showgrid=False),
                          legend=dict(orientation="h", y=-0.2))
        _chart(fig, 440)

    st.subheader("High-value transactions")
    st.caption(f"High value = amount >= ${cfg.HIGH_VALUE_THRESHOLD:,.0f} (`HIGH_VALUE_THRESHOLD` in config/config.py).")
    bands = pd.cut(df[cfg.AMOUNT_COL], [0, 25, 50, 100, 250, cfg.HIGH_VALUE_THRESHOLD, np.inf],
                   labels=["<25", "25-50", "50-100", "100-250", f"250-{cfg.HIGH_VALUE_THRESHOLD:.0f}",
                           f">={cfg.HIGH_VALUE_THRESHOLD:.0f}"], right=False)
    band = _rate_table(df.assign(amount_band=bands), "amount_band")
    left, right = st.columns([2, 1])
    with left:
        fig = px.bar(band, x="amount_band", y="fraud_rate_pct", text="fraud_count", hover_data=["transactions"],
                     title="Fraud rate by amount band (labels = fraud count)")
        _chart(fig.update_traces(marker_color=FRAUD_COLOR).update_layout(xaxis_title="Amount", yaxis_title="Fraud rate %"))
    with right:
        hv = _rate_table(df, "is_high_value").replace({"is_high_value": {True: "High value", False: "Normal"}})
        st.dataframe(hv.rename(columns={"is_high_value": "segment"}), hide_index=True, width="stretch",
                     column_config={"fraud_rate_pct": st.column_config.NumberColumn(format="%.2f%%"),
                                    "total_amount": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("Merchant and customer summaries")
    top_n = st.slider("Rows to show", 5, 50, 15)
    left, right = st.columns(2)
    with left:
        merch = _rate_table(df, cfg.MERCHANT_COL)
        merch["fraud_amount"] = df[df[cfg.LABEL_COL] == 1].groupby(cfg.MERCHANT_COL)[cfg.AMOUNT_COL].sum() \
            .reindex(merch[cfg.MERCHANT_COL]).fillna(0).to_numpy()
        sort = st.selectbox("Sort merchants by", ["fraud_count", "fraud_amount", "fraud_rate_pct", "transactions"])
        st.dataframe(merch.sort_values(sort, ascending=False).head(top_n), hide_index=True, width="stretch",
                     column_config={"fraud_rate_pct": st.column_config.NumberColumn(format="%.2f%%"),
                                    "total_amount": st.column_config.NumberColumn(format="%.2f"),
                                    "fraud_amount": st.column_config.NumberColumn(format="%.2f")})
    with right:
        cust = _rate_table(df, cfg.CARD_NUM_COL)
        cust[cfg.CARD_NUM_COL] = cust[cfg.CARD_NUM_COL].map(_mask_card)
        sort = st.selectbox("Sort customers by", ["fraud_count", "total_amount", "transactions", "fraud_rate_pct"])
        st.dataframe(cust.rename(columns={cfg.CARD_NUM_COL: "card"}).sort_values(sort, ascending=False).head(top_n),
                     hide_index=True, width="stretch",
                     column_config={"fraud_rate_pct": st.column_config.NumberColumn(format="%.2f%%"),
                                    "total_amount": st.column_config.NumberColumn(format="%.2f")})

    _render_pipeline_outputs()


def _render_pipeline_outputs():
    st.subheader("Big Data pipeline outputs")
    st.caption("Results written by the MapReduce, Spark and Hive (Spark SQL) stages, over all rows of both Kaggle "
               "files (unfiltered).")
    sources = {"Hive / Spark SQL queries": data.HIVE_DIR, "Spark DataFrame jobs": data.SPARK_DIR,
               "MapReduce jobs": data.MAPREDUCE_DIR}
    left, right = st.columns([1, 2])
    source = left.radio("Stage", list(sources))
    files = data.list_pipeline_outputs(sources[source])
    if not files:
        st.info(f"No outputs in `{sources[source].relative_to(cfg.PROJECT_ROOT)}` yet. Run `.\\scripts\\run_pipeline.ps1`.")
        return
    name = right.selectbox("Output file", files)
    out = data.load_pipeline_output(sources[source], name)
    if "cc_num" in out.columns:
        out = out.assign(cc_num=out["cc_num"].map(_mask_card))
    st.dataframe(out, hide_index=True, width="stretch")


# --------------------------------------------------------------------------
# C. Fraud detection (single transaction)
# --------------------------------------------------------------------------
# A plausible everyday purchase in Columbus, OH (values within the dataset's ranges).
FORM_DEFAULTS = {
    "f_amt": 64.50, "f_date": date(2020, 9, 12), "f_time": time(14, 30), "f_category": "grocery_pos",
    "f_state": "OH", "f_gender": "F", "f_dob": date(1985, 3, 12), "f_city_pop": 905748,
    "f_lat": 39.9612, "f_long": -82.9988, "f_merch_lat": 40.1200, "f_merch_long": -83.1500,
}


def _load_example(df, fraud: bool):
    row = df[df[cfg.LABEL_COL] == int(fraud)].sample(1).iloc[0]
    known = inf.known_categorical_values(data.load_model()[0])
    ts = row[cfg.TIMESTAMP_COL]
    st.session_state.update({
        "f_amt": float(row[cfg.AMOUNT_COL]), "f_date": ts.date(), "f_time": ts.time(),
        "f_category": str(row[cfg.CATEGORY_COL]), "f_state": str(row[cfg.STATE_COL]) if str(row[cfg.STATE_COL]) in known["state"]
        else FORM_DEFAULTS["f_state"], "f_gender": str(row[cfg.GENDER_COL]),
        "f_dob": pd.Timestamp(row[cfg.DOB_COL]).date(), "f_city_pop": int(row[cfg.CITY_POP_COL]),
        "f_lat": float(row[cfg.LAT_COL]), "f_long": float(row[cfg.LONG_COL]),
        "f_merch_lat": float(row[cfg.MERCH_LAT_COL]), "f_merch_long": float(row[cfg.MERCH_LONG_COL]),
        "f_example": f"`{row[cfg.TRANSACTION_ID_COL]}` from fraudTest.csv (actual label: {'fraud' if fraud else 'legitimate'})",
    })


def render_prediction():
    st.header("Fraud detection")
    bundle, threshold = data.load_model()
    known = inf.known_categorical_values(bundle)
    st.caption(f"Model: **{MODEL_LABELS.get(bundle['model_name'], bundle['model_name'])}** "
               f"(`{inf.MODEL_PATH.relative_to(cfg.PROJECT_ROOT)}`). Features are derived with the same "
               f"code used in training (`add_derived_features` + `build_features`). Trained on `fraudTrain.csv`; "
               f"decision threshold {threshold:.3f} chosen on the validation set.")

    for k, v in FORM_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    df = data.load_transactions("test")
    b = st.columns([1, 1, 3])
    if b[0].button("Load a legitimate example"):
        _load_example(df, fraud=False)
    if b[1].button("Load a fraud example"):
        _load_example(df, fraud=True)
    if st.session_state.get("f_example"):
        b[2].caption(f"Form filled from dataset transaction {st.session_state['f_example']}.")

    with st.form("predict"):
        c = st.columns(3)
        c[0].number_input("Amount (USD)", min_value=0.0, step=1.0, format="%.2f", key="f_amt")
        c[1].date_input("Transaction date", key="f_date")
        c[2].time_input("Transaction time", key="f_time", step=60)
        c = st.columns(3)
        c[0].selectbox("Merchant category", known["category"], key="f_category")
        c[1].selectbox("Customer state", known["state"], key="f_state")
        c[2].selectbox("Customer gender", known["gender"], key="f_gender")
        c = st.columns(2)
        c[0].date_input("Customer date of birth", min_value=date(1900, 1, 1), max_value=date.today(), key="f_dob")
        c[1].number_input("Customer city population", min_value=0, step=1000, key="f_city_pop")
        c = st.columns(4)
        c[0].number_input("Customer latitude", -90.0, 90.0, format="%.4f", key="f_lat")
        c[1].number_input("Customer longitude", -180.0, 180.0, format="%.4f", key="f_long")
        c[2].number_input("Merchant latitude", -90.0, 90.0, format="%.4f", key="f_merch_lat")
        c[3].number_input("Merchant longitude", -180.0, 180.0, format="%.4f", key="f_merch_long")
        submitted = st.form_submit_button("Assess transaction", type="primary")

    if not submitted:
        return
    s = st.session_state
    txn = pd.DataFrame([{
        cfg.TIMESTAMP_COL: datetime.combine(s.f_date, s.f_time), cfg.AMOUNT_COL: s.f_amt,
        cfg.CATEGORY_COL: s.f_category, cfg.STATE_COL: s.f_state, cfg.GENDER_COL: s.f_gender,
        cfg.DOB_COL: datetime.combine(s.f_dob, time()), cfg.CITY_POP_COL: s.f_city_pop,
        cfg.LAT_COL: s.f_lat, cfg.LONG_COL: s.f_long, cfg.MERCH_LAT_COL: s.f_merch_lat, cfg.MERCH_LONG_COL: s.f_merch_long,
    }])
    scored, errors = inf.predict(txn, bundle, threshold)
    if errors:
        st.error("The transaction could not be scored: " + errors[0]["error"])
        return

    r = scored.iloc[0]
    p, risk = float(r["fraud_probability"]), r["risk_level"]
    st.caption("Model prediction (not a verified outcome):")
    c = st.columns([1, 1, 2])
    c[0].metric("Fraud probability", f"{p:.1%}")
    c[1].markdown(f"**Risk level**<br><span style='font-size:1.8rem;font-weight:600;color:{RISK_COLORS[risk]}'>{risk}</span>",
                  unsafe_allow_html=True)
    c[2].markdown(
        f"- **High**: probability >= {threshold:.3f}, the F1-optimal threshold on the validation set "
        f"(latest 20% of fraudTrain.csv) - predicted fraud\n"
        f"- **Elevated**: {inf.elevated_cutoff(threshold):.3f} to {threshold:.3f}, worth a second look\n"
        f"- **Low**: below {inf.elevated_cutoff(threshold):.3f}")
    st.progress(min(max(p, 0.0), 1.0))

    with st.expander("Derived model inputs"):
        st.dataframe(pd.DataFrame({
            "hour": [int(r["txn_hour"])], "month": [int(r["txn_month"])], "weekend": [bool(r["is_weekend"])],
            "night (00-04h)": [bool(r["is_night_txn"])], "customer age": [int(r["customer_age"])],
            "customer-merchant distance km": [float(r["customer_merchant_distance_km"])],
            "high value": [bool(r["is_high_value"])],
        }), hide_index=True)
    st.info(PREDICTION_DISCLAIMER)
    test = data.load_metrics().get("test")
    if test and test.get("model") == bundle["model_name"]:
        st.caption(f"On the final test set (fraudTest.csv), {test['precision_at_threshold']:.0%} of transactions "
                   f"scored High were actually fraud, and {test['recall_at_threshold']:.0%} of fraud "
                   f"reached High (see Model performance).")


# --------------------------------------------------------------------------
# D. Model performance
# --------------------------------------------------------------------------
def _confusion_chart(cm, title):
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = px.imshow(z, x=["Predicted legitimate", "Predicted fraud"], y=["Actual legitimate", "Actual fraud"],
                    text_auto=",", color_continuous_scale="Blues", aspect="auto")
    _chart(fig.update_layout(coloraxis_showscale=False, title=title), 350)


def render_performance():
    st.header("Model performance")
    metrics = data.load_metrics()
    best, test, split = metrics["best_model"], metrics["test"], metrics["split"]
    threshold = metrics["decision_threshold"]
    st.caption(f"Source: `{inf.METRICS_PATH.relative_to(cfg.PROJECT_ROOT)}`. Models are trained on `fraudTrain.csv` only. "
               f"Model selection and the decision threshold use a **validation** set; `fraudTest.csv` is scored once, "
               f"afterwards, as the **final test**.")

    st.subheader("Final test: fraudTest.csv")
    st.markdown(f"**{MODEL_LABELS.get(best, best)}** (selected on validation PR-AUC) scored on all "
                f"{test['rows']:,} test transactions ({test['fraud_rows']:,} actual fraud), at the validation-chosen "
                f"threshold **{threshold:.3f}**. Nothing was fitted or tuned on this data.")
    cm = test["confusion_matrix_at_threshold"]
    c = st.columns(7)
    c[0].metric("ROC-AUC", f"{test['roc_auc']:.4f}")
    c[1].metric("PR-AUC (AP)", f"{test['pr_auc']:.4f}")
    c[2].metric("Precision", f"{test['precision_at_threshold']:.3f}")
    c[3].metric("Recall", f"{test['recall_at_threshold']:.3f}")
    c[4].metric("F1", f"{test['f1_at_threshold']:.3f}")
    c[5].metric("False positives", f"{cm['fp']:,}")
    c[6].metric("False negatives", f"{cm['fn']:,}")

    try:
        y_test, proba = data.load_test_set_scores()
    except data.MissingArtifact as e:
        st.warning(str(e))
        return
    left, mid, right = st.columns(3)
    with left:
        _confusion_chart(cm, f"Confusion matrix @ {threshold:.3f}")
    fpr, tpr, _ = roc_curve(y_test, proba)
    with mid:
        fig = px.area(x=fpr, y=tpr, title=f"ROC curve (AUC = {roc_auc_score(y_test, proba):.4f})")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        _chart(fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate"), 350)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    with right:
        fig = px.line(x=rec, y=prec, title=f"Precision-recall curve (AP = {average_precision_score(y_test, proba):.4f})")
        fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="grey",
                      annotation_text=f"random baseline ({y_test.mean():.2%})")
        _chart(fig.update_layout(xaxis_title="Recall", yaxis_title="Precision"), 350)

    with st.expander("Explore other thresholds on the test set"):
        st.caption("For illustration only: the deployed threshold was fixed on the validation set. Choosing it here "
                   "would tune on the test labels.")
        t = st.slider("Decision threshold", 0.05, 0.99, float(round(threshold, 2)), 0.01)
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        c = st.columns(6)
        c[0].metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
        c[1].metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
        c[2].metric("F1", f"{f1_score(y_test, pred, zero_division=0):.3f}")
        c[3].metric("Flagged", f"{tp + fp:,}")
        c[4].metric("Fraud caught", f"{tp:,} / {tp + fn:,}")
        c[5].metric("False alarms", f"{fp:,}")

    st.subheader("Model selection: validation set")
    st.caption(f"Validation = the latest {cfg.VALIDATION_FRACTION:.0%} of `fraudTrain.csv` by time "
               f"({split['validation_period'][0][:10]} to {split['validation_period'][1][:10]}, "
               f"{split['validation_rows']:,} rows, {split['validation_fraud']:,} fraud). Models were fit on the earlier "
               f"{split['fit_rows']:,} rows ({split['fit_period'][0][:10]} to {split['fit_period'][1][:10]}).")
    rows = pd.DataFrame(metrics["models"])
    table = rows[["model", "roc_auc", "pr_auc", "precision_at_0.5", "recall_at_0.5", "f1_at_0.5",
                  "decision_threshold", "precision_at_threshold", "recall_at_threshold", "f1_at_threshold",
                  "train_seconds"]].copy()
    table["model"] = table["model"].map(lambda m: MODEL_LABELS.get(m, m) + (" (selected)" if m == best else ""))
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in table.columns
                                if c not in ("model", "train_seconds")})

    left, right = st.columns(2)
    with left:
        long = rows.melt(id_vars="model", value_vars=["roc_auc", "pr_auc", "precision_at_threshold",
                                                      "recall_at_threshold", "f1_at_threshold"],
                         var_name="metric", value_name="value")
        long["model"] = long["model"].map(lambda m: MODEL_LABELS.get(m, m))
        long["metric"] = long["metric"].str.replace("_at_threshold", " @thr").str.replace("_", "-").str.upper()
        _chart(px.bar(long, x="metric", y="value", color="model", barmode="group", title="Validation comparison")
               .update_layout(xaxis_title="", yaxis_title="", yaxis_range=[0, 1], legend=dict(orientation="h", y=-0.2)), 400)
    with right:
        choice = st.selectbox("Validation confusion matrix for", rows["model"], format_func=lambda m: MODEL_LABELS.get(m, m),
                              index=int(rows.index[rows["model"] == best][0]))
        m = next(m for m in metrics["models"] if m["model"] == choice)
        _confusion_chart(m["confusion_matrix_at_threshold"],
                         f"{MODEL_LABELS.get(choice, choice)} @ {m['decision_threshold']:.3f} (validation)")

    st.info("**Why PR-AUC rather than ROC-AUC or accuracy?** With under 1% fraud, a model that never flags "
            "anything is over 99% accurate, and ROC-AUC is dominated by the huge legitimate class. PR-AUC "
            "only measures how well fraud is ranked, which is why it is used to select the model. " + DATA_NOTE)


# --------------------------------------------------------------------------
# E. Fraud alerts (MongoDB)
# --------------------------------------------------------------------------
def render_alerts():
    st.header("Fraud alerts")
    try:
        collection, backend = data.load_alerts_collection()
    except data.MissingArtifact as e:
        st.warning(str(e))
        return
    except MongoUnavailable as e:
        st.error(f"MongoDB is not available, so no fraud alerts can be shown. {e}")
        return
    total = collection.count_documents({})
    if backend == "mongodb":
        st.success(f"Connected to MongoDB at `{cfg.MONGO_DB_NAME}.fraud_alerts` ({total:,} alert documents).")
    else:
        st.info(f"BDA_MONGO_URI is not set and no MongoDB server answered at `{cfg.MONGO_URI}`, so this page uses "
                f"the in-memory **mongomock** fallback (set `BDA_MONGO_FALLBACK=0` to disable it), "
                f"loaded with the {total:,} top-scored alerts by `src/nosql/store_analytics.py`.")
    st.caption("Alerts are the highest model scores among **fraudTest.csv** transactions (out-of-sample). "
               "`predicted_fraud` is the model's decision at the validation-chosen threshold; `actual_label` is the "
               "dataset's real `is_fraud` label.")

    cats = sorted(collection.distinct("category"))
    states = sorted(collection.distinct("customer.state"))
    c = st.columns(4)
    sel_cats = c[0].multiselect("Category", cats)
    sel_states = c[1].multiselect("State", states)
    label = c[2].selectbox("Actual label (is_fraud)", ["All", "Fraud", "Legitimate"])
    min_p = c[3].slider("Min. fraud probability", 0.0, 1.0, 0.0, 0.01)
    c = st.columns(3)
    sort_labels = {"Fraud probability": "predicted_fraud_probability", "Amount": "amount", "Timestamp": "timestamp"}
    sort = c[0].selectbox("Sort by", list(sort_labels))
    descending = c[1].radio("Order", ["Descending", "Ascending"], horizontal=True) == "Descending"
    limit = c[2].number_input("Max rows", 10, 5000, 200, 10)

    actual = {"All": None, "Fraud": 1, "Legitimate": 0}[label]
    alerts, query = data.query_alerts(collection, sel_cats, sel_states, min_p, actual, sort_labels[sort], descending, limit)
    with st.expander("MongoDB query"):
        st.code(f"db.fraud_alerts.find({json.dumps(query)}).sort({{'{sort_labels[sort]}': {-1 if descending else 1}}}).limit({limit})",
                language="javascript")
    if alerts.empty:
        st.warning("No alerts match these filters.")
        return

    precision = alerts["actual_label"].mean()
    c = st.columns(3)
    c[0].metric("Alerts shown", f"{len(alerts):,}")
    c[1].metric("Actually fraud (label)", f"{int(alerts['actual_label'].sum()):,}")
    c[2].metric("Alert precision", f"{precision:.1%}")
    st.dataframe(alerts, hide_index=True, width="stretch",
                 column_config={"predicted_fraud_probability": st.column_config.ProgressColumn(
                     "fraud probability", min_value=0.0, max_value=1.0, format="%.3f"),
                     "amount": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Download alerts (CSV)", alerts.to_csv(index=False), "fraud_alerts.csv", "text/csv")

    by_cat = alerts.groupby("category").agg(alerts=("trans_num", "size"), actual_fraud=("actual_label", "sum")).reset_index()
    fig = px.bar(by_cat.sort_values("alerts"), x=["alerts", "actual_fraud"], y="category", orientation="h",
                 barmode="overlay", title="Alerts by category", color_discrete_sequence=[LEGIT_COLOR, FRAUD_COLOR])
    _chart(fig.update_layout(xaxis_title="", yaxis_title=""), 420)


# --------------------------------------------------------------------------
# F. CSV analysis
# --------------------------------------------------------------------------
def render_csv_analysis():
    st.header("Batch CSV analysis")
    bundle, threshold = data.load_model()
    known = inf.known_categorical_values(bundle)
    st.markdown(f"Upload transactions in the Kaggle schema (for example rows of `fraudTest.csv`; extra columns such "
                f"as the unnamed index, `street` or names are ignored). "
                f"Required columns: `{'`, `'.join(inf.REQUIRED_INPUT_COLUMNS)}`. Category, state and gender must be "
                f"values the model was trained on. Rows that fail validation are reported and skipped. If an "
                f"`{cfg.LABEL_COL}` column is present, the scores are evaluated against it.")
    template = pd.DataFrame([{
        cfg.TIMESTAMP_COL: "2020-09-12 14:30:00", cfg.AMOUNT_COL: 64.5, cfg.CATEGORY_COL: known["category"][0],
        cfg.STATE_COL: known["state"][0], cfg.GENDER_COL: known["gender"][0], cfg.DOB_COL: "1985-03-12",
        cfg.CITY_POP_COL: 905748, cfg.LAT_COL: 39.9612, cfg.LONG_COL: -82.9988, cfg.MERCH_LAT_COL: 40.12, cfg.MERCH_LONG_COL: -83.15,
    }])
    st.download_button("Download a template CSV", template.to_csv(index=False), "transaction_template.csv", "text/csv")

    upload = st.file_uploader("Transaction CSV", type=["csv"])
    if upload is None:
        return
    try:
        df = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
        return
    try:
        with st.spinner(f"Scoring {len(df):,} rows..."):
            scored, errors = inf.predict(df, bundle, threshold)
    except ValueError as e:
        st.error(str(e))
        return

    c = st.columns(4)
    c[0].metric("Rows uploaded", f"{len(df):,}")
    c[1].metric("Scored", f"{len(scored):,}")
    c[2].metric("Rejected", f"{len(errors):,}")
    c[3].metric("Predicted fraud (High risk)", f"{int(scored['predicted_fraud'].sum()) if len(scored) else 0:,}")
    if errors:
        with st.expander(f"{len(errors):,} rejected rows"):
            st.dataframe(pd.DataFrame(errors).head(1000), hide_index=True, width="stretch")
    if scored.empty:
        return

    left, right = st.columns(2)
    with left:
        counts = scored["risk_level"].value_counts().reindex(["Low", "Elevated", "High"], fill_value=0).reset_index()
        fig = px.bar(counts, x="risk_level", y="count", color="risk_level", color_discrete_map=RISK_COLORS, text="count",
                     title="Risk level distribution")
        _chart(fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Transactions"))
    with right:
        _chart(px.histogram(scored, x="fraud_probability", nbins=50, title="Fraud probability distribution")
               .update_layout(xaxis_title="Fraud probability", yaxis_title="Transactions"))

    if cfg.LABEL_COL in scored.columns:
        y = pd.to_numeric(scored[cfg.LABEL_COL], errors="coerce")
        ok = y.isin([0, 1])
        y, p = y[ok].astype(int), scored.loc[ok, "fraud_probability"]
        if y.nunique() == 2:
            pred = (p >= threshold).astype(int)
            c = st.columns(4)
            c[0].metric("ROC-AUC", f"{roc_auc_score(y, p):.3f}")
            c[1].metric("PR-AUC", f"{average_precision_score(y, p):.3f}")
            c[2].metric(f"Precision @ {threshold:.3f}", f"{precision_score(y, pred, zero_division=0):.3f}")
            c[3].metric(f"Recall @ {threshold:.3f}", f"{recall_score(y, pred, zero_division=0):.3f}")
            st.caption(f"Computed against the uploaded `{cfg.LABEL_COL}` labels. Rows from `fraudTrain.csv` were seen "
                       "in training, so their scores are optimistic; `fraudTest.csv` rows are out-of-sample.")
        else:
            st.caption(f"`{cfg.LABEL_COL}` contains only one class, so evaluation metrics are skipped.")

    out_cols = [c for c in df.columns if c in scored.columns] + ["fraud_probability", "predicted_fraud", "risk_level"]
    result = scored[out_cols].sort_values("fraud_probability", ascending=False)
    st.dataframe(result.head(1000), hide_index=True, width="stretch")
    st.download_button("Download scored transactions (CSV)", result.to_csv(index=False), "scored_upload.csv", "text/csv")
    st.info(PREDICTION_DISCLAIMER)
