"""
Streamlit Demo — Uplift Modeling for Email Marketing
=======================================================
This mirrors the exact pipeline built across Phases 1-4 of your notebook:
data loading -> binary subset prep -> T-Learner training -> uplift scoring
-> business value translation. Same functions, same logic, just wrapped in
an interactive UI.

Run with: streamlit run app.py
(make sure hillstrom.csv is in the same folder as this file)
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklift.models import TwoModels
from sklift.metrics import qini_auc_score, qini_curve
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Uplift Marketing Demo", layout="wide")

DATA_PATH = "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
TARGET = "conversion"


# ---------------------------------------------------------------
# Same functions as Phase 2 / Phase 3 of your notebook
# ---------------------------------------------------------------

def prepare_binary_subset(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["segment"].isin(["Mens E-Mail", "No E-Mail"])].copy()
    subset["treatment"] = (subset["segment"] == "Mens E-Mail").astype(int)
    subset = pd.get_dummies(subset, columns=["zip_code", "channel"], drop_first=True)
    return subset


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = ["segment", "treatment", "visit", "conversion", "spend", "history_segment"]
    return [c for c in df.columns if c not in exclude]


def make_base_estimator(random_state=42):
    return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          eval_metric="logloss", random_state=random_state)


def business_value_by_targeting_level(uplift_pred, trt_test, spend_test,
                                       k_values=(0.1, 0.2, 0.3, 0.5, 1.0),
                                       cost_per_email=1.0):
    n = len(uplift_pred)
    order = np.argsort(-uplift_pred)
    rows = []
    for k in k_values:
        top_n = max(1, int(n * k))
        idx = order[:top_n]
        bucket_trt = trt_test.values[idx]
        bucket_spend = spend_test.values[idx]
        treated_avg = bucket_spend[bucket_trt == 1].mean()
        control_avg = bucket_spend[bucket_trt == 0].mean()
        inc = treated_avg - control_avg
        proj = inc * top_n
        cost = cost_per_email * top_n
        net = proj - cost
        roi = proj / cost if cost > 0 else np.nan
        rows.append({
            "targeted_pct": f"{int(k * 100)}%", "n_emailed": top_n,
            "incremental_$_per_customer": round(inc, 3),
            "projected_incremental_revenue": round(proj, 2),
            "campaign_cost": round(cost, 2), "net_value": round(net, 2),
            "roi": round(roi, 2) if not np.isnan(roi) else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Cached data loading and model training (only runs once per session)
# ---------------------------------------------------------------

@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)


@st.cache_resource
def train_t_learner(_sub, feats):
    """Same T-Learner (TwoModels) as Phase 3 of the notebook. Cached so the
    model only trains once, not on every widget interaction."""
    X = _sub[feats]
    y = _sub[TARGET]
    trt = _sub["treatment"]
    X_train, X_test, y_train, y_test, trt_train, trt_test = train_test_split(
        X, y, trt, test_size=0.3, random_state=42, stratify=trt
    )
    model = TwoModels(
        estimator_trmnt=make_base_estimator(42),
        estimator_ctrl=make_base_estimator(43),
        method="vanilla",
    )
    model.fit(X_train, y_train, trt_train)
    return model, X_test, y_test, trt_test


# ---------------------------------------------------------------
# App layout
# ---------------------------------------------------------------

st.title("Uplift Modeling — Who Should Get the Marketing Email?")
st.caption(
    "T-Learner uplift model (scikit-uplift) trained on the Hillstrom E-Mail dataset — "
    "identifies customers who buy *because* they were emailed, not just customers likely to buy anyway."
)

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Place the Hillstrom CSV in the same folder as this app, then refresh."
    )
    st.stop()

sub = prepare_binary_subset(df)
feats = get_feature_columns(sub)

with st.spinner("Training T-Learner (only happens once per session)..."):
    model, X_test, y_test, trt_test = train_t_learner(sub, feats)

spend_test = sub.loc[X_test.index, "spend"]
uplift_pred_test = model.predict(X_test)

tab1, tab2, tab3 = st.tabs(["Score a Customer", "Business Value", "Model Performance"])

# ---------------- Tab 1: score a single customer ----------------
with tab1:
    st.subheader("Predict uplift for a single customer profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.slider("Recency (months since last purchase)", 1, 12, 6)
        history = st.number_input("Lifetime spend history ($)", 0.0, 3000.0, 250.0, step=10.0)
    with col2:
        mens = st.selectbox("Has bought men's merchandise before", [0, 1], index=0)
        womens = st.selectbox("Has bought women's merchandise before", [0, 1], index=1)
        newbie = st.selectbox("New customer (newbie)", [0, 1], index=0)
    with col3:
        zip_code = st.selectbox("Zip code area", ["Urban", "Surburban", "Rural"])
        channel = st.selectbox("Purchase channel", ["Web", "Phone", "Multichannel"])

    # Build a single-row dataframe matching the training feature columns exactly
    row = {c: 0 for c in feats}
    for col_name, val in [("recency", recency), ("history", history), ("mens", mens),
                           ("womens", womens), ("newbie", newbie)]:
        if col_name in row:
            row[col_name] = val
    zip_col = f"zip_code_{zip_code}"
    if zip_col in row:
        row[zip_col] = 1
    channel_col = f"channel_{channel}"
    if channel_col in row:
        row[channel_col] = 1

    X_single = pd.DataFrame([row])[feats]
    score = float(model.predict(X_single)[0])

    st.metric("Predicted uplift score", f"{score:.4f}")

    pct = float((uplift_pred_test < score).mean() * 100)
    if score > np.percentile(uplift_pred_test, 80):
        label = "Persuadable — strong candidate to target"
    elif score > np.percentile(uplift_pred_test, 50):
        label = "Mild responder — target if budget allows"
    elif score > 0:
        label = "Weak / unclear effect — low priority"
    else:
        label = "Possible Sleeping Dog — consider excluding from campaign"

    st.markdown(f"### {label}")
    st.caption(f"This customer's predicted uplift ranks higher than {pct:.0f}% of test customers.")

# ---------------- Tab 2: business value ----------------
with tab2:
    st.subheader("What would this model be worth if deployed?")
    cost_per_email = st.slider("Assumed cost per email ($)", 0.1, 5.0, 1.0, step=0.1)
    biz = business_value_by_targeting_level(uplift_pred_test, trt_test, spend_test,
                                             cost_per_email=cost_per_email)
    st.dataframe(biz, width="stretch")

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(biz["targeted_pct"], biz["net_value"], color="seagreen", alpha=0.7)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_ylabel("Net value ($)", color="seagreen")
    ax2 = ax1.twinx()
    ax2.plot(biz["targeted_pct"], biz["roi"], color="darkorange", marker="o")
    ax2.axhline(1, color="darkorange", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("ROI", color="darkorange")
    st.pyplot(fig)

    best_row = biz.loc[biz["net_value"].idxmax()]
    st.success(
        f"Highest total net value: targeting **{best_row['targeted_pct']}** of customers → "
        f"**${best_row['net_value']:,.0f}** net value at **{best_row['roi']}x** ROI "
        f"(assuming ${cost_per_email:.2f}/email)."
    )

# ---------------- Tab 3: model performance ----------------
with tab3:
    st.subheader("Qini Curve — model vs. random targeting")
    n = len(y_test)
    order, qini_values = qini_curve(y_test.values, uplift_pred_test, trt_test.values)
    x_pct = np.array(order) / n * 100

    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_pct, qini_values, label="T-Learner", linewidth=2)
    ax.plot([0, 100], [0, qini_values[-1]], linestyle="--", color="gray", label="Random targeting")
    ax.set_xlabel("% of population targeted")
    ax.set_ylabel("Cumulative incremental responders")
    ax.legend()
    st.pyplot(fig2)

    qini_score = qini_auc_score(y_test, uplift_pred_test, trt_test)
    st.metric("Qini AUC", f"{qini_score:.4f}")
