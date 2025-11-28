# app.py
# Streamlit dashboard for the Market Prediction Dataset
# Loads ../data/processed/market_dataset.csv and shows interactive charts.
# Run it in terminal with:  streamlit run app.py

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

DATA_PATH = os.path.join("data", "processed", "market_dataset.csv")
MODEL_PATH = os.path.join("models", "next_day_model.pkl")  # optional

# ---------- Page config ----------
st.set_page_config(page_title="News→Signal Dashboard", layout="wide")

# ---------- Sidebar: data loader ----------
st.sidebar.header("Data")
st.sidebar.write("This dashboard reads the processed dataset:")
st.sidebar.code(DATA_PATH, language="text")

uploaded = st.sidebar.file_uploader("Or upload a CSV with the same columns", type=["csv"])

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, parse_dates=["date"])
    expected = {
        "date", "close", "close_t+1", "next_day_direction",
        "return_1d", "ma_5", "vol_5", "sent_mean", "sent_count", "sent_max"
    }
    missing = expected - set(df.columns)
    if missing:
        st.warning(f"Missing columns: {missing}")
    df = df.sort_values("date").reset_index(drop=True)
    return df

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
else:
    if not os.path.exists(DATA_PATH):
        st.error("Can't find the dataset. Make sure it exists at data/processed/market_dataset.csv")
        st.stop()
    df = load_data(DATA_PATH)

# ---------- Sidebar: date filter ----------
st.sidebar.header("Filters")
min_d, max_d = df["date"].min(), df["date"].max()
f_start, f_end = st.sidebar.date_input(
    "Date range",
    value=(min_d.to_pydatetime().date(), max_d.to_pydatetime().date()),
    min_value=min_d.to_pydatetime().date(),
    max_value=max_d.to_pydatetime().date()
)

mask = (df["date"] >= pd.to_datetime(f_start)) & (df["date"] <= pd.to_datetime(f_end))
dff = df.loc[mask].copy()
if dff.empty:
    st.warning("No rows in the selected date range.")
    st.stop()

# ---------- Top metrics ----------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{len(dff):,}")
with col2:
    up_rate = dff["next_day_direction"].mean() if "next_day_direction" in dff else np.nan
    st.metric("Up-days (%)", f"{100*up_rate:.1f}" if pd.notna(up_rate) else "—")
with col3:
    st.metric("Avg Sentiment", f"{dff['sent_mean'].mean():.3f}")
with col4:
    st.metric("Avg 1D Return", f"{dff['return_1d'].mean():.4f}")

st.markdown("---")

# ---------- Row 1: Price & Sentiment ----------
c1, c2 = st.columns(2)
with c1:
    st.subheader("SPY Close Price (adjusted)")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(dff["date"], dff["close"])
    ax.set_xlabel("Date"); ax.set_ylabel("Close")
    fig.tight_layout()
    st.pyplot(fig)
with c2:
    st.subheader("Daily Mean Sentiment")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(dff["date"], dff["sent_mean"])
    ax.set_xlabel("Date"); ax.set_ylabel("sent_mean")
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Row 2: Headline count & class distribution ----------
c3, c4 = st.columns(2)
with c3:
    st.subheader("Headline Count per Day")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(dff["date"], dff["sent_count"])
    ax.set_xlabel("Date"); ax.set_ylabel("sent_count")
    fig.tight_layout()
    st.pyplot(fig)
with c4:
    st.subheader("Next-Day Direction (0/1)")
    fig, ax = plt.subplots(figsize=(5,3))
    counts = dff["next_day_direction"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Row 3: Sentiment vs Return & Correlation ----------
c5, c6 = st.columns(2)
with c5:
    st.subheader("Sentiment vs Next-Day Return")
    dff["next_day_return"] = (dff["close_t+1"] / dff["close"]) - 1

    # Drop NaN or infinite values before regression
    clean = dff[["sent_mean", "next_day_return"]].replace([np.inf, -np.inf], np.nan).dropna()

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(clean["sent_mean"], clean["next_day_return"], alpha=0.6)
    ax.set_xlabel("sent_mean")
    ax.set_ylabel("next_day_return")

    # Only fit line if we have enough valid points
    if len(clean) > 2:
        m = np.polyfit(clean["sent_mean"], clean["next_day_return"], 1)
        xline = np.linspace(clean["sent_mean"].min(), clean["sent_mean"].max(), 100)
        yline = m[0]*xline + m[1]
        ax.plot(xline, yline, color='orange', linewidth=2, label="Trend line")
        ax.legend()
    else:
        st.warning("Not enough valid points to fit a trend line.")

    fig.tight_layout()
    st.pyplot(fig)

with c6:
    st.subheader("Feature Correlation")
    feat_cols = ["sent_mean","sent_count","sent_max","return_1d","ma_5","vol_5","next_day_direction"]
    corr = dff[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr, interpolation="nearest")
    ax.set_xticks(range(len(feat_cols))); ax.set_xticklabels(feat_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(feat_cols))); ax.set_yticklabels(feat_cols)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)

# ---------- Optional model snapshot ----------
st.markdown("---")
st.subheader("Optional: Model Snapshot")
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        st.write("Loaded model:", type(model).__name__)
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_,
                            index=["sent_mean","sent_count","sent_max","return_1d","ma_5","vol_5"]).sort_values(ascending=False)
            st.dataframe(imp.to_frame("importance"))
        else:
            st.info("This model type does not expose feature_importances_.")
    except Exception as e:
        st.warning(f"Could not load model: {e}")
else:
    st.info("No trained next-day model found at models/next_day_model.pkl (will appear after Notebook 03).")

st.caption("© 2025 News→Signal Portfolio • Built with Streamlit")
