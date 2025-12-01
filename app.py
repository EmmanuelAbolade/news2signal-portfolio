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
import yfinance as yf


# ---------- Page config ----------
st.set_page_config(page_title="News→Signal Dashboard", layout="wide")

# ------------------- Portfolio Header -------------------
st.title("📰 News→Signal: Financial Sentiment & Market Insight")
st.markdown("""
### **Name:** Emmanuel Abolade  
**Institution:** South East Technological University 
**Course:** Software Development 
**Module:** Data Science & Machine Learning Portfolio  
**Date:** November 2025  

---

This portfolio demonstrates the application of **Data Science** and **Machine Learning**
to explore how financial news sentiment influences stock market movements.

It includes:
- A sentiment analysis model trained on the *Financial PhraseBank* dataset.  
- Integration of daily news sentiment with market data (SPY, AAPL, etc.).  
- Visual analytics through this interactive Streamlit dashboard.  
- Real-time price tracking powered by Yahoo Finance.  

---

**Aim:**  
To evaluate whether daily financial news sentiment can serve as a signal 
for predicting short-term market direction.

---
""")

# ---------- About this Portfolio (expandable section) ----------
with st.expander("ℹ️ **About this Portfolio**", expanded=False):
    st.markdown("""
    ### Overview  
    This project, **News→Signal**, demonstrates how **Data Science** and **Machine Learning** 
    can be applied to **financial text analysis** — exploring how the *sentiment* of daily 
    financial news correlates with *market price movements*.

    ### What This App Shows  
    - The relationship between **daily sentiment** and **next-day stock direction**.  
    - Interactive charts showing **price trends**, **headline counts**, and **sentiment strength**.  
    - Correlation analysis between **news sentiment** and **market returns**.  
    - Real-time stock snapshots from Yahoo Finance.

    ### Workflow Summary  
    - **Notebook 01** → Trained a sentiment classifier using the *Financial PhraseBank* dataset.  
    - **Notebook 02** → Merged sentiment with market data to build a unified dataset.  
    - **Streamlit Dashboard** → Visualizes both datasets and provides live updates.  

    ### Key Tools  
    - **Python**, **Pandas**, **NumPy**, **Scikit-Learn**, **YFinance**, **Matplotlib**, and **Streamlit**.  

    ### Interpretation  
    Periods of higher **positive sentiment** often correlate with an **increase** in next-day 
    returns, while negative sentiment tends to precede price drops.  
    The model and charts here help visualize that subtle relationship.

    ---
    **Created by:** *Emmanuel Abolade (SETU Carlow)*  
    **Portfolio Date:** November 2025  
    """)


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

# ---------- Sidebar: Select Ticker for Display ----------
st.sidebar.header("Market Selection")
available_tickers = ["SPY", "AAPL", "MSFT", "AMZN", "GOOG", "TSLA"]
ticker = st.sidebar.selectbox("Choose Ticker", available_tickers, index=0)

# Download fresh price data for selected ticker (using same date range)
px = yf.download(ticker, start=str(f_start), end=str(f_end), auto_adjust=True).reset_index()

# --- Clean up any multi-index columns before merging ---
if isinstance(px.columns, pd.MultiIndex):
    px.columns = ['_'.join([str(c) for c in col if c not in (None, '')]) for col in px.columns]

if isinstance(dff.columns, pd.MultiIndex):
    dff.columns = ['_'.join([str(c) for c in col if c not in (None, '')]) for col in dff.columns]

# --- Normalize date and close columns ---
if "Close" in px.columns:
    close_col = "Close"
elif "Adj Close" in px.columns:
    close_col = "Adj Close"
else:
    close_candidates = [c for c in px.columns if str(c).lower().endswith("close")]
    close_col = close_candidates[0] if close_candidates else None

if close_col:
    px = px.rename(columns={"Date": "date", close_col: "close"})[["date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"]).dt.date

    # --- Merge safely ---
    if "date" in dff.columns:
        dff = pd.merge(dff, px, on="date", how="left", suffixes=("", "_new"))
        dff["close"] = dff["close_new"].fillna(dff["close"])
        dff.drop(columns=["close_new"], inplace=True, errors="ignore")
else:
    st.warning(f"No close column found for {ticker}, skipping merge.")



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
    st.subheader(f"{ticker} Close Price (adjusted)")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(dff["date"], dff["close"])
    ax.set_xlabel("Date"); ax.set_ylabel("Close")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("""
This line chart visualizes the daily closing price of the SPY index, which tracks the S&P 500. 
It provides context for the overall market movement during the selected date range. 
A steady upward trend suggests bullish market conditions, while steep drops indicate periods of correction or volatility.
""")

with c2:
    st.subheader(f"{ticker} Close Price (adjusted)")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(dff["date"], dff["sent_mean"])
    ax.set_xlabel("Date"); ax.set_ylabel("sent_mean")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("""
This plot captures the average sentiment of financial headlines per day, as computed by the sentiment analysis model. 
Higher values represent more positive news tone (optimism in markets), while lower values reflect negative or cautious outlooks. 
Fluctuations here often mirror investor mood and anticipation around earnings, policy decisions, or economic reports.
""")


# ---------- Row 2: Headline count & class distribution ----------
c3, c4 = st.columns(2)
with c3:
    st.subheader(f"{ticker} Close Price (adjusted)")
    fig, ax = plt.subplots(figsize=(8,3))
    ax.bar(dff["date"], dff["sent_count"])
    ax.set_xlabel("Date"); ax.set_ylabel("sent_count")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("""
This bar chart indicates how many news headlines were recorded each day in the dataset. 
Spikes in headline volume usually correspond to major financial events, policy announcements, or corporate earnings seasons. 
Periods with high news intensity can drive short-term market volatility and sentiment swings.
""")


with c4:
    st.subheader(f"{ticker} Close Price (adjusted)")
    fig, ax = plt.subplots(figsize=(5,3))
    counts = dff["next_day_direction"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Class"); ax.set_ylabel("Count")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("""
This histogram displays the distribution of the target variable — market direction for the following day. 
A balanced dataset (roughly equal 0s and 1s) ensures fair model learning, while imbalance could bias predictions toward one class. 
Here, Class 1 indicates a price increase and Class 0 represents a price decline.
""")


# ---------- Row 3: Sentiment vs Return & Correlation ----------
c5, c6 = st.columns(2)
with c5:
    st.subheader(f"{ticker} Close Price (adjusted)")
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
    st.caption("""
Each point represents one trading day, plotting daily sentiment against the next day's percentage return. 
The orange trend line shows the general relationship: when sentiment is more positive, the market tends to deliver slightly higher returns the next day. 
Although not perfectly predictive, this pattern hints that investor optimism captured in headlines may carry over into short-term market momentum.
""")


with c6:
    st.subheader(f"{ticker} Close Price (adjusted)")
    feat_cols = ["sent_mean","sent_count","sent_max","return_1d","ma_5","vol_5","next_day_direction"]
    corr = dff[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr, interpolation="nearest")
    ax.set_xticks(range(len(feat_cols))); ax.set_xticklabels(feat_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(feat_cols))); ax.set_yticklabels(feat_cols)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("""
This correlation heatmap quantifies how sentiment and technical features relate to market outcomes. 
For instance, strong correlations between `sent_mean` and `next_day_direction` suggest that news tone may contain predictive signals. 
Weaker correlations with volatility or moving averages reveal where sentiment diverges from technical indicators.
""")


# ---------- Model Snapshot & Explanation ----------
st.markdown("---")
st.subheader("Sentiment Model Snapshot")

if os.path.exists("models/sentiment_baseline.pkl"):
    st.markdown("""
    This section shows information about the **Financial Sentiment Model**
    trained in **Notebook 01** using the *Financial PhraseBank* dataset.
    The model uses a **TF-IDF vectorizer** and **Logistic Regression** to classify
    sentences as **positive**, **neutral**, or **negative**.
    """)

    try:
        model = joblib.load("models/sentiment_baseline.pkl")
        st.success("Sentiment model successfully loaded from models/sentiment_baseline.pkl")

        st.markdown("""
        **Model Type:** Logistic Regression  
        **Vectorizer:** TF-IDF (max features = 5000)  
        **Purpose:** Predict the tone of financial news headlines or sentences.  
        """)
        st.info("""
        Although this dashboard currently focuses on market-level correlations,
        this model can be extended to generate daily sentiment automatically from new headlines.
        """)
    except Exception as e:
        st.warning(f"Could not load sentiment model: {e}")
else:
    st.info("""
    The trained **sentiment model** is not required for dashboard operation,
    but when available it provides richer analytics and automated news scoring.
    To include it, ensure that `models/sentiment_baseline.pkl` exists.
    """)

st.markdown("---")
with st.expander("**Live Prototype: Generate Sentiment from Today’s Headlines**", expanded=False):
    st.markdown("""
    This section demonstrates how this project can evolve into a **production-ready application**.
    Instead of relying on static CSV files, the model can fetch *live financial headlines*,
    analyze them in real-time using the trained sentiment model, and generate updated daily sentiment scores.
    """)

    if os.path.exists("models/sentiment_baseline.pkl"):
        model = joblib.load("models/sentiment_baseline.pkl")

        # --- Try to fetch live business headlines (requires internet connection) ---
        try:
            import yfinance as yf
            import requests
            import pandas as pd
            from datetime import datetime

            # --- diagnostic block ---
            st.sidebar.write("Secrets content:", dict(st.secrets))
            api_key_value = st.secrets.get("NEWS_API_KEY")
            st.sidebar.write("NEWS_API_KEY:", api_key_value)
            
            api_url = "https://newsapi.org/v2/top-headlines"
            params = {
                "category": "business",
                "language": "en",
                "apiKey": st.secrets.get("NEWS_API_KEY", "DEMO_KEY")  # optional secret key
            }
            response = requests.get(api_url, params=params)
            if response.status_code == 200:
                data = response.json()
                headlines = [a["title"] for a in data["articles"] if a["title"]]
                df_live = pd.DataFrame({"headline": headlines})
                st.write(f"Fetched {len(df_live)} live headlines.")
                preds = model.predict(df_live["headline"])
                df_live["sentiment"] = preds
                st.dataframe(df_live.head(10))

                st.success("Real-time sentiment predictions generated.")
            else:
                st.warning("Could not fetch live news — API key or internet may be missing.")

        except Exception as e:
            st.warning(f"Live fetch unavailable: {e}")
    else:
        st.info("The live sentiment model requires models/sentiment_baseline.pkl to be available.")



st.markdown("---")
st.caption("""
**© 2025 – News→Signal Portfolio by Emmanuel Abolade**  
Built with using Streamlit, Pandas, Scikit-Learn, and YFinance.
""")

