# News→Signal: Financial Sentiment & Market Insight  

## Student: Emmanuel Abolade  
Institution: South East Technological University, Carlow  
Module: Data Science & Machine Learning Portfolio  
Date: November 2025  

---

## Abstract  

This portfolio project investigates the relationship between **financial news sentiment** and **stock market movements** using machine learning and data science techniques.  
It begins with building a sentiment classifier on the *Financial PhraseBank dataset*, proceeds to merge sentiment scores with daily market data, and concludes with an interactive dashboard and a live prototype that fetches real-time financial headlines.  
The workflow reflects the complete data science lifecycle — from raw data processing and feature engineering to visualization, model evaluation, and live deployment.

---

## Project Overview  

**News→Signal** demonstrates how data science can be applied to financial text analysis.  
The project explores whether the *tone of daily financial news* influences short-term market direction.  
By integrating **NLP**, **market data analysis**, and **interactive visualization**, it shows how textual sentiment can complement traditional financial indicators.

---

## Objectives  

- Train a sentiment classifier using the Financial PhraseBank dataset.  
- Combine predicted sentiment with historical stock data from Yahoo Finance.  
- Derive analytical features such as daily returns, moving averages, and volatility.  
- Explore relationships between sentiment and market direction through visual analytics.  
- Develop a live prototype that generates sentiment automatically from current business headlines.  

---

## Project Workflow  

### 1. Dataset Preparation (Notebook 01)  
- The *Financial PhraseBank dataset* was used to train a Logistic Regression classifier.  
- Preprocessing steps included tokenization, text cleaning, and TF–IDF vectorization.  
- The final model was saved as `models/sentiment_baseline.pkl` using Joblib.  

### 2. Market Dataset Construction (Notebook 02)  
- Downloaded market data for SPY, AAPL, TSLA, and others using `yfinance`.  
- Merged news sentiment scores with technical indicators such as daily return, moving average (MA5), and volatility (Vol5).  
- Created the unified dataset `data/processed/market_dataset.csv`.  

### 3. Interactive Dashboard (Streamlit)  
- The dashboard visualizes sentiment and market data interactively:  
  - Daily price trends  
  - Sentiment averages and counts  
  - Feature correlations  
  - Relationship between sentiment and next-day returns  

### 4. Live Sentiment Prototype  
- Integrated the NewsAPI to fetch live business headlines.  
- Applied the trained sentiment model in real time to classify the tone of incoming headlines.  
- Demonstrated how this can extend into a production-level application.

---

## Key Features  

- Interactive dashboard powered by Streamlit.  
- Real-time sentiment prediction using the trained Logistic Regression model.  
- Clear data processing pipeline across structured notebooks.  
- Reusable modular design for future extensions (multi-stock support, forecasting).  

---

## Project Structure  

news2signal/
│
├── app.py
├── requirements.txt
├── .streamlit/
│ └── secrets.toml
│
├── data/
│ ├── raw/
│ └── processed/
│ └── market_dataset.csv
│
├── models/
│ ├── sentiment_baseline.pkl
│ └── next_day_model.pkl
│
├── notebooks/
│ ├── 01_train_sentiment_baseline.ipynb
│ └── 02_build_market_dataset.ipynb
│
└── README.md


---

## Technologies Used  

- Python  
- Pandas, NumPy  
- Scikit-Learn, Joblib  
- Matplotlib, Seaborn  
- Streamlit  
- YFinance (market data)  
- NewsAPI (live news integration)

---

## How to Run Locally  

1. Clone the repository  
git clone https://github.com/EmmanuelAbolade/news2signal-portfolio.git

cd news2signal-portfolio


2. Create and activate a virtual environment  

conda create -n news2signal python=3.11
conda activate 


3. Install dependencies  

pip install -r requirements.txt


4. Add your NewsAPI key  
Create a file `.streamlit/secrets.toml` and insert:

[general]
NEWS_API_KEY = "your_api_key_here"


5. Run the app  
streamlit run app.py


---

## Deployed Version  

(Replace this link after deployment)  
https://share.streamlit.io/EmmanuelAbolade/news2signal-portfolio/main/app.py  

---

## References  

- Malo, P., Sinha, A., Takala, P., Korhonen, P., & Wallenius, J. (2014). *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts*.  
- Yahoo Finance API (via `yfinance`).  
- NewsAPI.org – Business & Financial Headlines.  
- Scikit-Learn Documentation.  

---

## Reflection  

This portfolio represents a complete end-to-end data science process:  
from dataset acquisition and feature engineering to model training, evaluation, visualization, and deployment.  
It demonstrates both technical and conceptual understanding of how sentiment analysis can inform market prediction — bridging finance, machine learning, and communication between data and decision-making.
