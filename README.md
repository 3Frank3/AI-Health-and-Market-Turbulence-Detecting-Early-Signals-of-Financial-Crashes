# AI–Health–Finance Early Warning (AHF-EW)

A compact data science project predicting short-term market downturn risks using trends in AI attention and health disruptions.

## 📊 Overview
- **Goal:** Detect early warning signals before market downturns (S&P 500)
- **Data Sources:**
  - [Yahoo Finance](https://finance.yahoo.com) — S&P 500 (^GSPC)
  - [Google Trends](https://trends.google.com) — AI-related search terms
  - [Our World in Data](https://ourworldindata.org) — Health indicators (COVID, etc.)

## ⚙️ Tech Stack
- Python 3.10, pandas, numpy, scikit-learn, plotly, yfinance, pytrends, streamlit

## 🚀 How to Run
```bash
git clone https://github.com/yourname/ahf-ew.git
cd ahf-ew
pip install -r requirements.txt
python -m src.features
python -m src.train
python -m src.evaluate
streamlit run app/app.py
