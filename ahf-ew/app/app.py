import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="AHF-EW Dashboard", layout="wide")
df = pd.read_parquet("data/processed/features.parquet")

st.title("AI–Health–Finance Early Warning (AHF-EW)")
split_date = "2023-01-01"
features = [c for c in df.columns if c not in ["date","close","ret_1w","NextWeekDown","HighRiskNextWeek"]]
TARGET = "NextWeekDown"

logit = joblib.load("data/interim/model_logit.pkl")

latest = df.iloc[-1][features].to_frame().T
prob = float(logit.predict_proba(latest)[:,1])
st.metric("Next-week Down Probability", f"{prob:.2%}")

st.line_chart(df.set_index("date")[["close"]].rename(columns={"close":"S&P500 Close"}))
st.line_chart(df.set_index("date")[["AI_Index","Health_Index"]])

st.subheader("Recent snapshot")
st.dataframe(df.tail(10))
