import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
from datetime import datetime

def fetch_sp500(start="2018-01-01", end=None):
    end = end or datetime.today().strftime("%Y-%m-%d")
    df = yf.download("^GSPC", start=start, end=end, interval="1d")[["Adj Close"]].rename(columns={"Adj Close":"close"})
    df.index = pd.to_datetime(df.index)
    df["ret_1d"] = df["close"].pct_change()
    # 轉週頻（週五對齊）
    w = df.resample("W-FRI").agg({"close":"last"}).dropna()
    w["ret_1w"] = w["close"].pct_change()
    return w.reset_index().rename(columns={"index":"date"})

def fetch_google_trends(keywords=("AI","machine learning","ChatGPT"), geo="US", timeframe="2018-01-01 2025-10-31"):
    pytrends = TrendReq(hl='en-US', tz=360)
    frames = []
    for kw in keywords:
        pytrends.build_payload([kw], geo=geo, timeframe=timeframe)
        df = pytrends.interest_over_time().reset_index()
        df = df.rename(columns={kw: f"{kw}_trend"})
        frames.append(df[["date", f"{kw}_trend"]])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    # 合成 AI_Index：z-score 後平均
    trend_cols = [c for c in out.columns if c.endswith("_trend")]
    out[trend_cols] = out[trend_cols].fillna(method="ffill")
    z = (out[trend_cols] - out[trend_cols].mean()) / out[trend_cols].std(ddof=0)
    out["AI_Index"] = z.mean(axis=1)
    # 週頻
    out = out.set_index("date").resample("W-FRI").mean().reset_index()
    return out

def fetch_owid_health(country="United States"):
    # 直接讀 OWID COVID（簡易做法：讀官方 CSV）
    url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv(url, parse_dates=["date"])
    df = df[df["location"] == country]
    keep = ["date","new_cases","new_deaths","stringency_index"]
    df = df[keep].copy()
    # 週頻聚合（求和/均值混合：cases/deaths 用和、stringency 用均值）
    weekly = df.set_index("date").resample("W-FRI").agg({
        "new_cases":"sum",
        "new_deaths":"sum",
        "stringency_index":"mean"
    }).reset_index()
    # Health_Index：z-score 合成
    zc = (weekly["new_cases"] - weekly["new_cases"].mean())/weekly["new_cases"].std(ddof=0)
    zd = (weekly["new_deaths"] - weekly["new_deaths"].mean())/weekly["new_deaths"].std(ddof=0)
    zs = (weekly["stringency_index"] - weekly["stringency_index"].mean())/weekly["stringency_index"].std(ddof=0)
    weekly["Health_Index"] = pd.concat([zc, zd, zs], axis=1).mean(axis=1)
    return weekly
