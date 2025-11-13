import pandas as pd
import numpy as np
from .data_fetch import fetch_sp500, fetch_google_trends, fetch_owid_health

def rolling_drawdown(close, window=20):
    # 近似 4 週 max drawdown（以週資料計算）
    roll_max = close.rolling(window, min_periods=1).max()
    dd = (close / roll_max) - 1.0
    return dd

def build_feature_table(start="2018-01-01", end=None, owid_country="United States"):
    mkt = fetch_sp500(start, end)                     # close, ret_1w
    ai  = fetch_google_trends()
    hlth= fetch_owid_health(owid_country)

    df = mkt.merge(ai[["date","AI_Index"]], on="date", how="left") \
            .merge(hlth[["date","Health_Index"]], on="date", how="left")

    # 技術指標
    df["vol_4w"] = df["ret_1w"].rolling(4).std()
    df["drawdown_4w"] = rolling_drawdown(df["close"], window=4)

    # 滯後特徵
    for col in ["AI_Index","Health_Index","ret_1w","vol_4w","drawdown_4w"]:
        for l in [1,2,4]:
            df[f"{col}_lag{l}"] = df[col].shift(l)

    # 目標：下週是否下跌
    df["NextWeekDown"] = (df["ret_1w"].shift(-1) < 0).astype(int)
    # 高風險：未來一週最大回撤 <= -7%（簡化用下週報酬近似；可改進為高頻回撤）
    df["HighRiskNextWeek"] = (df["ret_1w"].shift(-1) <= -0.07).astype(int)

    # 清理
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    feat = build_feature_table()
    feat.to_parquet("data/processed/features.parquet")
    print("Saved -> data/processed/features.parquet", feat.shape)
import pandas as pd
import numpy as np
from .data_fetch import fetch_sp500, fetch_google_trends, fetch_owid_health

def rolling_drawdown(close, window=20):
    # 近似 4 週 max drawdown（以週資料計算）
    roll_max = close.rolling(window, min_periods=1).max()
    dd = (close / roll_max) - 1.0
    return dd

def build_feature_table(start="2018-01-01", end=None, owid_country="United States"):
    mkt = fetch_sp500(start, end)                     # close, ret_1w
    ai  = fetch_google_trends()
    hlth= fetch_owid_health(owid_country)

    df = mkt.merge(ai[["date","AI_Index"]], on="date", how="left") \
            .merge(hlth[["date","Health_Index"]], on="date", how="left")

    # 技術指標
    df["vol_4w"] = df["ret_1w"].rolling(4).std()
    df["drawdown_4w"] = rolling_drawdown(df["close"], window=4)

    # 滯後特徵
    for col in ["AI_Index","Health_Index","ret_1w","vol_4w","drawdown_4w"]:
        for l in [1,2,4]:
            df[f"{col}_lag{l}"] = df[col].shift(l)

    # 目標：下週是否下跌
    df["NextWeekDown"] = (df["ret_1w"].shift(-1) < 0).astype(int)
    # 高風險：未來一週最大回撤 <= -7%（簡化用下週報酬近似；可改進為高頻回撤）
    df["HighRiskNextWeek"] = (df["ret_1w"].shift(-1) <= -0.07).astype(int)

    # 清理
    df = df.dropna().reset_index(drop=True)
    return df

if __name__ == "__main__":
    feat = build_feature_table()
    feat.to_parquet("data/processed/features.parquet")
    print("Saved -> data/processed/features.parquet", feat.shape)
