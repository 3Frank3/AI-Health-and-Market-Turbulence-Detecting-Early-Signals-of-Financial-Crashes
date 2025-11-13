import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import joblib

TARGET = "NextWeekDown"  # 或 "HighRiskNextWeek"

def train_models(path="data/processed/features.parquet"):
    df = pd.read_parquet(path)
    # 時間切分
    split_date = "2023-01-01"
    tr = df[df["date"] < split_date].copy()
    te = df[df["date"] >= split_date].copy()

    features = [c for c in df.columns if c not in ["date","close","ret_1w","NextWeekDown","HighRiskNextWeek"]]
    Xtr, ytr = tr[features], tr[TARGET]
    Xte, yte = te[features], te[TARGET]

    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))
    ]).fit(Xtr, ytr)

    rf = RandomForestClassifier(n_estimators=400, max_depth=5, random_state=42, class_weight="balanced")
    rf.fit(Xtr, ytr)

    preds = {
        "logit": logit.predict_proba(Xte)[:,1],
        "rf": rf.predict_proba(Xte)[:,1]
    }

    metrics = {}
    for name, p in preds.items():
        metrics[name] = {
            "roc_auc": float(roc_auc_score(yte, p)),
            "pr_auc": float(average_precision_score(yte, p)),
            "f1@0.5": float(f1_score(yte, (p>=0.5).astype(int)))
        }

    joblib.dump(logit, "data/interim/model_logit.pkl")
    joblib.dump(rf, "data/interim/model_rf.pkl")
    with open("data/interim/metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved models & metrics:", metrics)

if __name__ == "__main__":
    train_models()
