import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, calibration_curve
import joblib

def evaluation_report(feat_path="data/processed/features.parquet", metrics_path="data/interim/metrics.json"):
    df = pd.read_parquet(feat_path)
    split_date = "2023-01-01"
    te = df[df["date"] >= split_date].copy()
    features = [c for c in df.columns if c not in ["date","close","ret_1w","NextWeekDown","HighRiskNextWeek"]]
    TARGET = "NextWeekDown"

    Xte, yte = te[features], te[TARGET]
    logit = joblib.load("data/interim/model_logit.pkl")
    rf    = joblib.load("data/interim/model_rf.pkl")

    p_logit = logit.predict_proba(Xte)[:,1]
    p_rf    = rf.predict_proba(Xte)[:,1]

    # ROC
    RocCurveDisplay.from_predictions(yte, p_logit); plt.title("ROC - Logistic"); plt.show()
    RocCurveDisplay.from_predictions(yte, p_rf);    plt.title("ROC - RandomForest"); plt.show()

    # PR
    PrecisionRecallDisplay.from_predictions(yte, p_logit); plt.title("PR - Logistic"); plt.show()
    PrecisionRecallDisplay.from_predictions(yte, p_rf);    plt.title("PR - RandomForest"); plt.show()

    # Calibration
    for name, p in [("Logistic", p_logit), ("RandomForest", p_rf)]:
        prob_true, prob_pred = calibration_curve(yte, p, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0,1],[0,1],"--")
    plt.title("Calibration")
    plt.xlabel("Predicted prob.")
    plt.ylabel("Observed freq.")
    plt.legend(); plt.show()

    with open(metrics_path) as f:
        print("Metrics:", json.load(f))

if __name__ == "__main__":
    evaluation_report()
