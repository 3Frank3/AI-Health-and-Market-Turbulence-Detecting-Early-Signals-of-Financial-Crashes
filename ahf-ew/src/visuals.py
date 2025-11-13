import pandas as pd
import plotly.graph_objects as go

def timeline_plot(feat_path="data/processed/features.parquet"):
    df = pd.read_parquet(feat_path)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="S&P500 Close", mode="lines"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["AI_Index"], name="AI Index (z)", yaxis="y2", mode="lines"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["Health_Index"], name="Health Index (z)", yaxis="y2", mode="lines"))
    fig.update_layout(
        title="AI & Health Indices vs S&P500",
        xaxis=dict(domain=[0.0, 1.0]),
        yaxis=dict(title="Close"),
        yaxis2=dict(title="Indices (z)", overlaying="y", side="right")
    )
    fig.show()

if __name__ == "__main__":
    timeline_plot()
