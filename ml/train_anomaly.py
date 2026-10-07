"""Anomaly detection model (Isolation Forest, unsupervised).

Trained ONLY on early-life cycles (assumed healthy) with no labels.
Evaluated afterwards against RUL purely to check the scores are meaningful.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path

PROC = Path("../data/processed")
MODELS = Path("../models")
MODELS.mkdir(exist_ok=True)

ID_COLS = ["unit_number", "time_cycles"]
LABEL_COLS = ["RUL", "failure_within_horizon"]

HEALTHY_CYCLES = 30    # first N cycles per engine assumed healthy
CONTAMINATION = 0.05


def main():
    train = pd.read_parquet(PROC / "train_FD001_features.parquet")
    test = pd.read_parquet(PROC / "test_FD001_features.parquet")
    features = [c for c in train.columns if c not in ID_COLS + LABEL_COLS]

    # Train only on early-life data — no labels used anywhere here.
    healthy = train[train.time_cycles <= HEALTHY_CYCLES]
    print(f"Training on {len(healthy)} healthy rows "
          f"(first {HEALTHY_CYCLES} cycles of each engine)")
    print(f"Features: {len(features)}")

    scaler = StandardScaler().fit(healthy[features])

    model = IsolationForest(
        n_estimators=200, contamination=CONTAMINATION,
        random_state=42, n_jobs=-1,
    ).fit(scaler.transform(healthy[features]))

    # score_samples: higher = more normal. Negate so higher = more anomalous.
    def anomaly_score(df):
        return -model.score_samples(scaler.transform(df[features]))

    train_scores = anomaly_score(train)
    test_scores = anomaly_score(test)

    threshold = float(np.percentile(anomaly_score(healthy), 95))

    print("\n" + "=" * 52)
    print("ANOMALY DETECTION RESULTS")
    print("=" * 52)
    print(f"Threshold (95th pct of healthy): {threshold:.4f}")
    print(f"Flagged in train: {(train_scores > threshold).mean():.1%}")
    print(f"Flagged in test:  {(test_scores > threshold).mean():.1%}")

    # Sanity check only — labels used for evaluation, not training.
    auc = roc_auc_score(train["failure_within_horizon"], train_scores)
    print(f"\nROC-AUC vs failure label (sanity check): {auc:.4f}")

    print("\nMean anomaly score by RUL band (train):")
    bands = pd.cut(train["RUL"], [-1, 20, 50, 90, 125],
                   labels=["0-20 (critical)", "21-50", "51-90", "91-125 (healthy)"])
    summary = pd.DataFrame({
        "mean_score": train_scores, "band": bands,
    }).groupby("band", observed=True)["mean_score"].agg(["mean", "count"])
    print(summary.round(4).to_string())

    print("\nEngine 1 anomaly score over life (every 25th cycle):")
    e1 = train[train.unit_number == 1].copy()
    e1["score"] = anomaly_score(e1)
    e1["flag"] = e1["score"] > threshold
    print(e1[["time_cycles", "RUL", "score", "flag"]]
          .iloc[::25].round(4).to_string(index=False))

    joblib.dump({"model": model, "scaler": scaler,
                 "features": features, "threshold": threshold},
                MODELS / "anomaly_iforest.pkl")
    print(f"\nSaved model -> {MODELS / 'anomaly_iforest.pkl'}")


if __name__ == "__main__":
    main()
