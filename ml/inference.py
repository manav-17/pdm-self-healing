"""Inference module — runs the three trained models on a feature record.

SDD 4.1 AI/ML Layer: "Runs anomaly detection, failure prediction, and RUL
prediction on incoming feature data."

Loads the models saved by train_*.py and exposes a single predict() call
that the Agent Orchestrator consumes.
"""
from pathlib import Path

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_store

def _load():
    """Load models from Cloud Storage via model_registry (falls back to local)."""
    return {
        "rul":     model_store.load("rul_xgboost"),
        "failure": model_store.load("failure_xgboost"),
        "anomaly": model_store.load("anomaly_iforest"),
    }


def _to_frame(features, expected):
    """Build a single-row frame in the exact column order the model expects.

    Missing features (e.g. during window warm-up) default to 0.0 rather than
    raising, so early cycles still produce a prediction. TC-ML-05 covers the
    related null-handling case.
    """
    row = {c: float(features.get(c, 0.0)) for c in expected}
    return pd.DataFrame([row], columns=expected)


def predict(features):
    """Run all three models on one feature dict.

    Returns anomaly_score (0-1), failure_probability (0-1), rul_estimate
    (cycles) and the top contributing sensors.
    """
    m = _load()

    # --- RUL ---------------------------------------------------------------
    rul_bundle = m["rul"]
    X = _to_frame(features, rul_bundle["features"])
    rul = float(np.clip(rul_bundle["model"].predict(X)[0], 0, 125))

    # --- Failure probability ----------------------------------------------
    f_bundle = m["failure"]
    Xf = _to_frame(features, f_bundle["features"])
    failure_prob = float(f_bundle["model"].predict_proba(Xf)[0, 1])

    # --- Anomaly -----------------------------------------------------------
    a_bundle = m["anomaly"]
    Xa = _to_frame(features, a_bundle["features"])
    raw = -a_bundle["model"].score_samples(a_bundle["scaler"].transform(Xa))[0]
    thr = a_bundle["threshold"]
    # Normalise around the trained threshold so 0.5 == the decision boundary,
    # giving the Risk Agent a consistent 0-1 scale.
    anomaly_score = float(np.clip(0.5 * (raw / thr), 0.0, 1.0))

    # --- Top contributing sensors -----------------------------------------
    imp = pd.Series(rul_bundle["model"].feature_importances_,
                    index=rul_bundle["features"]).sort_values(ascending=False)
    top_sensors, seen = [], set()
    for feat in imp.index:
        base = feat.split("_mean")[0].split("_std")[0].replace("_delta", "")
        if base.startswith("sensor_") and base not in seen:
            seen.add(base)
            top_sensors.append(base)
        if len(top_sensors) == 5:
            break

    return {
        "anomaly_score": round(anomaly_score, 4),
        "anomaly_flag": bool(raw > thr),
        "failure_probability": round(failure_prob, 4),
        "rul_estimate": round(rul, 1),
        "top_sensors": top_sensors,
    }


if __name__ == "__main__":
    proc = Path(__file__).resolve().parents[1] / "data" / "processed"
    test = pd.read_parquet(proc / "test_FD001_features.parquet")

    print("Inference check — last cycle of 5 test engines\n")
    print(f"{'engine':>7} {'true_RUL':>9} {'pred_RUL':>9} "
          f"{'P(fail)':>8} {'anomaly':>8} {'flag':>6}")

    last = test.groupby("unit_number").tail(1).head(5)
    for row in last.itertuples(index=False):
        feats = {c: getattr(row, c) for c in test.columns}
        p = predict(feats)
        print(f"{int(row.unit_number):>7} {int(row.RUL):>9} "
              f"{p['rul_estimate']:>9.1f} {p['failure_probability']:>8.3f} "
              f"{p['anomaly_score']:>8.3f} {str(p['anomaly_flag']):>6}")

    print(f"\nTop sensors driving RUL: {predict(feats)['top_sensors']}")
