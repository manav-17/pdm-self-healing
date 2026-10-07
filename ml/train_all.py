"""Retrain all three models on the combined FD001-FD004 dataset."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from pathlib import Path
import joblib

PROC   = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(exist_ok=True)

ID_COLS    = ["unit_number", "time_cycles", "dataset", "condition"]
LABEL_COLS = ["RUL", "failure_within_horizon"]
RUL_CAP    = 125
HEALTHY_CYCLES = 30


def load():
    train = pd.read_parquet(PROC / "train_ALL_features.parquet")
    test  = pd.read_parquet(PROC / "test_ALL_features.parquet")
    drop  = [c for c in ID_COLS if c in train.columns] + LABEL_COLS
    features = [c for c in train.columns if c not in drop]
    return train, test, features


def train_rul(train, test, features):
    print("\n[RUL model]")
    X_tr, y_tr = train[features], train["RUL"]
    last = test.groupby("unit_number").tail(1)
    X_te, y_te = last[features], last["RUL"]
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=40,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    pred = np.clip(model.predict(X_te), 0, RUL_CAP)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(np.mean(np.abs(y_te - pred)))
    print(f"  RMSE: {rmse:.2f}  MAE: {mae:.2f}")
    bundle = {"model": model, "features": features}
    joblib.dump(bundle, MODELS / "rul_xgboost.pkl")
    return rmse, mae


def train_failure(train, test, features):
    print("\n[Failure model]")
    X_tr, y_tr = train[features], train["failure_within_horizon"]
    X_te, y_te = test[features],  test["failure_within_horizon"]
    pos_weight  = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=42, early_stopping_rounds=40,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    auc   = float(roc_auc_score(y_te, proba))
    print(f"  ROC-AUC: {auc:.4f}")
    bundle = {"model": model, "features": features, "threshold": 0.5}
    joblib.dump(bundle, MODELS / "failure_xgboost.pkl")
    return auc


def train_anomaly(train, features):
    print("\n[Anomaly model]")
    healthy = train[train.time_cycles <= HEALTHY_CYCLES]
    print(f"  training on {len(healthy):,} healthy rows")
    scaler  = StandardScaler().fit(healthy[features])
    model   = IsolationForest(
        n_estimators=200, contamination=0.05,
        random_state=42, n_jobs=-1,
    ).fit(scaler.transform(healthy[features]))
    threshold = float(np.percentile(
        -model.score_samples(scaler.transform(healthy[features])), 95
    ))
    print(f"  threshold (95th pct): {threshold:.4f}")
    bundle = {"model": model, "scaler": scaler,
              "features": features, "threshold": threshold}
    joblib.dump(bundle, MODELS / "anomaly_iforest.pkl")
    return threshold


def main():
    print("Training on combined FD001+FD002+FD003+FD004 dataset")
    print("=" * 55)
    train, test, features = load()
    print(f"Train: {len(train):,} rows  Features: {len(features)}")

    rmse, mae = train_rul(train, test, features)
    auc       = train_failure(train, test, features)
    threshold = train_anomaly(train, features)

    print("\n" + "=" * 55)
    print("All models trained and saved to models/")
    print(f"  RUL:     RMSE={rmse:.2f}  MAE={mae:.2f}")
    print(f"  Failure: ROC-AUC={auc:.4f}")
    print(f"  Anomaly: threshold={threshold:.4f}")


if __name__ == "__main__":
    main()
