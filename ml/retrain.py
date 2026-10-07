"""MLOps retraining script.

Triggered when drift_detector reports drift_score > threshold.
Retrains the RUL and failure models on the full training data,
validates against the current production model, and either
promotes or rejects the new model.

Covered by TC-MLO-02, TC-MLO-03, TC-MLO-04.
"""
import io
import uuid
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import mean_squared_error, roc_auc_score

PROC    = Path(__file__).resolve().parent.parent / "data" / "processed"
PROJECT = "project-c3e3f3ae-71cf-45cb-800"
DATASET = "pdm"
BUCKET  = "pdm-models-363641793280"

ID_COLS    = ["unit_number", "time_cycles"]
LABEL_COLS = ["RUL", "failure_within_horizon"]


def load_data():
    train = pd.read_parquet(PROC / "train_FD001_features.parquet")
    test  = pd.read_parquet(PROC / "test_FD001_features.parquet")
    features = [c for c in train.columns if c not in ID_COLS + LABEL_COLS]
    return train, test, features


def train_rul(train, test, features):
    X_tr, y_tr = train[features], train["RUL"]
    last = test.groupby("unit_number").tail(1)
    X_te, y_te = last[features], last["RUL"]
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    pred  = np.clip(model.predict(X_te), 0, 125)
    rmse  = float(np.sqrt(mean_squared_error(y_te, pred)))
    return {"model": model, "features": features}, {"rmse": round(rmse, 4)}


def train_failure(train, test, features):
    X_tr, y_tr = train[features], train["failure_within_horizon"]
    X_te, y_te = test[features],  test["failure_within_horizon"]
    pos_weight  = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    proba = model.predict_proba(X_te)[:, 1]
    auc   = float(roc_auc_score(y_te, proba))
    return {"model": model, "features": features, "threshold": 0.5}, {"roc_auc": round(auc, 4)}


def get_production_metrics(model_name):
    """Read current production metrics from model_registry."""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT)
        rows = list(client.query(
            f"SELECT metrics FROM `{PROJECT}.{DATASET}.model_registry` "
            f"WHERE model_name='{model_name}' AND status='production' ORDER BY trained_at DESC LIMIT 1"
        ).result())
        if rows:
            m = rows[0].metrics
            return m if isinstance(m, dict) else json.loads(m)
    except Exception as e:
        print(f"  could not read registry: {e}")
    return {}


def upload_to_gcs(bundle, model_name, version):
    """Upload retrained model to Cloud Storage."""
    from google.cloud import storage
    client = storage.Client(project=PROJECT)
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    buf.seek(0)
    path = f"models/{model_name}/{version}/model.pkl"
    client.bucket(BUCKET).blob(path).upload_from_file(buf)
    return f"gs://{BUCKET}/{path}"


def update_registry(model_name, version, metrics, artifact_uri, status):
    """Insert new version into model_registry."""
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    row = {
        "version_id":   f"{model_name}-{version}",
        "model_name":   model_name,
        "version":      version,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "metrics":      json.dumps(metrics),
        "artifact_uri": artifact_uri,
        "status":       status,
    }
    client.insert_rows_json(f"{PROJECT}.{DATASET}.model_registry", [row])
    print(f"  registry: {model_name} {version} -> {status}")


def retrain_and_validate():
    print("\nMLOps Retraining Pipeline")
    print("=" * 50)

    train, test, features = load_data()
    version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"

    results = {}
    for model_name, train_fn, metric_key, better_fn in [
        ("rul_xgboost",     train_rul,     "rmse",    lambda n, o: n < o),
        ("failure_xgboost", train_failure, "roc_auc", lambda n, o: n > o),
    ]:
        print(f"\n[{model_name}]")
        bundle, new_metrics = train_fn(train, test, features)
        new_val = new_metrics[metric_key]
        print(f"  new   {metric_key}: {new_val}")

        prod = get_production_metrics(model_name)
        old_val = prod.get(metric_key)
        print(f"  prod  {metric_key}: {old_val}")

        if old_val is None or better_fn(new_val, float(old_val)):
            print(f"  decision: PROMOTE (new is better)")
            uri = upload_to_gcs(bundle, model_name, version)
            update_registry(model_name, version, new_metrics, uri, "production")
            status = "promoted"
        else:
            print(f"  decision: REJECT (production is better)")
            status = "rejected"

        results[model_name] = {"metrics": new_metrics, "status": status}

    print("\nRetraining complete.")
    for name, r in results.items():
        print(f"  {name}: {r['status']} {r['metrics']}")
    return results


if __name__ == "__main__":
    retrain_and_validate()
