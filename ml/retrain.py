"""MLOps retraining (TC-MLO-02, TC-MLO-03, TC-MLO-04), v3 feature space.

Retrains the RUL and failure models on FD001-FD004 (validation split by engine
for early stopping), scores them on the official test sets, and promotes a
candidate only if it beats the production model of the SAME feature space.
Models are never promoted automatically across feature spaces.
"""
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import FEATURES, SPACE  # noqa: E402
import data_store  # noqa: E402

PROJECT, DATASET, BUCKET = "project-c3e3f3ae-71cf-45cb-800", "pdm", "pdm-models-363641793280"
RUL_CAP = 125


def _split(train):
    units = np.sort(train.unit_number.unique())
    val = np.random.default_rng(42).choice(units, size=len(units) // 5, replace=False)
    return train[~train.unit_number.isin(val)], train[train.unit_number.isin(val)]


def train_rul(tr, va, test):
    m = xgb.XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=50,
                         random_state=42, n_jobs=-1)
    m.fit(tr[FEATURES], tr["RUL"], eval_set=[(va[FEATURES], va["RUL"])], verbose=False)
    last = test.groupby("unit_number").tail(1)
    err = np.clip(m.predict(last[FEATURES]), 0, RUL_CAP) - last["RUL"].values
    return ({"model": m, "features": FEATURES, "space": SPACE},
            {"rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
             "mae": round(float(np.mean(np.abs(err))), 2), "space": SPACE})


def train_failure(tr, va, test):
    pos = tr["failure_within_horizon"].mean()
    m = xgb.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", scale_pos_weight=(1 - pos) / pos,
                          eval_metric="aucpr", early_stopping_rounds=50, random_state=42, n_jobs=-1)
    m.fit(tr[FEATURES], tr["failure_within_horizon"],
          eval_set=[(va[FEATURES], va["failure_within_horizon"])], verbose=False)
    auc = roc_auc_score(test["failure_within_horizon"], m.predict_proba(test[FEATURES])[:, 1])
    return ({"model": m, "features": FEATURES, "threshold": 0.5, "space": SPACE},
            {"roc_auc": round(float(auc), 4), "space": SPACE})


def production(model_name):
    from google.cloud import bigquery
    rows = list(bigquery.Client(project=PROJECT).query(
        f"SELECT version, metrics FROM `{PROJECT}.{DATASET}.model_registry` "
        f"WHERE model_name='{model_name}' AND status='production' AND artifact_uri != '' "
        f"ORDER BY trained_at DESC LIMIT 1").result())
    if not rows:
        return None, {}
    m = rows[0].metrics
    return rows[0].version, (m if isinstance(m, dict) else json.loads(m or "{}"))


def upload(bundle, name, version):
    from google.cloud import storage
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    buf.seek(0)
    path = f"models/{name}/{version}/model.pkl"
    storage.Client(project=PROJECT).bucket(BUCKET).blob(path).upload_from_file(buf, timeout=600)
    return f"gs://{BUCKET}/{path}"


def register(name, version, metrics, uri):
    from google.cloud import bigquery
    err = bigquery.Client(project=PROJECT).insert_rows_json(f"{PROJECT}.{DATASET}.model_registry", [{
        "version_id": f"{name}-{version}", "model_name": name, "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(), "metrics": json.dumps(metrics),
        "artifact_uri": uri, "status": "production"}])
    if err:
        raise RuntimeError(f"registry insert failed: {err}")


def retrain_and_validate():
    print("\nMLOps retraining pipeline (v3 feature space)\n" + "=" * 50)
    train, test = data_store.load("train"), data_store.load("test")
    tr, va = _split(train)
    version = "v3-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    results = {}
    for name, fn, key, better in (("rul_xgboost", train_rul, "rmse", lambda n, o: n < o),
                                  ("failure_xgboost", train_failure, "roc_auc", lambda n, o: n > o)):
        bundle, metrics = fn(tr, va, test)
        prod_version, prod = production(name)
        print(f"[{name}] candidate {key}={metrics[key]}  production {prod_version} "
              f"{key}={prod.get(key)} space={prod.get('space')}")
        if prod_version is not None and prod.get("space") != SPACE:
            status = "rejected (production uses a different feature space)"
        elif prod.get(key) is not None and not better(metrics[key], float(prod[key])):
            status = "rejected"
        else:
            register(name, version, metrics, upload(bundle, name, version))
            status = "promoted"
        print(f"  decision: {status}")
        results[name] = {"status": status, "metrics": metrics}
    return results


if __name__ == "__main__":
    retrain_and_validate()
