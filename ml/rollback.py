"""Rollback (TC-MLO-05, TC-MLO-06), feature-space aware.

model_registry is append-only, so a version's status is the status on its
newest row; the live model is the newest version whose status is 'production'.
Rollback re-promotes the most recent earlier version that is 'production' or
'archived' AND uses the same feature space as the current model; it never
restores a model trained on a different feature space (e.g. raw-feature v1).
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from google.cloud import bigquery

PROJECT = "project-c3e3f3ae-71cf-45cb-800"
TABLE = f"{PROJECT}.pdm.model_registry"
MODELS = ("rul_xgboost", "failure_xgboost", "anomaly_iforest", "normalizer")


def _versions(client, model):
    rows = client.query(
        f"SELECT version, status, trained_at, metrics, artifact_uri FROM `{TABLE}` "
        f"WHERE model_name = '{model}' ORDER BY trained_at DESC").result()
    info = {}
    for r in rows:
        m = r.metrics if isinstance(r.metrics, dict) else json.loads(r.metrics or "{}")
        v = info.setdefault(r.version, {"version": r.version, "status": r.status, "metrics": m,
                                        "space": m.get("space"), "uri": ""})
        if not v["uri"] and r.artifact_uri:
            v["uri"] = r.artifact_uri
    return list(info.values())


def current_production(model, client=None):
    client = client or bigquery.Client(project=PROJECT)
    return next((v for v in _versions(client, model) if v["status"] == "production" and v["uri"]), None)


def eligible_previous(model, client=None):
    client = client or bigquery.Client(project=PROJECT)
    vs = _versions(client, model)
    cur = next((v for v in vs if v["status"] == "production" and v["uri"]), None)
    if cur is None:
        return None
    return next((v for v in vs if v["version"] != cur["version"] and v["uri"]
                 and v["space"] == cur["space"] and v["status"] in ("production", "archived")), None)


def _row(model, v, status, ts):
    return {"version_id": f"{model}-{v['version']}-{status}-{ts:%Y%m%d%H%M%S}",
            "model_name": model, "version": v["version"], "trained_at": ts.isoformat(),
            "metrics": json.dumps(v["metrics"]), "artifact_uri": v["uri"], "status": status}


def rollback(model):
    client = bigquery.Client(project=PROJECT)
    cur, prev = current_production(model, client), eligible_previous(model, client)
    if cur is None:
        print(f"[rollback] {model}: no production version"); return False
    if prev is None:
        print(f"[rollback] {model}: no earlier version in feature space {cur['space']}; nothing to restore")
        return False
    now = datetime.now(timezone.utc)
    err = client.insert_rows_json(TABLE, [_row(model, cur, "archived", now),
                                          _row(model, prev, "production", now + timedelta(seconds=1))])
    print(f"[rollback] {model}: {cur['version']} -> {prev['version']}" if not err else f"error: {err}")
    return not err


if __name__ == "__main__":
    if len(sys.argv) > 1:
        rollback(sys.argv[1])
    c = bigquery.Client(project=PROJECT)
    for m in MODELS:
        v, p = current_production(m, c), eligible_previous(m, c)
        print(f"{m:<17} production = {v['version'] if v else None:<20} space = {v['space'] if v else None}  "
              f"rollback target = {p['version'] if p else 'none'}")
