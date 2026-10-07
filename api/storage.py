"""BigQuery storage layer.

Replaces the local JSONL sinks with the tables defined in SDD 5.2.
Falls back to a no-op (with a warning) if BigQuery is unreachable, so local
development still works without cloud access.
"""
import json
import os
import uuid
from datetime import datetime, timezone

PROJECT = os.getenv("GCP_PROJECT", "project-c3e3f3ae-71cf-45cb-800")
DATASET = os.getenv("BQ_DATASET", "pdm")

_client = None
_available = None


def client():
    global _client, _available
    if _available is None:
        try:
            from google.cloud import bigquery
            _client = bigquery.Client(project=PROJECT)
            _available = True
        except Exception as e:
            print(f"[storage] BigQuery unavailable ({e}); writes will be skipped")
            _available = False
    return _client if _available else None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _insert(table, rows):
    c = client()
    if c is None:
        return False
    errors = c.insert_rows_json(f"{PROJECT}.{DATASET}.{table}", rows)
    if errors:
        print(f"[storage] insert errors on {table}: {errors}")
        return False
    return True



def save_raw(machine_id, cycle, sensors, timestamp=None):
    """SDD 5.2 raw_sensor_data: raw readings as received from the edge."""
    return _insert("raw_sensor_data", [{
        "reading_id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "cycle": cycle,
        "timestamp": timestamp or _now(),
        "sensors": json.dumps(sensors),
    }])


def save_features(machine_id, cycle, features, timestamp=None):
    return _insert("feature_data", [{
        "feature_id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "cycle": cycle,
        "timestamp": timestamp or _now(),
        "features": json.dumps(features),
    }])


def save_prediction(machine_id, cycle, pred, model_version="v1", timestamp=None):
    return _insert("predictions", [{
        "prediction_id": str(uuid.uuid4()),
        "machine_id": machine_id,
        "cycle": cycle,
        "timestamp": timestamp or _now(),
        "anomaly_score": pred["anomaly_score"],
        "anomaly_flag": pred["anomaly_flag"],
        "failure_probability": pred["failure_probability"],
        "rul_estimate": pred["rul_estimate"],
        "model_version": model_version,
    }])


def save_decision(decision: dict):
    keep = ("decision_id", "machine_id", "cycle", "anomaly_score",
            "failure_probability", "rul_estimate", "diagnosis_summary",
            "diagnosis_source", "likely_failure_mode", "confidence",
            "risk_score", "risk_level", "risk_explanation",
            "recommended_action", "urgency", "maintenance_window_cycles",
            "justification", "approval_status", "run_id")
    row = {k: decision.get(k) for k in keep}
    row["timestamp"] = _now()
    return _insert("agent_decisions", [row])


def save_action(action: dict, approval_status, approver, note=""):
    return _insert("maintenance_actions", [{
        "action_id": action.get("action_id", str(uuid.uuid4())),
        "decision_id": action.get("decision_id"),
        "machine_id": action.get("machine_id"),
        "action_taken": action.get("action_taken"),
        "approval_status": approval_status,
        "approved_by": approver,
        "approval_note": note,
        "timestamp": _now(),
    }])


def query(sql):
    c = client()
    if c is None:
        return None
    return c.query(sql).to_dataframe()


if __name__ == "__main__":
    print(f"Project: {PROJECT}   Dataset: {DATASET}")
    ok = save_decision({
        "decision_id": str(uuid.uuid4()), "machine_id": "test_engine",
        "cycle": 1, "anomaly_score": 0.5, "failure_probability": 0.5,
        "rul_estimate": 50.0, "diagnosis_summary": "connectivity test",
        "diagnosis_source": "template", "likely_failure_mode": "none",
        "confidence": "low", "risk_score": 0.4, "risk_level": "MEDIUM",
        "risk_explanation": "test", "recommended_action": "CONTINUE_MONITORING",
        "urgency": 1, "maintenance_window_cycles": None,
        "justification": "test", "approval_status": "PENDING", "run_id": "test",
    })
    print("write:", "ok" if ok else "failed")

    df = query(f"SELECT machine_id, cycle, risk_level FROM `{PROJECT}.{DATASET}.agent_decisions` LIMIT 5")
    print("\nread back:")
    print(df if df is not None else "query failed")


def _plain(d):
    import datetime as _dt
    return {k: (v.isoformat() if isinstance(v, (_dt.datetime, _dt.date)) else v)
            for k, v in d.items()}


def pending_decisions(limit=200):
    """Decisions still awaiting a human decision, shared across API instances."""
    c = client()
    if c is None:
        return None
    sql = (f"SELECT d.* FROM `{PROJECT}.{DATASET}.agent_decisions` d "
           f"LEFT JOIN (SELECT DISTINCT decision_id AS aid FROM `{PROJECT}.{DATASET}.maintenance_actions`) a "
           f"ON a.aid = d.decision_id "
           f"WHERE d.approval_status = 'PENDING' AND a.aid IS NULL "
           f"ORDER BY d.risk_score DESC LIMIT {int(limit)}")
    return [_plain(dict(r.items())) for r in c.query(sql).result()]


def get_decision(decision_id):
    """One decision, plus whether a human has already acted on it."""
    c = client()
    if c is None:
        return None
    from google.cloud import bigquery
    sql = (f"SELECT d.*, a.aid IS NOT NULL AS acted "
           f"FROM `{PROJECT}.{DATASET}.agent_decisions` d "
           f"LEFT JOIN (SELECT DISTINCT decision_id AS aid FROM `{PROJECT}.{DATASET}.maintenance_actions`) a "
           f"ON a.aid = d.decision_id WHERE d.decision_id = @id LIMIT 1")
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("id", "STRING", decision_id)])
    rows = list(c.query(sql, job_config=cfg).result())
    return _plain(dict(rows[0].items())) if rows else None
