"""FastAPI services.

Implements the five endpoints defined in SDD 6.1. Run locally as one app;
at deployment these split into separate Cloud Run services.

Covered by TC-API-01 .. TC-API-06.
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
for sub in ("ml", "agents", "edge", "api"):
    sys.path.insert(0, str(ROOT / sub))

import inference          # noqa: E402
import orchestrator       # noqa: E402
import storage            # noqa: E402

DATA = ROOT / "data"
FEATURES_SINK = DATA / "feature_data.jsonl"
DECISIONS_SINK = DATA / "decisions.jsonl"
APPROVALS_SINK = DATA / "maintenance_actions.jsonl"

app = FastAPI(
    title="Risk-Aware Multi-Agent PdM API",
    description="SDD 6.1 internal APIs",
    version="1.0.0",
)

# In-memory index of pending decisions (BigQuery replaces this later).
_pending: dict[str, dict] = {}


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class FeatureRecord(BaseModel):
    machine_id: str = Field(min_length=1)
    cycle: int = Field(ge=0)
    timestamp: float | None = None
    features: dict[str, float]


class PredictRequest(BaseModel):
    machine_id: str = Field(min_length=1)
    cycle: int = Field(ge=0)
    features: dict[str, float]


class EvaluateRequest(BaseModel):
    machine_id: str = Field(min_length=1)
    cycle: int = Field(ge=0)
    anomaly_score: float = Field(ge=0, le=1)
    failure_probability: float = Field(ge=0, le=1)
    rul_estimate: float = Field(ge=0)
    top_sensors: list[str] = []
    asset_criticality: float = Field(default=1.0, ge=0, le=1)
    use_llm: bool = False


class ApprovalRequest(BaseModel):
    decision: str = Field(pattern="^(APPROVED|REJECTED)$")
    approver: str = Field(min_length=1)
    note: str = ""


def _append(path, record):
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "pdm-api", "revision": os.getenv("K_REVISION", "local")}


@app.post("/ingest", status_code=status.HTTP_201_CREATED)
def ingest(record: FeatureRecord):
    """SDD 6.1 Ingestion API. TC-API-01 / TC-API-02.

    Writes the feature record to storage (JSONL locally, BigQuery later).
    """
    row = record.model_dump()
    row["ingested_at"] = time.time()
    _append(FEATURES_SINK, row)
    storage.save_features(record.machine_id, record.cycle, record.features)
    return {"status": "stored", "machine_id": record.machine_id,
            "cycle": record.cycle}


@app.post("/predict")
def predict(req: PredictRequest):
    """SDD 6.1 Inference API. TC-API-03.

    Returns anomaly score, RUL estimate and failure probability.
    """
    try:
        result = inference.predict(req.features)
    except Exception as e:
        raise HTTPException(500, f"inference failed: {e}")

    return {"machine_id": req.machine_id, "cycle": req.cycle, **result}


@app.post("/agents/evaluate")
def evaluate(req: EvaluateRequest):
    """SDD 6.1 Agent API. TC-API-04.

    Runs Diagnosis -> Risk -> Maintenance and returns a PENDING decision.
    """
    decision = orchestrator.run(
        machine_id=req.machine_id,
        cycle=req.cycle,
        anomaly_score=req.anomaly_score,
        failure_probability=req.failure_probability,
        rul_estimate=req.rul_estimate,
        top_sensors=req.top_sensors,
        asset_criticality=req.asset_criticality,
        use_llm=req.use_llm,
    )
    row = decision.to_dict()
    _pending[decision.decision_id] = row
    _append(DECISIONS_SINK, row)
    storage.save_decision(row)
    return row


@app.post("/decisions", status_code=status.HTTP_201_CREATED)
def submit_decision(decision: dict):
    """Accept a decision already computed by the pipeline.

    The pipeline runs the agent chain itself, so re-running it here would
    produce a second, divergent record for the same cycle (and a duplicate
    LLM call). This endpoint stores the decision as-is, keeping one
    authoritative record per event.
    """
    did = decision.get("decision_id")
    if not did:
        raise HTTPException(422, "decision_id is required")

    decision.setdefault("approval_status", "PENDING")
    _pending[did] = decision
    # The pipeline already wrote this to the local history file; writing it
    # again here would duplicate the row.
    storage.save_decision(decision)
    return {"decision_id": did, "approval_status": decision["approval_status"]}


@app.get("/decisions/pending")
def pending():
    """Support endpoint for the dashboard: list decisions awaiting approval."""
    return {"count": len(_pending), "decisions": list(_pending.values())}


@app.post("/approvals/{decision_id}")
def approve(decision_id: str, req: ApprovalRequest):
    """SDD 6.1 Approval API. TC-API-05 / TC-HITL-02 / TC-HITL-03.

    Records the human decision. Only an APPROVED decision produces an action.
    """
    decision = _pending.get(decision_id)
    if decision is None:
        raise HTTPException(404, f"no pending decision {decision_id}")

    decision["approval_status"] = req.decision
    decision["approved_by"] = req.approver
    decision["approved_at"] = time.time()
    decision["approval_note"] = req.note

    action = None
    if req.decision == "APPROVED":
        action = {
            "action_id": str(uuid.uuid4()),
            "decision_id": decision_id,
            "machine_id": decision["machine_id"],
            "action_taken": decision["recommended_action"],
            "approved_by": req.approver,
            "timestamp": time.time(),
        }

    _append(APPROVALS_SINK, {**decision, "action": action})
    storage.save_action(
        action or {"decision_id": decision_id,
                   "machine_id": decision["machine_id"],
                   "action_taken": decision["recommended_action"]},
        req.decision, req.approver, req.note)
    del _pending[decision_id]

    return {"decision_id": decision_id, "approval_status": req.decision,
            "action_generated": action is not None, "action": action}


@app.post("/mlops/check-drift")
def check_drift():
    """SDD 6.1 MLOps job. Triggered by Cloud Scheduler. TC-MLO-07.

    Runs drift detection, triggers retraining if needed,
    validates new model, promotes or rolls back.
    """
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    for sub in ("ml",):
        if str(ROOT / sub) not in sys.path:
            sys.path.insert(0, str(ROOT / sub))

    import drift_detector, retrain as retrain_mod

    drift_score, triggered, drifted = drift_detector.check(verbose=False)

    result = {
        "drift_score": drift_score,
        "drifted_features": drifted,
        "triggered_retrain": triggered,
        "retrain_results": None,
    }

    if triggered:
        retrain_results = retrain_mod.retrain_and_validate()
        result["retrain_results"] = {
            k: {"status": v["status"], "metrics": v["metrics"]}
            for k, v in retrain_results.items()
        }

    return result
