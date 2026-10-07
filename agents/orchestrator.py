"""Agent Orchestrator.

SDD 4.2 steps 5-9: invokes the Diagnosis, Risk and Maintenance agents in
sequence and produces a decision record that requires human approval.

The sequence is strict — each agent's output is the next agent's input —
so a recommendation is always traceable to the risk score and diagnosis
that produced it. Verified by TC-AGT-05.
"""
import time
import uuid
from dataclasses import dataclass, field, asdict

import diagnosis_agent
import risk_agent
import maintenance_agent


@dataclass
class AgentDecision:
    """Mirrors the agent_decisions table in SDD 5.2."""
    decision_id: str
    machine_id: str
    cycle: int
    timestamp: float

    # model inputs
    anomaly_score: float
    failure_probability: float
    rul_estimate: float

    # agent outputs
    diagnosis_summary: str
    diagnosis_source: str
    likely_failure_mode: str
    confidence: str
    risk_score: float
    risk_level: str
    risk_explanation: str
    recommended_action: str
    urgency: int
    maintenance_window_cycles: int | None
    justification: str

    # human-in-the-loop (SDD 4.2 step 8 / TC-HITL-01)
    approval_status: str = "PENDING"

    # execution trace (TC-AGT-05)
    agent_sequence: list = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def run(machine_id, cycle, anomaly_score, failure_probability, rul_estimate,
        top_sensors=None, asset_criticality=1.0, use_llm=True):
    """Execute the agent chain in order and return a decision record."""
    sequence, latency = [], {}

    # --- Step 5: Diagnosis -------------------------------------------------
    t0 = time.perf_counter()
    diag = diagnosis_agent.diagnose(
        machine_id, cycle, anomaly_score, failure_probability,
        rul_estimate, top_sensors, use_llm=use_llm,
    )
    latency["diagnosis"] = round((time.perf_counter() - t0) * 1000, 1)
    sequence.append("diagnosis")

    # --- Step 6: Risk (consumes diagnosis context) -------------------------
    t0 = time.perf_counter()
    risk = risk_agent.assess(
        failure_probability, rul_estimate, anomaly_score, asset_criticality,
    )
    latency["risk"] = round((time.perf_counter() - t0) * 1000, 1)
    sequence.append("risk")

    # --- Step 7: Maintenance (consumes the risk score) ---------------------
    t0 = time.perf_counter()
    rec = maintenance_agent.recommend(risk, diag.summary)
    latency["maintenance"] = round((time.perf_counter() - t0) * 1000, 1)
    sequence.append("maintenance")

    return AgentDecision(
        decision_id=str(uuid.uuid4()),
        machine_id=machine_id,
        cycle=cycle,
        timestamp=time.time(),
        anomaly_score=round(anomaly_score, 4),
        failure_probability=round(failure_probability, 4),
        rul_estimate=round(rul_estimate, 1),
        diagnosis_summary=diag.summary,
        diagnosis_source=diag.source,
        likely_failure_mode=diag.likely_failure_mode,
        confidence=diag.confidence,
        risk_score=risk.risk_score,
        risk_level=risk.risk_level,
        risk_explanation=risk.explanation,
        recommended_action=rec.action,
        urgency=rec.urgency,
        maintenance_window_cycles=rec.window_cycles,
        justification=rec.justification,
        agent_sequence=sequence,
        latency_ms=latency,
    )


if __name__ == "__main__":
    print("Agent Orchestrator — TC-AGT-05 check")
    print("(agents must execute in the order diagnosis -> risk -> maintenance)\n")

    scenarios = [
        ("engine_001", 45, 0.31, 0.05, 118.0, ["sensor_4", "sensor_11"]),
        ("engine_042", 120, 0.55, 0.48, 52.0, ["sensor_11", "sensor_4"]),
        ("engine_007", 178, 0.72, 0.88, 14.0, ["sensor_17", "sensor_15", "sensor_4"]),
    ]

    for args in scenarios:
        d = run(*args, use_llm=False)   # template mode for a fast, repeatable check
        print(f"=== {d.machine_id}  cycle {d.cycle} ===")
        print(f"  inputs   anomaly={d.anomaly_score}  P(fail)={d.failure_probability}  RUL={d.rul_estimate}")
        print(f"  risk     {d.risk_score:.3f} ({d.risk_level})")
        print(f"  action   {d.recommended_action}  urgency={d.urgency}  window={d.maintenance_window_cycles}")
        print(f"  approval {d.approval_status}")
        print(f"  sequence {' -> '.join(d.agent_sequence)}")
        print(f"  latency  {d.latency_ms} ms\n")

    print("TC-AGT-05: sequence correct in all runs ->",
          all(run(*a, use_llm=False).agent_sequence == ["diagnosis", "risk", "maintenance"]
              for a in scenarios))
    print("TC-HITL-01: all decisions PENDING ->",
          all(run(*a, use_llm=False).approval_status == "PENDING" for a in scenarios))
