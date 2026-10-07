"""Maintenance Agent — recommends an action from the risk assessment.

SDD 4.2 step 7: "Risk score passes to the Maintenance Agent, which
recommends a specific action."

Rule-based mapping, so the recommendation is traceable to the risk score
that produced it. Verified by TC-AGT-04 (higher risk -> more urgent action).
"""
from dataclasses import dataclass, asdict

# Ordered by urgency (ascending). Higher risk band -> more urgent action.
ACTION_MAP = {
    "LOW":    {"action": "CONTINUE_MONITORING", "urgency": 1,
               "window_cycles": None,
               "detail": "No intervention required; continue routine monitoring."},
    "MEDIUM": {"action": "SCHEDULE_MAINTENANCE", "urgency": 2,
               "window_cycles": 30,
               "detail": "Plan maintenance within the next 30 operating cycles."},
    "HIGH":   {"action": "IMMEDIATE_INSPECTION", "urgency": 3,
               "window_cycles": 5,
               "detail": "Inspect within 5 cycles; consider taking the unit offline."},
}


@dataclass
class MaintenanceRecommendation:
    action: str
    urgency: int
    window_cycles: int | None
    detail: str
    requires_approval: bool
    justification: str

    def to_dict(self):
        return asdict(self)


def recommend(risk_assessment, diagnosis_summary=""):
    """Map a RiskAssessment to a maintenance recommendation."""
    spec = ACTION_MAP[risk_assessment.risk_level]

    justification = (
        f"Risk level {risk_assessment.risk_level} "
        f"(score {risk_assessment.risk_score:.3f}) -> {spec['action']}. "
        f"{risk_assessment.explanation}"
    )
    if diagnosis_summary:
        justification += f" Diagnosis: {diagnosis_summary}"

    return MaintenanceRecommendation(
        action=spec["action"],
        urgency=spec["urgency"],
        window_cycles=spec["window_cycles"],
        detail=spec["detail"],
        # SDD 4.2 step 8 / TC-HITL-01: nothing executes without human approval.
        requires_approval=True,
        justification=justification,
    )


if __name__ == "__main__":
    import risk_agent

    print("Maintenance Agent — TC-AGT-04 check")
    print("(higher risk must produce a more urgent action)\n")

    cases = [("low", 0.05, 115, 0.10),
             ("medium", 0.55, 45, 0.55),
             ("high", 0.92, 10, 0.80)]

    prev_urgency = 0
    for name, f, rul, a in cases:
        risk = risk_agent.assess(f, rul, a)
        rec = recommend(risk, f"simulated {name}-risk condition")
        more_urgent = rec.urgency > prev_urgency
        print(f"{name:<7} risk={risk.risk_score:.3f} {risk.risk_level:<7} "
              f"-> {rec.action:<22} urgency={rec.urgency} "
              f"window={rec.window_cycles}  more_urgent={more_urgent}")
        prev_urgency = rec.urgency

    print("\nTC-HITL-01: every recommendation requires approval ->",
          all(recommend(risk_agent.assess(f, r, a)).requires_approval
              for _, f, r, a in cases))
