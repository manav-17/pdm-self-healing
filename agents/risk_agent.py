"""Risk Agent — deterministic risk scoring.

SDD 4.2 step 6: "computes a deterministic risk score".
Kept rule-based (not LLM) per SDD 11.3, so risk scoring never depends
on LLM availability and every score is auditable.

Verified by TC-AGT-02 (determinism) and TC-AGT-03 (monotonicity).
"""
from dataclasses import dataclass, asdict

# Weights sum to 1.0. Failure probability dominates because it is the most
# direct evidence of imminent failure; RUL and anomaly corroborate it.
W_FAILURE = 0.50
W_RUL = 0.30
W_ANOMALY = 0.20

RUL_HORIZON = 125.0  # matches the RUL cap used in training

RISK_BANDS = [(0.70, "HIGH"), (0.40, "MEDIUM"), (0.00, "LOW")]


@dataclass
class RiskAssessment:
    risk_score: float
    risk_level: str
    failure_component: float
    rul_component: float
    anomaly_component: float
    criticality: float
    explanation: str

    def to_dict(self):
        return asdict(self)


def assess(failure_probability, rul_estimate, anomaly_score,
           asset_criticality=1.0):
    """Compute an auditable risk score in [0, 1].

    failure_probability : 0-1 from the failure classifier
    rul_estimate        : predicted cycles remaining
    anomaly_score       : 0-1 normalised anomaly score
    asset_criticality   : 0-1 multiplier for how important this machine is
    """
    # Urgency rises as remaining life shrinks.
    rul_urgency = max(0.0, 1.0 - (min(rul_estimate, RUL_HORIZON) / RUL_HORIZON))

    f_comp = W_FAILURE * failure_probability
    r_comp = W_RUL * rul_urgency
    a_comp = W_ANOMALY * min(max(anomaly_score, 0.0), 1.0)

    base = f_comp + r_comp + a_comp
    score = round(min(1.0, base * asset_criticality), 4)

    level = next(lbl for thr, lbl in RISK_BANDS if score >= thr)

    explanation = (
        f"risk={score:.3f} ({level}) = "
        f"{W_FAILURE}*P(fail)={f_comp:.3f} + "
        f"{W_RUL}*RUL_urgency={r_comp:.3f} + "
        f"{W_ANOMALY}*anomaly={a_comp:.3f}, "
        f"criticality x{asset_criticality}"
    )

    return RiskAssessment(score, level, round(f_comp, 4), round(r_comp, 4),
                          round(a_comp, 4), asset_criticality, explanation)


if __name__ == "__main__":
    print("Risk Agent — behaviour check\n")

    cases = [
        ("healthy engine",   0.02, 120, 0.10),
        ("early degradation",0.20,  80, 0.35),
        ("mid degradation",  0.55,  45, 0.55),
        ("near failure",     0.92,  10, 0.80),
    ]
    for name, f, rul, a in cases:
        r = assess(f, rul, a)
        print(f"{name:<20} P(fail)={f:<5} RUL={rul:<4} anom={a:<5} "
              f"-> {r.risk_score:.3f} {r.risk_level}")

    print("\nTC-AGT-02 (determinism): same inputs run twice ->")
    a1, a2 = assess(0.7, 30, 0.6), assess(0.7, 30, 0.6)
    print(f"  {a1.risk_score} == {a2.risk_score}  -> {a1.risk_score == a2.risk_score}")

    print("\nTC-AGT-03 (monotonic in failure probability):")
    prev = -1
    for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = assess(f, 60, 0.4).risk_score
        print(f"  P(fail)={f:<5} -> risk={s:.3f}  rising={s > prev}")
        prev = s

    print("\nTC-AGT-03 (monotonic as RUL falls):")
    prev = -1
    for rul in [125, 90, 60, 30, 5]:
        s = assess(0.5, rul, 0.4).risk_score
        print(f"  RUL={rul:<4} -> risk={s:.3f}  rising={s > prev}")
        prev = s
