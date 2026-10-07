"""Operator dashboard.

SDD 4.1 Dashboard Layer: "Streamlit interface showing live sensor status,
predictions, risk scores, RUL, and agent recommendations."

Reads decisions from the API and provides the approve/reject control that
gates every maintenance action (SDD 4.2 step 8).
Covered by TC-DSH-01 and TC-DSH-02.
"""
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"
ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "data" / "decisions.jsonl"

RISK_COLOR = {"LOW": "#2e7d32", "MEDIUM": "#ef6c00", "HIGH": "#c62828"}

st.set_page_config(page_title="Predictive Maintenance", layout="wide")


def api_up():
    try:
        return requests.get(f"{API}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def load_history():
    if not DECISIONS.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in DECISIONS.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


st.title("Predictive maintenance")
st.caption("Risk-aware multi-agent decision support — every action needs your approval")

if not api_up():
    st.error("The API isn't responding. Start it with "
             "`uvicorn api.main:app --port 8000`, then reload this page.")
    st.stop()

tab_review, tab_history = st.tabs(["Awaiting your decision", "Machine history"])

# ---------------------------------------------------------------- review ---
with tab_review:
    pending = requests.get(f"{API}/decisions/pending", timeout=5).json()
    decisions = pending["decisions"]

    if not decisions:
        st.info("Nothing needs your attention right now. "
                "Run the pipeline to generate new recommendations.")
    else:
        st.write(f"{len(decisions)} recommendation(s) waiting.")

    # Most urgent first.
    for d in sorted(decisions, key=lambda x: -x["risk_score"]):
        colour = RISK_COLOR[d["risk_level"]]
        with st.container(border=True):
            head, metrics = st.columns([2, 3])

            with head:
                st.markdown(f"### {d['machine_id']} · cycle {d['cycle']}")
                st.markdown(
                    f"<span style='color:{colour};font-size:1.6rem;font-weight:600'>"
                    f"{d['risk_level']} risk · {d['risk_score']:.2f}</span>",
                    unsafe_allow_html=True)
                st.markdown(f"**{d['recommended_action'].replace('_', ' ').title()}**")
                if d["maintenance_window_cycles"]:
                    st.caption(f"Within {d['maintenance_window_cycles']} cycles")

            with metrics:
                a, b, c = st.columns(3)
                a.metric("Remaining life", f"{d['rul_estimate']:.0f} cycles")
                b.metric("Failure risk", f"{d['failure_probability']:.0%}")
                c.metric("Anomaly", f"{d['anomaly_score']:.2f}")

            st.write(d["diagnosis_summary"])

            with st.expander("How this score was calculated"):
                st.code(d["risk_explanation"], language=None)
                st.caption(f"Diagnosis from: {d['diagnosis_source']} · "
                           f"confidence {d['confidence']} · "
                           f"agents ran {' → '.join(d['agent_sequence'])}")

            approve, reject, _ = st.columns([1, 1, 4])
            if approve.button("Approve", key=f"a{d['decision_id']}",
                              type="primary"):
                r = requests.post(
                    f"{API}/approvals/{d['decision_id']}",
                    json={"decision": "APPROVED", "approver": "operator",
                          "note": "approved from dashboard"}, timeout=5).json()
                st.success(f"Approved. Work order {r['action']['action_id'][:8]} created.")
                st.rerun()

            if reject.button("Reject", key=f"r{d['decision_id']}"):
                requests.post(
                    f"{API}/approvals/{d['decision_id']}",
                    json={"decision": "REJECTED", "approver": "operator",
                          "note": "rejected from dashboard"}, timeout=5)
                st.warning("Rejected. No maintenance action was created.")
                st.rerun()

# --------------------------------------------------------------- history ---
with tab_history:
    hist = load_history()
    if hist.empty:
        st.info("No decision history yet. Run `python3 edge/pipeline.py` "
                "with the simulator to populate it.")
    else:
        # Show only the most recent run so repeated runs don't overlay.
        if "run_id" in hist.columns and hist.run_id.notna().any():
            latest = hist.dropna(subset=["run_id"]).run_id.iloc[-1]
            hist = hist[hist.run_id == latest]

        machine = st.selectbox("Machine", sorted(hist.machine_id.unique()))
        m = hist[hist.machine_id == machine].sort_values("cycle")

        a, b, c = st.columns(3)
        a.metric("Cycles recorded", len(m))
        b.metric("Latest predicted life", f"{m.rul_estimate.iloc[-1]:.0f}")
        c.metric("Current risk", m.risk_level.iloc[-1])

        st.subheader("Predicted remaining life")
        st.line_chart(m.set_index("cycle")[["rul_estimate"]])

        st.subheader("Risk score")
        st.line_chart(m.set_index("cycle")[["risk_score"]])

        st.subheader("Model outputs")
        st.line_chart(m.set_index("cycle")[["failure_probability", "anomaly_score"]])

        st.subheader("Recent decisions")
        st.dataframe(
            m.tail(20)[["cycle", "rul_estimate", "failure_probability",
                        "anomaly_score", "risk_score", "risk_level",
                        "recommended_action", "approval_status"]],
            hide_index=True, use_container_width=True)
