"""Operator dashboard (SDD 4.1, TC-DSH-01, TC-DSH-02).

Pending decisions come from the API (backed by BigQuery); machine history
comes from BigQuery. Approve / Reject calls the API's approvals endpoint.
Access is protected by a password stored in Secret Manager.
"""
import hmac
import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))
import risk_agent  # noqa: E402

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
PROJECT = os.getenv("GCP_PROJECT", "project-c3e3f3ae-71cf-45cb-800")
DATASET = os.getenv("BQ_DATASET", "pdm")
RISK_COLOR = {"LOW": "#2e7d32", "MEDIUM": "#ef6c00", "HIGH": "#c62828"}

st.set_page_config(page_title="Predictive maintenance", layout="wide")


def tbl(name):
    return f"`{PROJECT}.{DATASET}.{name}`"


@st.cache_resource(show_spinner=False)
def dashboard_password():
    if os.getenv("DASHBOARD_PASSWORD"):
        return os.getenv("DASHBOARD_PASSWORD")
    from google.cloud import secretmanager
    name = f"projects/{PROJECT}/secrets/dashboard-password/versions/latest"
    return secretmanager.SecretManagerServiceClient().access_secret_version(
        request={"name": name}).payload.data.decode()


def require_login():
    if st.session_state.get("authed"):
        return
    st.title("Predictive maintenance")
    pw = st.text_input("Access password", type="password")
    if pw:
        if hmac.compare_digest(pw.encode(), dashboard_password().encode()):
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


@st.cache_data(ttl=1800, show_spinner=False)
def api_headers():
    if not API.startswith("https://"):
        return {}
    import google.auth.transport.requests
    import google.oauth2.id_token
    token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), API)
    return {"Authorization": f"Bearer {token}"}


def api(method, path, **kw):
    return requests.request(method, f"{API}{path}", headers=api_headers(), timeout=60, **kw)


@st.cache_resource(show_spinner=False)
def bq():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT)


@st.cache_data(ttl=60, show_spinner=False)
def query(sql):
    return bq().query(sql).to_dataframe()


require_login()
st.title("Predictive maintenance")
st.caption("Risk-aware multi-agent decision support. Every action needs your approval.")

try:
    api_ok = api("GET", "/health").ok
except requests.RequestException:
    api_ok = False
if not api_ok:
    st.error(f"The API at {API} isn't responding. Try again in a minute.")
    st.stop()

tab_review, tab_history = st.tabs(["Awaiting your decision", "Machine history"])

with tab_review:
    decisions = api("GET", "/decisions/pending").json().get("decisions", [])
    if not decisions:
        st.info("Nothing needs your attention right now.")
    else:
        st.write(f"{len(decisions)} recommendation(s) waiting. Showing the 20 most urgent.")
    for d in decisions[:20]:
        colour = RISK_COLOR.get(d.get("risk_level"), "#555555")
        with st.container(border=True):
            head, metrics = st.columns([2, 3])
            with head:
                st.markdown(f"### {d['machine_id']} · cycle {d['cycle']}")
                st.markdown(f"<span style='color:{colour};font-size:1.6rem;font-weight:600'>"
                            f"{d['risk_level']} risk · {d['risk_score']:.2f}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"**{str(d['recommended_action']).replace('_', ' ').title()}**")
                if d.get("maintenance_window_cycles"):
                    st.caption(f"Within {d['maintenance_window_cycles']} cycles")
            with metrics:
                a, b, c = st.columns(3)
                a.metric("Remaining life", f"{d['rul_estimate']:.0f} cycles")
                b.metric("Failure risk", f"{d['failure_probability']:.0%}")
                c.metric("Anomaly", f"{d['anomaly_score']:.2f}")
            st.write(d.get("diagnosis_summary", ""))
            with st.expander("How this score was calculated"):
                st.code(d.get("risk_explanation", ""), language=None)
                st.caption(f"Diagnosis from: {d.get('diagnosis_source')} · confidence {d.get('confidence')}")
            col_a, col_r, _ = st.columns([1, 1, 4])
            for col, label, verdict in ((col_a, "Approve", "APPROVED"), (col_r, "Reject", "REJECTED")):
                if col.button(label, key=f"{verdict}-{d['decision_id']}",
                              type="primary" if verdict == "APPROVED" else "secondary"):
                    r = api("POST", f"/approvals/{d['decision_id']}",
                            json={"decision": verdict, "approver": "operator",
                                  "note": f"{verdict.lower()} from dashboard"})
                    if r.ok:
                        st.cache_data.clear()
                        st.rerun()
                    st.error(f"API error {r.status_code}: {r.text[:200]}")

with tab_history:
    machines = query(f"SELECT DISTINCT machine_id FROM {tbl('predictions')} ORDER BY machine_id")
    if machines.empty:
        st.info("No prediction history yet. Run the pipeline to populate it.")
    else:
        machine = st.selectbox("Machine", machines.machine_id.tolist())
        h = query(f"""
            SELECT cycle, rul_estimate, failure_probability, anomaly_score
            FROM {tbl('predictions')} WHERE machine_id = '{machine}'
            QUALIFY ROW_NUMBER() OVER (PARTITION BY cycle ORDER BY timestamp DESC) = 1
            ORDER BY cycle""")
        h["risk_score"] = [risk_agent.assess(float(f), float(r), float(a)).risk_score
                           for f, r, a in zip(h.failure_probability, h.rul_estimate, h.anomaly_score)]
        last = h.iloc[-1]
        a, b, c = st.columns(3)
        a.metric("Cycles recorded", len(h))
        b.metric("Latest predicted life", f"{last.rul_estimate:.0f}")
        c.metric("Current risk", risk_agent.assess(float(last.failure_probability),
                                                   float(last.rul_estimate),
                                                   float(last.anomaly_score)).risk_level)
        st.subheader("Predicted remaining life")
        st.line_chart(h.set_index("cycle")[["rul_estimate"]])
        st.subheader("Risk score")
        st.line_chart(h.set_index("cycle")[["risk_score"]])
        st.subheader("Model outputs")
        st.line_chart(h.set_index("cycle")[["failure_probability", "anomaly_score"]])
        st.subheader("Escalated decisions")
        st.dataframe(query(f"""
            SELECT d.cycle, d.risk_score, d.risk_level, d.recommended_action,
                   COALESCE(a.status, 'PENDING') AS approval_status
            FROM {tbl('agent_decisions')} d
            LEFT JOIN (SELECT decision_id, ANY_VALUE(approval_status) AS status
                       FROM {tbl('maintenance_actions')} GROUP BY decision_id) a
              ON a.decision_id = d.decision_id
            WHERE d.machine_id = '{machine}'
            ORDER BY d.timestamp DESC LIMIT 20"""), hide_index=True)
