"""PdM Control Room — showcase dashboard for the Risk-Aware Multi-Agent
Self-Healing Predictive Maintenance system.

Pages
  1. Fleet overview      live status of every engine the pipeline has seen
  2. Approval queue      human-in-the-loop: approve / reject agent decisions
  3. Machine detail      RUL, risk, failure probability, anomaly, operating condition
  4. Models & MLOps      model registry, v3 results per dataset, drift checks
  5. System architecture end-to-end flow and how each box is implemented

Data comes from BigQuery (read-only) and the pdm-api Cloud Run service
(pending decisions, approvals, health). Every query is defensive about
column names so a schema tweak does not take the page down.

Local preview:
  DASHBOARD_PASSWORD=test \
  API_URL=https://pdm-api-363641793280.asia-south1.run.app \
  API_TOKEN=$(gcloud auth print-identity-token) \
  streamlit run dashboard/app.py
"""
import hmac
import json
import os
import sys
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT = os.getenv("GCP_PROJECT", "project-c3e3f3ae-71cf-45cb-800")
DATASET = os.getenv("BQ_DATASET", "pdm")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

RISK_COLORS = {"LOW": "#2F7D5B", "MEDIUM": "#C77D1A", "HIGH": "#B3362B"}
ALTITUDES_KFT = [0, 10, 20, 25, 35, 42]
FLEETS = {
    "FD001": ("Sea level, 1 condition", "HPC degradation"),
    "FD002": ("6 operating conditions", "HPC degradation"),
    "FD003": ("Sea level, 1 condition", "HPC + fan degradation"),
    "FD004": ("6 operating conditions", "HPC + fan degradation"),
}
INK = "#1C2430"
MUTED = "#5B6573"
LINE = "#D9D6CC"

st.set_page_config(page_title="PdM Control Room", page_icon="⚙️", layout="wide")


# ----------------------------------------------------------------- styling
def inject_css():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
code, .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.01em; color: #1C2430; }
.eyebrow { font: 500 0.72rem 'IBM Plex Mono', monospace; letter-spacing: .12em;
  text-transform: uppercase; color: #1F4E79; margin-bottom: .15rem; }
.page-title { font-size: 1.85rem; font-weight: 700; color: #1C2430; margin: 0 0 .2rem 0; }
.page-sub { color: #5B6573; margin-bottom: 1.2rem; max-width: 70ch; }
.kpi { background: #FFFFFF; border: 1px solid #D9D6CC; border-left: 4px solid #1F4E79;
  border-radius: 6px; padding: .8rem 1rem; height: 100%; }
.kpi .label { font: 500 .72rem 'IBM Plex Mono', monospace; letter-spacing: .08em;
  text-transform: uppercase; color: #5B6573; }
.kpi .value { font-size: 1.9rem; font-weight: 700; color: #1C2430; line-height: 1.15; }
.kpi .note { font-size: .8rem; color: #5B6573; }
.kpi.high { border-left-color: #B3362B; } .kpi.med { border-left-color: #C77D1A; }
.kpi.low { border-left-color: #2F7D5B; }
.pill { display: inline-block; padding: .12rem .55rem; border-radius: 999px;
  font: 600 .72rem 'IBM Plex Mono', monospace; letter-spacing: .06em; color: #fff; }
.tag { display: inline-block; padding: .1rem .5rem; border-radius: 4px; background: #ECEAE3;
  font: 500 .72rem 'IBM Plex Mono', monospace; color: #1C2430; margin-right: .3rem; }
.diag { background: #F6F5F1; border-left: 3px solid #1F4E79; padding: .6rem .8rem;
  border-radius: 4px; color: #1C2430; font-size: .93rem; }
.small { font-size: .82rem; color: #5B6573; }
.status-dot { display:inline-block; width:.6rem; height:.6rem; border-radius:50%; margin-right:.4rem; }
section[data-testid="stSidebar"] { background: #1C2430; }
section[data-testid="stSidebar"] * { color: #E7E5DE !important; }
section[data-testid="stSidebar"] input { color: #1C2430 !important; }
section[data-testid="stSidebar"] .brand { font: 700 1.15rem 'IBM Plex Sans'; letter-spacing: -.01em; }
section[data-testid="stSidebar"] .brand-sub { font: 500 .7rem 'IBM Plex Mono'; letter-spacing: .1em;
  text-transform: uppercase; opacity: .7; }
div[data-testid="stMetricValue"] { font-weight: 700; }
.stApp, [data-testid="stAppViewContainer"] { background: #F6F5F1; color: #1C2430; }
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: #1F4E79 !important; border-color: #1F4E79 !important; color: #fff !important; }
</style>""",
        unsafe_allow_html=True,
    )


def header(eyebrow, title, sub):
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div>'
        f'<div class="page-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, note="", tone=""):
    col.markdown(
        f'<div class="kpi {tone}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def pill(level):
    level = (level or "").upper()
    c = RISK_COLORS.get(level, "#5B6573")
    return f'<span class="pill" style="background:{c}">{level or "—"}</span>'


# ----------------------------------------------------------------- auth
def _secret(name):
    try:
        from google.cloud import secretmanager
        sm = secretmanager.SecretManagerServiceClient()
        path = f"projects/{PROJECT}/secrets/{name}/versions/latest"
        return sm.access_secret_version(request={"name": path}).payload.data.decode().strip()
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def expected_password():
    return os.getenv("DASHBOARD_PASSWORD") or _secret("dashboard-password")


def password_gate():
    if st.session_state.get("authed"):
        return True
    expected = expected_password()
    if not expected:
        st.error("Dashboard password is not configured (DASHBOARD_PASSWORD or Secret Manager).")
        st.stop()
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Restricted · Operator access</div>'
                    '<div class="page-title">PdM Control Room</div>'
                    '<div class="page-sub">Risk-aware multi-agent predictive maintenance '
                    'for turbofan engines (NASA C-MAPSS FD001–FD004).</div>',
                    unsafe_allow_html=True)
        with st.form("login"):
            pw = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if ok:
            if hmac.compare_digest(pw.encode(), expected.encode()):
                st.session_state["authed"] = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()


# ----------------------------------------------------------------- API
def api_headers():
    tok = os.getenv("API_TOKEN")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    if API_URL.startswith("https://"):
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token
            req = google.auth.transport.requests.Request()
            t = google.oauth2.id_token.fetch_id_token(req, API_URL)
            return {"Authorization": f"Bearer {t}"}
        except Exception:
            return {}
    return {}


def api_get(path, timeout=20):
    try:
        r = requests.get(f"{API_URL}{path}", headers=api_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def api_post(path, payload, timeout=30):
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, headers=api_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        detail = ""
        try:
            detail = e.response.text[:300]  # type: ignore[attr-defined]
        except Exception:
            pass
        return None, f"{e} {detail}".strip()


@st.cache_data(ttl=30, show_spinner=False)
def api_health():
    data, err = api_get("/health", timeout=8)
    return data, err


@st.cache_data(ttl=20, show_spinner=False)
def api_pending():
    data, err = api_get("/decisions/pending")
    if data is None:
        return None, err
    if isinstance(data, dict):
        for k in ("pending", "decisions", "items", "results"):
            if isinstance(data.get(k), list):
                return data[k], None
        return [], None
    return list(data), None


# ----------------------------------------------------------------- BigQuery
@st.cache_resource(show_spinner=False)
def bq():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT)


def T(name):
    return f"`{PROJECT}.{DATASET}.{name}`"


@st.cache_data(ttl=60, show_spinner=False)
def query(sql, params=None):
    """Run SQL -> DataFrame. Returns empty frame + error text on failure."""
    from google.cloud import bigquery
    try:
        cfg = None
        if params:
            cfg = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in params.items()])
        rows = [dict(r) for r in bq().query(sql, job_config=cfg).result(timeout=60)]
        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), str(e)


def pick(df, *names):
    """First column in df matching one of names (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def as_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith(("{", "[")):
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


def deep_get(obj, *keys, default=None):
    """Look for any of keys at top level or one level down in nested dicts."""
    obj = as_dict(obj) if not isinstance(obj, dict) else obj
    for k in keys:
        if k in obj and obj[k] not in (None, ""):
            return obj[k]
    for v in obj.values():
        if isinstance(v, (dict, str)):
            d = as_dict(v)
            for k in keys:
                if k in d and d[k] not in (None, ""):
                    return d[k]
    return default


def fleet_of(mid):
    m = str(mid).lower()
    return m[:5].upper() if m.startswith("fd00") else "FD001"


def machine_filter(include_legacy):
    return "TRUE" if include_legacy else "machine_id LIKE 'fd%'"


def risk_of(p, rul, anom):
    try:
        from agents.risk_agent import risk_agent
        r = risk_agent.assess(failure_probability=float(p), rul=float(rul), anomaly=float(anom))
        return float(r.risk_score), str(r.risk_level)
    except Exception:
        urg = 1 - min(max(float(rul), 0.0), 125.0) / 125.0
        s = 0.5 * float(p) + 0.3 * urg + 0.2 * float(anom)
        return s, ("HIGH" if s >= 0.7 else "MEDIUM" if s >= 0.4 else "LOW")


def time_col(df):
    return pick(df, "timestamp", "created_at", "predicted_at", "decided_at",
                "event_time", "ingested_at", "ts", "detected_at", "checked_at")


# ----------------------------------------------------------------- data loaders
def load_predictions(include_legacy):
    df, err = query(f"SELECT * FROM {T('predictions')} WHERE {machine_filter(include_legacy)}")
    if df.empty:
        return df, err
    tc = time_col(df)
    if tc:
        df = df.sort_values(tc)
    df = df.drop_duplicates(subset=["machine_id", "cycle"], keep="last")
    pc = pick(df, "failure_probability", "failure_prob")
    rc = pick(df, "rul_estimate", "rul")
    ac = pick(df, "anomaly_score", "anomaly")
    df = df.rename(columns={pc: "failure_probability", rc: "rul_estimate", ac: "anomaly_score"})
    scores = [risk_of(p, r, a) for p, r, a in
              zip(df["failure_probability"], df["rul_estimate"], df["anomaly_score"])]
    df["risk_score"] = [s for s, _ in scores]
    df["risk_level"] = [lvl for _, lvl in scores]
    df["fleet"] = df["machine_id"].map(fleet_of)
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    return df.sort_values(["machine_id", "cycle"]), None


def load_decisions(include_legacy):
    d, err = query(f"SELECT * FROM {T('agent_decisions')} WHERE {machine_filter(include_legacy)}")
    a, _ = query(f"SELECT * FROM {T('maintenance_actions')} WHERE TRUE")
    return d, a, err


def action_outcomes(actions):
    """decision_id -> (decision, approver, note, time)."""
    if actions.empty or "decision_id" not in actions.columns:
        return {}
    tc = time_col(actions)
    if tc:
        actions = actions.sort_values(tc)
    dc = pick(actions, "decision", "status", "outcome", "action")
    ac = pick(actions, "approver", "approved_by", "operator")
    nc = pick(actions, "note", "notes", "comment")
    out = {}
    for _, r in actions.iterrows():
        out[r["decision_id"]] = (
            str(r.get(dc, "") or "").upper(), r.get(ac, "") if ac else "",
            r.get(nc, "") if nc else "", r.get(tc) if tc else None)
    return out


# ----------------------------------------------------------------- sidebar
def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand">PdM Control Room</div>'
                    '<div class="brand-sub">Self-healing · Multi-agent</div>',
                    unsafe_allow_html=True)
        st.write("")
        page = st.radio("Navigate", ["Fleet overview", "Approval queue", "Machine detail",
                                     "Models & MLOps", "System architecture"],
                        label_visibility="collapsed")
        st.divider()
        approver = st.text_input("Approver name", value=st.session_state.get("approver", "operator"))
        st.session_state["approver"] = approver.strip() or "operator"
        legacy = st.toggle("Include legacy FD001-only runs", value=False,
                           help="Early runs used ids like engine_001 before multi-dataset support.")
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        h, err = api_health()
        if h:
            rev = h.get("revision") or h.get("K_REVISION") or "—"
            st.markdown(f'<span class="status-dot" style="background:#3FB37F"></span>API online'
                        f'<br><span class="small mono">{rev}</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot" style="background:#B3362B"></span>API unreachable',
                        unsafe_allow_html=True)
        st.markdown(f'<span class="small mono">{PROJECT}<br>dataset: {DATASET} · asia-south1</span>',
                    unsafe_allow_html=True)
        st.write("")
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return page, legacy


# ----------------------------------------------------------------- pages
def page_overview(legacy):
    header("01 · Fleet overview", "Engine fleet status",
           "Latest model outputs for every engine streamed through the edge pipeline. "
           "Risk = 0.5·P(failure) + 0.3·RUL urgency + 0.2·anomaly. "
           "MEDIUM and HIGH are escalated to the agents and wait for a human decision.")
    preds, err = load_predictions(legacy)
    if preds.empty:
        st.info("No predictions in BigQuery yet. Run the edge pipeline (run_demo.sh) to stream data."
                + (f"\n\n`{err}`" if err else ""))
        return
    latest = preds.groupby("machine_id").tail(1).copy()
    pending, perr = api_pending()
    d, a, _ = load_decisions(legacy)
    outcomes = action_outcomes(a)
    approved = sum(1 for v in outcomes.values() if v[0].startswith("APPROV"))
    rejected = sum(1 for v in outcomes.values() if v[0].startswith("REJECT"))

    c = st.columns(5)
    kpi(c[0], "Engines monitored", f"{latest['machine_id'].nunique()}",
        f"{latest['fleet'].nunique()} C-MAPSS fleets")
    kpi(c[1], "Readings scored", f"{len(preds):,}", "unique engine-cycles")
    kpi(c[2], "Escalated decisions", f"{len(d):,}" if not d.empty else "0", "MEDIUM / HIGH risk")
    kpi(c[3], "Awaiting approval", "—" if pending is None else f"{len(pending)}",
        "from pdm-api" if pending is not None else "API unreachable", "med")
    kpi(c[4], "Approved · Rejected", f"{approved} · {rejected}", "human decisions logged", "low")
    st.write("")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Current state per engine")
        show = latest[["machine_id", "fleet", "cycle", "rul_estimate", "failure_probability",
                       "anomaly_score", "risk_score", "risk_level"]].sort_values(
            "risk_score", ascending=False)
        st.dataframe(
            show, hide_index=True, use_container_width=True,
            column_config={
                "machine_id": "Engine", "fleet": "Fleet",
                "cycle": st.column_config.NumberColumn("Last cycle", format="%d"),
                "rul_estimate": st.column_config.NumberColumn("RUL (cycles)", format="%.0f"),
                "failure_probability": st.column_config.ProgressColumn(
                    "P(failure ≤30 cycles)", min_value=0.0, max_value=1.0, format="%.2f"),
                "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%.2f"),
                "risk_score": st.column_config.ProgressColumn(
                    "Risk score", min_value=0.0, max_value=1.0, format="%.2f"),
                "risk_level": "Level",
            })
    with right:
        st.subheader("Risk by engine")
        chart = alt.Chart(latest).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("machine_id:N", sort="-x", title=None),
            x=alt.X("risk_score:Q", scale=alt.Scale(domain=[0, 1]), title="Risk score"),
            color=alt.Color("risk_level:N", title="Level",
                            scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values()))),
            tooltip=["machine_id", "fleet", "cycle",
                     alt.Tooltip("rul_estimate:Q", format=".0f", title="RUL"),
                     alt.Tooltip("risk_score:Q", format=".2f")])
        rules = alt.Chart(pd.DataFrame({"x": [0.4, 0.7]})).mark_rule(
            strokeDash=[4, 4], color=MUTED).encode(x="x:Q")
        st.altair_chart((chart + rules).properties(height=max(220, 34 * len(latest))),
                        use_container_width=True)

    st.subheader("Risk level mix across all scored readings")
    mix = preds.groupby(["fleet", "risk_level"]).size().reset_index(name="readings")
    st.altair_chart(alt.Chart(mix).mark_bar().encode(
        x=alt.X("fleet:N", title=None), y=alt.Y("readings:Q", stack="normalize", title="Share"),
        color=alt.Color("risk_level:N", scale=alt.Scale(domain=list(RISK_COLORS),
                                                        range=list(RISK_COLORS.values())), title="Level"),
        tooltip=["fleet", "risk_level", "readings"]).properties(height=220),
        use_container_width=True)


def _decision_fields(item):
    did = deep_get(item, "decision_id", "id")
    mid = deep_get(item, "machine_id")
    cyc = deep_get(item, "cycle")
    lvl = str(deep_get(item, "risk_level", default="")).upper()
    score = deep_get(item, "risk_score")
    action = deep_get(item, "recommended_action", "action", "maintenance_action", default="—")
    p = deep_get(item, "failure_probability")
    rul = deep_get(item, "rul_estimate", "rul")
    an = deep_get(item, "anomaly_score")
    diag = deep_get(item, "diagnosis")
    dd = as_dict(diag) if diag is not None else {}
    summary = dd.get("summary") or (diag if isinstance(diag, str) and not dd else None) \
        or deep_get(item, "summary", default="")
    mode = dd.get("likely_failure_mode") or deep_get(item, "likely_failure_mode", default="")
    src = dd.get("source") or deep_get(item, "source", default="")
    conf = dd.get("confidence") or deep_get(item, "confidence", default="")
    return dict(did=did, mid=mid, cycle=cyc, level=lvl, score=score, action=action, p=p,
                rul=rul, anom=an, summary=summary, mode=mode, src=src, conf=conf,
                created=deep_get(item, "created_at", "timestamp"))


def _fmt(v, f="{:.2f}"):
    try:
        return f.format(float(v))
    except Exception:
        return "—"


def page_queue(legacy):
    header("02 · Human-in-the-loop", "Approval queue",
           "Decisions the agents escalated. Nothing is actioned until an operator approves it — "
           "every approval or rejection is written to BigQuery (maintenance_actions) as an audit trail.")
    pending, err = api_pending()
    if pending is None:
        st.error(f"Could not reach pdm-api at {API_URL}. Approvals need the API.\n\n`{err}`")
        return
    items = [_decision_fields(x) for x in pending]
    if not legacy:
        items = [x for x in items if str(x["mid"]).lower().startswith("fd")] or items
    if not items:
        st.success("Queue is clear — no decisions awaiting approval.")
        return

    f1, f2, f3 = st.columns([1, 1, 2])
    machines = sorted({str(x["mid"]) for x in items})
    sel_m = f1.selectbox("Engine", ["All"] + machines)
    sel_l = f2.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    f3.markdown(f'<div class="small" style="padding-top:2rem">{len(items)} pending · '
                f'signed in as <b>{st.session_state["approver"]}</b></div>', unsafe_allow_html=True)
    view = [x for x in items if (sel_m == "All" or str(x["mid"]) == sel_m)
            and (not sel_l or x["level"] in sel_l)]
    view.sort(key=lambda x: -float(x["score"] or 0))

    for x in view[:50]:
        with st.container(border=True):
            top = st.columns([3, 1.1, 1.1, 1.1, 1.1])
            top[0].markdown(
                f'{pill(x["level"])} &nbsp; <b style="font-size:1.05rem">{x["mid"]}</b> '
                f'<span class="small">· cycle {x["cycle"]} · {fleet_of(x["mid"])}</span><br>'
                f'<span class="tag">{x["action"]}</span>'
                f'<span class="small mono">{x["did"]}</span>', unsafe_allow_html=True)
            top[1].metric("Risk", _fmt(x["score"]))
            top[2].metric("RUL", _fmt(x["rul"], "{:.0f}"))
            top[3].metric("P(fail)", _fmt(x["p"]))
            top[4].metric("Anomaly", _fmt(x["anom"]))
            if x["summary"]:
                meta = " · ".join(s for s in [x["mode"], f"confidence {x['conf']}" if x["conf"] else "",
                                              f"by {x['src']}" if x["src"] else ""] if s)
                st.markdown(f'<div class="diag">{x["summary"]}<br>'
                            f'<span class="small">{meta}</span></div>', unsafe_allow_html=True)
            with st.expander("How this score was calculated"):
                try:
                    p, rul, an = float(x["p"]), float(x["rul"]), float(x["anom"])
                    urg = 1 - min(max(rul, 0), 125) / 125
                    st.markdown(
                        f"- Failure probability term: 0.5 × {p:.2f} = **{0.5*p:.3f}**\n"
                        f"- RUL urgency term: 0.3 × (1 − {min(rul,125):.0f}/125) = 0.3 × {urg:.2f} = **{0.3*urg:.3f}**\n"
                        f"- Anomaly term: 0.2 × {an:.2f} = **{0.2*an:.3f}**\n"
                        f"- Thresholds: LOW < 0.40 ≤ MEDIUM < 0.70 ≤ HIGH")
                except Exception:
                    st.caption("Component values not available for this decision.")
            note = st.text_input("Note (optional)", key=f"note_{x['did']}",
                                 placeholder="e.g. Inspection booked for next shift")
            b1, b2, _ = st.columns([1, 1, 4])
            for btn, label, decision, kind in [(b1, "Approve", "APPROVED", "primary"),
                                               (b2, "Reject", "REJECTED", "secondary")]:
                if btn.button(label, key=f"{decision}_{x['did']}", type=kind, use_container_width=True):
                    res, perr = api_post(f"/approvals/{x['did']}", {
                        "decision": decision, "approver": st.session_state["approver"], "note": note})
                    if perr:
                        st.error(f"Failed: {perr}")
                    else:
                        st.toast(f"{x['mid']} cycle {x['cycle']}: {decision.lower()}")
                        st.cache_data.clear()
                        st.rerun()
    if len(view) > 50:
        st.caption(f"Showing top 50 of {len(view)} by risk score.")


def _bands():
    return alt.Chart(pd.DataFrame({
        "y0": [0, 0.4, 0.7], "y1": [0.4, 0.7, 1.0], "level": ["LOW", "MEDIUM", "HIGH"]})).mark_rect(
        opacity=0.10).encode(y="y0:Q", y2="y1:Q", color=alt.Color(
            "level:N", scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values())),
            legend=None))


def page_machine(legacy):
    header("03 · Engine drill-down", "Machine detail",
           "Cycle-by-cycle history of one engine: how its remaining life shrinks, how risk "
           "crosses the escalation thresholds, and which operating condition it was flying in.")
    preds, err = load_predictions(legacy)
    if preds.empty:
        st.info("No predictions yet." + (f"\n\n`{err}`" if err else ""))
        return
    machines = sorted(preds["machine_id"].unique())
    default = st.session_state.get("machine", machines[0])
    mid = st.selectbox("Engine", machines, index=machines.index(default) if default in machines else 0)
    st.session_state["machine"] = mid
    m = preds[preds["machine_id"] == mid].copy()
    last = m.iloc[-1]
    fl = fleet_of(mid)
    c = st.columns(5)
    tone = {"HIGH": "high", "MEDIUM": "med", "LOW": "low"}.get(last["risk_level"], "")
    kpi(c[0], "Fleet", fl, " · ".join(FLEETS.get(fl, ("", ""))))
    kpi(c[1], "Cycles observed", f"{int(m['cycle'].max())}", f"{len(m)} scored readings")
    kpi(c[2], "Latest RUL", f"{last['rul_estimate']:.0f}", "cycles remaining (capped at 125)")
    kpi(c[3], "Latest risk", f"{last['risk_score']:.2f}", last["risk_level"], tone)
    first_hi = m[m["risk_level"] == "HIGH"]["cycle"].min()
    kpi(c[4], "First HIGH alert", "—" if pd.isna(first_hi) else f"c{int(first_hi)}",
        "—" if pd.isna(first_hi) else f"{int(m['cycle'].max() - first_hi)} cycles of warning", "high")
    st.write("")

    base = alt.Chart(m).encode(x=alt.X("cycle:Q", title="Operating cycle"))
    l, r = st.columns(2)
    with l:
        st.markdown("**Remaining useful life**")
        st.altair_chart(base.mark_line(color="#1F4E79", strokeWidth=2).encode(
            y=alt.Y("rul_estimate:Q", title="RUL (cycles)"),
            tooltip=["cycle", alt.Tooltip("rul_estimate:Q", format=".1f")]).properties(height=260),
            use_container_width=True)
    with r:
        st.markdown("**Risk score vs escalation thresholds**")
        line = base.mark_line(color=INK, strokeWidth=2).encode(
            y=alt.Y("risk_score:Q", scale=alt.Scale(domain=[0, 1]), title="Risk"),
            tooltip=["cycle", alt.Tooltip("risk_score:Q", format=".2f"), "risk_level"])
        st.altair_chart((_bands() + line).properties(height=260), use_container_width=True)

    l, r = st.columns(2)
    with l:
        st.markdown("**Failure probability and anomaly score**")
        long = m.melt(id_vars=["cycle"], value_vars=["failure_probability", "anomaly_score"],
                      var_name="signal", value_name="value")
        st.altair_chart(alt.Chart(long).mark_line(strokeWidth=1.8).encode(
            x=alt.X("cycle:Q", title="Operating cycle"),
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color("signal:N", scale=alt.Scale(
                domain=["failure_probability", "anomaly_score"], range=["#B3362B", "#7A5AA6"]),
                legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["cycle", "signal", alt.Tooltip("value:Q", format=".3f")]).properties(height=260),
            use_container_width=True)
    with r:
        st.markdown("**Operating condition (altitude regime)**")
        cond, _ = query(
            f"SELECT cycle, ANY_VALUE(JSON_VALUE(features, '$.condition')) AS cond "
            f"FROM {T('feature_data')} WHERE machine_id = @m GROUP BY cycle ORDER BY cycle",
            {"m": mid})
        if cond.empty or cond["cond"].isna().all():
            st.caption("No operating-condition data recorded for this engine.")
        else:
            cond["cond"] = pd.to_numeric(cond["cond"], errors="coerce")
            cond["altitude_kft"] = cond["cond"].map(
                lambda i: ALTITUDES_KFT[int(i)] if pd.notna(i) and 0 <= int(i) < 6 else None)
            st.altair_chart(alt.Chart(cond).mark_line(interpolate="step-after", color="#0F766E").encode(
                x=alt.X("cycle:Q", title="Operating cycle"),
                y=alt.Y("altitude_kft:Q", title="Altitude (kft)", scale=alt.Scale(domain=[0, 45])),
                tooltip=["cycle", "cond", "altitude_kft"]).properties(height=260),
                use_container_width=True)
            st.caption(f"{int(cond['cond'].nunique())} distinct condition(s). Sensors are normalised "
                       "per condition (cond_norm_v3) so the models see degradation, not altitude.")

    st.subheader("Agent decisions for this engine")
    d, _ = query(f"SELECT * FROM {T('agent_decisions')} WHERE machine_id = @m", {"m": mid})
    a, _ = query(f"SELECT * FROM {T('maintenance_actions')} WHERE TRUE")
    if d.empty:
        st.caption("No escalated decisions for this engine (all readings were LOW risk).")
        return
    tc = time_col(d)
    if tc:
        d = d.sort_values(tc)
    if "cycle" in d.columns:
        d = d.drop_duplicates(subset=["cycle"], keep="last").sort_values("cycle")
    outcomes = action_outcomes(a)
    rows = []
    for _, r in d.iterrows():
        f = _decision_fields(r.to_dict())
        o = outcomes.get(f["did"], ("PENDING", "", "", None))
        rows.append({"cycle": f["cycle"], "risk": f["level"], "score": f["score"],
                     "action": f["action"], "diagnosis": f["summary"], "source": f["src"],
                     "human decision": o[0] or "PENDING", "approver": o[1]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"score": st.column_config.NumberColumn("Risk score", format="%.2f"),
                                "diagnosis": st.column_config.TextColumn("Diagnosis", width="large")})


def page_mlops():
    header("04 · MLOps", "Models, registry and drift",
           "Production models are loaded from Cloud Storage via the BigQuery model registry. "
           "A Cloud Scheduler job runs drift detection nightly; drift triggers retraining, and a "
           "challenger is promoted only if it beats the production model.")
    reg, err = query(f"SELECT * FROM {T('model_registry')} WHERE TRUE")
    if reg.empty:
        st.info("Model registry is empty or unreachable." + (f"\n\n`{err}`" if err else ""))
        return
    nc = pick(reg, "model_name", "model")
    vc = pick(reg, "version")
    sc = pick(reg, "status")
    tc = pick(reg, "trained_at", "registered_at", "created_at", "timestamp")
    uc = pick(reg, "artifact_uri", "uri")
    mc = pick(reg, "metrics")
    if tc:
        reg = reg.sort_values(tc)
    reg["_m"] = reg[mc].map(as_dict) if mc else [{}] * len(reg)

    prod = reg[(reg[sc] == "production")] if sc else reg
    if uc:
        prod = prod[prod[uc].fillna("") != ""]
    prod = prod.groupby(nc).tail(1)

    st.subheader("In production")
    cards = st.columns(max(1, len(prod)))
    labels = {"rul_xgboost": ("RUL regressor", "rmse", "RMSE"),
              "failure_xgboost": ("Failure classifier", "roc_auc", "ROC-AUC"),
              "anomaly_iforest": ("Anomaly detector", "roc_auc", "ROC-AUC"),
              "normalizer": ("Condition normaliser", None, None)}
    for col, (_, r) in zip(cards, prod.iterrows()):
        name = r[nc]
        title, key, klabel = labels.get(name, (name, None, None))
        mets = r["_m"]
        val = None
        if key:
            val = mets.get(key) or mets.get(f"test_{key}") or deep_get(mets, key, f"combined_{key}")
        note = f"{klabel} {_fmt(val, '{:.4f}' if 'AUC' in (klabel or '') else '{:.2f}')}" if key else \
            f"space {mets.get('space', 'cond_norm_v3')}"
        kpi(col, title, r[vc], f"{name} · {note}", "low")
    st.write("")

    st.subheader("v3 accuracy on the official NASA test sets")
    ds_rows = {}
    for _, r in reg[reg[vc] == "v3"].iterrows():
        per = r["_m"].get("per_dataset") or {}
        per = as_dict(per) if not isinstance(per, dict) else per
        for ds, mets in per.items():
            mets = as_dict(mets) if not isinstance(mets, dict) else mets
            row = ds_rows.setdefault(ds, {"dataset": ds})
            if r[nc] == "rul_xgboost":
                row.update({"RUL RMSE": mets.get("rmse"), "RUL MAE": mets.get("mae"),
                            "NASA score": mets.get("nasa_score") or mets.get("score")})
            elif r[nc] == "failure_xgboost":
                row["Failure ROC-AUC"] = mets.get("roc_auc") or mets.get("auc")
            elif r[nc] == "anomaly_iforest":
                row["Anomaly ROC-AUC"] = mets.get("roc_auc") or mets.get("auc")
    if not ds_rows:  # documented final results if per-dataset metrics are not in the registry
        ds_rows = {k: dict(zip(["dataset", "RUL RMSE", "RUL MAE", "NASA score", "Failure ROC-AUC",
                                "Anomaly ROC-AUC"], v)) for k, v in {
            "FD001": ["FD001", 13.69, 10.06, 293.4, 0.9970, 0.9939],
            "FD002": ["FD002", 14.29, 10.33, 1215.0, 0.9944, 0.9836],
            "FD003": ["FD003", 15.60, 11.39, 561.8, 0.9971, 0.9743],
            "FD004": ["FD004", 16.25, 11.88, 1331.2, 0.9945, 0.9537]}.items()}
    ds = pd.DataFrame(sorted(ds_rows.values(), key=lambda x: x["dataset"]))
    l, r = st.columns([1.2, 1])
    with l:
        st.dataframe(ds, hide_index=True, use_container_width=True, column_config={
            c: st.column_config.NumberColumn(c, format="%.4f" if "AUC" in c else "%.2f")
            for c in ds.columns if c != "dataset"})
        st.caption("Combined: RUL RMSE 15.11 · MAE 10.99 · failure ROC-AUC 0.9951 · anomaly ROC-AUC 0.9676 "
                   "(707 test engines). FD002/FD004 fly six operating conditions; FD003/FD004 have two fault modes.")
    with r:
        if "RUL RMSE" in ds:
            st.altair_chart(alt.Chart(ds).mark_bar(color="#1F4E79", cornerRadiusEnd=3).encode(
                x=alt.X("dataset:N", title=None), y=alt.Y("RUL RMSE:Q", title="RUL RMSE (cycles, lower is better)"),
                tooltip=list(ds.columns)).properties(height=240), use_container_width=True)

    st.subheader("Drift checks")
    dr, derr = query(f"SELECT * FROM {T('drift_events')} WHERE TRUE")
    if dr.empty:
        st.caption("No drift checks logged yet." + (f" ({derr})" if derr else ""))
    else:
        dt = time_col(dr)
        scol = pick(dr, "drift_score", "score", "drifted_fraction")
        trig = pick(dr, "retrain_triggered", "triggered", "drift_detected")
        if dt:
            dr = dr.sort_values(dt)
        c = st.columns(3)
        kpi(c[0], "Checks run", f"{len(dr)}", "scheduled nightly 02:00 IST")
        kpi(c[1], "Retrains triggered", f"{int(dr[trig].astype(bool).sum())}" if trig else "—",
            "drift > 30% of features")
        kpi(c[2], "Last check", str(dr[dt].iloc[-1])[:16] if dt else "—", "KS test on 14 sensor means")
        st.write("")
        if dt and scol:
            dr[scol] = pd.to_numeric(dr[scol], errors="coerce")
            st.altair_chart((alt.Chart(dr).mark_line(point=True, color="#1F4E79").encode(
                x=alt.X(f"{dt}:T", title=None), y=alt.Y(f"{scol}:Q", title="Share of features drifted",
                                                         scale=alt.Scale(domain=[0, 1])),
                tooltip=[dt, scol] + ([trig] if trig else []))
                + alt.Chart(pd.DataFrame({"y": [0.3]})).mark_rule(strokeDash=[4, 4], color="#B3362B")
                .encode(y="y:Q")).properties(height=220), use_container_width=True)
        st.caption("A single engine always looks drifted against a whole-fleet baseline; the retrain "
                   "step guards against that by only promoting a challenger that beats production.")

    st.subheader("Registry history")
    hist_cols = [c for c in [tc, nc, vc, sc, uc] if c]
    st.dataframe(reg[hist_cols].iloc[::-1].head(40), hide_index=True, use_container_width=True)


ARCH_DOT = """
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11,
        color="#1C2430", fillcolor="#FFFFFF", penwidth=1.2, margin="0.18,0.08"];
  edge [color="#5B6573", arrowsize=0.7, penwidth=1.1];

  data [label="Industrial data\\nC-MAPSS FD001–FD004\\n+ engine simulator", fillcolor="#ECEAE3"];
  mqtt [label="MQTT broker\\nplant/+/sensors"];
  edge_api [label="Edge / API\\nPython + FastAPI\\nvalidation + condition\\nnormalisation (56 features)"];
  bq [label="BigQuery\\nraw · features · predictions\\ndecisions · actions\\nregistry · drift", fillcolor="#E4EDF5"];
  ml [label="AI / ML\\nIsolation Forest (anomaly)\\nXGBoost (failure ≤30)\\nXGBoost (RUL)", fillcolor="#E4EDF5"];
  orch [label="Agent orchestrator", fillcolor="#1F4E79", fontcolor="#FFFFFF", color="#1F4E79"];
  diag [label="1 · Diagnosis agent\\nGemini + template fallback"];
  risk [label="2 · Risk agent\\n0.5P + 0.3U + 0.2A"];
  maint [label="3 · Maintenance agent\\naction + priority"];
  dec [label="Maintenance decision"];
  human [label="Human approval\\n(this dashboard)", fillcolor="#FBEFD9", color="#C77D1A"];
  act [label="Action / alert\\nlogged to BigQuery", fillcolor="#E3F1EA", color="#2F7D5B"];

  data -> mqtt -> edge_api -> bq -> ml -> orch -> diag -> risk -> maint -> dec -> human -> act;

  subgraph cluster_mlops { label="MLOps / CI-CD"; fontname="Helvetica"; fontsize=11; color="#C7C3B6"; style="rounded,dashed";
    drift [label="Drift check\\n(Cloud Scheduler)"]; retrain [label="Retrain"]; validate [label="Validate vs\\nproduction"];
    version [label="Version →\\nregistry + GCS"]; cicd [label="Cloud Build\\nCI/CD"]; rollback [label="Rollback"];
    drift -> retrain -> validate -> version -> cicd; validate -> rollback [style=dashed];
  }
  bq -> drift [style=dashed]; version -> ml [style=dashed, label=" new model", fontsize=9];

  subgraph cluster_gcp { label="Google Cloud (asia-south1)"; fontname="Helvetica"; fontsize=11; color="#C7C3B6"; style="rounded,dashed";
    run [label="Cloud Run\\npdm-api · pdm-dashboard"]; gcs [label="Cloud Storage\\nmodel artifacts"];
    iam [label="IAM + Secret Manager"]; obs [label="Monitoring · Logging\\n· Trace · alerts"];
  }
}
"""


def page_architecture():
    header("05 · Architecture", "How the system fits together",
           "Each box below is implemented and deployed. Sensor data flows left to right; the MLOps "
           "loop keeps the models healthy; everything runs on Google Cloud with least-privilege service accounts.")
    st.graphviz_chart(ARCH_DOT, use_container_width=True)

    st.subheader("What implements each box")
    st.dataframe(pd.DataFrame([
        ("Industrial data", "NASA C-MAPSS FD001–FD004 (709 train / 707 test engines) replayed by edge/simulator.py"),
        ("MQTT", "Eclipse Mosquitto; topic plant/<machine>/sensors; queue + worker so slow agents never drop the broker"),
        ("Edge / API", "Pydantic validation, per-condition z-score normalisation (6 altitude regimes), 10-cycle rolling mean/std/delta → 56 features"),
        ("BigQuery", "7 tables: raw_sensor_data, feature_data, predictions, agent_decisions, maintenance_actions, model_registry, drift_events"),
        ("AI / ML", "Isolation Forest anomaly score, XGBoost failure-within-30-cycles classifier, XGBoost RUL regressor (cap 125)"),
        ("Agents", "Diagnosis (Gemini flash-lite, 10 s timeout, template fallback) → Risk (weighted score, thresholds 0.4 / 0.7) → Maintenance (action + priority)"),
        ("Human approval", "This dashboard → POST /approvals/{id} on pdm-api → maintenance_actions audit row"),
        ("MLOps", "Nightly KS-test drift check → retrain → validate vs production → version in registry + GCS → rollback if worse"),
        ("CI/CD", "GitHub push to main → Cloud Build: build, smoke-test, push to Artifact Registry, deploy both Cloud Run services"),
        ("GCP", "Cloud Run, Cloud Storage, BigQuery, Secret Manager, IAM service accounts, Cloud Monitoring alert on API error rate"),
    ], columns=["Component", "Implementation"]), hide_index=True, use_container_width=True,
        column_config={"Implementation": st.column_config.TextColumn(width="large")})

    st.subheader("Validation highlights")
    c = st.columns(4)
    kpi(c[0], "RUL RMSE", "13.69–16.25", "per dataset · official test sets")
    kpi(c[1], "Failure ROC-AUC", "≥ 0.994", "all four datasets", "low")
    kpi(c[2], "Live run", "926 readings", "4 fleets · 0 rejected · 0 errors")
    kpi(c[3], "Train/serve parity", "1e-14", "max feature difference", "low")


# ----------------------------------------------------------------- main
def main():
    inject_css()
    password_gate()
    page, legacy = sidebar()
    {"Fleet overview": lambda: page_overview(legacy),
     "Approval queue": lambda: page_queue(legacy),
     "Machine detail": lambda: page_machine(legacy),
     "Models & MLOps": page_mlops,
     "System architecture": page_architecture}[page]()
    st.markdown(f'<div class="small" style="margin-top:2.5rem;border-top:1px solid {LINE};padding-top:.6rem">'
                f'Manav Mabian · M.Tech Data Science, NMIMS · Risk-Aware Multi-Agent AI for Self-Healing '
                f'Industrial Predictive Maintenance · rendered {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</div>',
                unsafe_allow_html=True)


main()