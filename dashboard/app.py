"""Operator console for the Risk-Aware Multi-Agent Self-Healing PdM system.

Two views only:
  1. Awaiting your decision  - escalated agent decisions, approve / reject
  2. Machine history         - one engine's life: RUL, risk, decisions

Data: pdm-api (pending decisions, approvals, health) + BigQuery (history).

Local run:
  DASHBOARD_PASSWORD=test \
  API_URL=https://pdm-api-363641793280.asia-south1.run.app \
  API_TOKEN=$(gcloud auth print-identity-token) \
  streamlit run dashboard/app.py
"""
import hmac
import json
import re
import os
import sys

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT = os.getenv("GCP_PROJECT", "project-c3e3f3ae-71cf-45cb-800")
DATASET = os.getenv("BQ_DATASET", "pdm")
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

RISK = {"LOW": "#2E9E73", "MEDIUM": "#E8A33D", "HIGH": "#E25C55"}
INK, PAPER, RULE, MUTED, STEEL = "#1B2240", "#F6F7FB", "#E3E6F0", "#6B7393", "#4C5BD4"
NAVY, CARD, ACCENT = "#161C3A", "#FFFFFF", "#8B9BFF"

st.set_page_config(page_title="PdM Operator Console", page_icon="🛠️", layout="wide",
                   initial_sidebar_state="collapsed")


# ======================================================================= style
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
html, body, .stApp, [class*="css"], button, input, textarea, select, label, p, li {{
  font-family: 'Space Grotesk', system-ui, sans-serif !important; }}
.stApp, [data-testid="stAppViewContainer"] {{
  background-color: {PAPER}; color: {INK};
  background-image: repeating-linear-gradient(0deg, transparent 0 31px, rgba(76,91,212,.03) 31px 32px); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
.mono {{ font-family: 'JetBrains Mono', monospace !important; }}

/* top band */
.band {{ background: linear-gradient(120deg, {NAVY} 0%, #262D63 55%, #3B3F8F 100%); color: #F1F3FF; border-radius: 10px; padding: 1.1rem 1.4rem;
  display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;
  box-shadow: 0 10px 30px -12px rgba(22,28,58,.45); border-bottom: 3px solid {ACCENT}; margin-bottom: 1.6rem; }}
.band .t {{ font-size: 1.45rem; font-weight: 700; letter-spacing: -.02em; }}
.band .s {{ font: 500 .7rem 'JetBrains Mono', monospace; letter-spacing: .16em; text-transform: uppercase;
  color: {ACCENT}; }}
.band .stat {{ text-align: right; font: 500 .72rem 'JetBrains Mono', monospace; color: #C3C9EE; line-height: 1.5; }}
.dot {{ display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .35rem;
  vertical-align: middle; }}
.dot.on {{ background: #4ADE9B; box-shadow: 0 0 0 3px rgba(74,222,155,.28); }}
.dot.off {{ background: {RISK['HIGH']}; }}

/* tabs as a segmented switch */
.stTabs [role="tablist"] {{ gap: .3rem !important; background: {CARD}; border: 1px solid {RULE} !important;
  border-radius: 14px; padding: .35rem; width: fit-content; max-width: 100%;
  box-shadow: 0 4px 14px -8px rgba(27,34,64,.18); margin-bottom: .4rem; }}
.stTabs [role="tablist"] > div:not(:has([role="tab"])) {{ display: none !important; }}
.stTabs [role="tab"] {{ height: auto !important; padding: .6rem 1.25rem !important; margin: 0 !important;
  border: none !important; border-radius: 10px !important; background: transparent !important;
  color: {MUTED} !important; transition: background .15s ease, color .15s ease; }}
.stTabs [role="tab"] p, .stTabs [role="tab"] span {{ font-size: .95rem !important; font-weight: 600 !important;
  color: inherit !important; margin: 0 !important; }}
.stTabs [role="tab"]:hover {{ background: #EEF0FB !important; color: {INK} !important; }}
.stTabs [role="tab"][aria-selected="true"] {{
  background: linear-gradient(135deg, {STEEL} 0%, #6A5AE0 100%) !important; color: #FFFFFF !important;
  box-shadow: 0 6px 16px -6px rgba(76,91,212,.55); }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [role="tabpanel"] {{ padding-top: .6rem; }}

/* segmented control (risk filter) + focus rings: indigo instead of default red */
button[data-testid="stBaseButton-segmented_control"], button[kind="segmented_control"] {{
  background: {CARD} !important; color: {MUTED} !important; border-color: {RULE} !important; }}
button[data-testid="stBaseButton-segmented_controlActive"], button[kind="segmented_controlActive"] {{
  background: #EEF0FD !important; color: {STEEL} !important; border-color: {STEEL} !important;
  font-weight: 600 !important; }}
button[data-testid^="stBaseButton-segmented_control"] p {{ color: inherit !important; }}
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within {{
  border-color: {STEEL} !important; box-shadow: 0 0 0 1px {STEEL} !important; }}
.srcbadge {{ display: inline-block; padding: .08rem .5rem; border-radius: 999px; font: 700 .66rem 'JetBrains Mono', monospace;
  letter-spacing: .06em; text-transform: uppercase; margin-left: .35rem; vertical-align: middle; }}
.srcbadge.llm {{ background: #EEF0FD; color: {STEEL}; border: 1px solid #C9CEF5; }}
.srcbadge.template {{ background: #FFF4E2; color: #A86A12; border: 1px solid #F3D9A8; }}
.srcbadge.auto {{ background: #F0F2F6; color: {MUTED}; border: 1px solid {RULE}; }}

/* decision ticket */
.ticket {{ background: {CARD}; border: 1px solid {RULE}; border-radius: 12px; overflow: hidden;
  margin-top: .9rem; box-shadow: 0 4px 16px -8px rgba(27,34,64,.14); }}
.ticket .head {{ display: flex; align-items: stretch; border-bottom: 1.5px dashed {RULE}; }}
.ticket .lvl {{ writing-mode: vertical-rl; transform: rotate(180deg); color: #fff;
  font: 700 .72rem 'JetBrains Mono', monospace; letter-spacing: .2em; padding: .7rem .45rem;
  text-align: center; }}
.ticket .main {{ padding: .8rem 1rem; flex: 1; }}
.ticket .id {{ font-size: 1.2rem; font-weight: 700; letter-spacing: -.01em; }}
.ticket .meta {{ font: 500 .74rem 'JetBrains Mono', monospace; color: {MUTED}; margin-top: .1rem; }}
.ticket .act {{ display: inline-block; margin-top: .45rem; padding: .18rem .55rem; border: 1px solid {STEEL}; color: {STEEL}; background: #EEF0FD;
  border-radius: 5px; font: 700 .72rem 'JetBrains Mono', monospace; letter-spacing: .05em; }}
.ticket .nums {{ display: grid; grid-template-columns: repeat(4, minmax(80px, 1fr)); border-left: 1.5px dashed {RULE}; }}
.ticket .num {{ padding: .7rem .9rem; border-left: 1px solid #F0F2F8; }}
.ticket .num:first-child {{ border-left: none; }}
.ticket .num .k {{ font: 500 .64rem 'JetBrains Mono', monospace; letter-spacing: .12em; text-transform: uppercase;
  color: {MUTED}; }}
.ticket .num .v {{ font: 700 1.35rem 'JetBrains Mono', monospace; }}
.ticket .body {{ padding: .8rem 1rem .9rem 1rem; }}
.ticket .diag {{ font-size: .95rem; line-height: 1.5; }}
.ticket .src {{ font: 500 .7rem 'JetBrains Mono', monospace; color: {MUTED}; margin-top: .35rem; }}
.stack {{ display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: #ECEEF6;
  margin-top: .7rem; position: relative; }}
.stack span {{ height: 100%; }}
.legend {{ display: flex; gap: 1rem; flex-wrap: wrap; font: 500 .68rem 'JetBrains Mono', monospace;
  color: {MUTED}; margin-top: .35rem; }}
.legend i {{ display: inline-block; width: .6rem; height: .6rem; border-radius: 2px; margin-right: .3rem;
  vertical-align: middle; }}
.marks {{ position: relative; height: 0; }}
.marks b {{ position: absolute; top: -12px; width: 2px; height: 12px; background: {NAVY}; opacity: .6; }}

/* machine history */
.strip {{ display: flex; height: 26px; border: 1px solid {RULE}; border-radius: 6px; overflow: hidden; }}
.strip span {{ flex: 1; }}
.scale {{ display: flex; justify-content: space-between; font: 500 .68rem 'JetBrains Mono', monospace;
  color: {MUTED}; margin-top: .25rem; }}
.facts {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; border: 1px solid {RULE};
  border-radius: 12px; background: {CARD}; box-shadow: 0 4px 16px -8px rgba(27,34,64,.14); margin: .6rem 0 1.2rem 0; }}
.facts div {{ padding: .75rem 1rem; border-left: 1.5px dashed {RULE}; }}
.facts div:first-child {{ border-left: none; }}
.facts .k {{ font: 500 .64rem 'JetBrains Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: {MUTED}; }}
.facts .v {{ font: 700 1.5rem 'JetBrains Mono', monospace; }}
.facts .n {{ font-size: .78rem; color: {MUTED}; }}
.section {{ font: 700 .74rem 'JetBrains Mono', monospace; letter-spacing: .16em; text-transform: uppercase;
  color: {STEEL}; margin: 1.3rem 0 .4rem 0; }}
.empty {{ border: 1.5px dashed {RULE}; border-radius: 12px; padding: 2.2rem; text-align: center;
  background: {CARD}; }}
.empty .big {{ font-size: 1.3rem; font-weight: 700; }}

button[kind="primary"], button[data-testid="stBaseButton-primary"] {{
  background: {STEEL} !important; border: 1px solid {STEEL} !important; color: #FFFFFF !important; font-weight: 600; }}
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {{
  background: {CARD} !important; border: 1px solid {RISK['HIGH']} !important; color: {RISK['HIGH']} !important; font-weight: 600; }}
@media (max-width: 760px) {{
  .ticket .head {{ flex-wrap: wrap; }} .ticket .nums {{ grid-template-columns: repeat(2, 1fr); width: 100%;
  border-left: none; border-top: 1.5px dashed {RULE}; }}
  .facts {{ grid-template-columns: repeat(2, 1fr); }} }}
</style>""", unsafe_allow_html=True)


# ======================================================================= auth
def _secret(name):
    try:
        from google.cloud import secretmanager
        sm = secretmanager.SecretManagerServiceClient()
        p = f"projects/{PROJECT}/secrets/{name}/versions/latest"
        return sm.access_secret_version(request={"name": p}).payload.data.decode().strip()
    except Exception:
        return None


@st.cache_resource(ttl=300, show_spinner=False)
def user_accounts():
    """username -> password.

    Read from the Secret Manager secret `dashboard-users`, a JSON object such as
    {"manav": "...", "reviewer": "..."}  (or env DASHBOARD_USERS for local runs).
    """
    raw = os.getenv("DASHBOARD_USERS") or _secret("dashboard-users")
    try:
        users = json.loads(raw) if raw else {}
        return {str(k).strip().lower(): str(v) for k, v in users.items() if str(v)}
    except Exception:
        return {}


def check_login(username, password):
    users = user_accounts()
    expected = users.get(username.strip().lower())
    # always compare, even for unknown users, so timing does not reveal valid usernames
    ok = hmac.compare_digest(password.encode(), (expected or "\x00invalid").encode())
    return ok and expected is not None


def password_gate():
    if st.session_state.get("authed"):
        return
    if not user_accounts():
        st.error("No user accounts configured. Create the `dashboard-users` secret in Secret Manager.")
        st.stop()
    _, mid, _ = st.columns([1, 1.1, 1])
    with mid:
        st.markdown("<div style='height:14vh'></div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="band" style="flex-direction:column;align-items:flex-start;gap:.2rem">
            <div class="s">Restricted · operator access</div>
            <div class="t">PdM Operator Console</div></div>""", unsafe_allow_html=True)
        with st.form("login"):
            user = st.text_input("Username", placeholder="your username")
            pw = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if ok:
            if user.strip() and check_login(user, pw):
                st.session_state["authed"] = True
                st.session_state["approver"] = user.strip().lower()
                st.rerun()
            st.error("Incorrect username or password.")
    st.stop()


# ======================================================================= API
def api_headers():
    tok = os.getenv("API_TOKEN")
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    if API_URL.startswith("https://"):
        try:
            import google.auth.transport.requests
            import google.oauth2.id_token
            t = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), API_URL)
            return {"Authorization": f"Bearer {t}"}
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=30, show_spinner=False)
def api_health():
    try:
        r = requests.get(f"{API_URL}/health", headers=api_headers(), timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def api_pending():
    try:
        r = requests.get(f"{API_URL}/decisions/pending", headers=api_headers(), timeout=25)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return None, str(e)
    if isinstance(data, dict):
        for k in ("pending", "decisions", "items", "results"):
            if isinstance(data.get(k), list):
                return data[k], None
        return [], None
    return list(data), None


def api_approve(decision_id, decision, note):
    try:
        r = requests.post(f"{API_URL}/approvals/{decision_id}", headers=api_headers(), timeout=30,
                          json={"decision": decision, "approver": st.session_state.get("approver", "operator"),
                                "note": note})
        r.raise_for_status()
        return None
    except Exception as e:
        body = getattr(getattr(e, "response", None), "text", "") or ""
        return f"{e} {body[:200]}".strip()


# ======================================================================= BigQuery
@st.cache_resource(show_spinner=False)
def bq():
    from google.cloud import bigquery
    return bigquery.Client(project=PROJECT)


def T(name):
    return f"`{PROJECT}.{DATASET}.{name}`"


@st.cache_data(ttl=60, show_spinner=False)
def query(sql, params=None):
    from google.cloud import bigquery
    try:
        cfg = None
        if params:
            cfg = bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING", v) for k, v in params.items()])
        return pd.DataFrame([dict(r) for r in bq().query(sql, job_config=cfg).result(timeout=60)]), None
    except Exception as e:
        return pd.DataFrame(), str(e)


# ======================================================================= helpers
def pick(df, *names):
    low = {c.lower(): c for c in df.columns}
    return next((low[n.lower()] for n in names if n.lower() in low), None)


def as_dict(v):
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            return json.loads(v)
        except Exception:
            pass
    return {}


def deep_get(obj, *keys, default=None):
    obj = obj if isinstance(obj, dict) else as_dict(obj)
    for k in keys:
        if obj.get(k) not in (None, ""):
            return obj[k]
    for v in obj.values():
        d = as_dict(v)
        for k in keys:
            if d.get(k) not in (None, ""):
                return d[k]
    return default


def num(v, f="{:.2f}"):
    try:
        return f.format(float(v))
    except Exception:
        return "—"


def fleet_of(mid):
    m = str(mid).lower()
    return m[:5].upper() if m.startswith("fd00") else "FD001"


def time_col(df):
    return pick(df, "timestamp", "created_at", "predicted_at", "decided_at", "event_time", "ingested_at", "ts")


def risk_of(p, rul, anom):
    try:
        from agents.risk_agent import risk_agent
        r = risk_agent.assess(failure_probability=float(p), rul=float(rul), anomaly=float(anom))
        return float(r.risk_score), str(r.risk_level)
    except Exception:
        u = 1 - min(max(float(rul), 0.0), 125.0) / 125.0
        s = 0.5 * float(p) + 0.3 * u + 0.2 * float(anom)
        return s, ("HIGH" if s >= 0.7 else "MEDIUM" if s >= 0.4 else "LOW")


TEXT_KEYS = ("summary", "diagnosis_summary", "diagnosis_text", "explanation", "narrative", "text")


def find_text(obj, depth=0):
    """Search any depth (dicts, lists, JSON strings) for the diagnosis sentence."""
    if depth > 5 or obj is None:
        return ""
    if isinstance(obj, str):
        d = as_dict(obj)
        if d:
            return find_text(d, depth + 1)
        if obj.strip().startswith("["):
            try:
                return find_text(json.loads(obj), depth + 1)
            except Exception:
                return ""
        return ""
    if isinstance(obj, list):
        for v in obj:
            t = find_text(v, depth + 1)
            if t:
                return t
        return ""
    if isinstance(obj, dict):
        for k in TEXT_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and len(v.strip()) > 15 and not v.strip().startswith(("{", "[")):
                return v.strip()
        d = obj.get("diagnosis")
        if isinstance(d, str) and len(d.strip()) > 15 and not d.strip().startswith(("{", "[")):
            return d.strip()
        for v in obj.values():
            t = find_text(v, depth + 1)
            if t:
                return t
    return ""


def auto_summary(mid, cycle, p, rul, an):
    """Readable fallback built from the model outputs when no text was stored."""
    try:
        p, rul, an = float(p), float(rul), float(an)
    except Exception:
        return ""
    state = ("advanced degradation" if p >= 0.7 or rul <= 20 else
             "progressing degradation" if p >= 0.3 or rul <= 60 else "early signs of wear")
    return (f"{mid} at cycle {cycle} shows {state}: about {rul:.0f} cycle{'' if round(rul) == 1 else 's'} of useful life remain{'s' if round(rul) == 1 else ''}, "
            f"a {p:.0%} chance of failure within 30 cycles, and sensor behaviour that is "
            f"{'clearly' if an >= 0.6 else 'mildly' if an >= 0.4 else 'not'} abnormal.")


TEMPLATE_RE = re.compile(r"Anomaly score [0-9.]+, failure probability [0-9.]+, estimated RUL", re.I)


def find_key(obj, keys, depth=0):
    if depth > 5 or obj is None:
        return None
    if isinstance(obj, str):
        d = as_dict(obj)
        return find_key(d, keys, depth + 1) if d else None
    if isinstance(obj, list):
        for v in obj:
            r = find_key(v, keys, depth + 1)
            if r is not None:
                return r
        return None
    if isinstance(obj, dict):
        for k in keys:
            if obj.get(k) not in (None, ""):
                return obj[k]
        for v in obj.values():
            r = find_key(v, keys, depth + 1)
            if r is not None:
                return r
    return None


def detect_source(item, summary):
    """'llm' or 'template': use the stored flag if present, otherwise recognise the
    diagnosis agent's fixed template sentence."""
    v = find_key(item, ("source", "diagnosis_source", "diag_source"))
    if isinstance(v, str) and v.strip().lower() in ("llm", "template", "gemini"):
        return "llm" if v.strip().lower() in ("llm", "gemini") else "template"
    flag = find_key(item, ("llm_used", "used_llm"))
    if isinstance(flag, bool):
        return "llm" if flag else "template"
    if summary:
        return "template" if TEMPLATE_RE.search(summary) else "llm"
    return ""


def fields(item):
    diag = deep_get(item, "diagnosis")
    dd = as_dict(diag)
    summary = find_text(item)
    auto = False
    if not summary:
        summary = auto_summary(deep_get(item, "machine_id"), deep_get(item, "cycle"),
                               deep_get(item, "failure_probability"), deep_get(item, "rul_estimate", "rul"),
                               deep_get(item, "anomaly_score"))
        auto = bool(summary)
    return dict(
        did=deep_get(item, "decision_id", "id"), mid=deep_get(item, "machine_id"),
        cycle=deep_get(item, "cycle"), level=str(deep_get(item, "risk_level", default="")).upper(),
        score=deep_get(item, "risk_score"),
        action=deep_get(item, "recommended_action", "action", "maintenance_action", default="REVIEW"),
        p=deep_get(item, "failure_probability"), rul=deep_get(item, "rul_estimate", "rul"),
        anom=deep_get(item, "anomaly_score"), summary=summary,
        mode=dd.get("likely_failure_mode") or deep_get(item, "likely_failure_mode", default=""),
        conf=dd.get("confidence") or deep_get(item, "confidence", default=""),
        src=("auto" if auto else detect_source(item, summary)))


# ======================================================================= top band
def top_band(pending):
    h = api_health()
    rev = (h or {}).get("revision") or ""
    n = "—" if pending is None else len(pending)
    st.markdown(f"""<div class="band">
      <div><div class="s">Self-healing predictive maintenance · C-MAPSS FD001–FD004</div>
           <div class="t">Operator Console</div></div>
      <div class="stat"><span class="dot {'on' if h else 'off'}"></span>{'API online' if h else 'API unreachable'}
           {f'· {rev}' if rev else ''}<br>{n} decision(s) waiting · signed in as
           <b style="color:#FFFFFF">{st.session_state.get('approver', 'operator')}</b></div>
    </div>""", unsafe_allow_html=True)


# ======================================================================= view 1
def ticket_html(x):
    lvl = x["level"] or "—"
    color = RISK.get(lvl, MUTED)
    stack = ""
    try:
        p, rul, an = float(x["p"]), float(x["rul"]), float(x["anom"])
        u = 1 - min(max(rul, 0), 125) / 125
        parts = [(0.5 * p, "#E25C55", f"failure 0.5×{p:.2f}"),
                 (0.3 * u, "#4C5BD4", f"RUL urgency 0.3×{u:.2f}"),
                 (0.2 * an, "#2BAFC4", f"anomaly 0.2×{an:.2f}")]
        bars = "".join(f'<span style="width:{w*100:.1f}%;background:{c}"></span>' for w, c, _ in parts)
        leg = "".join(f'<span><i style="background:{c}"></i>{t} = {w:.2f}</span>' for w, c, t in parts)
        stack = (f'<div class="stack">{bars}</div>'
                 f'<div class="marks"><b style="left:40%"></b><b style="left:70%"></b></div>'
                 f'<div class="legend">{leg}<span>| ticks = MEDIUM 0.40 · HIGH 0.70</span></div>')
    except Exception:
        pass
    meta = " · ".join(s for s in [x["mode"], f"confidence {x['conf']}" if x["conf"] else ""] if s)
    badge = {"llm": ("llm", "Gemini LLM"), "template": ("template", "Template fallback"),
             "auto": ("auto", "Auto summary")}.get(x["src"])
    if badge:
        meta += f'<span class="srcbadge {badge[0]}">{badge[1]}</span>'
    return f"""<div class="ticket">
      <div class="head">
        <div class="lvl" style="background:{color}">{lvl}</div>
        <div class="main"><div class="id">{x['mid']}</div>
          <div class="meta">cycle {x['cycle']} · {fleet_of(x['mid'])} · {x['did']}</div>
          <div class="act">{x['action']}</div></div>
        <div class="nums">
          <div class="num"><div class="k">Risk</div><div class="v" style="color:{color}">{num(x['score'])}</div></div>
          <div class="num"><div class="k">RUL</div><div class="v">{num(x['rul'], '{:.0f}')}</div></div>
          <div class="num"><div class="k">P(fail)</div><div class="v">{num(x['p'])}</div></div>
          <div class="num"><div class="k">Anomaly</div><div class="v">{num(x['anom'])}</div></div>
        </div>
      </div>
      <div class="body"><div class="diag">{x['summary'] or 'No diagnosis text recorded.'}</div>
        <div class="src">{meta}</div>{stack}</div>
    </div>"""


def view_queue(pending, err):
    if pending is None:
        st.markdown(f'<div class="empty"><div class="big">Cannot reach the API</div>'
                    f'<div class="mono" style="color:{MUTED};font-size:.8rem">{err}</div></div>',
                    unsafe_allow_html=True)
        return
    items = [fields(x) for x in pending]
    if not items:
        st.markdown('<div class="empty"><div class="big">All clear</div>'
                    'No decisions are waiting for approval.</div>', unsafe_allow_html=True)
        return

    counts = {k: sum(1 for x in items if x["level"] == k) for k in RISK}
    c1, c2, c3 = st.columns([1.3, 1.2, 1])
    levels = c1.segmented_control(
        "Risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"], selection_mode="multi",
        format_func=lambda k: f"{k} · {counts[k]}")
    machines = sorted({str(x["mid"]) for x in items})
    eng = c2.selectbox("Engine", ["All engines"] + machines)
    order = c3.selectbox("Sort by", ["Highest risk", "Lowest RUL", "Engine"])

    view = [x for x in items if (not levels or x["level"] in levels)
            and (eng == "All engines" or str(x["mid"]) == eng)]
    key = {"Highest risk": lambda x: -float(x["score"] or 0),
           "Lowest RUL": lambda x: float(x["rul"] if x["rul"] is not None else 1e9),
           "Engine": lambda x: (str(x["mid"]), float(x["cycle"] or 0))}[order]
    view.sort(key=key)
    if not view:
        st.info("Nothing matches these filters.")
        return

    for x in view[:40]:
        st.markdown(ticket_html(x), unsafe_allow_html=True)
        n, a, r = st.columns([4, 1, 1], vertical_alignment="bottom")
        note = n.text_input("Note", key=f"n_{x['did']}", label_visibility="collapsed",
                            placeholder="Optional note for the audit log")
        for col, label, dec, kind in [(a, "Approve", "APPROVED", "primary"),
                                      (r, "Reject", "REJECTED", "secondary")]:
            if col.button(label, key=f"{dec}_{x['did']}", type=kind, use_container_width=True):
                e = api_approve(x["did"], dec, note)
                if e:
                    st.error(f"Could not record decision: {e}")
                else:
                    st.toast(f"{x['mid']} · cycle {x['cycle']} → {dec.lower()}")
                    st.cache_data.clear()
                    st.rerun()
    if len(view) > 40:
        st.caption(f"Showing 40 of {len(view)}. Use the filters to narrow down.")


# ======================================================================= view 2
def load_history(mid):
    df, err = query(f"SELECT * FROM {T('predictions')} WHERE machine_id = @m", {"m": mid})
    if df.empty:
        return df, err
    tc = time_col(df)
    if tc:
        df = df.sort_values(tc)
    df = df.drop_duplicates(subset=["cycle"], keep="last")
    df = df.rename(columns={pick(df, "failure_probability", "failure_prob"): "p",
                            pick(df, "rul_estimate", "rul"): "rul",
                            pick(df, "anomaly_score", "anomaly"): "anom"})
    df["cycle"] = pd.to_numeric(df["cycle"], errors="coerce")
    rs = [risk_of(p, r, a) for p, r, a in zip(df["p"], df["rul"], df["anom"])]
    df["risk"] = [s for s, _ in rs]
    df["level"] = [l for _, l in rs]
    return df.sort_values("cycle"), None


def view_history():
    eng, err = query(f"SELECT DISTINCT machine_id FROM {T('predictions')} "
                     f"WHERE machine_id LIKE 'fd%' ORDER BY machine_id")
    if eng.empty:
        eng, err = query(f"SELECT DISTINCT machine_id FROM {T('predictions')} ORDER BY machine_id")
    if eng.empty:
        st.markdown(f'<div class="empty"><div class="big">No history yet</div>'
                    f'Run the edge pipeline to stream an engine.<br><span class="mono" '
                    f'style="font-size:.75rem;color:{MUTED}">{err or ""}</span></div>', unsafe_allow_html=True)
        return
    machines = eng["machine_id"].tolist()
    mid = st.selectbox("Engine", machines, format_func=lambda m: f"{m}  ·  {fleet_of(m)}")
    m, err = load_history(mid)
    if m.empty:
        st.warning(f"No predictions for {mid}. {err or ''}")
        return

    last = m.iloc[-1]
    first_high = m.loc[m["level"] == "HIGH", "cycle"].min()
    warn = "—" if pd.isna(first_high) else f"c{int(first_high)}"
    lead = "never reached HIGH" if pd.isna(first_high) else \
        f"{int(m['cycle'].max() - first_high)} cycles of warning"
    st.markdown(f"""<div class="facts">
      <div><div class="k">Cycles seen</div><div class="v">{int(m['cycle'].max())}</div>
           <div class="n">{len(m)} scored readings</div></div>
      <div><div class="k">Latest RUL</div><div class="v">{num(last['rul'], '{:.0f}')}</div>
           <div class="n">cycles remaining</div></div>
      <div><div class="k">Latest risk</div><div class="v" style="color:{RISK.get(last['level'], INK)}">
           {num(last['risk'])}</div><div class="n">{last['level']}</div></div>
      <div><div class="k">First HIGH alert</div><div class="v" style="color:{RISK['HIGH']}">{warn}</div>
           <div class="n">{lead}</div></div>
    </div>""", unsafe_allow_html=True)

    # risk timeline strip: one slice per cycle
    st.markdown('<div class="section">Risk level, cycle by cycle</div>', unsafe_allow_html=True)
    slices = "".join(f'<span style="background:{RISK.get(l, RULE)}" title="c{int(c)} {l}"></span>'
                     for c, l in zip(m["cycle"], m["level"]))
    st.markdown(f'<div class="strip">{slices}</div><div class="scale"><span>c{int(m["cycle"].min())}'
                f'</span><span>LOW · MEDIUM · HIGH</span><span>c{int(m["cycle"].max())}</span></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section">Remaining useful life and risk</div>', unsafe_allow_html=True)
    base = alt.Chart(m).encode(x=alt.X("cycle:Q", title="Operating cycle"))
    rul = base.mark_area(line={"color": STEEL, "strokeWidth": 2}, color=alt.Gradient(
        gradient="linear", x1=1, x2=1, y1=1, y2=0,
        stops=[alt.GradientStop(color="rgba(76,91,212,0.02)", offset=0),
               alt.GradientStop(color="rgba(76,91,212,0.22)", offset=1)])).encode(
        y=alt.Y("rul:Q", title="RUL (cycles)"),
        tooltip=["cycle", alt.Tooltip("rul:Q", format=".0f", title="RUL")])
    bands = alt.Chart(pd.DataFrame({"a": [0, .4, .7], "b": [.4, .7, 1], "l": list(RISK)})).mark_rect(
        opacity=.13).encode(y="a:Q", y2="b:Q", color=alt.Color(
            "l:N", scale=alt.Scale(domain=list(RISK), range=list(RISK.values())), legend=None))
    risk = base.mark_line(color=INK, strokeWidth=2).encode(
        y=alt.Y("risk:Q", title="Risk score", scale=alt.Scale(domain=[0, 1])),
        tooltip=["cycle", alt.Tooltip("risk:Q", format=".2f"), "level"])
    l, r = st.columns(2)
    l.altair_chart(rul.properties(height=250), use_container_width=True)
    r.altair_chart((bands + risk).properties(height=250), use_container_width=True)

    st.markdown('<div class="section">Decisions taken on this engine</div>', unsafe_allow_html=True)
    d, _ = query(f"SELECT * FROM {T('agent_decisions')} WHERE machine_id = @m", {"m": mid})
    if d.empty:
        st.caption("No escalations — every reading stayed LOW risk.")
        return
    a, _ = query(f"SELECT * FROM {T('maintenance_actions')} WHERE TRUE")
    outcome = {}
    if not a.empty and "decision_id" in a.columns:
        tc = time_col(a)
        a = a.sort_values(tc) if tc else a
        dc, pc = pick(a, "decision", "status", "outcome"), pick(a, "approver", "approved_by")
        for _, row in a.iterrows():
            outcome[row["decision_id"]] = (str(row.get(dc) or "").upper(), row.get(pc) if pc else "")
    tc = time_col(d)
    d = d.sort_values(tc) if tc else d
    if "cycle" in d.columns:
        d = d.drop_duplicates(subset=["cycle"], keep="last").sort_values("cycle", ascending=False)
    rows = []
    for _, row in d.iterrows():
        f = fields(row.to_dict())
        o = outcome.get(f["did"], ("PENDING", ""))
        rows.append({"Cycle": f["cycle"], "Risk": f["level"], "Score": f["score"], "Action": f["action"],
                     "Diagnosis": f["summary"],
                     "Written by": {"llm": "Gemini LLM", "template": "Template", "auto": "Auto"}.get(f["src"], ""),
                     "Decision": o[0] or "PENDING", "By": o[1]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
        "Score": st.column_config.NumberColumn(format="%.2f"),
        "Diagnosis": st.column_config.TextColumn(width="large")})


# ======================================================================= main
password_gate()
pending, perr = api_pending()
top_band(pending)
_, refresh, out = st.columns([6, 1, 1])
if refresh.button("Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
if out.button("Sign out", use_container_width=True):
    st.session_state.clear()
    st.rerun()

waiting = "" if pending is None else f"  ·  {len(pending)}"
t1, t2 = st.tabs([f":material/pending_actions: Awaiting your decision{waiting}",
                  ":material/monitoring: Machine history"])
with t1:
    view_queue(pending, perr)
with t2:
    view_history()