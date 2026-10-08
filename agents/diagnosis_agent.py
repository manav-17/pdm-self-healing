"""Diagnosis Agent: interprets the ML outputs as a readable diagnosis.

SDD 4.2 step 5. LLM-based (Gemini). Per SDD 11.3 the LLM must never block or
break the pipeline: each call is capped at TIMEOUT_S seconds and any failure
falls back to a deterministic template. Verified by TC-AGT-01.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
TIMEOUT_S = 10
SECRET = "projects/project-c3e3f3ae-71cf-45cb-800/secrets/gemini-api-key/versions/latest"

FLEETS = {
    "FD001": ("one operating condition", "High-Pressure Compressor (HPC) degradation"),
    "FD002": ("six operating conditions", "High-Pressure Compressor (HPC) degradation"),
    "FD003": ("one operating condition", "HPC or fan degradation (this fleet has two fault modes)"),
    "FD004": ("six operating conditions", "HPC or fan degradation (this fleet has two fault modes)"),
}


@dataclass
class Diagnosis:
    summary: str
    likely_failure_mode: str
    confidence: str
    source: str          # "llm" or "template"

    def to_dict(self):
        return asdict(self)


def fleet(machine_id):
    prefix = machine_id.split("_")[0].upper()
    return prefix if prefix in FLEETS else "FD001"


def _confidence(failure_probability, anomaly_score):
    combined = 0.6 * failure_probability + 0.4 * anomaly_score
    if combined >= 0.7:
        return "high"
    if combined >= 0.4:
        return "medium"
    return "low"


def _template_diagnosis(machine_id, cycle, anomaly_score, failure_probability,
                        rul_estimate, top_sensors):
    """Deterministic fallback, used when the LLM is unavailable (SDD 11.3)."""
    mode = FLEETS[fleet(machine_id)][1]
    if failure_probability >= 0.7 or rul_estimate <= 20:
        state = "advanced degradation"
    elif failure_probability >= 0.3 or rul_estimate <= 60:
        state = "progressing degradation"
    else:
        state = "normal operation"
    sensors = ", ".join(top_sensors[:3]) if top_sensors else "no dominant sensor"
    summary = (f"{machine_id} at cycle {cycle} shows {state}. "
               f"Anomaly score {anomaly_score:.3f}, failure probability "
               f"{failure_probability:.3f}, estimated RUL {rul_estimate:.0f} cycles. "
               f"Dominant signals: {sensors}.")
    return Diagnosis(summary, mode, _confidence(failure_probability, anomaly_score), "template")


def _api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    from google.cloud import secretmanager
    sm = secretmanager.SecretManagerServiceClient()
    return sm.access_secret_version(request={"name": SECRET}).payload.data.decode()


def _llm_diagnosis(machine_id, cycle, anomaly_score, failure_probability,
                   rul_estimate, top_sensors):
    """Call Gemini. Raises on any failure so the caller can fall back."""
    from google import genai
    ds = fleet(machine_id)
    conditions, mode = FLEETS[ds]
    client = genai.Client(api_key=_api_key())
    prompt = f"""You are a turbofan engine diagnostics assistant for a predictive
maintenance system. Interpret these model outputs into a concise engineering
diagnosis.

Machine: {machine_id}, operating cycle {cycle}
Fleet: NASA C-MAPSS {ds} ({conditions}); documented fault mode: {mode}
Anomaly score (0-1, higher = more abnormal): {anomaly_score:.3f}
Failure probability within 30 cycles: {failure_probability:.1%}
Estimated Remaining Useful Life: {rul_estimate:.0f} cycles
Most influential sensors: {', '.join(top_sensors[:5]) or 'none'}

Reply with exactly 2-3 sentences describing the engine's condition and what
the signals indicate. No preamble, no bullet points. Do not repeat the numeric
values above; the operator already sees them next to your text."""
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(client.models.generate_content, model=MODEL_NAME, contents=prompt)
    pool.shutdown(wait=False)
    resp = fut.result(timeout=TIMEOUT_S)
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("empty LLM response")
    return Diagnosis(text, mode, _confidence(failure_probability, anomaly_score), "llm")


def diagnose(machine_id, cycle, anomaly_score, failure_probability,
             rul_estimate, top_sensors=None, use_llm=True):
    """Produce a diagnosis, falling back to the template if the LLM fails."""
    top_sensors = top_sensors or []
    args = (machine_id, cycle, anomaly_score, failure_probability, rul_estimate, top_sensors)
    if use_llm:
        try:
            return _llm_diagnosis(*args)
        except Exception as e:
            print(f"  [diagnosis] LLM unavailable ({type(e).__name__}: {e}); using template fallback")
    return _template_diagnosis(*args)


if __name__ == "__main__":
    print("Diagnosis Agent: TC-AGT-01 check\n")
    for args in (("fd001_engine_001", 45, 0.31, 0.05, 118.0, ["sensor_4", "sensor_11"]),
                 ("fd004_engine_001", 300, 0.72, 0.88, 14.0, ["sensor_17", "sensor_15", "sensor_4"])):
        d = diagnose(*args)
        print(f"--- {args[0]} cycle {args[1]} ---\n  source: {d.source}\n  mode:   {d.likely_failure_mode}\n"
              f"  conf:   {d.confidence}\n  summary: {d.summary}\n")
    d = diagnose("fd004_engine_001", 300, 0.72, 0.88, 14.0, ["sensor_17"], use_llm=False)
    print(f"--- fallback path (use_llm=False) ---\n  source: {d.source}\n  summary: {d.summary}")
