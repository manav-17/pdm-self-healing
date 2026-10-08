"""Diagnosis Agent — interprets raw ML outputs into a readable diagnosis.

SDD 4.2 step 5: "Agent Orchestrator invokes the Diagnosis Agent, which
interprets the predictions."

LLM-based (Gemini). Per SDD 11.3, LLM latency or unavailability must not
break the pipeline, so a deterministic template fallback is always available.
Verified by TC-AGT-01 (structured diagnosis referencing the input signals).
"""
import os
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
TIMEOUT_S = 10


@dataclass
class Diagnosis:
    summary: str
    likely_failure_mode: str
    confidence: str
    source: str          # "llm" or "template"

    def to_dict(self):
        return asdict(self)


def _infer_failure_mode(top_sensors):
    """FD001 has a single documented fault mode (HPC degradation, per the
    dataset readme), so this maps observed sensors to that context."""
    if not top_sensors:
        return "Unspecified degradation"
    return "High-Pressure Compressor (HPC) degradation"


def _confidence(failure_probability, anomaly_score):
    combined = 0.6 * failure_probability + 0.4 * anomaly_score
    if combined >= 0.7:
        return "high"
    if combined >= 0.4:
        return "medium"
    return "low"


def _template_diagnosis(machine_id, cycle, anomaly_score, failure_probability,
                        rul_estimate, top_sensors):
    """Deterministic fallback — used when the LLM is unavailable (SDD 11.3)."""
    mode = _infer_failure_mode(top_sensors)
    conf = _confidence(failure_probability, anomaly_score)

    if failure_probability >= 0.7 or rul_estimate <= 20:
        state = "advanced degradation"
    elif failure_probability >= 0.3 or rul_estimate <= 60:
        state = "progressing degradation"
    else:
        state = "normal operation"

    sensors = ", ".join(top_sensors[:3]) if top_sensors else "no dominant sensor"
    summary = (
        f"{machine_id} at cycle {cycle} shows {state}. "
        f"Anomaly score {anomaly_score:.3f}, failure probability "
        f"{failure_probability:.3f}, estimated RUL {rul_estimate:.0f} cycles. "
        f"Dominant signals: {sensors}."
    )
    return Diagnosis(summary, mode, conf, "template")


def _llm_diagnosis(machine_id, cycle, anomaly_score, failure_probability,
                   rul_estimate, top_sensors):
    """Call Gemini. Raises on any failure so the caller can fall back."""
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            from google.cloud import secretmanager
            sm = secretmanager.SecretManagerServiceClient()
            name = "projects/project-c3e3f3ae-71cf-45cb-800/secrets/gemini-api-key/versions/latest"
            api_key = sm.access_secret_version(request={"name": name}).payload.data.decode()
        except Exception as e:
            raise RuntimeError(f"GEMINI_API_KEY not in env and Secret Manager failed: {e}")

    client = genai.Client(api_key=api_key)

    prompt = f"""You are a turbofan engine diagnostics assistant for a predictive
maintenance system. Interpret these model outputs into a concise engineering
diagnosis. Do not invent numbers not given.

Machine: {machine_id}, operating cycle {cycle}
Anomaly score (0-1, higher = more abnormal): {anomaly_score:.3f}
Failure probability within 30 cycles: {failure_probability:.1%}
Estimated Remaining Useful Life: {rul_estimate:.0f} cycles
Most influential sensors: {', '.join(top_sensors[:5]) or 'none'}

Context: NASA C-MAPSS FD001, whose documented fault mode is
High-Pressure Compressor (HPC) degradation.

Reply with exactly 2-3 sentences describing the engine's condition and what
the signals indicate. No preamble, no bullet points. Do not repeat the numeric values above; the operator already sees them next to your text."""

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(client.models.generate_content, model=MODEL_NAME, contents=prompt)
    pool.shutdown(wait=False)
    resp = fut.result(timeout=TIMEOUT_S)
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("empty LLM response")

    return Diagnosis(text, _infer_failure_mode(top_sensors),
                     _confidence(failure_probability, anomaly_score), "llm")


def diagnose(machine_id, cycle, anomaly_score, failure_probability,
             rul_estimate, top_sensors=None, use_llm=True):
    """Produce a diagnosis, falling back to the template if the LLM fails."""
    top_sensors = top_sensors or []
    args = (machine_id, cycle, anomaly_score, failure_probability,
            rul_estimate, top_sensors)

    if use_llm:
        try:
            return _llm_diagnosis(*args)
        except Exception as e:
            print(f"  [diagnosis] LLM unavailable ({type(e).__name__}: {e}); "
                  f"using template fallback")

    return _template_diagnosis(*args)


if __name__ == "__main__":
    print("Diagnosis Agent — TC-AGT-01 check\n")

    scenarios = [
        ("engine_001", 45, 0.31, 0.05, 118.0, ["sensor_4", "sensor_11"]),
        ("engine_007", 178, 0.72, 0.88, 14.0,
         ["sensor_17", "sensor_15", "sensor_4"]),
    ]

    for args in scenarios:
        print(f"--- {args[0]} cycle {args[1]} ---")
        d = diagnose(*args)
        print(f"  source: {d.source}")
        print(f"  mode:   {d.likely_failure_mode}")
        print(f"  conf:   {d.confidence}")
        print(f"  summary: {d.summary}\n")

    print("--- fallback path (use_llm=False) ---")
    d = diagnose(*scenarios[1], use_llm=False)
    print(f"  source: {d.source}\n  summary: {d.summary}")
