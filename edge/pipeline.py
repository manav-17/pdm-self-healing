"""End-to-end edge pipeline: MQTT -> features -> ML -> agents -> decision.

Runs at the edge. Every cycle is scored, and its raw reading, features and
predictions are written to BigQuery. Decisions at or above the escalation
threshold are submitted to the API (local or Cloud Run), which stores them
and places them in the operator's approval queue.

With USE_LLM=true, the LLM diagnosis is requested only for escalated
decisions. The risk score is deterministic and never depends on the LLM,
so routine LOW-risk cycles don't need an LLM explanation.
"""
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("ml", "agents", "edge", "api"):
    sys.path.insert(0, str(ROOT / sub))

import paho.mqtt.client as mqtt  # noqa: E402
import requests  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from edge_service import SensorReading, RollingWindow, BROKER, PORT, TOPIC  # noqa: E402
import inference  # noqa: E402
import orchestrator  # noqa: E402
import storage  # noqa: E402

OUT = ROOT / "data" / "decisions.jsonl"
API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "6"))
ESCALATE = {"MEDIUM", "HIGH"}


class Pipeline:
    def __init__(self):
        self.run_id = uuid.uuid4().hex[:8]
        self.windows = defaultdict(RollingWindow)
        self.stats = defaultdict(int)
        self.last_llm = 0.0
        self.headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
        OUT.parent.mkdir(exist_ok=True)
        self.sink = OUT.open("a")
        print(f"run_id={self.run_id}  LLM={'on (escalated only)' if USE_LLM else 'off'}  API={API}", flush=True)

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"Connected (rc={rc}); subscribing to {TOPIC}\n", flush=True)
        client.subscribe(TOPIC, qos=1)

    def on_message(self, client, userdata, msg):
        self.stats["received"] += 1
        try:
            reading = SensorReading(**json.loads(msg.payload.decode()))
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
            self.stats["rejected"] += 1
            print(f"REJECTED on {msg.topic}: {type(e).__name__}", flush=True)
            return

        storage.save_raw(reading.machine_id, reading.cycle, reading.sensors)
        feats = self.windows[reading.machine_id].update(reading.sensors)
        storage.save_features(reading.machine_id, reading.cycle, feats)
        pred = inference.predict(feats)
        storage.save_prediction(reading.machine_id, reading.cycle, pred)

        args = dict(machine_id=reading.machine_id, cycle=reading.cycle,
                    anomaly_score=pred["anomaly_score"],
                    failure_probability=pred["failure_probability"],
                    rul_estimate=pred["rul_estimate"], top_sensors=pred["top_sensors"])
        decision = orchestrator.run(**args, use_llm=False)
        escalate = decision.risk_level in ESCALATE
        if escalate and USE_LLM:
            wait = LLM_MIN_INTERVAL - (time.time() - self.last_llm)
            if wait > 0:
                time.sleep(wait)
            decision = orchestrator.run(**args, use_llm=True)
            self.last_llm = time.time()

        self.stats["decisions"] += 1
        self.stats[decision.risk_level] += 1
        row = decision.to_dict()
        row["run_id"] = self.run_id
        note = self._escalate(row) if escalate else ""
        self.sink.write(json.dumps(row) + "\n")
        self.sink.flush()

        print(f"{decision.machine_id} c{decision.cycle:>3} | "
              f"RUL={pred['rul_estimate']:>5.1f} P(fail)={pred['failure_probability']:.3f} "
              f"anom={pred['anomaly_score']:.3f} | risk={decision.risk_score:.3f} "
              f"{decision.risk_level:<6} | {decision.recommended_action:<21}{note}", flush=True)

    def _escalate(self, row):
        try:
            r = requests.post(f"{API}/decisions", json=row, headers=self.headers, timeout=30)
            r.raise_for_status()
            self.stats["escalated"] += 1
            return f"  -> queued [{row['diagnosis_source']}]"
        except requests.RequestException as e:
            self.stats["api_errors"] += 1
            code = getattr(getattr(e, "response", None), "status_code", "")
            return f"  -> API error {code} ({type(e).__name__})"

    def run(self):
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        c.on_connect = self.on_connect
        c.on_message = self.on_message
        c.connect(BROKER, PORT, 60)
        try:
            c.loop_forever()
        except KeyboardInterrupt:
            print(f"\nrun_id={self.run_id}  stats={dict(self.stats)}", flush=True)
            self.sink.close()
            c.disconnect()


if __name__ == "__main__":
    print("Pipeline: MQTT -> features -> ML -> agents -> decision\n", flush=True)
    Pipeline().run()
