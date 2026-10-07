"""End-to-end local pipeline: MQTT -> features -> ML -> agents -> decision."""
import json
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("ml", "agents", "edge", "api"):
    sys.path.insert(0, str(ROOT / sub))

import paho.mqtt.client as mqtt
import requests
from pydantic import ValidationError

from edge_service import SensorReading, RollingWindow, BROKER, PORT, TOPIC
import inference
import orchestrator
import storage

OUT = ROOT / "data" / "decisions.jsonl"
API = os.getenv("API_URL", "http://localhost:8000")
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"
ESCALATE = {"MEDIUM", "HIGH"}


class Pipeline:
    def __init__(self):
        self.run_id = uuid.uuid4().hex[:8]
        self.windows = defaultdict(RollingWindow)
        self.stats = defaultdict(int)
        OUT.parent.mkdir(exist_ok=True)
        self.sink = OUT.open("a")
        print(f"run_id={self.run_id}   LLM={'on' if USE_LLM else 'off'}   API={API}")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"Connected (rc={rc}); subscribing to {TOPIC}\n")
        client.subscribe(TOPIC, qos=1)

    def on_message(self, client, userdata, msg):
        self.stats["received"] += 1
        try:
            reading = SensorReading(**json.loads(msg.payload.decode()))
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
            self.stats["rejected"] += 1
            print(f"REJECTED on {msg.topic}: {type(e).__name__}")
            return

        # Step 3: persist the raw reading and the windowed features
        storage.save_raw(reading.machine_id, reading.cycle, reading.sensors)
        feats = self.windows[reading.machine_id].update(reading.sensors)
        storage.save_features(reading.machine_id, reading.cycle, feats)

        # Step 4: predictions, written back to BigQuery
        pred = inference.predict(feats)
        storage.save_prediction(reading.machine_id, reading.cycle, pred)

        decision = orchestrator.run(
            machine_id=reading.machine_id,
            cycle=reading.cycle,
            anomaly_score=pred["anomaly_score"],
            failure_probability=pred["failure_probability"],
            rul_estimate=pred["rul_estimate"],
            top_sensors=pred["top_sensors"],
            use_llm=USE_LLM,
        )

        self.stats["decisions"] += 1
        self.stats[decision.risk_level] += 1

        row = decision.to_dict()
        row["run_id"] = self.run_id

        note = ""
        if decision.risk_level in ESCALATE:
            note = self._escalate(row)

        self.sink.write(json.dumps(row) + "\n")
        self.sink.flush()

        print(f"{decision.machine_id} c{decision.cycle:>3} | "
              f"RUL={pred['rul_estimate']:>5.1f} "
              f"P(fail)={pred['failure_probability']:.3f} "
              f"anom={pred['anomaly_score']:.3f} | "
              f"risk={decision.risk_score:.3f} {decision.risk_level:<6} | "
              f"{decision.recommended_action:<21}{note}")

    def _escalate(self, row):
        """Submit the decision this pipeline already computed.

        Sends the finished record rather than raw inputs, so the API stores
        exactly what was decided here — one record per cycle, one LLM call.
        """
        try:
            r = requests.post(f"{API}/decisions", json=row, timeout=5)
            r.raise_for_status()
            self.stats["escalated"] += 1
            return "  -> queued for approval"
        except requests.RequestException as e:
            self.stats["api_errors"] += 1
            return f"  -> API unavailable ({type(e).__name__})"

    def run(self):
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        c.on_connect = self.on_connect
        c.on_message = self.on_message
        c.connect(BROKER, PORT, 60)
        try:
            c.loop_forever()
        except KeyboardInterrupt:
            print(f"\n\nrun_id={self.run_id}  stats={dict(self.stats)}")
            self.sink.close()
            c.disconnect()


if __name__ == "__main__":
    print("Pipeline: MQTT -> features -> ML -> agents -> decision\n")
    Pipeline().run()
