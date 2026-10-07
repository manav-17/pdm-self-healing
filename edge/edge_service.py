"""Edge / API Layer.

Subscribes to MQTT sensor telemetry, validates each reading, maintains a
per-machine rolling window, and emits engineered feature records.

The rolling-window logic mirrors ml/features.py so that features computed
live match the features the models were trained on.
"""
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean, stdev

import paho.mqtt.client as mqtt
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from prepare_data import USEFUL_SENSORS  # noqa: E402

BROKER, PORT = "localhost", 1883
TOPIC = "plant/+/sensors"
WINDOW = 10  # must match ml/features.py


class SensorReading(BaseModel):
    """Schema for an incoming telemetry message. Rejects malformed readings."""
    machine_id: str = Field(min_length=1)
    cycle: int = Field(ge=0)
    timestamp: float
    sensors: dict[str, float]


class RollingWindow:
    """Per-machine sliding window; computes mean/std/delta like features.py."""

    def __init__(self, size=WINDOW):
        self.size = size
        self.history = defaultdict(lambda: deque(maxlen=size))
        self.baseline = {}

    def update(self, sensors):
        feats = {}
        for name, value in sensors.items():
            buf = self.history[name]
            buf.append(value)
            if name not in self.baseline:
                self.baseline[name] = value  # first reading = engine baseline

            feats[name] = value
            feats[f"{name}_mean{self.size}"] = mean(buf)
            feats[f"{name}_std{self.size}"] = stdev(buf) if len(buf) > 1 else 0.0
            feats[f"{name}_delta"] = value - self.baseline[name]
        return feats


class EdgeService:
    def __init__(self):
        self.windows = defaultdict(RollingWindow)
        self.stats = {"received": 0, "valid": 0, "rejected": 0}

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"Connected to broker (rc={rc}); subscribing to {TOPIC}")
        client.subscribe(TOPIC, qos=1)

    def on_message(self, client, userdata, msg):
        self.stats["received"] += 1
        try:
            reading = SensorReading(**json.loads(msg.payload.decode()))
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError) as e:
            self.stats["rejected"] += 1
            print(f"  REJECTED on {msg.topic}: {type(e).__name__}")
            return

        missing = set(USEFUL_SENSORS) - set(reading.sensors)
        if missing:
            self.stats["rejected"] += 1
            print(f"  REJECTED {reading.machine_id} c{reading.cycle}: missing {sorted(missing)}")
            return

        self.stats["valid"] += 1
        feats = self.windows[reading.machine_id].update(reading.sensors)

        record = {
            "machine_id": reading.machine_id,
            "cycle": reading.cycle,
            "timestamp": reading.timestamp,
            "features": feats,
        }
        self.emit(record)

    def emit(self, record):
        """Local: print. Later: write to BigQuery / call the inference API."""
        f = record["features"]
        print(f"{record['machine_id']} c{record['cycle']:>3} | "
              f"s11={f['sensor_11']:.2f} "
              f"mean={f['sensor_11_mean10']:.3f} "
              f"std={f['sensor_11_std10']:.3f} "
              f"delta={f['sensor_11_delta']:+.3f} | "
              f"{len(f)} features")

    def run(self):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(BROKER, PORT, 60)
        try:
            client.loop_forever()
        except KeyboardInterrupt:
            print(f"\n\nStats: {self.stats}")
            client.disconnect()


if __name__ == "__main__":
    EdgeService().run()
