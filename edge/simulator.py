"""Replay NASA C-MAPSS engines as live MQTT telemetry (FD001-FD004).

Each message carries the three operating settings and the 14 informative
sensors. Machine IDs encode the fleet, e.g. fd004_engine_001.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "edge"))
from conditions import SENSORS, OP_COLS  # noqa: E402
from telemetry import BROKER, PORT  # noqa: E402

COLUMNS = ["unit_number", "time_cycles"] + OP_COLS + [f"sensor_{i}" for i in range(1, 22)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--engines", type=int, default=1)
    ap.add_argument("--start-engine", type=int, default=1)
    ap.add_argument("--rate", type=float, default=0.2, help="seconds between messages")
    a = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / "raw" / f"train_{a.dataset}.txt",
                     sep=r"\s+", header=None, names=COLUMNS)
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.connect(BROKER, PORT, 60)
    c.loop_start()
    print(f"Connected to MQTT broker at {BROKER}:{PORT}\n")

    total = 0
    for unit in range(a.start_engine, a.start_engine + a.engines):
        eng = df[df.unit_number == unit].sort_values("time_cycles")
        if eng.empty:
            print(f"engine {unit} is not in {a.dataset}; skipping")
            continue
        mid = f"{a.dataset.lower()}_engine_{unit:03d}"
        topic = f"plant/{mid}/sensors"
        n = len(eng)
        print(f"Replaying {mid}  ({n} cycles) -> {topic}")
        for _, r in eng.iterrows():
            cyc = int(r["time_cycles"])
            payload = {"machine_id": mid, "cycle": cyc,
                       "timestamp": datetime.now(timezone.utc).isoformat(),
                       "op_settings": {k: float(r[k]) for k in OP_COLS},
                       "sensors": {s: float(r[s]) for s in SENSORS}}
            c.publish(topic, json.dumps(payload), qos=1).wait_for_publish(timeout=10)
            total += 1
            if cyc % 25 == 0:
                print(f"  cycle {cyc:>3}  true RUL={min(n - cyc, 125):>3}")
            time.sleep(a.rate)
        print(f"LAST machine={mid} cycle={n}")

    c.loop_stop()
    c.disconnect()
    print(f"\nPublished {total} messages.")


if __name__ == "__main__":
    main()
