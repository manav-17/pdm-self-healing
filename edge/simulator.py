"""Sensor simulator: replays C-MAPSS engine data over MQTT.

Stands in for real industrial sensors. Each row of the dataset is published
as one telemetry message on plant/<machine_id>/sensors, at a configurable
speed so a 200-cycle engine run can be replayed in seconds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import paho.mqtt.client as mqtt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from prepare_data import USEFUL_SENSORS  # noqa: E402

BROKER, PORT = "localhost", 1883
TOPIC = "plant/{machine_id}/sensors"
DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "train_FD001.parquet"


def build_payload(row):
    return {
        "machine_id": f"engine_{int(row.unit_number):03d}",
        "cycle": int(row.time_cycles),
        "timestamp": time.time(),
        "sensors": {s: float(getattr(row, s)) for s in USEFUL_SENSORS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", type=int, default=3, help="how many engines to replay")
    ap.add_argument("--rate", type=float, default=0.1, help="seconds between messages")
    ap.add_argument("--limit", type=int, default=None, help="max cycles per engine")
    args = ap.parse_args()

    df = pd.read_parquet(DATA)
    engines = sorted(df.unit_number.unique())[: args.engines]

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    print(f"Connected to MQTT broker at {BROKER}:{PORT}")

    sent = 0
    try:
        for eng in engines:
            rows = df[df.unit_number == eng].sort_values("time_cycles")
            if args.limit:
                rows = rows.head(args.limit)

            topic = TOPIC.format(machine_id=f"engine_{int(eng):03d}")
            print(f"\nReplaying engine_{int(eng):03d}  ({len(rows)} cycles) -> {topic}")

            for row in rows.itertuples(index=False):
                client.publish(topic, json.dumps(build_payload(row)), qos=1)
                sent += 1
                if row.time_cycles % 25 == 0:
                    print(f"  cycle {int(row.time_cycles):>3}  RUL={int(row.RUL):>3}")
                time.sleep(args.rate)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"\nPublished {sent} messages.")


if __name__ == "__main__":
    main()
