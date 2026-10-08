"""Telemetry contract between the plant (MQTT publishers) and the edge.

Every reading must carry the three operating settings and the 14 informative
sensors: the edge needs the operating settings to identify the flight
condition before normalising (FD002/FD004 run at six conditions).
"""
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from conditions import SENSORS, OP_COLS  # noqa: E402

BROKER = os.getenv("MQTT_HOST", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = "plant/+/sensors"


class SensorReading(BaseModel):
    machine_id: str = Field(min_length=1, max_length=64)
    cycle: int = Field(ge=1)
    timestamp: Optional[str] = None
    op_settings: Dict[str, float]
    sensors: Dict[str, float]

    @field_validator("sensors")
    @classmethod
    def _sensors(cls, v):
        missing = [s for s in SENSORS if s not in v]
        if missing:
            raise ValueError(f"missing sensors: {missing}")
        bad = [k for k, x in v.items() if not math.isfinite(x)]
        if bad:
            raise ValueError(f"non-finite values: {bad}")
        return v

    @field_validator("op_settings")
    @classmethod
    def _ops(cls, v):
        missing = [k for k in OP_COLS if k not in v]
        if missing:
            raise ValueError(f"missing operating settings: {missing}")
        if not all(math.isfinite(x) for x in v.values()):
            raise ValueError("non-finite operating setting")
        return v
