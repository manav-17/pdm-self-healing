"""Operating-condition handling, shared by training and the live edge pipeline.

C-MAPSS FD002 and FD004 run engines at six flight conditions. Raw sensor
values are only comparable within one condition, so every reading is
  1. assigned to its condition (nearest of the six operating altitudes), and
  2. z-scored with that condition's training statistics.
FD001 and FD003 run at sea level, which is one of the six conditions, so all
four subsets share a single feature space.

Training (vectorised, pandas) and the live edge (one reading at a time) both
use this module; prepare_data_v3.py proves the two produce identical features.
"""
import numpy as np
import pandas as pd

SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9",
           "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
           "sensor_17", "sensor_20", "sensor_21"]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
WINDOW = 10
SPACE = "cond_norm_v3"
ENGINEERED = [f"{s}_{k}" for s in SENSORS for k in (f"mean{WINDOW}", f"std{WINDOW}", "delta")]
FEATURES = SENSORS + ENGINEERED


class Normalizer:
    def __init__(self, centers, mean, std):
        self.centers = np.asarray(centers, dtype=float)
        self.mean = {int(c): dict(v) for c, v in mean.items()}
        self.std = {int(c): dict(v) for c, v in std.items()}

    @staticmethod
    def find_centers(op1):
        centers = np.unique(np.round(np.asarray(op1, dtype=float)))
        if len(centers) != 6:
            raise ValueError(f"expected 6 operating altitudes, found {centers.tolist()}")
        return centers

    def condition(self, op1):
        return int(np.argmin(np.abs(self.centers - float(op1))))

    def conditions(self, op1):
        op1 = np.asarray(op1, dtype=float)
        return np.abs(op1[:, None] - self.centers[None, :]).argmin(axis=1)

    @classmethod
    def fit(cls, df):
        multi = df[df["dataset"].isin(["FD002", "FD004"])]
        norm = cls(cls.find_centers(multi["op_setting_1"]), {}, {})
        cond = norm.conditions(df["op_setting_1"].values)
        for c in range(len(norm.centers)):
            g = df.loc[cond == c, SENSORS]
            if g.empty:
                raise ValueError(f"no training rows for condition {c}")
            norm.mean[c] = {s: float(g[s].mean()) for s in SENSORS}
            norm.std[c] = {s: (float(g[s].std(ddof=0)) or 1.0) for s in SENSORS}
        return norm

    def transform_df(self, df):
        out = df.copy()
        cond = self.conditions(df["op_setting_1"].values)
        out["condition"] = cond
        mu = pd.DataFrame(self.mean).T.loc[cond, SENSORS].values
        sd = pd.DataFrame(self.std).T.loc[cond, SENSORS].values
        for i, s in enumerate(SENSORS):
            out[s] = (df[s].astype(float).values - mu[:, i]) / sd[:, i]
        return out

    def transform_row(self, sensors, op_setting_1):
        c = self.condition(op_setting_1)
        return c, {s: (float(sensors[s]) - self.mean[c][s]) / self.std[c][s] for s in SENSORS}

    def to_dict(self):
        return {"centers": self.centers.tolist(), "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d):
        return cls(d["centers"], d["mean"], d["std"])


def add_features(df):
    """Rolling features on normalised sensors (training, vectorised)."""
    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)
    g = df.groupby("unit_number", sort=False)
    cols = {}
    for s in SENSORS:
        r = g[s].rolling(WINDOW, min_periods=1)
        cols[f"{s}_mean{WINDOW}"] = r.mean().reset_index(level=0, drop=True)
        cols[f"{s}_std{WINDOW}"] = r.std().reset_index(level=0, drop=True).fillna(0.0)
        cols[f"{s}_delta"] = df[s] - g[s].transform("first")
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


class RollingFeatures:
    """The same features, one reading at a time (live edge)."""
    def __init__(self):
        self.history = {s: [] for s in SENSORS}
        self.first = None

    def update(self, norm):
        if self.first is None:
            self.first = dict(norm)
        out = {}
        for s in SENSORS:
            h = self.history[s]
            h.append(float(norm[s]))
            if len(h) > WINDOW:
                del h[0]
            arr = np.asarray(h)
            out[s] = float(norm[s])
            out[f"{s}_mean{WINDOW}"] = float(arr.mean())
            out[f"{s}_std{WINDOW}"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[f"{s}_delta"] = float(norm[s]) - float(self.first[s])
        return out
