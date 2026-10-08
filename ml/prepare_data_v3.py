"""Prepare FD001-FD004 for the multi-dataset models (v3).

Replaces prepare_data_multi.py, which learned the operating conditions from
FD001 (a single-condition subset). Here the six conditions come from the
multi-condition subsets, the normaliser is fitted once on all training data,
and the same conditions.py code is used later at the edge.
"""
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from conditions import SENSORS, OP_COLS, FEATURES, SPACE, Normalizer, RollingFeatures, add_features  # noqa: E402

RAW, PROC, MODELS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "models"
COLUMNS = ["unit_number", "time_cycles"] + OP_COLS + [f"sensor_{i}" for i in range(1, 22)]
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
OFFSET = {"FD001": 1000, "FD002": 2000, "FD003": 3000, "FD004": 4000}
RUL_CAP, HORIZON = 125, 30


def read(prefix, ds):
    df = pd.read_csv(RAW / f"{prefix}_{ds}.txt", sep=r"\s+", header=None, names=COLUMNS)
    df["dataset"] = ds
    df["unit_number"] = df["unit_number"] + OFFSET[ds]
    return df


def main():
    train = pd.concat([read("train", d) for d in DATASETS], ignore_index=True)
    last = train.groupby("unit_number")["time_cycles"].transform("max")
    train["RUL"] = (last - train["time_cycles"]).clip(upper=RUL_CAP)
    train["failure_within_horizon"] = (train["RUL"] <= HORIZON).astype(int)

    parts = []
    for d in DATASETS:
        t = read("test", d)
        rul = pd.read_csv(RAW / f"RUL_{d}.txt", sep=r"\s+", header=None, names=["RUL"])["RUL"].values
        final = rul[(t["unit_number"] - OFFSET[d]).values - 1]
        tlast = t.groupby("unit_number")["time_cycles"].transform("max").values
        t["RUL"] = np.minimum(final + tlast - t["time_cycles"].values, RUL_CAP)
        parts.append(t)
    test = pd.concat(parts, ignore_index=True)
    test["failure_within_horizon"] = (test["RUL"] <= HORIZON).astype(int)

    norm = Normalizer.fit(train)
    print("Operating altitudes (kft) per condition:", norm.centers.tolist())
    train_n = add_features(norm.transform_df(train))
    test_n = add_features(norm.transform_df(test))

    print("\nTraining rows per dataset and condition:")
    print(pd.crosstab(train_n["dataset"], train_n["condition"]).to_string())

    print("\nParity test: live edge features vs training features")
    for unit in (OFFSET["FD001"] + 1, OFFSET["FD004"] + 1):
        raw = train[train.unit_number == unit].sort_values("time_cycles")
        feat = train_n[train_n.unit_number == unit].sort_values("time_cycles")
        rf, worst = RollingFeatures(), 0.0
        for (_, r), (_, f) in zip(raw.iterrows(), feat.iterrows()):
            c, z = norm.transform_row({s: r[s] for s in SENSORS}, r["op_setting_1"])
            if c != int(f["condition"]):
                raise SystemExit(f"PARITY FAILED: condition mismatch for unit {unit}")
            live = rf.update(z)
            worst = max(worst, max(abs(live[k] - float(f[k])) for k in FEATURES))
        status = "PASS" if worst < 1e-6 else "FAIL"
        print(f"  unit {unit} ({raw['dataset'].iloc[0]}, {len(raw)} cycles): max difference {worst:.2e}  {status}")
        if status == "FAIL":
            raise SystemExit("PARITY FAILED")

    keep = ["unit_number", "time_cycles", "dataset", "condition"] + FEATURES + ["RUL", "failure_within_horizon"]
    for name, df in (("train", train_n), ("test", test_n)):
        out = df[keep].copy()
        out[FEATURES] = out[FEATURES].astype("float32")
        path = PROC / f"{name}_v3.parquet"
        out.to_parquet(path, index=False)
        print(f"\nsaved {path.name}: {len(out):,} rows, {path.stat().st_size / 1e6:.1f} MB, "
              f"failure rate {out['failure_within_horizon'].mean():.1%}")

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"normalizer": norm.to_dict(), "sensors": SENSORS, "features": FEATURES, "space": SPACE},
                MODELS / "normalizer.pkl")
    print("saved models/normalizer.pkl")


if __name__ == "__main__":
    main()
