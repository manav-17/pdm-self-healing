"""Multi-dataset preparation: FD001 + FD002 + FD003 + FD004.

Key difference from prepare_data.py:
- FD002 and FD004 have SIX operating conditions (op_settings vary).
  Raw sensor values are NOT comparable across conditions, so we
  cluster each row into one of 6 regimes and z-score sensors
  WITHIN each regime before feature engineering.
- FD001 and FD003 have ONE condition — no clustering needed,
  but we apply the same pipeline for consistency.

This produces a wider, more representative training distribution
that fixes the single-engine drift-detection baseline problem.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW  = Path(__file__).resolve().parent.parent / "data" / "raw"
PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

RUL_CAP          = 125
FAILURE_HORIZON  = 30
WINDOW           = 10
N_CONDITIONS     = 6   # number of operating regimes for clustering

# Sensors that are constant in FD001 — re-check per dataset below
ALWAYS_DROP = ["sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

COLUMNS = (
    ["unit_number", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}"    for i in range(1, 22)]
)


# ── 1. Raw loading ─────────────────────────────────────────────────────────

def load_raw(dataset):
    def _load(prefix):
        return pd.read_csv(
            RAW / f"{prefix}_{dataset}.txt",
            sep=r"\s+", header=None, names=COLUMNS,
        )
    train = _load("train")
    test  = _load("test")
    rul   = pd.read_csv(RAW / f"RUL_{dataset}.txt",
                        sep=r"\s+", header=None, names=["RUL"])
    train["dataset"] = dataset
    test["dataset"]  = dataset
    # make unit_number unique across datasets
    offset = {"FD001": 0, "FD002": 1000, "FD003": 2000, "FD004": 3000}[dataset]
    train["unit_number"] += offset
    test["unit_number"]  += offset
    return train, test, rul


# ── 2. Operating-condition clustering ──────────────────────────────────────

def assign_conditions(df, kmeans=None):
    """Cluster rows into N_CONDITIONS regimes using op_settings."""
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
    X = df[op_cols].values
    if kmeans is None:
        kmeans = KMeans(n_clusters=N_CONDITIONS, random_state=42, n_init=10)
        kmeans.fit(X)
    df = df.copy()
    df["condition"] = kmeans.predict(X)
    return df, kmeans


# ── 3. Per-condition z-score normalization ─────────────────────────────────

def normalize_per_condition(train_df, test_df, sensor_cols):
    """Z-score each sensor within each operating condition.
    
    Scalers are fit on training data only.
    """
    scalers = {}
    train_out = train_df.copy().astype({c: float for c in sensor_cols})
    test_out  = test_df.copy().astype({c: float for c in sensor_cols})

    for cond in train_df["condition"].unique():
        mask_tr = train_df["condition"] == cond
        mask_te = test_df["condition"]  == cond
        scaler  = StandardScaler()
        train_out.loc[mask_tr, sensor_cols] = scaler.fit_transform(
            train_df.loc[mask_tr, sensor_cols]
        )
        if mask_te.any():
            test_out.loc[mask_te, sensor_cols] = scaler.transform(
                test_df.loc[mask_te, sensor_cols]
            )
        scalers[cond] = scaler

    return train_out, test_out, scalers


# ── 4. Drop low-variance sensors ───────────────────────────────────────────

def select_sensors(df, threshold=0.01):
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    useful = [c for c in sensor_cols
              if c not in ALWAYS_DROP and df[c].std() > threshold]
    return useful


# ── 5. RUL + failure labels ────────────────────────────────────────────────

def add_train_rul(df):
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    df = df.copy()
    df["RUL"] = (max_cycle - df["time_cycles"]).clip(upper=RUL_CAP)
    df["failure_within_horizon"] = (df["RUL"] <= FAILURE_HORIZON).astype(int)
    return df


def add_test_rul(df, rul_df):
    df = df.copy()
    # map rul_df index (0-based) to unit_number
    unit_ids = sorted(df["unit_number"].unique())
    rul_map  = {uid: rul_df["RUL"].iloc[i] for i, uid in enumerate(unit_ids)}
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    final_rul = df["unit_number"].map(rul_map)
    df["RUL"] = (final_rul + (max_cycle - df["time_cycles"])).clip(upper=RUL_CAP)
    df["failure_within_horizon"] = (df["RUL"] <= FAILURE_HORIZON).astype(int)
    return df


# ── 6. Rolling-window features ─────────────────────────────────────────────

def add_rolling_features(df, sensors):
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    g  = df.groupby("unit_number")
    new_cols = {}
    for s in sensors:
        r = g[s].rolling(window=WINDOW, min_periods=1)
        new_cols[f"{s}_mean{WINDOW}"] = r.mean().reset_index(level=0, drop=True)
        new_cols[f"{s}_std{WINDOW}"]  = r.std().reset_index(level=0, drop=True).fillna(0)
        new_cols[f"{s}_delta"]        = df[s] - g[s].transform("first")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


# ── 7. Main build ───────────────────────────────────────────────────────────

def build_all():
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    all_train, all_test = [], []
    kmeans = None   # fit once on FD002 (multi-condition), reuse for others

    for ds in datasets:
        print(f"\nProcessing {ds}...")
        train_raw, test_raw, rul = load_raw(ds)

        # cluster into operating conditions (fit on first multi-condition dataset)
        train_c, kmeans = assign_conditions(train_raw, kmeans)
        test_c,  _      = assign_conditions(test_raw,  kmeans)

        # find useful sensors for this dataset
        sensors = select_sensors(train_c)
        print(f"  useful sensors: {len(sensors)}")

        # per-condition normalization
        train_n, test_n, _ = normalize_per_condition(train_c, test_c, sensors)

        # labels
        train_l = add_train_rul(train_n)
        test_l  = add_test_rul(test_n, rul)

        # rolling features
        train_f = add_rolling_features(train_l, sensors)
        test_f  = add_rolling_features(test_l,  sensors)

        n_pos = train_f.failure_within_horizon.mean()
        print(f"  train rows: {len(train_f):,}  failure rate: {n_pos:.1%}")

        all_train.append(train_f)
        all_test.append(test_f)

    train_combined = pd.concat(all_train, ignore_index=True)
    test_combined  = pd.concat(all_test,  ignore_index=True)

    print(f"\nCombined train: {len(train_combined):,} rows")
    print(f"Combined test:  {len(test_combined):,} rows")
    print(f"Datasets: {train_combined.dataset.value_counts().to_dict()}")

    PROC.mkdir(exist_ok=True)
    train_combined.to_parquet(PROC / "train_ALL_features.parquet", index=False)
    test_combined.to_parquet( PROC / "test_ALL_features.parquet",  index=False)
    print(f"\nSaved to {PROC}/")
    return train_combined, test_combined


if __name__ == "__main__":
    build_all()
