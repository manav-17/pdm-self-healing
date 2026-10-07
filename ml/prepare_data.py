"""Label preparation for C-MAPSS: RUL targets, failure labels, feature selection."""
import pandas as pd
from data_loader import load_train, load_test, load_rul

# Dropped after variance analysis (see explore.py):
#   constant:      sensor_1, 10, 18, 19
#   near-constant: sensor_5, 6, 16
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                "sensor_16", "sensor_18", "sensor_19"]

# FD001 has a single operating condition, so settings carry no information.
DROP_OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]

USEFUL_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8",
                  "sensor_9", "sensor_11", "sensor_12", "sensor_13",
                  "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]

RUL_CAP = 125          # standard in C-MAPSS literature
FAILURE_HORIZON = 30   # "will this engine fail within 30 cycles?"


def add_train_rul(df, cap=RUL_CAP):
    """For training data, RUL = (last cycle of that engine) - (current cycle)."""
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    df = df.copy()
    df["RUL"] = max_cycle - df["time_cycles"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(df, rul_df, cap=RUL_CAP):
    """For test data, the true RUL at the *final* cycle comes from RUL_FD00X.txt.
    RUL at earlier cycles = final_RUL + (last_cycle - current_cycle)."""
    df = df.copy()
    rul_map = rul_df["RUL"].to_dict()          # index 0..99 -> engine 1..100
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    final_rul = df["unit_number"].map(lambda u: rul_map[u - 1])
    df["RUL"] = final_rul + (max_cycle - df["time_cycles"])
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_failure_label(df, horizon=FAILURE_HORIZON):
    """Binary target for the failure-prediction model."""
    df = df.copy()
    df["failure_within_horizon"] = (df["RUL"] <= horizon).astype(int)
    return df


def select_features(df):
    """Keep identifiers, useful sensors, and any labels present."""
    keep = ["unit_number", "time_cycles"] + USEFUL_SENSORS
    keep += [c for c in ("RUL", "failure_within_horizon") if c in df.columns]
    return df[keep]


def build(dataset="FD001"):
    train = add_failure_label(add_train_rul(load_train(dataset)))
    test = add_failure_label(add_test_rul(load_test(dataset), load_rul(dataset)))
    return select_features(train), select_features(test)


if __name__ == "__main__":
    train, test = build()

    print(f"Train: {train.shape}   Test: {test.shape}")
    print(f"Features (14 sensors + 2 ids + 2 labels) = {len(train.columns)} cols")

    print(f"\nRUL range  train: {train.RUL.min()} - {train.RUL.max()}  (capped at {RUL_CAP})")
    print(f"RUL range  test:  {test.RUL.min()} - {test.RUL.max()}")

    pos = train.failure_within_horizon.mean()
    print(f"\nFailure label (RUL <= {FAILURE_HORIZON} cycles):")
    print(f"  positive rate: {pos:.1%}   ({train.failure_within_horizon.sum()} of {len(train)} rows)")

    print("\nEngine 1, first 3 and last 3 cycles:")
    e1 = train[train.unit_number == 1]
    print(pd.concat([e1.head(3), e1.tail(3)])[
        ["unit_number", "time_cycles", "sensor_2", "sensor_11", "RUL", "failure_within_horizon"]
    ].to_string(index=False))

    OUT = "../data/processed"
    train.to_parquet(f"{OUT}/train_FD001.parquet", index=False)
    test.to_parquet(f"{OUT}/test_FD001.parquet", index=False)
    print(f"\nSaved to {OUT}/")
