"""Rolling-window feature engineering.

These features mirror what the Edge Layer computes in the live system,
so training-time and inference-time features stay consistent.
"""
import pandas as pd
from prepare_data import USEFUL_SENSORS, build

WINDOW = 10  # cycles


def add_rolling_features(df, sensors=USEFUL_SENSORS, window=WINDOW):
    """Per-engine rolling mean, std, and delta-from-first-reading."""
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    g = df.groupby("unit_number")

    new_cols = {}
    for s in sensors:
        r = g[s].rolling(window=window, min_periods=1)
        new_cols[f"{s}_mean{window}"] = r.mean().reset_index(level=0, drop=True)
        new_cols[f"{s}_std{window}"] = r.std().reset_index(level=0, drop=True).fillna(0)
        new_cols[f"{s}_delta"] = df[s] - g[s].transform("first")

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


if __name__ == "__main__":
    train, test = build()
    train_f = add_rolling_features(train)
    test_f = add_rolling_features(test)

    print(f"Before: {train.shape[1]} cols   After: {train_f.shape[1]} cols")
    print(f"Added {train_f.shape[1] - train.shape[1]} features "
          f"({len(USEFUL_SENSORS)} sensors x 3)")
    print(f"Nulls: {train_f.isnull().sum().sum()}")

    print("\nEngine 1 - sensor_11 degradation (every 30th cycle):")
    e1 = train_f[train_f.unit_number == 1]
    print(e1[["time_cycles", "sensor_11", "sensor_11_mean10",
              "sensor_11_std10", "sensor_11_delta", "RUL"]]
          .iloc[::30].round(3).to_string(index=False))

    OUT = "../data/processed"
    train_f.to_parquet(f"{OUT}/train_FD001_features.parquet", index=False)
    test_f.to_parquet(f"{OUT}/test_FD001_features.parquet", index=False)
    print(f"\nSaved to {OUT}/")
