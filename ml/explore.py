"""Exploratory analysis of the C-MAPSS FD001 dataset."""
from data_loader import load_train

df = load_train("FD001")

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
op_cols = [c for c in df.columns if c.startswith("op_setting_")]

print("=" * 70)
print("SENSOR VARIANCE CHECK")
print("=" * 70)
stats = df[sensor_cols].agg(["min", "max", "mean", "std"]).T
stats["range"] = stats["max"] - stats["min"]
print(stats.round(4).to_string())

constant = stats[stats["std"] == 0].index.tolist()
print(f"\nCONSTANT sensors (zero variance) -> drop these: {constant}")

near_constant = stats[(stats["std"] > 0) & (stats["std"] < 0.01)].index.tolist()
print(f"NEAR-constant sensors (std < 0.01) -> likely drop: {near_constant}")

useful = [c for c in sensor_cols if c not in constant + near_constant]
print(f"\nUSEFUL sensors ({len(useful)}): {useful}")

print("\n" + "=" * 70)
print("OPERATIONAL SETTINGS")
print("=" * 70)
print(df[op_cols].agg(["min", "max", "std"]).T.round(4).to_string())

print("\n" + "=" * 70)
print("ENGINE LIFETIMES (cycles to failure)")
print("=" * 70)
lifetimes = df.groupby("unit_number").time_cycles.max()
print(lifetimes.describe().round(1).to_string())
