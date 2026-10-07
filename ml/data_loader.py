"""Data loader for NASA C-MAPSS turbofan degradation dataset."""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

COLUMNS = (
    ["unit_number", "time_cycles"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_train(dataset="FD001"):
    """Load a training set (run-to-failure trajectories)."""
    df = pd.read_csv(
        RAW_DIR / f"train_{dataset}.txt",
        sep=r"\s+", header=None, names=COLUMNS,
    )
    return df


def load_test(dataset="FD001"):
    """Load a test set (trajectories truncated before failure)."""
    df = pd.read_csv(
        RAW_DIR / f"test_{dataset}.txt",
        sep=r"\s+", header=None, names=COLUMNS,
    )
    return df


def load_rul(dataset="FD001"):
    """Load true RUL values for the test set."""
    df = pd.read_csv(
        RAW_DIR / f"RUL_{dataset}.txt",
        sep=r"\s+", header=None, names=["RUL"],
    )
    df.index.name = "unit_number_idx"
    return df


if __name__ == "__main__":
    train = load_train()
    test = load_test()
    rul = load_rul()

    print(f"Train shape: {train.shape}   (expect 20631 rows x 26 cols)")
    print(f"Test  shape: {test.shape}")
    print(f"RUL   shape: {rul.shape}     (expect 100 rows)")
    print(f"Train engines: {train.unit_number.nunique()}  (expect 100)")
    print(f"Nulls in train: {train.isnull().sum().sum()}  (expect 0)")
    print("\nFirst 3 rows:")
    print(train.head(3).to_string())
