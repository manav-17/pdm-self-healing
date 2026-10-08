"""v3 training data for drift detection and retraining.

Local copies are used when present (development). On Cloud Run the files are
downloaded once from Cloud Storage and cached in /tmp.
"""
import os
from pathlib import Path
import pandas as pd

PROJECT = os.getenv("GCP_PROJECT", "project-c3e3f3ae-71cf-45cb-800")
BUCKET = os.getenv("MODEL_BUCKET", "pdm-models-363641793280")
LOCAL = Path(__file__).resolve().parents[1] / "data" / "processed"
CACHE = Path("/tmp/pdm-data")
_frames = {}


def load(name):
    """name: 'train' or 'test'."""
    if name in _frames:
        return _frames[name]
    path = LOCAL / f"{name}_v3.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{name}_v3.parquet"
        if not path.exists():
            from google.cloud import storage
            blob = storage.Client(project=PROJECT).bucket(BUCKET).blob(f"data/v3/{name}_v3.parquet")
            blob.download_to_filename(str(path), timeout=600)
    _frames[name] = pd.read_parquet(path)
    return _frames[name]
