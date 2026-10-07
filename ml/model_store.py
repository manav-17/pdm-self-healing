"""Model store — loads models from Cloud Storage based on model_registry.

Replaces the local joblib.load() calls in inference.py.
Falls back to local files if Cloud Storage is unavailable,
so local development still works without cloud access.
"""
import io
import os
import joblib
from pathlib import Path

PROJECT  = os.getenv("GCP_PROJECT", "project-c3e3f3ae-71cf-45cb-800")
DATASET  = os.getenv("BQ_DATASET",  "pdm")
LOCAL    = Path(__file__).resolve().parent.parent / "models"

_cache = {}


def _load_from_gcs(uri):
    """Download a .pkl artifact from Cloud Storage and deserialise it."""
    from google.cloud import storage
    client = storage.Client(project=PROJECT)
    bucket_name, blob_name = uri.replace("gs://", "").split("/", 1)
    blob = client.bucket(bucket_name).blob(blob_name)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)


def _production_uri(model_name):
    """Query model_registry for the production artifact URI."""
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    rows = list(client.query(
        f"SELECT artifact_uri FROM `{PROJECT}.{DATASET}.model_registry` "
        f"WHERE model_name = '{model_name}' AND status = 'production' AND artifact_uri != '' "
        f"ORDER BY trained_at DESC LIMIT 1"
    ).result())
    return rows[0].artifact_uri if rows else None


def load(model_name):
    """Return the production model, using a local cache to avoid re-downloading."""
    if model_name in _cache:
        return _cache[model_name]

    try:
        uri = _production_uri(model_name)
        if uri:
            print(f"[model_store] loading {model_name} from {uri}")
            obj = _load_from_gcs(uri)
            _cache[model_name] = obj
            return obj
    except Exception as e:
        print(f"[model_store] GCS unavailable ({e}); falling back to local")

    # Fallback: local file (keeps local dev working)
    local_path = LOCAL / f"{model_name}.pkl"
    print(f"[model_store] loading {model_name} from {local_path}")
    obj = joblib.load(local_path)
    _cache[model_name] = obj
    return obj


if __name__ == "__main__":
    for name in ("rul_xgboost", "failure_xgboost", "anomaly_iforest"):
        m = load(name)
        print(f"{name}: loaded, type={type(m)}")
