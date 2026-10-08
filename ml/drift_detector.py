"""Drift detection (SDD 8, TC-MLO-01, TC-MLO-07), v3 feature space.

Two-sample KS test per feature: recent live features vs the v3 training
baseline. Both are condition-normalised, so data from all four fleets is
comparable. A feature is drifted only if p < P_THRESHOLD AND the KS statistic
exceeds KS_STAT_THRESHOLD (with large samples, p-values alone flag trivial
differences).
"""
import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import SENSORS, SPACE, WINDOW  # noqa: E402
import data_store  # noqa: E402

PROJECT, DATASET = "project-c3e3f3ae-71cf-45cb-800", "pdm"
P_THRESHOLD, KS_STAT_THRESHOLD, RETRAIN_FRACTION, RECENT_WINDOW = 0.05, 0.2, 0.3, 300
FEATURE_COLS = [f"{s}_mean{WINDOW}" for s in SENSORS]


def load_recent_from_bigquery(n=RECENT_WINDOW):
    try:
        import pandas as pd
        from google.cloud import bigquery
        rows = bigquery.Client(project=PROJECT).query(
            f"SELECT features FROM `{PROJECT}.{DATASET}.feature_data` "
            f"WHERE JSON_VALUE(features, '$._space') = '{SPACE}' "
            f"ORDER BY timestamp DESC LIMIT {int(n)}").result()
        recs = []
        for r in rows:
            f = json.loads(r.features) if isinstance(r.features, str) else r.features
            if all(c in f for c in FEATURE_COLS):
                recs.append({c: f[c] for c in FEATURE_COLS})
        return pd.DataFrame(recs) if recs else None
    except Exception as e:
        print(f"[drift] BigQuery unavailable: {e}")
        return None


def run_ks_test(baseline, recent):
    per, drifted = {}, []
    for c in FEATURE_COLS:
        stat, p = stats.ks_2samp(baseline[c].dropna(), recent[c].dropna())
        d = bool(p < P_THRESHOLD and stat > KS_STAT_THRESHOLD)
        per[c] = {"ks_stat": round(float(stat), 4), "p_value": float(p), "drifted": d}
        if d:
            drifted.append(c)
    return round(len(drifted) / len(FEATURE_COLS), 4), drifted, per


def log_drift_event(score, triggered, metric="ks_test"):
    try:
        from google.cloud import bigquery
        err = bigquery.Client(project=PROJECT).insert_rows_json(
            f"{PROJECT}.{DATASET}.drift_events", [{
                "event_id": str(uuid.uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": "all", "drift_metric": metric, "drift_score": score,
                "threshold": KS_STAT_THRESHOLD, "triggered_retrain": triggered}])
        print(f"[drift] logged ({metric}, score={score})" if not err else f"[drift] log error: {err}")
    except Exception as e:
        print(f"[drift] could not log: {e}")


def check(verbose=True, log=True, baseline=None, recent=None,
          metric="ks_test", title="Live check: BigQuery feature_data"):
    if baseline is None:
        baseline = data_store.load("train")[FEATURE_COLS]
    if recent is None:
        recent = load_recent_from_bigquery()
        if recent is None or recent.empty:
            print(f"[drift] no live data in feature space {SPACE}; skipping")
            return 0.0, False, []
    score, drifted, per = run_ks_test(baseline, recent)
    triggered = score > RETRAIN_FRACTION
    if verbose:
        print(f"\n{title}\n" + "=" * 60)
        print(f"baseline={len(baseline):,} rows  recent={len(recent):,} rows  "
              f"drifted={len(drifted)}/{len(FEATURE_COLS)}  score={score:.0%}  retrain={triggered}")
        for c, r in sorted(per.items(), key=lambda x: -x[1]["ks_stat"])[:5]:
            print(f"  {c:<18} ks={r['ks_stat']:.3f} p={r['p_value']:.2g} {'DRIFT' if r['drifted'] else 'ok'}")
    if log:
        log_drift_event(score, triggered, metric)
    return score, triggered, drifted


def selftest():
    """TC-MLO-01: held-out engines (expect no drift) vs injected 3-sigma shift (expect drift)."""
    df = data_store.load("train")
    ids = df.unit_number.unique()
    hold = set(np.random.default_rng(42).choice(ids, size=len(ids) // 5, replace=False))
    base = df[~df.unit_number.isin(hold)][FEATURE_COLS]
    test = df[df.unit_number.isin(hold)][FEATURE_COLS]
    _, t1, _ = check(baseline=base, recent=test, metric="selftest_no_drift",
                     title="Self-test 1: held-out engines, all four fleets (expect NO drift)")
    _, t2, _ = check(baseline=base, recent=test + 3 * base.std(), metric="selftest_injected_shift",
                     title="Self-test 2: same engines + injected 3-sigma shift (expect DRIFT)")
    ok = (not t1) and t2
    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'} (no-drift retrain={t1}, injected retrain={t2})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else check()
