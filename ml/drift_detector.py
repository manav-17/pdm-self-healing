"""Drift detection (SDD §8, TC-MLO-01, TC-MLO-07).

Two-sample KS test per feature, live data vs a raw training baseline.
A feature is drifted only if p < P_THRESHOLD AND KS statistic > KS_STAT_THRESHOLD.
With thousands of rows, p-values alone flag trivial differences, so the
effect-size threshold is what makes the result meaningful.
"""
import json, uuid, argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
PROJECT, DATASET = "project-c3e3f3ae-71cf-45cb-800", "pdm"
P_THRESHOLD, KS_STAT_THRESHOLD, RETRAIN_FRACTION, RECENT_WINDOW = 0.05, 0.2, 0.3, 192
SENSORS = ["sensor_2","sensor_3","sensor_4","sensor_7","sensor_8","sensor_9","sensor_11",
           "sensor_12","sensor_13","sensor_14","sensor_15","sensor_17","sensor_20","sensor_21"]
FEATURE_COLS = [f"{s}_mean10" for s in SENSORS]


def load_baseline_df():
    """Raw FD001 training features: same scale as the live pipeline."""
    return pd.read_parquet(PROC / "train_FD001_features.parquet")


def load_recent_from_bigquery(n=RECENT_WINDOW):
    try:
        from google.cloud import bigquery
        rows = bigquery.Client(project=PROJECT).query(
            f"SELECT features FROM `{PROJECT}.{DATASET}.feature_data` "
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
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_name": "all", "drift_metric": metric, "drift_score": score,
                "threshold": KS_STAT_THRESHOLD, "triggered_retrain": triggered}])
        print(f"[drift] logged ({metric}, score={score})" if not err else f"[drift] log error: {err}")
    except Exception as e:
        print(f"[drift] could not log: {e}")


def check(verbose=True, log=True, baseline=None, recent=None,
          metric="ks_test", title="Live check: BigQuery feature_data"):
    if baseline is None:
        baseline = load_baseline_df()[FEATURE_COLS]
    if recent is None:
        recent = load_recent_from_bigquery()
        if recent is None or recent.empty:
            print("[drift] no live data in BigQuery, skipping")
            return 0.0, False, []
    score, drifted, per = run_ks_test(baseline, recent)
    triggered = score > RETRAIN_FRACTION
    if verbose:
        print(f"\n{title}\n" + "=" * 60)
        print(f"baseline={len(baseline):,} rows  recent={len(recent):,} rows  "
              f"drifted={len(drifted)}/{len(FEATURE_COLS)}  score={score:.0%}  retrain={triggered}")
        for c, r in sorted(per.items(), key=lambda x: -x[1]["ks_stat"])[:5]:
            print(f"  {c:<18} ks={r['ks_stat']:.3f} p={r['p_value']:.2g} "
                  f"{'DRIFT' if r['drifted'] else 'ok'}")
    if log:
        log_drift_event(score, triggered, metric)
    return score, triggered, drifted


def selftest():
    """TC-MLO-01: held-out engines (expect no drift) vs injected 3-sigma shift (expect drift)."""
    df = load_baseline_df()
    ids = df.unit_number.unique()
    hold_ids = set(np.random.default_rng(42).choice(ids, size=len(ids) // 5, replace=False))
    base = df[~df.unit_number.isin(hold_ids)][FEATURE_COLS]
    hold = df[df.unit_number.isin(hold_ids)][FEATURE_COLS]
    _, t1, _ = check(baseline=base, recent=hold, metric="selftest_no_drift",
                     title="Self-test 1: 20 held-out engines, same distribution (expect NO drift)")
    _, t2, _ = check(baseline=base, recent=hold + 3 * base.std(), metric="selftest_injected_shift",
                     title="Self-test 2: same engines + injected 3-sigma shift (expect DRIFT)")
    ok = (not t1) and t2
    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'} (no-drift retrain={t1}, injected retrain={t2})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else check()
