"""Train and evaluate the v3 multi-dataset models (FD001-FD004).

Early stopping uses a validation split of training engines (20%, by engine);
the official test sets are used only for the reported scores.
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from conditions import FEATURES, SPACE  # noqa: E402
import model_store  # noqa: E402

PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"
BUCKET = "pdm-models-363641793280"
RUL_CAP, HEALTHY = 125, 30


def nasa_score(err):
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))


def main():
    train = pd.read_parquet(PROC / "train_v3.parquet")
    test = pd.read_parquet(PROC / "test_v3.parquet")
    units = np.sort(train.unit_number.unique())
    val_units = np.random.default_rng(42).choice(units, size=len(units) // 5, replace=False)
    tr = train[~train.unit_number.isin(val_units)]
    va = train[train.unit_number.isin(val_units)]
    print(f"train engines {tr.unit_number.nunique()}, validation engines {va.unit_number.nunique()}, "
          f"test engines {test.unit_number.nunique()}")

    rul = xgb.XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, tree_method="hist", early_stopping_rounds=50,
                           random_state=42, n_jobs=-1)
    rul.fit(tr[FEATURES], tr["RUL"], eval_set=[(va[FEATURES], va["RUL"])], verbose=False)
    print(f"RUL model: best iteration {rul.best_iteration}")

    pos = tr["failure_within_horizon"].mean()
    clf = xgb.XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", scale_pos_weight=(1 - pos) / pos,
                            eval_metric="aucpr", early_stopping_rounds=50, random_state=42, n_jobs=-1)
    clf.fit(tr[FEATURES], tr["failure_within_horizon"],
            eval_set=[(va[FEATURES], va["failure_within_horizon"])], verbose=False)
    print(f"Failure model: best iteration {clf.best_iteration}")

    healthy = tr[tr.time_cycles <= HEALTHY][FEATURES]
    scaler = StandardScaler().fit(healthy)
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(scaler.transform(healthy))
    threshold = float(np.percentile(-iso.score_samples(scaler.transform(healthy)), 95))
    print(f"Anomaly model: {len(healthy):,} healthy rows, threshold {threshold:.4f}")

    last = test.groupby("unit_number").tail(1)
    rows = []
    for ds in sorted(test.dataset.unique()):
        g = last[last.dataset == ds]
        err = np.clip(rul.predict(g[FEATURES]), 0, RUL_CAP) - g["RUL"].values
        tg = test[test.dataset == ds]
        rows.append({
            "dataset": ds, "engines": len(g),
            "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
            "mae": round(float(np.mean(np.abs(err))), 2),
            "nasa_score": round(nasa_score(err), 1),
            "failure_auc": round(float(roc_auc_score(tg.failure_within_horizon,
                                                     clf.predict_proba(tg[FEATURES])[:, 1])), 4),
            "anomaly_auc": round(float(roc_auc_score(tg.failure_within_horizon,
                                                     -iso.score_samples(scaler.transform(tg[FEATURES])))), 4),
        })
    err = np.clip(rul.predict(last[FEATURES]), 0, RUL_CAP) - last["RUL"].values
    combined = {
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "mae": round(float(np.mean(np.abs(err))), 2),
        "failure_auc": round(float(roc_auc_score(test.failure_within_horizon,
                                                 clf.predict_proba(test[FEATURES])[:, 1])), 4),
        "anomaly_auc": round(float(roc_auc_score(test.failure_within_horizon,
                                                 -iso.score_samples(scaler.transform(test[FEATURES])))), 4),
    }
    table = pd.DataFrame(rows)
    print("\nResults on the official test sets (last cycle per engine for RUL; all cycles for AUC):")
    print(table.to_string(index=False))
    print(f"\nCombined: RMSE {combined['rmse']}  MAE {combined['mae']}  "
          f"failure AUC {combined['failure_auc']}  anomaly AUC {combined['anomaly_auc']}")

    bundles = {
        "rul_xgboost": {"model": rul, "features": FEATURES, "space": SPACE},
        "failure_xgboost": {"model": clf, "features": FEATURES, "threshold": 0.5, "space": SPACE},
        "anomaly_iforest": {"model": iso, "scaler": scaler, "features": FEATURES,
                            "threshold": threshold, "space": SPACE},
    }
    print("\nCompatibility check against the v1 bundles the inference code already serves:")
    for name, b in bundles.items():
        try:
            v1 = model_store._load_from_gcs(f"gs://{BUCKET}/models/{name}/v1/model.pkl")
            missing = set(v1) - set(b)
            if missing:
                raise SystemExit(f"  {name}: v1 has keys {sorted(missing)} that v3 lacks. Stop and tell Claude.")
            print(f"  {name}: OK (v1 keys {sorted(v1)} all present)")
        except SystemExit:
            raise
        except Exception as e:
            raise SystemExit(f"  {name}: could not load v1 for comparison ({type(e).__name__}). Re-run on a stable network.")

    for name, b in bundles.items():
        joblib.dump(b, MODELS / f"{name}.pkl")
    metrics = {
        "rul_xgboost": {"rmse": combined["rmse"], "mae": combined["mae"], "space": SPACE,
                        "per_dataset": {r["dataset"]: {"rmse": r["rmse"], "nasa_score": r["nasa_score"]} for r in rows}},
        "failure_xgboost": {"roc_auc": combined["failure_auc"], "space": SPACE,
                            "per_dataset": {r["dataset"]: r["failure_auc"] for r in rows}},
        "anomaly_iforest": {"roc_auc": combined["anomaly_auc"], "threshold": round(threshold, 4), "space": SPACE},
        "normalizer": {"space": SPACE},
    }
    (MODELS / "v3_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nsaved models/rul_xgboost.pkl, failure_xgboost.pkl, anomaly_iforest.pkl, v3_metrics.json")


if __name__ == "__main__":
    main()
