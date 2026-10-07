"""Failure prediction model (XGBoost classifier).

Binary target: will this engine fail within FAILURE_HORIZON cycles?
Evaluated with precision / recall / F1 / ROC-AUC, per TC-ML-04.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support)
import joblib
from pathlib import Path

PROC = Path("../data/processed")
MODELS = Path("../models")
MODELS.mkdir(exist_ok=True)

ID_COLS = ["unit_number", "time_cycles"]
LABEL_COLS = ["RUL", "failure_within_horizon"]


def main():
    train = pd.read_parquet(PROC / "train_FD001_features.parquet")
    test = pd.read_parquet(PROC / "test_FD001_features.parquet")

    features = [c for c in train.columns if c not in ID_COLS + LABEL_COLS]
    X_train, y_train = train[features], train["failure_within_horizon"]
    X_test, y_test = test[features], test["failure_within_horizon"]

    # Handle the 15% / 85% class imbalance rather than resampling.
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print(f"Features: {len(features)}")
    print(f"Train: {len(X_train)} rows, {y_train.mean():.1%} positive")
    print(f"Test:  {len(X_test)} rows, {y_test.mean():.1%} positive")
    print(f"scale_pos_weight: {pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr", random_state=42,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n" + "=" * 52)
    print("FAILURE PREDICTION RESULTS (within 30 cycles)")
    print("=" * 52)
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print("\n" + classification_report(y_test, pred,
                                       target_names=["healthy", "failing"], digits=3))

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print("Confusion matrix:")
    print(f"  True negatives : {tn:>5}   False positives: {fp:>5}")
    print(f"  False negatives: {fn:>5}   True positives : {tp:>5}")
    print(f"\n  Missed failures (FN) = {fn}  <- the costly errors")

    print("\nThreshold sweep:")
    print(f"{'thresh':>7} {'precision':>10} {'recall':>8} {'f1':>7}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, (proba >= t).astype(int), average="binary", zero_division=0)
        print(f"{t:>7.1f} {p:>10.3f} {r:>8.3f} {f1:>7.3f}")

    print("\nTop 10 features:")
    print(pd.Series(model.feature_importances_, index=features)
            .sort_values(ascending=False).head(10).round(4).to_string())

    joblib.dump({"model": model, "features": features, "threshold": 0.5},
                MODELS / "failure_xgboost.pkl")
    print(f"\nSaved model -> {MODELS / 'failure_xgboost.pkl'}")


if __name__ == "__main__":
    main()
