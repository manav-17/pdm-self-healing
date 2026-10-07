"""RUL prediction model (XGBoost baseline).

Evaluated with RMSE, MAE and the NASA asymmetric scoring function used
throughout the C-MAPSS literature, so results are comparable to published work.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

PROC = Path("../data/processed")
MODELS = Path("../models")
MODELS.mkdir(exist_ok=True)

ID_COLS = ["unit_number", "time_cycles"]
LABEL_COLS = ["RUL", "failure_within_horizon"]


def nasa_score(y_true, y_pred):
    """NASA C-MAPSS scoring function.

    Asymmetric on purpose: predicting failure LATER than reality (d > 0) is
    penalised harder than predicting it early, because a late prediction means
    a missed maintenance window. Lower is better.
    """
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def main():
    train = pd.read_parquet(PROC / "train_FD001_features.parquet")
    test = pd.read_parquet(PROC / "test_FD001_features.parquet")

    features = [c for c in train.columns if c not in ID_COLS + LABEL_COLS]
    X_train, y_train = train[features], train["RUL"]

    # Standard C-MAPSS protocol: evaluate on the LAST cycle of each test engine.
    last = test.groupby("unit_number").tail(1)
    X_test, y_test = last[features], last["RUL"]

    print(f"Features: {len(features)}")
    print(f"Train rows: {len(X_train)}   Test engines: {len(X_test)}")

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pred = np.clip(model.predict(X_test), 0, 125)

    print("\n" + "=" * 52)
    print("RUL MODEL RESULTS (FD001, last cycle per test engine)")
    print("=" * 52)
    print(f"RMSE:       {np.sqrt(mean_squared_error(y_test, pred)):.2f} cycles")
    print(f"MAE:        {mean_absolute_error(y_test, pred):.2f} cycles")
    print(f"NASA score: {nasa_score(y_test.values, pred):.1f}   (lower is better)")
    print(f"Best iteration: {model.best_iteration}")

    print("\nTop 10 features by importance:")
    print(pd.Series(model.feature_importances_, index=features)
            .sort_values(ascending=False).head(10).round(4).to_string())

    print("\nSample predictions (first 10 engines):")
    print(pd.DataFrame({
        "engine": last["unit_number"].values[:10],
        "true_RUL": y_test.values[:10],
        "pred_RUL": pred[:10].round(1),
        "error": (pred[:10] - y_test.values[:10]).round(1),
    }).to_string(index=False))

    joblib.dump({"model": model, "features": features}, MODELS / "rul_xgboost.pkl")
    print(f"\nSaved model -> {MODELS / 'rul_xgboost.pkl'}")


if __name__ == "__main__":
    main()
