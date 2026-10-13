"""Model training and walk-forward validation for CrowdWisdomTrading.

Implements:
- Expanding-window walk-forward validation (chronological only)
- Multiple candidate models: Naive baseline, Ridge, Random Forest,
  XGBoost (if available), HistGradientBoosting
- Direct multi-horizon forecasting (separate model per horizon)
- Target: log return for each horizon
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.processing.feature_engineering import get_feature_columns, create_targets
from src.modeling.metrics import calculate_all_metrics
from src.utils.logging_config import logger
from config.settings import RANDOM_STATE

# Try XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost not available, using HistGradientBoosting as alternative")


def get_candidate_models() -> dict:
    """Return dict of model_name -> unfitted estimator."""
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGBR": HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, random_state=RANDOM_STATE,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=0,
        )
    return models


class NaiveBaseline:
    """Random-walk baseline: predicts zero return (price stays the same)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def prepare_model_data(feature_df: pd.DataFrame, horizon: int):
    """Prepare X and y for a specific horizon.

    Returns (X, y, timestamps, feature_cols) with NaN rows dropped.
    """
    target_col = f"target_return_{horizon}"
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in feature_df.columns]

    df = feature_df.copy()

    # Create targets if not present
    if target_col not in df.columns:
        df = create_targets(df, horizons=[horizon])

    # Remove rows where target is NaN (future data not available)
    df = df.dropna(subset=[target_col])

    if df.empty:
        return None, None, None, available_cols

    X = df[available_cols].copy()
    y = df[target_col].values

    # Fill NaN features with 0 and mark via sentiment_missing
    X = X.fillna(0)

    timestamps = df["timestamp_utc"].values if "timestamp_utc" in df.columns else None

    return X, y, timestamps, available_cols


def walk_forward_validate(feature_df: pd.DataFrame, horizon: int,
                          n_folds: int = 5, test_ratio: float = 0.2,
                          gap: int = None) -> dict:
    """Expanding-window walk-forward cross-validation.

    1. Sort by timestamp
    2. Hold out final test_ratio as untouched test set
    3. Split remaining into n_folds expanding windows
    4. Train on expanding window, validate on next fold
    5. Select best model based on validation MAE
    6. Evaluate on final test set

    Gap: number of observations to skip between train and validation
         to prevent target leakage for overlapping horizons.
         Default: horizon + 1
    """
    if gap is None:
        gap = horizon + 1

    X, y, timestamps, feature_cols = prepare_model_data(feature_df, horizon)
    if X is None or len(X) < 30:
        logger.warning("Insufficient data for horizon %d: %d rows",
                       horizon, 0 if X is None else len(X))
        return {"error": "insufficient_data", "n_obs": 0 if X is None else len(X)}

    n = len(X)
    test_start = int(n * (1 - test_ratio))
    if test_start < 20:
        test_start = max(20, n - 10)

    X_dev, y_dev = X.iloc[:test_start], y[:test_start]
    X_test, y_test = X.iloc[test_start:], y[test_start:]

    logger.info(
        "Walk-forward: horizon=%d, total=%d, dev=%d, test=%d, gap=%d",
        horizon, n, len(X_dev), len(X_test), gap,
    )

    # Expanding window folds
    fold_size = max(len(X_dev) // (n_folds + 1), 10)
    min_train = max(30, fold_size)

    models = get_candidate_models()
    models["Naive"] = NaiveBaseline()

    # Track validation metrics per model
    val_results = {name: [] for name in models}

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        val_start = train_end + gap
        val_end = min(val_start + fold_size, len(X_dev))

        if train_end > len(X_dev) or val_start >= len(X_dev) or val_end <= val_start:
            break

        X_train = X_dev.iloc[:train_end]
        y_train = y_dev[:train_end]
        X_val = X_dev.iloc[val_start:val_end]
        y_val = y_dev[val_start:val_end]

        if len(X_val) < 3:
            continue

        # Scale features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val), columns=feature_cols, index=X_val.index
        )

        for name, model in models.items():
            try:
                if name == "Naive":
                    y_pred = model.predict(X_val_scaled)
                else:
                    model_copy = _clone_model(model)
                    model_copy.fit(X_train_scaled, y_train)
                    y_pred = model_copy.predict(X_val_scaled)

                metrics = calculate_all_metrics(y_val, y_pred)
                metrics["fold"] = fold
                val_results[name].append(metrics)
            except Exception as e:
                logger.warning("Model %s failed on fold %d: %s", name, fold, e)

    # Aggregate validation metrics
    model_summary = {}
    for name, fold_metrics in val_results.items():
        if not fold_metrics:
            continue
        avg_metrics = {}
        for key in fold_metrics[0]:
            if key == "fold":
                continue
            values = [m[key] for m in fold_metrics if not np.isnan(m.get(key, float("nan")))]
            avg_metrics[key] = float(np.mean(values)) if values else float("nan")
        avg_metrics["n_folds"] = len(fold_metrics)
        model_summary[name] = avg_metrics

    # Select best model by validation MAE (lower is better)
    best_model_name = "Naive"
    best_mae = float("inf")
    for name, metrics in model_summary.items():
        m = metrics.get("mae", float("inf"))
        if not np.isnan(m) and m < best_mae:
            best_mae = m
            best_model_name = name

    # Refit best model on full dev set and evaluate on test
    scaler_final = StandardScaler()
    X_dev_scaled = pd.DataFrame(
        scaler_final.fit_transform(X_dev), columns=feature_cols, index=X_dev.index
    )
    X_test_scaled = pd.DataFrame(
        scaler_final.transform(X_test), columns=feature_cols, index=X_test.index
    )

    if best_model_name == "Naive":
        best_model = NaiveBaseline()
        y_test_pred = best_model.predict(X_test_scaled)
    else:
        best_model = _clone_model(models[best_model_name])
        best_model.fit(X_dev_scaled, y_dev)
        y_test_pred = best_model.predict(X_test_scaled)

    test_metrics = calculate_all_metrics(y_test, y_test_pred)

    # Also get naive baseline test metrics for comparison
    naive_pred = np.zeros(len(X_test))
    naive_test_metrics = calculate_all_metrics(y_test, naive_pred)

    return {
        "horizon": horizon,
        "best_model": best_model_name,
        "best_model_obj": best_model,
        "scaler": scaler_final,
        "feature_cols": feature_cols,
        "validation_metrics": model_summary,
        "test_metrics": test_metrics,
        "naive_test_metrics": naive_test_metrics,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "n_train": len(X_dev),
        "n_test": len(X_test),
        "test_timestamps": timestamps[test_start:] if timestamps is not None else None,
    }


def train_final_model(feature_df: pd.DataFrame, horizon: int,
                      model_name: str, feature_cols: list[str]):
    """Train the selected model on all available data for production forecasting."""
    X, y, timestamps, fcols = prepare_model_data(feature_df, horizon)
    if X is None or len(X) < 10:
        return None, None

    # Use only the specified feature columns
    available = [c for c in feature_cols if c in X.columns]
    X = X[available].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_name == "Naive":
        model = NaiveBaseline()
    else:
        models = get_candidate_models()
        model = _clone_model(models.get(model_name, models.get("Ridge")))
        model.fit(X_scaled, y)

    return model, scaler


def _clone_model(model):
    """Create a fresh copy of a model with same params."""
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        return model.__class__(**model.get_params())
