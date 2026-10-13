"""Model training, walk-forward validation, and fair experiment comparison for CrowdWisdomTrading.

Implements:
- Expanding-window walk-forward validation (strictly chronological)
- Zero-leakage horizon-aware purge at fold boundaries AND development/test boundary:
  Purge gap = horizon + 1 (or max(horizon, 1)), ensuring training targets never overlap validation/test periods.
- Scaler fitting strictly on training partitions only.
- Direct multi-horizon forecasting (n+1 through n+5).
- Candidate estimators: Naive baseline, Ridge, RandomForest, HistGradientBoosting, XGBoost (if available).
- Multi-variant reproducible experiment framework:
  * Experiment A: Naive baseline (predicted log return = 0)
  * Experiment B: Price-only model (price features only)
  * Experiment C: Primary assessment model (price + point-in-time X sentiment)
  * Experiment D: Engagement-enhanced model (price + X + engagement weights)
  * Experiment E: Optional news-enhanced model (price + X + news sentiment)
- Evaluation on strictly identical chronological test windows with Relative MAE improvement.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.processing.feature_engineering import (
    get_feature_columns,
    create_targets,
    PRICE_FEATURE_COLS,
    X_SENTIMENT_FEATURE_COLS,
    ENGAGEMENT_FEATURE_COLS,
    NEWS_FEATURE_COLS,
)
from src.modeling.metrics import calculate_all_metrics
from src.modeling.backtesting import run_backtest
from src.utils.logging_config import logger
from config.settings import RANDOM_STATE

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_candidate_models() -> dict:
    """Return dict of model_name -> unfitted estimator."""
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGBR": HistGradientBoostingRegressor(
            max_iter=150, max_depth=5, random_state=RANDOM_STATE,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=0,
        )
    return models


class NaiveBaseline:
    """Zero future log-return baseline: predicts no price change from reference price."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def prepare_model_data(
    feature_df: pd.DataFrame,
    horizon: int,
    feature_mode: str = "all",
    feature_cols_subset: Optional[List[str]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Prepare X and y for a specific horizon and feature subset.

    Args:
        feature_df: DataFrame containing features and close prices.
        horizon: forecast horizon (1..5).
        feature_mode: 'price_only', 'price_sentiment', 'engagement_enhanced', 'news_enhanced', or 'all'.
        feature_cols_subset: explicit list of feature columns (overrides feature_mode).

    Returns:
        (X, y, timestamps, feature_cols)
    """
    target_col = f"target_return_{horizon}"
    if feature_cols_subset is not None:
        selected_cols = feature_cols_subset
    else:
        selected_cols = get_feature_columns(mode=feature_mode)

    df = feature_df.copy()
    if target_col not in df.columns:
        df = create_targets(df, horizons=[horizon])

    # Drop rows where target is NaN (at the right edge of series where future data is not yet observed)
    df = df.dropna(subset=[target_col])
    if df.empty:
        return None, None, None, selected_cols

    available_cols = [c for c in selected_cols if c in df.columns]
    if not available_cols:
        logger.warning("No requested feature columns present in DataFrame for mode %s", feature_mode)
        return None, None, None, []

    X = df[available_cols].copy().fillna(0)
    y = df[target_col].values
    timestamps = df["timestamp_utc"].values if "timestamp_utc" in df.columns else None

    return X, y, timestamps, available_cols


def walk_forward_validate(
    feature_df: pd.DataFrame,
    horizon: int,
    n_folds: int = 3,
    test_ratio: float = 0.2,
    gap: int = None,
    feature_mode: str = "all",
    feature_cols_subset: Optional[List[str]] = None,
) -> dict:
    """Expanding-window walk-forward cross-validation with zero-leakage horizon purge.

    Purge Rule:
    For horizon h, the training target at time t depends on the price at t + h.
    Therefore, observations whose target intervals reach into the next partition
    (validation or test) must be purged by applying an embargo gap >= h.

    Dev / Test split:
      X_dev strictly ends at (test_start - gap).
      X_test begins at test_start.
      This guarantees zero target overlap between training targets and test observations.
    """
    if gap is None:
        gap = max(horizon + 1, 2)

    X, y, timestamps, feature_cols = prepare_model_data(
        feature_df, horizon, feature_mode=feature_mode, feature_cols_subset=feature_cols_subset
    )
    if X is None or len(X) < 25:
        logger.warning("Insufficient data for horizon %d: %d rows", horizon, 0 if X is None else len(X))
        return {"error": "insufficient_data", "n_obs": 0 if X is None else len(X)}

    n = len(X)
    test_start = int(n * (1 - test_ratio))
    if test_start < 15:
        test_start = max(15, n - 8)

    # Apply embargo purge between development set and final test set
    dev_end = max(10, test_start - gap)
    X_dev, y_dev = X.iloc[:dev_end], y[:dev_end]
    X_test, y_test = X.iloc[test_start:], y[test_start:]

    logger.info(
        "Walk-forward (horizon=%d, mode=%s): total=%d, dev=%d (purged gap=%d), test=%d",
        horizon, feature_mode, n, len(X_dev), gap, len(X_test),
    )

    # Expanding window folds within development set
    fold_size = max(len(X_dev) // (n_folds + 1), 6)
    min_train = max(12, fold_size)

    models = get_candidate_models()
    models["Naive"] = NaiveBaseline()

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

        if len(X_val) < 2:
            continue

        # Feature scaling fitted strictly on train
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)

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

    # Select best model by validation MAE
    best_model_name = "Naive"
    best_mae = float("inf")
    for name, metrics in model_summary.items():
        m = metrics.get("mae", float("inf"))
        if not np.isnan(m) and m < best_mae:
            best_mae = m
            best_model_name = name

    # Refit selected best model on dev set (strictly ending at dev_end)
    scaler_final = StandardScaler()
    X_dev_scaled = pd.DataFrame(scaler_final.fit_transform(X_dev), columns=feature_cols, index=X_dev.index)
    X_test_scaled = pd.DataFrame(scaler_final.transform(X_test), columns=feature_cols, index=X_test.index)

    if best_model_name == "Naive":
        best_model = NaiveBaseline()
        y_test_pred = best_model.predict(X_test_scaled)
    else:
        best_model = _clone_model(models[best_model_name])
        best_model.fit(X_dev_scaled, y_dev)
        y_test_pred = best_model.predict(X_test_scaled)

    test_metrics = calculate_all_metrics(y_test, y_test_pred)

    # Naive baseline test metrics on exact same test set
    naive_pred = np.zeros(len(X_test))
    naive_test_metrics = calculate_all_metrics(y_test, naive_pred)

    return {
        "horizon": horizon,
        "feature_mode": feature_mode,
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
        "test_start_idx": test_start,
        "dev_end_idx": dev_end,
        "gap": gap,
        "test_timestamps": timestamps[test_start:] if timestamps is not None else None,
    }


def run_experiment_suite(
    feature_df: pd.DataFrame,
    horizons: List[int] = None,
    n_folds: int = 3,
    test_ratio: float = 0.2,
    asset: str = "BTCUSDT",
    timeframe: str = "1d",
    transaction_cost_bps: float = 10.0,
    has_real_sentiment: bool = True,
    has_engagement: bool = False,
    has_news: bool = False,
) -> Dict[str, Any]:
    """Execute the multi-experiment comparison framework across all horizons.

    Variants:
      - Experiment A: Naive baseline (predict log return = 0)
      - Experiment B: Price-only model
      - Experiment C: Primary assessment model (Price + X sentiment)
      - Experiment D: Engagement-enhanced model (when historical engagement available)
      - Experiment E: Optional news-enhanced model (when historical news available)

    Ensures all variants are evaluated on strictly identical chronological test sets.
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5]

    comparison_rows = []
    horizon_results = {}

    for h in horizons:
        # Run Walk-Forward Validation for each variant on identical test partitions
        # Experiment B: Price-only
        res_b = walk_forward_validate(
            feature_df, horizon=h, n_folds=n_folds, test_ratio=test_ratio, feature_mode="price_only"
        )
        # Experiment C: Price + X sentiment
        res_c = walk_forward_validate(
            feature_df, horizon=h, n_folds=n_folds, test_ratio=test_ratio, feature_mode="price_sentiment"
        )
        # Experiment D: Engagement-enhanced (if engagement present)
        res_d = None
        if has_engagement:
            res_d = walk_forward_validate(
                feature_df, horizon=h, n_folds=n_folds, test_ratio=test_ratio, feature_mode="engagement_enhanced"
            )
        # Experiment E: News-enhanced (if news present)
        res_e = None
        if has_news:
            res_e = walk_forward_validate(
                feature_df, horizon=h, n_folds=n_folds, test_ratio=test_ratio, feature_mode="news_enhanced"
            )

        if "error" in res_b or "error" in res_c:
            logger.warning("Could not complete experiment comparison for horizon %d: insufficient data", h)
            continue

        test_ts = res_b.get("test_timestamps")
        y_test = res_b.get("y_test")

        # 1. Experiment A: Naive Baseline
        pred_a = np.zeros(len(y_test))
        metrics_a = calculate_all_metrics(y_test, pred_a)
        bt_a = run_backtest(
            predictions=pred_a, actual_returns=y_test, timestamps=test_ts,
            threshold=0.0, transaction_cost_bps=transaction_cost_bps,
            asset=asset, timeframe=timeframe, model_name="Naive Baseline",
        )
        sm_a = bt_a.get("strategy_metrics", {})
        comparison_rows.append({
            "experiment": "Experiment A (Naive Baseline)",
            "experiment_id": "exp_a_naive",
            "asset": asset,
            "timeframe": timeframe,
            "horizon": h,
            "feature_set": "None (zero-return)",
            "best_model": "Naive",
            "validation_mae": np.nan,
            "test_mae": metrics_a.get("mae"),
            "test_rmse": metrics_a.get("rmse"),
            "directional_accuracy": metrics_a.get("directional_accuracy"),
            "sharpe_ratio": sm_a.get("sharpe_ratio"),
            "sortino_ratio": sm_a.get("sortino_ratio"),
            "max_drawdown": sm_a.get("max_drawdown"),
            "strategy_return": sm_a.get("cumulative_return"),
            "relative_mae_improvement_pct": 0.0,
            "data_mode": "baseline",
            "n_obs": len(y_test),
        })

        # 2. Experiment B: Price-Only Model
        pred_b = res_b["y_test_pred"]
        metrics_b = res_b["test_metrics"]
        bt_b = run_backtest(
            predictions=pred_b, actual_returns=y_test, timestamps=test_ts,
            threshold=0.0, transaction_cost_bps=transaction_cost_bps,
            asset=asset, timeframe=timeframe, model_name=f"PriceOnly_{res_b['best_model']}",
        )
        sm_b = bt_b.get("strategy_metrics", {})
        vm_b = res_b.get("validation_metrics", {}).get(res_b["best_model"], {})
        comparison_rows.append({
            "experiment": "Experiment B (Price-Only)",
            "experiment_id": "exp_b_price_only",
            "asset": asset,
            "timeframe": timeframe,
            "horizon": h,
            "feature_set": "Price features only",
            "best_model": res_b["best_model"],
            "validation_mae": vm_b.get("mae"),
            "test_mae": metrics_b.get("mae"),
            "test_rmse": metrics_b.get("rmse"),
            "directional_accuracy": metrics_b.get("directional_accuracy"),
            "sharpe_ratio": sm_b.get("sharpe_ratio"),
            "sortino_ratio": sm_b.get("sortino_ratio"),
            "max_drawdown": sm_b.get("max_drawdown"),
            "strategy_return": sm_b.get("cumulative_return"),
            "relative_mae_improvement_pct": 0.0,
            "data_mode": "price_only",
            "n_obs": len(y_test),
        })

        # 3. Experiment C: Primary Assessment (Price + X Sentiment)
        pred_c = res_c["y_test_pred"]
        metrics_c = res_c["test_metrics"]
        bt_c = run_backtest(
            predictions=pred_c, actual_returns=y_test, timestamps=test_ts,
            threshold=0.0, transaction_cost_bps=transaction_cost_bps,
            asset=asset, timeframe=timeframe, model_name=f"PriceSentiment_{res_c['best_model']}",
        )
        sm_c = bt_c.get("strategy_metrics", {})
        vm_c = res_c.get("validation_metrics", {}).get(res_c["best_model"], {})

        # Compute Relative MAE Improvement:
        # 100 * (Price-only MAE - Price-plus-X MAE) / Price-only MAE
        mae_b = metrics_b.get("mae", 0.0)
        mae_c = metrics_c.get("mae", 0.0)
        rel_improvement = (100.0 * (mae_b - mae_c) / mae_b) if (mae_b and mae_b > 0) else 0.0

        comparison_rows.append({
            "experiment": "Experiment C (Primary: Price + X)",
            "experiment_id": "exp_c_primary",
            "asset": asset,
            "timeframe": timeframe,
            "horizon": h,
            "feature_set": "Price + X sentiment (point-in-time)",
            "best_model": res_c["best_model"],
            "validation_mae": vm_c.get("mae"),
            "test_mae": metrics_c.get("mae"),
            "test_rmse": metrics_c.get("rmse"),
            "directional_accuracy": metrics_c.get("directional_accuracy"),
            "sharpe_ratio": sm_c.get("sharpe_ratio"),
            "sortino_ratio": sm_c.get("sortino_ratio"),
            "max_drawdown": sm_c.get("max_drawdown"),
            "strategy_return": sm_c.get("cumulative_return"),
            "relative_mae_improvement_pct": float(rel_improvement),
            "data_mode": "price_sentiment" if has_real_sentiment else "partial",
            "n_obs": len(y_test),
        })

        # 4. Experiment D: Engagement-Enhanced (if available)
        if res_d and "error" not in res_d:
            pred_d = res_d["y_test_pred"]
            metrics_d = res_d["test_metrics"]
            bt_d = run_backtest(
                predictions=pred_d, actual_returns=y_test, timestamps=test_ts,
                threshold=0.0, transaction_cost_bps=transaction_cost_bps,
                asset=asset, timeframe=timeframe, model_name=f"Engagement_{res_d['best_model']}",
            )
            sm_d = bt_d.get("strategy_metrics", {})
            vm_d = res_d.get("validation_metrics", {}).get(res_d["best_model"], {})
            mae_d = metrics_d.get("mae", 0.0)
            rel_imp_d = (100.0 * (mae_b - mae_d) / mae_b) if (mae_b and mae_b > 0) else 0.0
            comparison_rows.append({
                "experiment": "Experiment D (Engagement-Enhanced)",
                "experiment_id": "exp_d_engagement",
                "asset": asset,
                "timeframe": timeframe,
                "horizon": h,
                "feature_set": "Price + X + Engagement-weighted sentiment",
                "best_model": res_d["best_model"],
                "validation_mae": vm_d.get("mae"),
                "test_mae": metrics_d.get("mae"),
                "test_rmse": metrics_d.get("rmse"),
                "directional_accuracy": metrics_d.get("directional_accuracy"),
                "sharpe_ratio": sm_d.get("sharpe_ratio"),
                "sortino_ratio": sm_d.get("sortino_ratio"),
                "max_drawdown": sm_d.get("max_drawdown"),
                "strategy_return": sm_d.get("cumulative_return"),
                "relative_mae_improvement_pct": float(rel_imp_d),
                "data_mode": "engagement_enhanced",
                "n_obs": len(y_test),
            })

        # 5. Experiment E: News-Enhanced (if available)
        if res_e and "error" not in res_e:
            pred_e = res_e["y_test_pred"]
            metrics_e = res_e["test_metrics"]
            bt_e = run_backtest(
                predictions=pred_e, actual_returns=y_test, timestamps=test_ts,
                threshold=0.0, transaction_cost_bps=transaction_cost_bps,
                asset=asset, timeframe=timeframe, model_name=f"News_{res_e['best_model']}",
            )
            sm_e = bt_e.get("strategy_metrics", {})
            vm_e = res_e.get("validation_metrics", {}).get(res_e["best_model"], {})
            mae_e = metrics_e.get("mae", 0.0)
            rel_imp_e = (100.0 * (mae_b - mae_e) / mae_b) if (mae_b and mae_b > 0) else 0.0
            comparison_rows.append({
                "experiment": "Experiment E (News-Enhanced [Experimental])",
                "experiment_id": "exp_e_news",
                "asset": asset,
                "timeframe": timeframe,
                "horizon": h,
                "feature_set": "Price + X + News sentiment",
                "best_model": res_e["best_model"],
                "validation_mae": vm_e.get("mae"),
                "test_mae": metrics_e.get("mae"),
                "test_rmse": metrics_e.get("rmse"),
                "directional_accuracy": metrics_e.get("directional_accuracy"),
                "sharpe_ratio": sm_e.get("sharpe_ratio"),
                "sortino_ratio": sm_e.get("sortino_ratio"),
                "max_drawdown": sm_e.get("max_drawdown"),
                "strategy_return": sm_e.get("cumulative_return"),
                "relative_mae_improvement_pct": float(rel_imp_e),
                "data_mode": "news_enhanced",
                "n_obs": len(y_test),
            })

        horizon_results[h] = {
            "res_b": res_b,
            "res_c": res_c,
            "res_d": res_d,
            "res_e": res_e,
            "mae_price_only": mae_b,
            "mae_price_sentiment": mae_c,
            "relative_mae_improvement_pct": rel_improvement,
            "directional_acc_diff": (metrics_c.get("directional_accuracy", 0.0) - metrics_b.get("directional_accuracy", 0.0)),
        }

    comparison_df = pd.DataFrame(comparison_rows)
    return {
        "comparison_df": comparison_df,
        "horizon_results": horizon_results,
        "n_evaluated_horizons": len(horizon_results),
    }


def train_final_model(
    feature_df: pd.DataFrame,
    horizon: int,
    model_name: str,
    feature_cols: list[str],
):
    """Train the selected model on all available data for production forecasting."""
    X, y, timestamps, fcols = prepare_model_data(feature_df, horizon, feature_cols_subset=feature_cols)
    if X is None or len(X) < 10:
        return None, None

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
    """Create a fresh copy of an estimator with same parameters."""
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        return model.__class__(**model.get_params())
