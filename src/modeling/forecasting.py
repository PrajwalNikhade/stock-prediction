"""Forecasting module for CrowdWisdomTrading.

Generates the complete 6-slot forecast matrix (n+0 through n+5):
  - n+0: Clearly labeled observed reference price / nowcast
  - n+1..n+5: Horizon-specific forward predictions:
    predicted_price_h = reference_price * exp(predicted_log_return_h)

For each horizon, displays:
  - Target timestamp
  - Reference price
  - Predicted price
  - Predicted return
  - Model name
  - Forecast horizon
  - Data mode
  - Status
"""

from datetime import timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from src.processing.feature_engineering import get_feature_columns
from src.utils.logging_config import logger
from src.utils.time_utils import format_utc, utc_now


def generate_forecasts(
    feature_df: pd.DataFrame,
    models: dict,
    scalers: dict,
    asset: str,
    timeframe: str,
    reference_price: float = None,
    reference_timestamp: str = None,
    asset_class: str = None,
    data_mode: str = "live",
    model_names: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Generate the full 6-horizon forecast matrix.

    Args:
        feature_df: Feature DataFrame with features up to decision origin.
        models: dict mapping horizon (1..5) -> fitted model estimator.
        scalers: dict mapping horizon (1..5) -> fitted StandardScaler.
        asset: canonical ticker symbol (e.g. BTCUSDT, NVDA).
        timeframe: '1d' or '1h'.
        reference_price: last observed price at forecast origin.
        reference_timestamp: timestamp of last observed bar.
        asset_class: asset classification string ('crypto', 'us_equities', etc.).
        data_mode: 'live', 'partial', or 'demo'.
        model_names: optional dict mapping horizon -> model name string.

    Returns:
        List of 6 forecast dictionaries (n+0 through n+5).
    """
    if feature_df.empty:
        logger.warning("No feature data for forecasting")
        return []

    feature_cols = get_feature_columns()
    last_row = feature_df.iloc[-1:].copy()
    available_cols = [c for c in feature_cols if c in last_row.columns]
    X_latest = last_row[available_cols].fillna(0)

    if reference_price is None:
        reference_price = float(last_row["close"].iloc[0])
    if reference_timestamp is None:
        reference_timestamp = str(last_row["timestamp_utc"].iloc[0])

    # Determine if asset is equity vs 24/7 continuous crypto
    is_equity = False
    if asset_class:
        is_equity = asset_class.lower() in [
            "equity", "us_equities", "stock", "etf", "index", "commodity"
        ]
    else:
        try:
            from config.settings import get_asset_by_ticker
            cfg = get_asset_by_ticker(asset)
            if cfg and cfg.get("asset_class", "").lower() in [
                "equity", "us_equities", "stock", "etf", "index", "commodity"
            ]:
                is_equity = True
            elif cfg and cfg.get("provider") == "yfinance" and "BTC" not in asset and "ETH" not in asset:
                is_equity = True
        except Exception:
            is_equity = False

    ref_ts = pd.Timestamp(reference_timestamp)
    forecast_origin = format_utc(utc_now())

    forecasts = []

    for h in range(6):
        # Target timestamp calculation
        if h == 0:
            target_ts = ref_ts
        else:
            if timeframe == "1h":
                target_ts = ref_ts + timedelta(hours=h)
            else:
                if is_equity:
                    # Valid trading session: advance by business days (BDay) skipping weekends
                    target_ts = ref_ts + h * pd.tseries.offsets.BDay()
                else:
                    # Crypto trades 24/7 continuously
                    target_ts = ref_ts + timedelta(days=h)

        if hasattr(target_ts, "to_pydatetime"):
            target_ts_str = format_utc(target_ts.to_pydatetime())
        else:
            target_ts_str = str(target_ts)

        if h == 0:
            # n+0: Observed reference price or defined nowcast
            forecasts.append({
                "asset": asset,
                "timeframe": timeframe,
                "forecast_origin": forecast_origin,
                "horizon": 0,
                "horizon_label": "n+0",
                "target_timestamp": target_ts_str,
                "reference_price": float(reference_price),
                "predicted_return": 0.0,
                "predicted_price": float(reference_price),
                "model_name": "Observed Reference",
                "data_mode": data_mode,
                "status": "current/nowcast",
            })
        else:
            # n+1..n+5: Forward predictions
            model = models.get(h)
            scaler = scalers.get(h)

            m_name = None
            if model_names and h in model_names:
                m_name = model_names[h]
            elif model is not None:
                m_name = getattr(model, "__class__", type(model)).__name__
            else:
                m_name = models.get("name", "unavailable")

            if model is None:
                forecasts.append({
                    "asset": asset,
                    "timeframe": timeframe,
                    "forecast_origin": forecast_origin,
                    "horizon": h,
                    "horizon_label": f"n+{h}",
                    "target_timestamp": target_ts_str,
                    "reference_price": float(reference_price),
                    "predicted_return": None,
                    "predicted_price": None,
                    "model_name": "unavailable",
                    "data_mode": data_mode,
                    "status": "no_model",
                })
                continue

            try:
                if scaler is not None:
                    # Scale according to training feature columns
                    fcols = getattr(scaler, "feature_names_in_", None)
                    if fcols is not None:
                        sub_cols = [c for c in fcols if c in X_latest.columns]
                        X_sub = X_latest[sub_cols]
                        X_scaled = scaler.transform(X_sub)
                    else:
                        X_scaled = scaler.transform(X_latest)
                else:
                    X_scaled = X_latest.values

                predicted_return = float(model.predict(X_scaled)[0])
                predicted_price = float(reference_price * np.exp(predicted_return))

                forecasts.append({
                    "asset": asset,
                    "timeframe": timeframe,
                    "forecast_origin": forecast_origin,
                    "horizon": h,
                    "horizon_label": f"n+{h}",
                    "target_timestamp": target_ts_str,
                    "reference_price": float(reference_price),
                    "predicted_return": predicted_return,
                    "predicted_price": predicted_price,
                    "model_name": m_name,
                    "data_mode": data_mode,
                    "status": "forecast",
                })
            except Exception as e:
                logger.error("Forecast error for horizon %d: %s", h, e)
                forecasts.append({
                    "asset": asset,
                    "timeframe": timeframe,
                    "forecast_origin": forecast_origin,
                    "horizon": h,
                    "horizon_label": f"n+{h}",
                    "target_timestamp": target_ts_str,
                    "reference_price": float(reference_price),
                    "predicted_return": None,
                    "predicted_price": None,
                    "model_name": m_name,
                    "data_mode": data_mode,
                    "status": f"error: {e}",
                })

    return forecasts
