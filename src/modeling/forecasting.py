"""Forecasting module for CrowdWisdomTrading.

Generates n+0 through n+5 forecast slots.

Convention:
  - n+0: Current-period reference/nowcast (observed close or intra-bar nowcast)
  - n+1..n+5: Forward predictions using direct-horizon models
  - predicted_price_h = reference_price * exp(predicted_log_return_h)
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from src.processing.feature_engineering import get_feature_columns
from src.utils.logging_config import logger
from src.utils.time_utils import format_utc, utc_now


def generate_forecasts(feature_df: pd.DataFrame, models: dict,
                       scalers: dict, asset: str, timeframe: str,
                       reference_price: float = None,
                       reference_timestamp: str = None) -> list[dict]:
    """Generate 6-horizon forecast matrix.

    Args:
        feature_df: Full feature DataFrame
        models: dict of horizon -> fitted model
        scalers: dict of horizon -> fitted scaler
        asset: asset ticker
        timeframe: '1h' or '1d'
        reference_price: last observed close price
        reference_timestamp: timestamp of last observed bar

    Returns:
        List of 6 forecast dicts (n+0 through n+5)
    """
    if feature_df.empty:
        logger.warning("No feature data for forecasting")
        return []

    feature_cols = get_feature_columns()

    # Use last row as the current state
    last_row = feature_df.iloc[-1:].copy()
    available_cols = [c for c in feature_cols if c in last_row.columns]
    X_latest = last_row[available_cols].fillna(0)

    if reference_price is None:
        reference_price = float(last_row["close"].iloc[0])
    if reference_timestamp is None:
        reference_timestamp = str(last_row["timestamp_utc"].iloc[0])

    # Time delta per horizon step
    if timeframe == "1h":
        delta = timedelta(hours=1)
    else:
        delta = timedelta(days=1)

    ref_ts = pd.Timestamp(reference_timestamp)
    forecast_origin = format_utc(utc_now())

    forecasts = []

    for h in range(6):
        target_ts = ref_ts + h * delta
        horizon_label = f"n+{h}"

        if h == 0:
            # n+0: reference/nowcast - use actual observed close
            forecasts.append({
                "asset": asset,
                "timeframe": timeframe,
                "forecast_origin": forecast_origin,
                "horizon": h,
                "target_timestamp": format_utc(target_ts.to_pydatetime()) if hasattr(target_ts, 'to_pydatetime') else str(target_ts),
                "reference_price": reference_price,
                "predicted_return": 0.0,
                "predicted_price": reference_price,
                "model_name": "observed",
                "status": "current/nowcast",
            })
        else:
            # n+1..n+5: model prediction
            model = models.get(h)
            scaler = scalers.get(h)

            if model is None:
                forecasts.append({
                    "asset": asset,
                    "timeframe": timeframe,
                    "forecast_origin": forecast_origin,
                    "horizon": h,
                    "target_timestamp": format_utc(target_ts.to_pydatetime()) if hasattr(target_ts, 'to_pydatetime') else str(target_ts),
                    "reference_price": reference_price,
                    "predicted_return": None,
                    "predicted_price": None,
                    "model_name": "unavailable",
                    "status": "no_model",
                })
                continue

            try:
                if scaler is not None:
                    X_scaled = scaler.transform(X_latest)
                else:
                    X_scaled = X_latest.values

                predicted_return = float(model.predict(X_scaled)[0])
                predicted_price = reference_price * np.exp(predicted_return)

                forecasts.append({
                    "asset": asset,
                    "timeframe": timeframe,
                    "forecast_origin": forecast_origin,
                    "horizon": h,
                    "target_timestamp": format_utc(target_ts.to_pydatetime()) if hasattr(target_ts, 'to_pydatetime') else str(target_ts),
                    "reference_price": reference_price,
                    "predicted_return": predicted_return,
                    "predicted_price": predicted_price,
                    "model_name": models.get("name", "unknown"),
                    "status": "forecast",
                })
            except Exception as e:
                logger.error("Forecast error for horizon %d: %s", h, e)
                forecasts.append({
                    "asset": asset,
                    "timeframe": timeframe,
                    "forecast_origin": forecast_origin,
                    "horizon": h,
                    "target_timestamp": str(target_ts),
                    "reference_price": reference_price,
                    "predicted_return": None,
                    "predicted_price": None,
                    "model_name": "error",
                    "status": f"error: {e}",
                })

    return forecasts
