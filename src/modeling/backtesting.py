"""Trading strategy backtesting for CrowdWisdomTrading.

Strategy: Simple directional trading based on predicted returns.
  - Go long when predicted return > threshold
  - Stay in cash (flat) when predicted return <= threshold
  - Optional short when predicted return < -threshold

Transaction costs deducted on position changes.

All position decisions use only information available at decision time.
"""

import numpy as np
import pandas as pd
from src.modeling.metrics import calculate_trading_metrics
from src.utils.logging_config import logger


def run_backtest(predictions: np.ndarray, actual_returns: np.ndarray,
                 timestamps: np.ndarray = None,
                 threshold: float = 0.0,
                 allow_short: bool = False,
                 transaction_cost_bps: float = 10,
                 asset: str = "",
                 timeframe: str = "1d",
                 model_name: str = "") -> dict:
    """Run a simple directional trading backtest.

    Args:
        predictions: predicted future returns (log returns)
        actual_returns: realized future returns (log returns)
        timestamps: corresponding timestamps
        threshold: minimum predicted return to enter long position
        allow_short: if True, go short when prediction < -threshold
        transaction_cost_bps: cost in basis points per trade (one-way)
        asset, timeframe, model_name: metadata

    Returns:
        dict with backtest results DataFrame and trading metrics
    """
    n = len(predictions)
    if n < 5:
        logger.warning("Too few observations for backtest: %d", n)
        return {"error": "insufficient_data"}

    cost_rate = transaction_cost_bps / 10000.0

    # Generate positions
    positions = np.zeros(n)
    for i in range(n):
        if predictions[i] > threshold:
            positions[i] = 1.0  # Long
        elif allow_short and predictions[i] < -threshold:
            positions[i] = -1.0  # Short
        else:
            positions[i] = 0.0  # Cash

    # Calculate strategy returns
    strategy_returns = np.zeros(n)
    transaction_costs = np.zeros(n)

    for i in range(n):
        # Strategy return = position * actual_return
        strategy_returns[i] = positions[i] * actual_returns[i]

        # Transaction cost on position change
        if i == 0:
            pos_change = abs(positions[i])
        else:
            pos_change = abs(positions[i] - positions[i - 1])

        tc = pos_change * cost_rate
        transaction_costs[i] = tc
        strategy_returns[i] -= tc

    # Cumulative returns
    cumulative = np.cumprod(1 + strategy_returns)

    # Buy-and-hold benchmark
    bh_cumulative = np.cumprod(1 + actual_returns)

    # Determine annualization factor
    if timeframe == "1h":
        # Crypto: 24*365 = 8760 hours/year; Equities: ~252*6.5 ≈ 1638
        periods_per_year = 8760  # Conservative: use crypto assumption
    else:
        periods_per_year = 252

    # Calculate metrics
    metrics = calculate_trading_metrics(
        strategy_returns, positions,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
    )

    # Buy-and-hold metrics
    bh_metrics = calculate_trading_metrics(
        actual_returns, np.ones(n),
        transaction_cost_bps=0,
        periods_per_year=periods_per_year,
    )

    # Build results DataFrame
    results_df = pd.DataFrame({
        "timestamp_utc": timestamps if timestamps is not None else range(n),
        "signal": ["long" if p > 0 else ("short" if p < 0 else "cash") for p in positions],
        "position": positions,
        "predicted_return": predictions,
        "realized_return": actual_returns,
        "strategy_return": strategy_returns,
        "transaction_cost": transaction_costs,
        "cumulative_return": cumulative,
        "bh_cumulative": bh_cumulative,
        "asset": asset,
        "timeframe": timeframe,
        "model_name": model_name,
    })

    total_tc = transaction_costs.sum()

    return {
        "results_df": results_df,
        "strategy_metrics": metrics,
        "buyhold_metrics": bh_metrics,
        "total_transaction_costs": float(total_tc),
        "threshold": threshold,
        "allow_short": allow_short,
        "transaction_cost_bps": transaction_cost_bps,
        "periods_per_year": periods_per_year,
        "n_observations": n,
    }
