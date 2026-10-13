"""Trading strategy backtesting for CrowdWisdomTrading.

Key Quantitative Features:
- Mathematically consistent return conversion:
  Training targets are forward log returns: r = ln(P_{t+1} / P_t)
  Converted to simple return for trading: R_simple = exp(r) - 1
  Long return: exp(r) - 1
  Short return: -(exp(r) - 1) = 1 - exp(r)
  Cash return: 0.0
- Configurable transaction costs (default 10 bps one-way on position changes).
  Position change = abs(current_pos - prev_pos).
  Entry / exit = 1 unit of cost.
  Reversal (long to short or short to long) = 2 units of cost.
  Holding same position = 0 cost.
- Wealth index starting at 1.0; maximum drawdown evaluated including initial wealth.
- Asset-aware annualization factors:
  Continuous crypto: 365 daily, 8760 hourly.
  Equities / tradfi: 252 daily, 1638 hourly.
- Honest handling of degenerate strategies (cash strategies report NaN for Sharpe/Sortino).
- One-step-ahead (n+1) forecasts only for primary strategy to prevent overlapping return bias.
"""

import numpy as np
import pandas as pd
from src.modeling.metrics import calculate_trading_metrics
from src.utils.logging_config import logger


def run_backtest(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    timestamps: np.ndarray = None,
    threshold: float = 0.0,
    allow_short: bool = False,
    transaction_cost_bps: float = 10,
    asset: str = "",
    timeframe: str = "1d",
    model_name: str = "",
    is_crypto: bool = None,
) -> dict:
    """Run a directional trading strategy backtest with consistent return math.

    Args:
        predictions: predicted future returns (log returns, 1-step ahead)
        actual_returns: realized future returns (forward log returns: ln(P_{t+1}/P_t))
        timestamps: corresponding observation timestamps
        threshold: minimum predicted return to enter long position
        allow_short: if True, go short when prediction < -threshold
        transaction_cost_bps: cost in basis points per position change (10 bps = 0.0010)
        asset: ticker symbol
        timeframe: '1d' or '1h'
        model_name: estimator identifier
        is_crypto: optional boolean flag; auto-detected from asset ticker if None

    Returns:
        dict with results_df, strategy_metrics, buyhold_metrics, and parameters.
    """
    n = len(predictions)
    if n < 5:
        logger.warning("Too few observations for backtest: %d", n)
        return {"error": "insufficient_data"}

    cost_rate = float(transaction_cost_bps) / 10000.0

    # Determine asset-specific annualization factor
    if is_crypto is None:
        asset_upper = asset.upper()
        is_crypto = any(c in asset_upper for c in ["BTC", "ETH", "SOL", "USDT", "CRYPTO"]) or (
            "USD" in asset_upper and "." not in asset_upper and "-" not in asset_upper
        )

    if is_crypto:
        periods_per_year = 8760 if timeframe == "1h" else 365
    else:
        periods_per_year = 1638 if timeframe == "1h" else 252

    # 1. Generate position signals at decision origin t
    # Prediction available at time t decides position for period [t, t+1]
    positions = np.zeros(n)
    for i in range(n):
        pred = predictions[i]
        if pred > threshold:
            positions[i] = 1.0  # Long
        elif allow_short and pred < -threshold:
            positions[i] = -1.0  # Short
        else:
            positions[i] = 0.0  # Cash

    # 2. Convert forward log returns to simple returns
    # R_asset = exp(r) - 1
    # For long (+1): R_gross = exp(r) - 1
    # For short (-1): R_gross = -(exp(r) - 1)
    # For cash (0): R_gross = 0
    actual_simple_returns = np.exp(actual_returns) - 1.0
    strategy_returns = np.zeros(n)
    transaction_costs = np.zeros(n)

    prev_pos = 0.0  # Assumes portfolio begins in 100% cash prior to first bar

    for i in range(n):
        curr_pos = positions[i]

        # Gross simple return
        gross_return = curr_pos * actual_simple_returns[i]

        # Transaction cost on position change
        # delta = abs(curr_pos - prev_pos)
        # Entry / Exit: delta = 1 -> 1 * cost_rate
        # Reversal (+1 -> -1 or -1 -> +1): delta = 2 -> 2 * cost_rate
        # Maintained position: delta = 0 -> 0 cost
        pos_change = abs(curr_pos - prev_pos)
        tc = pos_change * cost_rate

        transaction_costs[i] = tc
        strategy_returns[i] = gross_return - tc

        prev_pos = curr_pos

    # 3. Compound wealth index starting at 1.0
    # W_t = W_{t-1} * (1 + strategy_returns[t])
    cumulative_wealth = np.cumprod(1.0 + strategy_returns)

    # 4. Buy-and-hold benchmark on identical observations
    # Buy-and-hold enters long at inception (1 position change cost if applied, or pure passive)
    # Here benchmark tracks passive asset simple return
    bh_returns = actual_simple_returns
    bh_cumulative = np.cumprod(1.0 + bh_returns)

    # 5. Compute performance metrics
    metrics = calculate_trading_metrics(
        strategy_returns,
        positions,
        transaction_cost_bps=transaction_cost_bps,
        periods_per_year=periods_per_year,
    )

    bh_metrics = calculate_trading_metrics(
        bh_returns,
        np.ones(n),
        transaction_cost_bps=0,
        periods_per_year=periods_per_year,
    )

    # 6. Build detailed results table
    results_df = pd.DataFrame({
        "timestamp_utc": timestamps if timestamps is not None else list(range(n)),
        "signal": ["long" if p > 0 else ("short" if p < 0 else "cash") for p in positions],
        "position": positions,
        "predicted_return": predictions,
        "realized_return": actual_returns,  # log return
        "simple_realized_return": actual_simple_returns,
        "strategy_return": strategy_returns,
        "transaction_cost": transaction_costs,
        "cumulative_return": cumulative_wealth,
        "bh_cumulative": bh_cumulative,
        "asset": asset,
        "timeframe": timeframe,
        "model_name": model_name,
    })

    total_tc = float(np.sum(transaction_costs))

    return {
        "results_df": results_df,
        "strategy_metrics": metrics,
        "buyhold_metrics": bh_metrics,
        "total_transaction_costs": total_tc,
        "threshold": threshold,
        "allow_short": allow_short,
        "transaction_cost_bps": transaction_cost_bps,
        "periods_per_year": periods_per_year,
        "is_crypto": is_crypto,
        "n_observations": n,
    }
