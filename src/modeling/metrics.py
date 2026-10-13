"""Prediction metrics for CrowdWisdomTrading.

All metrics are documented with their formula and applicable data type.
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error: (1/n) * sum(|y_true - y_pred|)"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error: sqrt((1/n) * sum((y_true - y_pred)^2))"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared: 1 - SS_res / SS_tot
    Only meaningful when SS_tot > 0."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-15:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.
    WARNING: Not reliable when y_true values approach zero.
    Only computed for |y_true| > 1e-8.
    """
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error:
    (100/n) * sum(|y_true - y_pred| / (|y_true| + |y_pred|)/2)
    """
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 1e-15
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional accuracy: fraction of predictions with correct sign.
    Uses sign of returns (positive/negative/zero).
    """
    if len(y_true) == 0:
        return float("nan")
    correct = np.sign(y_true) == np.sign(y_pred)
    return float(np.mean(correct) * 100)


def prediction_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean prediction error (bias): mean(y_pred - y_true)
    Positive = model over-predicts on average.
    """
    return float(np.mean(y_pred - y_true))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate all prediction metrics."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {k: float("nan") for k in
                ["mae", "rmse", "r_squared", "mape", "smape",
                 "directional_accuracy", "prediction_bias", "n_obs"]}

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r_squared": r_squared(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "prediction_bias": prediction_bias(y_true, y_pred),
        "n_obs": len(y_true),
    }


# =========================================================================
# Trading / Financial metrics
# =========================================================================

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                 periods_per_year: int = 252) -> float:
    """Annualized Sharpe Ratio.

    Formula: (mean(R - Rf) / std(R - Rf)) * sqrt(periods_per_year)
    Risk-free rate is expressed as per-period rate.

    Args:
        returns: array of per-period returns
        risk_free_rate: per-period risk-free rate (default 0)
        periods_per_year: annualization factor (252 daily, 8760 hourly crypto)
    """
    excess = returns - risk_free_rate
    if len(excess) < 2:
        return float("nan")
    std = np.std(excess, ddof=1)
    if std < 1e-15:
        return float("nan")
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, target_return: float = 0.0,
                  periods_per_year: int = 252) -> float:
    """Annualized Sortino Ratio.

    Formula: (mean(R) - target) / downside_deviation * sqrt(periods_per_year)
    Downside deviation uses only returns below the target.

    If zero downside deviation, returns nan (no downside risk observed).
    """
    excess = returns - target_return
    downside = np.minimum(excess, 0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd < 1e-15:
        return float("nan")
    return float(np.mean(excess) / dd * np.sqrt(periods_per_year))


def max_drawdown(cumulative_returns: np.ndarray) -> float:
    """Maximum drawdown from cumulative return series.

    cumulative_returns should be a wealth index (e.g., starting at 1.0).
    MDD = max((peak - trough) / peak)
    """
    if len(cumulative_returns) < 2:
        return float("nan")
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (peak - cumulative_returns) / np.where(peak > 0, peak, 1)
    return float(np.max(drawdowns))


def calculate_trading_metrics(returns: np.ndarray, positions: np.ndarray = None,
                               transaction_cost_bps: float = 10,
                               periods_per_year: int = 252) -> dict:
    """Calculate comprehensive trading strategy metrics.

    Args:
        returns: per-period strategy returns (after costs)
        positions: position array (for trade counting)
        transaction_cost_bps: cost in basis points per trade
        periods_per_year: for annualization
    """
    returns = np.array(returns, dtype=float)
    mask = np.isfinite(returns)
    returns = returns[mask]

    if len(returns) < 2:
        return {k: float("nan") for k in [
            "cumulative_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "annualized_volatility",
            "win_rate", "num_trades", "avg_return_per_trade",
        ]}

    # Cumulative return
    wealth = np.cumprod(1 + returns)
    cum_ret = wealth[-1] - 1

    # Annualized return
    n_periods = len(returns)
    if n_periods >= periods_per_year:
        ann_ret = (1 + cum_ret) ** (periods_per_year / n_periods) - 1
    else:
        ann_ret = cum_ret  # Too short for meaningful annualization

    # Volatility
    ann_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)

    # Win rate
    positive_returns = returns[returns > 0]
    win_rate = len(positive_returns) / len(returns) * 100 if len(returns) > 0 else 0

    # Trade count
    num_trades = 0
    if positions is not None:
        positions = np.array(positions, dtype=float)
        num_trades = int(np.sum(np.abs(np.diff(positions)) > 0))

    # Average return per trade
    active_returns = returns[returns != 0]
    avg_ret_per_trade = float(np.mean(active_returns)) if len(active_returns) > 0 else 0

    return {
        "cumulative_return": float(cum_ret),
        "annualized_return": float(ann_ret),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(wealth),
        "annualized_volatility": float(ann_vol),
        "win_rate": float(win_rate),
        "num_trades": num_trades,
        "avg_return_per_trade": float(avg_ret_per_trade),
        "n_periods": n_periods,
        "periods_per_year": periods_per_year,
    }
