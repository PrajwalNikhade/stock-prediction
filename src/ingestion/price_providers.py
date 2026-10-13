"""Price data providers for CrowdWisdomTrading.

Adapters: YahooFinance, Binance public API, CSV import.
All normalize to a common schema:
  timestamp_utc, open, high, low, close, volume, asset, timeframe, source
"""

import io
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import requests
from src.utils.logging_config import logger
from src.utils.time_utils import utc_now, format_utc

# ---------------------------------------------------------------------------
# Common schema
# ---------------------------------------------------------------------------
PRICE_COLUMNS = [
    "timestamp_utc", "open", "high", "low", "close", "volume",
    "asset", "timeframe", "source",
]


def _ensure_schema(df: pd.DataFrame, asset: str, timeframe: str, source: str) -> pd.DataFrame:
    """Normalize a price DataFrame to the standard schema."""
    df = df.copy()
    df["asset"] = asset
    df["timeframe"] = timeframe
    df["source"] = source
    for col in PRICE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    # Ensure timestamp is string in ISO format
    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    df = df[PRICE_COLUMNS].copy()
    df = df.dropna(subset=["close"])
    df = df.drop_duplicates(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df.attrs["source"] = source
    df.attrs["count"] = len(df)
    df.attrs["start_timestamp"] = df["timestamp_utc"].min() if not df.empty else None
    df.attrs["end_timestamp"] = df["timestamp_utc"].max() if not df.empty else None
    return df


# ---------------------------------------------------------------------------
# Yahoo Finance provider
# ---------------------------------------------------------------------------
def fetch_yfinance(ticker: str, asset_ticker: str, timeframe: str,
                   days: int = 180) -> pd.DataFrame:
    """Fetch historical data via yfinance.

    Args:
        ticker: yfinance ticker symbol (e.g. BTC-USD, NVDA)
        asset_ticker: canonical asset ticker for our system
        timeframe: '1h' or '1d'
        days: lookback days
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed")
        return pd.DataFrame(columns=PRICE_COLUMNS)

    period_map = {"1d": f"{days}d", "1h": "730d"}
    interval_map = {"1d": "1d", "1h": "1h"}

    interval = interval_map.get(timeframe, "1d")
    # yfinance hourly data limited to ~730 days for some tickers
    if timeframe == "1h":
        # yfinance only provides ~730 days of hourly data for some assets
        period = f"{min(days, 729)}d"
    else:
        period = f"{days}d"

    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval)
        if df.empty:
            logger.warning("yfinance returned empty data for %s", ticker)
            return pd.DataFrame(columns=PRICE_COLUMNS)

        df = df.reset_index()
        # Column mapping
        rename_map = {}
        for col in df.columns:
            cl = col.lower() if isinstance(col, str) else str(col).lower()
            if cl in ("date", "datetime", "index"):
                rename_map[col] = "timestamp_utc"
            elif cl == "open":
                rename_map[col] = "open"
            elif cl == "high":
                rename_map[col] = "high"
            elif cl == "low":
                rename_map[col] = "low"
            elif cl == "close":
                rename_map[col] = "close"
            elif cl == "volume":
                rename_map[col] = "volume"
        df = df.rename(columns=rename_map)

        if "timestamp_utc" not in df.columns:
            # Try first column as timestamp
            df = df.rename(columns={df.columns[0]: "timestamp_utc"})

        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        result = _ensure_schema(df, asset_ticker, timeframe, "yfinance")
        logger.info(
            "yfinance: fetched %d bars for %s (%s)", len(result), asset_ticker, timeframe
        )
        return result
    except Exception as e:
        logger.error("yfinance fetch error for %s: %s", ticker, e)
        return pd.DataFrame(columns=PRICE_COLUMNS)


# ---------------------------------------------------------------------------
# Binance public API provider
# ---------------------------------------------------------------------------
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

BINANCE_INTERVALS = {"1h": "1h", "1d": "1d"}


def fetch_binance(symbol: str, asset_ticker: str, timeframe: str,
                  days: int = 180) -> pd.DataFrame:
    """Fetch historical candlestick data from Binance public API.

    Supports pagination for large requests.
    """
    interval = BINANCE_INTERVALS.get(timeframe, "1d")
    end_ms = int(utc_now().timestamp() * 1000)
    start_ms = int((utc_now() - timedelta(days=days)).timestamp() * 1000)
    limit = 1000  # Binance max per request

    all_rows = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": limit,
        }
        try:
            resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_rows.extend(data)
            # Move start to after last candle
            last_close_time = data[-1][6]  # close time ms
            current_start = last_close_time + 1
            if len(data) < limit:
                break
        except requests.RequestException as e:
            logger.error("Binance API error for %s: %s", symbol, e)
            break

    if not all_rows:
        logger.warning("Binance returned no data for %s", symbol)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["timestamp_utc"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = _ensure_schema(df, asset_ticker, timeframe, "binance")
    logger.info("Binance: fetched %d bars for %s (%s)", len(result), asset_ticker, timeframe)
    return result


# ---------------------------------------------------------------------------
# CSV import provider
# ---------------------------------------------------------------------------
def import_csv(file_path_or_buffer, asset_ticker: str, timeframe: str) -> pd.DataFrame:
    """Import price data from a CSV file.

    Expected columns: timestamp (or date/datetime), open, high, low, close, volume (optional)
    """
    try:
        if isinstance(file_path_or_buffer, (str, bytes)):
            df = pd.read_csv(io.StringIO(file_path_or_buffer) if isinstance(file_path_or_buffer, str) and "\n" in file_path_or_buffer else file_path_or_buffer)
        else:
            df = pd.read_csv(file_path_or_buffer)
    except Exception as e:
        logger.error("CSV import error: %s", e)
        return pd.DataFrame(columns=PRICE_COLUMNS)

    # Try to find timestamp column
    ts_col = None
    for candidate in ["timestamp", "timestamp_utc", "date", "datetime", "Date", "Datetime", "time"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None and len(df.columns) > 0:
        ts_col = df.columns[0]

    if ts_col:
        df = df.rename(columns={ts_col: "timestamp_utc"})
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("open", "high", "low", "close", "volume"):
            col_map[c] = cl
    df = df.rename(columns=col_map)

    result = _ensure_schema(df, asset_ticker, timeframe, "csv")
    logger.info("CSV: imported %d bars for %s (%s)", len(result), asset_ticker, timeframe)
    return result


# ---------------------------------------------------------------------------
# Unified fetch
# ---------------------------------------------------------------------------
def fetch_price_data(asset_config: dict, timeframe: str,
                     days: int = 180) -> pd.DataFrame:
    """Fetch price data using the appropriate provider for the asset."""
    provider = asset_config.get("provider", "yfinance")
    ticker = asset_config["ticker"]
    yf_ticker = asset_config.get("yfinance_ticker", ticker)

    if provider == "binance":
        df = fetch_binance(ticker, ticker, timeframe, days)
        if df.empty:
            # Fallback to yfinance
            logger.info("Binance empty, falling back to yfinance for %s", ticker)
            df = fetch_yfinance(yf_ticker, ticker, timeframe, days)
    else:
        df = fetch_yfinance(yf_ticker, ticker, timeframe, days)

    return df


def get_price_coverage(df: pd.DataFrame) -> dict:
    """Extract source, timestamp range, and count metadata from a price DataFrame."""
    if df is None or df.empty:
        return {
            "source": "none",
            "count": 0,
            "date_range": "N/A",
            "start": None,
            "end": None,
        }
    source = df["source"].iloc[0] if "source" in df.columns else df.attrs.get("source", "unknown")
    start = str(df["timestamp_utc"].min())
    end = str(df["timestamp_utc"].max())
    return {
        "source": str(source),
        "count": len(df),
        "date_range": f"{start} to {end}",
        "start": start,
        "end": end,
    }


def test_price_connection(asset_ticker: str = "BTCUSDT", timeframe: str = "1d", days: int = 5) -> dict:
    """Verify live connectivity for price provider without requiring credentials."""
    if asset_ticker.upper().endswith("USDT"):
        df = fetch_binance(asset_ticker, asset_ticker, timeframe, days=days)
        provider_name = "binance"
    else:
        df = fetch_yfinance(asset_ticker, asset_ticker, timeframe, days=days)
        provider_name = "yfinance"

    if not df.empty:
        cov = get_price_coverage(df)
        return {
            "success": True,
            "provider": provider_name,
            "count": cov["count"],
            "date_range": cov["date_range"],
            "latest_close": float(df["close"].iloc[-1]),
            "error": None,
        }
    else:
        return {
            "success": False,
            "provider": provider_name,
            "count": 0,
            "date_range": "N/A",
            "latest_close": None,
            "error": f"Failed to retrieve data from {provider_name}",
        }
