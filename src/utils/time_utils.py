"""Time utility functions for CrowdWisdomTrading."""

from datetime import datetime, timezone, timedelta
import pandas as pd


def utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC. If naive, assume UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_timestamp(ts) -> datetime | None:
    """Parse various timestamp formats to UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return to_utc(ts)
    if isinstance(ts, (int, float)):
        # Unix timestamp (seconds or milliseconds)
        if ts > 1e12:
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        ts = ts.strip()
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%a %b %d %H:%M:%S %z %Y",
            "%a %b %d %H:%M:%S %Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                dt = datetime.strptime(ts, fmt)
                return to_utc(dt)
            except ValueError:
                continue
        # Try pandas as fallback
        try:
            return to_utc(pd.Timestamp(ts).to_pydatetime())
        except Exception:
            return None
    return None


def format_utc(dt: datetime) -> str:
    """Format datetime to ISO-8601 UTC string."""
    if dt is None:
        return ""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def days_ago(n: int) -> datetime:
    """Return UTC datetime n days ago."""
    return utc_now() - timedelta(days=n)
