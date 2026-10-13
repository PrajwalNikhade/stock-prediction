"""Data cleaning utilities for CrowdWisdomTrading."""

import re
import pandas as pd
from src.utils.logging_config import logger
from src.utils.time_utils import parse_timestamp, format_utc


def clean_post_text(text: str) -> str:
    """Normalize post text for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove @mentions but keep text
    text = re.sub(r"@\w+", "", text)
    # Keep $ cashtags
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate_posts(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Deduplicate posts by post_id. Returns (deduped_df, duplicate_count)."""
    if df.empty or "post_id" not in df.columns:
        return df, 0
    original_len = len(df)
    df = df.drop_duplicates(subset=["post_id"], keep="first")
    dup_count = original_len - len(df)
    if dup_count > 0:
        logger.info("Removed %d duplicate posts (%.1f%%)", dup_count, 100 * dup_count / original_len)
    return df.reset_index(drop=True), dup_count


def validate_timestamps(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    """Validate and filter rows with parseable timestamps."""
    if df.empty or ts_col not in df.columns:
        return df
    valid_mask = df[ts_col].notna() & (df[ts_col] != "")
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning("Removed %d rows with invalid timestamps", invalid_count)
    return df[valid_mask].reset_index(drop=True)


def filter_empty_text(df: pd.DataFrame, text_col: str = "post_text") -> pd.DataFrame:
    """Remove posts with empty text."""
    if df.empty or text_col not in df.columns:
        return df
    mask = df[text_col].fillna("").str.strip().str.len() > 0
    removed = (~mask).sum()
    if removed > 0:
        logger.info("Removed %d posts with empty text", removed)
    return df[mask].reset_index(drop=True)


def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure price data has correct types and no NaN close prices."""
    if df.empty:
        return df
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def data_quality_report(prices_df: pd.DataFrame, posts_df: pd.DataFrame) -> dict:
    """Generate a data quality summary."""
    report = {
        "price_rows": len(prices_df),
        "price_missing_close": int(prices_df["close"].isna().sum()) if "close" in prices_df.columns else 0,
        "price_date_range": "",
        "post_rows": len(posts_df),
        "post_missing_text": 0,
        "post_missing_timestamp": 0,
        "post_date_range": "",
    }
    if not prices_df.empty and "timestamp_utc" in prices_df.columns:
        report["price_date_range"] = f"{prices_df['timestamp_utc'].min()} to {prices_df['timestamp_utc'].max()}"
    if not posts_df.empty:
        if "post_text" in posts_df.columns:
            report["post_missing_text"] = int(posts_df["post_text"].fillna("").str.strip().str.len().eq(0).sum())
        if "timestamp_utc" in posts_df.columns:
            report["post_missing_timestamp"] = int(posts_df["timestamp_utc"].isna().sum())
            valid_ts = posts_df["timestamp_utc"].dropna()
            if len(valid_ts) > 0:
                report["post_date_range"] = f"{valid_ts.min()} to {valid_ts.max()}"
    return report
