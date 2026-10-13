"""General-purpose historical X dataset importer and date-coverage validator.

Supports:
- CSV imports with flexible column mappings (manual or auto-detected).
- HuggingFace dataset integration (StephanAkkerman/financial-tweets).
- Strict UTC timestamp normalization.
- Deduplication by post ID or content hash.
- Honest missing engagement tracking (no fabricated likes/views).
- Rigorous date coverage and overlap diagnostics against price histories.
"""

import io
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.utils.logging_config import logger
from src.utils.time_utils import format_utc


# Standard schema for raw posts
POST_COLUMNS = [
    "post_id", "asset", "timestamp_utc", "post_text", "author_id",
    "author_username", "author_followers", "likes", "reposts",
    "replies", "views", "post_url", "search_query",
    "collection_timestamp_utc", "source", "raw_json",
]

DEFAULT_COLUMN_CANDIDATES = {
    "timestamp_utc": [
        "timestamp_utc", "timestamp", "timestap", "created_at", "date",
        "datetime", "time", "published_at", "Date", "Datetime",
    ],
    "post_text": [
        "post_text", "tweet_text", "text", "description", "content",
        "body", "message", "full_text",
    ],
    "post_id": [
        "post_id", "id", "tweet_id", "status_id", "url", "tweet_url",
    ],
    "author_username": [
        "author_username", "author", "user", "username", "screen_name",
        "author_name", "embed_title",
    ],
    "likes": ["likes", "like_count", "favorite_count", "favorites"],
    "reposts": ["reposts", "retweet_count", "retweets", "shares"],
    "views": ["views", "view_count", "impressions"],
    "replies": ["replies", "reply_count"],
}


def auto_detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Auto-detect column mappings for a given DataFrame based on common names."""
    mapping = {}
    cols = list(df.columns)
    cols_lower = {str(c).lower().strip(): c for c in cols}

    for target_col, candidates in DEFAULT_COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand in cols:
                mapping[target_col] = cand
                break
            elif cand.lower() in cols_lower:
                mapping[target_col] = cols_lower[cand.lower()]
                break

    return mapping


def extract_id_from_url(url: str) -> Optional[str]:
    """Extract numeric status ID from a Twitter/X URL if present."""
    if not isinstance(url, str):
        return None
    match = re.search(r"/status(?:es)?/(\d+)", url)
    if match:
        return match.group(1)
    return None


def import_historical_x_csv(
    file_path_or_buffer: Union[str, Path, io.StringIO, bytes],
    asset: str,
    column_mapping: Optional[Dict[str, str]] = None,
    source_label: str = "csv_historical",
    filter_asset: bool = True,
) -> pd.DataFrame:
    """Import historical X posts from a CSV file or buffer.

    Args:
        file_path_or_buffer: path, buffer, or string containing CSV data.
        asset: canonical asset ticker (e.g., BTCUSDT, NVDA).
        column_mapping: optional dictionary mapping standard columns to CSV columns.
        source_label: description/provenance of the dataset.
        filter_asset: if True and asset column is present or filters needed, filter to asset.

    Returns:
        DataFrame conforming to standard raw_x_posts schema.
    """
    try:
        if isinstance(file_path_or_buffer, (str, Path)) and "\n" not in str(file_path_or_buffer):
            df = pd.read_csv(file_path_or_buffer, low_memory=False)
        elif isinstance(file_path_or_buffer, bytes):
            df = pd.read_csv(io.BytesIO(file_path_or_buffer), low_memory=False)
        else:
            df = pd.read_csv(io.StringIO(str(file_path_or_buffer)), low_memory=False)
    except Exception as e:
        logger.error("Failed to read CSV for historical X posts: %s", e)
        return pd.DataFrame(columns=POST_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=POST_COLUMNS)

    # Resolve column mappings
    detected = auto_detect_columns(df)
    final_mapping = {**detected, **(column_mapping or {})}

    # Rename mapped columns
    reverse_map = {v: k for k, v in final_mapping.items() if v in df.columns}
    df = df.rename(columns=reverse_map).copy()

    # Ensure required fields
    if "post_text" not in df.columns:
        logger.error("CSV lacks required post text column. Mappings: %s", final_mapping)
        return pd.DataFrame(columns=POST_COLUMNS)

    # Filter out empty text
    df = df.dropna(subset=["post_text"]).copy()
    df["post_text"] = df["post_text"].astype(str).str.strip()
    df = df[df["post_text"].str.len() > 0]

    # Timestamp normalization
    if "timestamp_utc" in df.columns:
        df["ts_raw"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True, format="mixed")
        df = df.dropna(subset=["ts_raw"]).copy()
        df["timestamp_utc"] = df["ts_raw"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df.drop(columns=["ts_raw"])
    else:
        logger.error("CSV lacks timestamp column")
        return pd.DataFrame(columns=POST_COLUMNS)

    # Asset identification
    df["asset"] = asset

    # Post ID normalization
    if "post_id" in df.columns:
        # Check if post_id is URL-like and extract status id
        def _clean_id(val):
            if pd.isna(val) or val is None:
                return None
            s = str(val).strip()
            if "status/" in s or "twitter.com" in s or "x.com" in s:
                extracted = extract_id_from_url(s)
                if extracted:
                    return extracted
            # Remove trailing .0 from float conversion
            if s.endswith(".0"):
                s = s[:-2]
            return s if s else None

        df["post_id"] = df["post_id"].apply(_clean_id)
    else:
        df["post_id"] = None

    # For rows where post_id is missing, synthesize deterministic hash ID based on timestamp and text
    missing_id_mask = df["post_id"].isna() | (df["post_id"] == "")
    if missing_id_mask.any():
        df.loc[missing_id_mask, "post_id"] = df.loc[missing_id_mask].apply(
            lambda r: f"hist_{hashlib.sha256((str(r['timestamp_utc']) + str(r['post_text'])).encode('utf-8')).hexdigest()[:16]}",
            axis=1,
        )

    # Engagement fields: strictly preserve missing as NaN / None
    for eng_col in ["likes", "reposts", "replies", "views", "author_followers"]:
        if eng_col in df.columns:
            df[eng_col] = pd.to_numeric(df[eng_col], errors="coerce")
        else:
            df[eng_col] = np.nan

    # Author identifiers
    if "author_username" not in df.columns:
        df["author_username"] = None
    if "author_id" not in df.columns:
        df["author_id"] = None
    if "post_url" not in df.columns:
        df["post_url"] = None
    if "search_query" not in df.columns:
        df["search_query"] = asset

    df["collection_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    df["source"] = source_label
    df["raw_json"] = None

    # Deduplicate by post_id and asset
    initial_count = len(df)
    df = df.drop_duplicates(subset=["post_id", "asset"]).reset_index(drop=True)
    dedup_count = len(df)
    if initial_count > dedup_count:
        logger.info("Deduplicated %d duplicate posts in CSV import", initial_count - dedup_count)

    # Sort chronologically
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Attach provenance attributes
    has_engagement = df["likes"].notna().any() or df["reposts"].notna().any()
    df.attrs["source"] = source_label
    df.attrs["count"] = len(df)
    df.attrs["has_engagement"] = bool(has_engagement)
    df.attrs["start_timestamp"] = df["timestamp_utc"].min() if not df.empty else None
    df.attrs["end_timestamp"] = df["timestamp_utc"].max() if not df.empty else None

    # Ensure all columns present
    for col in POST_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[POST_COLUMNS]


def extract_huggingface_posts(
    parquet_path: Union[str, Path],
    asset: str,
    search_terms: Optional[list] = None,
    max_posts: Optional[int] = None,
) -> pd.DataFrame:
    """Extract and normalize asset-specific posts from the StephanAkkerman/financial-tweets dataset.

    Args:
        parquet_path: path to the local parquet file.
        asset: canonical ticker (e.g. BTCUSDT, NVDA).
        search_terms: list of search keywords/regex patterns.
        max_posts: optional limit on extracted posts.
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        logger.error("HuggingFace parquet file does not exist at %s", parquet_path)
        return pd.DataFrame(columns=POST_COLUMNS)

    try:
        import pyarrow.parquet as pq
        table = pq.read_table(
            str(parquet_path),
            columns=["timestamp", "description", "url", "tweet_type", "financial_info", "sentiment"],
        )
        raw_df = table.to_pandas()
    except Exception as e:
        logger.error("Failed to read parquet %s: %s", parquet_path, e)
        return pd.DataFrame(columns=POST_COLUMNS)

    # Determine regex patterns for the asset
    if not search_terms:
        if "BTC" in asset.upper():
            search_terms = [r"\$BTC\b", r"\bBitcoin\b", r"#BTC\b"]
        elif "NVDA" in asset.upper():
            search_terms = [r"\$NVDA\b", r"\bNvidia\b", r"#NVDA\b"]
        elif "ETH" in asset.upper():
            search_terms = [r"\$ETH\b", r"\bEthereum\b", r"#ETH\b"]
        else:
            search_terms = [asset]

    pattern = "|".join(search_terms)
    mask = (
        raw_df["description"].astype(str).str.contains(pattern, case=False, na=False)
        | raw_df["financial_info"].astype(str).str.contains(pattern, case=False, na=False)
    )

    filtered = raw_df[mask].copy()
    if max_posts and len(filtered) > max_posts:
        filtered = filtered.tail(max_posts)

    # Map fields
    col_map = {
        "timestamp_utc": "timestamp",
        "post_text": "description",
        "post_url": "url",
    }
    posts_df = import_historical_x_csv(
        filtered.to_csv(index=False),
        asset=asset,
        column_mapping=col_map,
        source_label="huggingface_stephanakkerman",
    )
    return posts_df


def validate_date_coverage(
    price_df: pd.DataFrame,
    posts_df: pd.DataFrame,
    requested_start: Optional[str] = None,
    requested_end: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate and compare timestamp coverage between price history and X posts.

    Returns diagnostic metrics:
      - Requested date range
      - Actual tweet date range
      - Actual price date range
      - Earliest and latest overlapping timestamps
      - Number of relevant tweets
      - Number of price observations
      - Number of common aligned observations
      - Status and detailed warning message
    """
    diagnostics = {
        "requested_date_range": f"{requested_start} to {requested_end}" if requested_start and requested_end else "N/A",
        "actual_tweet_date_range": "N/A (0 tweets)",
        "actual_price_date_range": "N/A (0 prices)",
        "earliest_overlap": None,
        "latest_overlap": None,
        "n_tweets": len(posts_df) if posts_df is not None else 0,
        "n_prices": len(price_df) if price_df is not None else 0,
        "n_aligned_observations": 0,
        "has_adequate_overlap": False,
        "warning": None,
    }

    if price_df is None or price_df.empty:
        diagnostics["warning"] = "Price history is completely empty. Cannot evaluate overlap."
        return diagnostics

    p_min = pd.to_datetime(price_df["timestamp_utc"], utc=True, format="mixed").min()
    p_max = pd.to_datetime(price_df["timestamp_utc"], utc=True, format="mixed").max()
    diagnostics["actual_price_date_range"] = f"{p_min.strftime('%Y-%m-%d')} to {p_max.strftime('%Y-%m-%d')}"

    if posts_df is None or posts_df.empty:
        diagnostics["warning"] = (
            "Zero genuine X observations available. Cannot calculate sentiment or compare models. "
            "Supply a historical CSV via the importer to run sentiment experiments."
        )
        return diagnostics

    t_min = pd.to_datetime(posts_df["timestamp_utc"], utc=True, format="mixed").min()
    t_max = pd.to_datetime(posts_df["timestamp_utc"], utc=True, format="mixed").max()
    diagnostics["actual_tweet_date_range"] = f"{t_min.strftime('%Y-%m-%d')} to {t_max.strftime('%Y-%m-%d')}"

    # Overlap window
    overlap_start = max(p_min, t_min)
    overlap_end = min(p_max, t_max)

    if overlap_start > overlap_end:
        diagnostics["warning"] = (
            f"No temporal overlap between prices ({diagnostics['actual_price_date_range']}) "
            f"and tweets ({diagnostics['actual_tweet_date_range']}). "
            "Do NOT merge unrelated historical periods. Please supply a CSV matching the price period "
            "or fetch prices covering the historical dataset period."
        )
        return diagnostics

    diagnostics["earliest_overlap"] = overlap_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    diagnostics["latest_overlap"] = overlap_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Count price bars within overlap
    p_ts = pd.to_datetime(price_df["timestamp_utc"], utc=True)
    aligned_mask = (p_ts >= overlap_start) & (p_ts <= overlap_end)
    aligned_count = int(aligned_mask.sum())
    diagnostics["n_aligned_observations"] = aligned_count

    # Adequate overlap rule: at least 30 observations to support walk-forward evaluation
    if aligned_count < 30:
        diagnostics["has_adequate_overlap"] = False
        diagnostics["warning"] = (
            f"Insufficient overlapping observations ({aligned_count} bars). "
            f"At least 30 aligned observations are required for valid cross-validation and backtesting."
        )
    else:
        diagnostics["has_adequate_overlap"] = True

    return diagnostics
