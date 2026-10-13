"""Time-based feature engineering for CrowdWisdomTrading.

All features use backward-looking, point-in-time aggregation only.
No future data leakage is permitted.

Features are partitioned into strict quantitative subsets:
1. Price features only (Experiment B)
2. Primary assessment: Price + X sentiment point-in-time (Experiment C)
3. Engagement-enhanced: Price + X + Engagement-weighted sentiment (Experiment D)
4. News-enhanced (experimental extension): Price + X + News sentiment (Experiment E)
"""

import numpy as np
import pandas as pd
from src.utils.logging_config import logger


PRICE_FEATURE_COLS = [
    "close_price",
    "lagged_close_1", "lagged_close_2", "lagged_close_3", "lagged_close_5",
    "return_1", "return_2", "return_3", "return_5",
    "rolling_volatility", "rolling_mean_return", "momentum",
    "high_low_range", "open_close_return",
]

X_SENTIMENT_FEATURE_COLS = [
    "mean_sentiment", "median_sentiment",
    "positive_ratio", "negative_ratio", "neutral_ratio",
    "net_sentiment", "sentiment_std", "sentiment_change", "sentiment_momentum",
    "sentiment_lag_1", "sentiment_lag_2", "sentiment_lag_3", "sentiment_lag_5",
    "post_count", "relevant_post_count", "unique_author_count",
    "recent_sentiment_score",
    "posts_observed", "sentiment_missing",
]

ENGAGEMENT_FEATURE_COLS = [
    "engagement_weighted_sentiment",
    "likes_sum", "reposts_sum", "views_sum",
]

NEWS_FEATURE_COLS = [
    "news_mean_sentiment", "news_count", "news_positive_ratio",
]


def build_features(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    timeframe: str,
    news_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Build combined feature matrix from price, sentiment, and optional news data.

    Args:
        price_df: Normalized price bars with timestamp_utc, open, high, low, close, volume
        sentiment_df: Aggregated sentiment with period_start and metrics
        timeframe: '1d' or '1h'
        news_df: Optional news articles with published_at and sentiment_score

    Returns:
        Feature DataFrame indexed by timestamp with all features.
    """
    if price_df.empty:
        logger.warning("No price data for feature engineering")
        return pd.DataFrame()

    # Prepare price data
    pdf = price_df.copy()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True)
    pdf = pdf.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"]).reset_index(drop=True)

    # --- Price-derived features ---
    # close_price: current period close
    pdf["close_price"] = pdf["close"]

    # Lagged close prices: close at t-1, t-2, t-3, t-5
    for lag in [1, 2, 3, 5]:
        pdf[f"lagged_close_{lag}"] = pdf["close"].shift(lag)

    # Returns: log return over n periods: ln(close_t / close_{t-n})
    for n in [1, 2, 3, 5]:
        pdf[f"return_{n}"] = np.log(pdf["close"] / pdf["close"].shift(n))

    # rolling_volatility: std of 1-period log returns over 20 periods
    pdf["return_1_raw"] = np.log(pdf["close"] / pdf["close"].shift(1))
    pdf["rolling_volatility"] = pdf["return_1_raw"].rolling(20, min_periods=5).std()

    # rolling_mean_return: mean of 1-period log returns over 20 periods
    pdf["rolling_mean_return"] = pdf["return_1_raw"].rolling(20, min_periods=5).mean()

    # momentum: close_t / close_{t-10} - 1
    pdf["momentum"] = pdf["close"] / pdf["close"].shift(10) - 1

    # high_low_range: (high - low) / close
    pdf["high_low_range"] = (pdf["high"] - pdf["low"]) / pdf["close"].clip(lower=1e-8)

    # open_close_return: (close - open) / open
    pdf["open_close_return"] = (pdf["close"] - pdf["open"]) / pdf["open"].clip(lower=1e-8)

    pdf = pdf.drop(columns=["return_1_raw"], errors="ignore")

    # --- Merge sentiment features ---
    freq = "h" if timeframe == "1h" else "D"
    pdf["merge_key"] = pdf["timestamp_utc"].dt.floor(freq)

    if not sentiment_df.empty:
        sdf = sentiment_df.copy()
        sdf["period_start"] = pd.to_datetime(sdf["period_start"], utc=True)
        sdf = sdf.sort_values("period_start")
        sdf["merge_key"] = sdf["period_start"].dt.floor(freq)

        sent_cols = [
            "merge_key", "mean_sentiment", "median_sentiment",
            "positive_ratio", "negative_ratio", "neutral_ratio",
            "net_sentiment", "sentiment_std", "post_count",
            "relevant_post_count", "unique_author_count",
            "likes_sum", "reposts_sum", "views_sum",
            "engagement_weighted_sentiment",
        ]
        available_cols = [c for c in sent_cols if c in sdf.columns]
        sdf = sdf[available_cols].drop_duplicates(subset=["merge_key"])

        pdf = pdf.merge(sdf, on="merge_key", how="left")

        # Sentiment-derived dynamics
        pdf["sentiment_change"] = pdf["mean_sentiment"].diff()
        pdf["sentiment_momentum"] = pdf["mean_sentiment"] - pdf["mean_sentiment"].shift(5)

        for lag in [1, 2, 3, 5]:
            pdf[f"sentiment_lag_{lag}"] = pdf["mean_sentiment"].shift(lag)

        pdf["recent_sentiment_score"] = pdf["mean_sentiment"].ewm(halflife=3, min_periods=1).mean()
        pdf["posts_observed"] = pdf["post_count"].notna().astype(int)
        pdf["sentiment_missing"] = pdf["mean_sentiment"].isna().astype(int)

    else:
        # Explicit missing flags
        for col in [
            "mean_sentiment", "median_sentiment", "positive_ratio",
            "negative_ratio", "neutral_ratio", "net_sentiment",
            "sentiment_std", "post_count", "relevant_post_count",
            "unique_author_count", "engagement_weighted_sentiment",
            "sentiment_change", "sentiment_momentum",
            "sentiment_lag_1", "sentiment_lag_2", "sentiment_lag_3",
            "sentiment_lag_5", "recent_sentiment_score",
            "likes_sum", "reposts_sum", "views_sum",
        ]:
            pdf[col] = np.nan
        pdf["posts_observed"] = 0
        pdf["sentiment_missing"] = 1

    # --- Merge optional news features ---
    if news_df is not None and not news_df.empty and "published_at" in news_df.columns:
        ndf = news_df.copy()
        ndf["pub_ts"] = pd.to_datetime(ndf["published_at"], errors="coerce", utc=True)
        ndf = ndf.dropna(subset=["pub_ts"]).sort_values("pub_ts")
        ndf["merge_key"] = ndf["pub_ts"].dt.floor(freq)

        # Aggregate news point-in-time
        news_grp = ndf.groupby("merge_key").agg(
            news_mean_sentiment=("sentiment_score", "mean"),
            news_count=("sentiment_score", "count"),
            news_positive_ratio=("sentiment_label", lambda s: float((s == "positive").mean())),
        ).reset_index()

        pdf = pdf.merge(news_grp, on="merge_key", how="left")
        pdf["news_mean_sentiment"] = pdf["news_mean_sentiment"].fillna(0.0)
        pdf["news_count"] = pdf["news_count"].fillna(0)
        pdf["news_positive_ratio"] = pdf["news_positive_ratio"].fillna(0.0)
    else:
        pdf["news_mean_sentiment"] = np.nan
        pdf["news_count"] = 0
        pdf["news_positive_ratio"] = np.nan

    pdf = pdf.drop(columns=["merge_key"], errors="ignore")
    pdf["feature_timestamp"] = pdf["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info("Built %d feature rows with %d columns", len(pdf), len(pdf.columns))
    return pdf


def get_feature_columns(mode: str = "all") -> list[str]:
    """Return the list of feature column names for a specific experiment mode.

    Args:
        mode:
          - 'price_only' (Experiment B)
          - 'price_sentiment' / 'primary' (Experiment C)
          - 'engagement_enhanced' (Experiment D)
          - 'news_enhanced' (Experiment E)
          - 'all' (Full backward-compatible feature list)
    """
    if mode == "price_only":
        return list(PRICE_FEATURE_COLS)
    elif mode in ("price_sentiment", "primary"):
        return list(PRICE_FEATURE_COLS + X_SENTIMENT_FEATURE_COLS)
    elif mode == "engagement_enhanced":
        return list(PRICE_FEATURE_COLS + X_SENTIMENT_FEATURE_COLS + ENGAGEMENT_FEATURE_COLS)
    elif mode == "news_enhanced":
        return list(PRICE_FEATURE_COLS + X_SENTIMENT_FEATURE_COLS + NEWS_FEATURE_COLS)
    else:
        # Default 'all' preserves backward compatibility
        return list(
            PRICE_FEATURE_COLS
            + X_SENTIMENT_FEATURE_COLS
            + ENGAGEMENT_FEATURE_COLS
        )


def create_targets(feature_df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    """Create forward-looking target returns for each horizon.

    Target for horizon h:
      target_return_h = ln(close_{t+h} / close_t)

    This is future information used ONLY as the training target,
    never as an input feature.
    """
    if horizons is None:
        horizons = [0, 1, 2, 3, 4, 5]

    df = feature_df.copy()

    for h in horizons:
        if h == 0:
            df["target_return_0"] = df["open_close_return"]
        else:
            df[f"target_return_{h}"] = np.log(
                df["close"].shift(-h) / df["close"]
            )

    return df


def check_feature_leakage(features_df: pd.DataFrame, target_col: str) -> bool:
    """Check that target column is not correlated > 0.99 with any feature."""
    if target_col not in features_df.columns:
        return False

    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in features_df.columns]

    for col in available:
        if col == target_col:
            continue
        corr = features_df[[col, target_col]].dropna().corr().iloc[0, 1]
        if abs(corr) > 0.99:
            logger.warning("Potential leakage: %s corr with %s = %.4f", col, target_col, corr)
            return True
    return False
