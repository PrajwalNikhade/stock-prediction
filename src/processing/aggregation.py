"""Sentiment aggregation to hourly/daily periods for CrowdWisdomTrading.

Aggregation convention:
  - Daily: UTC day boundary (00:00 to 23:59 UTC)
  - Hourly: UTC hour boundary

Missing periods with no posts are distinguished from collection failures
via the posts_observed flag.
"""

import numpy as np
import pandas as pd
from src.utils.logging_config import logger


def aggregate_sentiment(posts_df: pd.DataFrame, timeframe: str,
                        asset: str) -> pd.DataFrame:
    """Aggregate processed post sentiment to time periods.

    Args:
        posts_df: DataFrame with processed posts (must have timestamp_utc,
                  final_sentiment_score, sentiment_label, engagement_weight)
        timeframe: '1h' or '1d'
        asset: asset ticker

    Returns:
        DataFrame with one row per period.
    """
    if posts_df.empty:
        return pd.DataFrame()

    df = posts_df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.dropna(subset=["timestamp_utc"])

    if df.empty:
        return pd.DataFrame()

    # Floor to period
    freq = "h" if timeframe == "1h" else "D"
    df["period_start"] = df["timestamp_utc"].dt.floor(freq)

    # Aggregate per period
    agg_kwargs = {
        "mean_sentiment": ("final_sentiment_score", "mean"),
        "median_sentiment": ("final_sentiment_score", "median"),
        "sentiment_std": ("final_sentiment_score", "std"),
        "post_count": ("final_sentiment_score", "count"),
    }
    if "is_relevant" in df.columns:
        agg_kwargs["relevant_post_count"] = ("is_relevant", "sum")
    else:
        agg_kwargs["relevant_post_count"] = ("final_sentiment_score", "count")

    if "author_username" in df.columns:
        agg_kwargs["unique_author_count"] = ("author_username", "nunique")
    else:
        agg_kwargs["unique_author_count"] = ("final_sentiment_score", "count")

    if "likes" in df.columns:
        agg_kwargs["likes_sum"] = ("likes", lambda x: x.dropna().sum() if x.notna().any() else 0.0)
    if "reposts" in df.columns:
        agg_kwargs["reposts_sum"] = ("reposts", lambda x: x.dropna().sum() if x.notna().any() else 0.0)
    if "views" in df.columns:
        agg_kwargs["views_sum"] = ("views", lambda x: x.dropna().sum() if x.notna().any() else 0.0)

    aggs = df.groupby("period_start").agg(**agg_kwargs).reset_index()

    for col in ["likes_sum", "reposts_sum", "views_sum"]:
        if col not in aggs.columns:
            aggs[col] = 0.0

    # Calculate ratios
    for period_start in aggs["period_start"].unique():
        mask = aggs["period_start"] == period_start
        period_posts = df[df["period_start"] == period_start]
        total = len(period_posts)
        if total > 0:
            pos_count = (period_posts["sentiment_label"] == "positive").sum()
            neg_count = (period_posts["sentiment_label"] == "negative").sum()
            neu_count = (period_posts["sentiment_label"] == "neutral").sum()
            aggs.loc[mask, "positive_ratio"] = pos_count / total
            aggs.loc[mask, "negative_ratio"] = neg_count / total
            aggs.loc[mask, "neutral_ratio"] = neu_count / total
            aggs.loc[mask, "net_sentiment"] = (pos_count - neg_count) / total
        else:
            aggs.loc[mask, "positive_ratio"] = 0
            aggs.loc[mask, "negative_ratio"] = 0
            aggs.loc[mask, "neutral_ratio"] = 0
            aggs.loc[mask, "net_sentiment"] = 0

    # Engagement-weighted sentiment
    for period_start in aggs["period_start"].unique():
        mask = aggs["period_start"] == period_start
        period_posts = df[df["period_start"] == period_start]
        weights = period_posts["engagement_weight"].fillna(1.0).values
        scores = period_posts["final_sentiment_score"].fillna(0).values
        total_weight = weights.sum()
        if total_weight > 0:
            ews = np.sum(scores * weights) / total_weight
        else:
            ews = 0.0
        aggs.loc[mask, "engagement_weighted_sentiment"] = ews

    # Author count fix if column not available
    if "author_username" not in df.columns:
        aggs["unique_author_count"] = aggs["post_count"]

    aggs["asset"] = asset
    aggs["timeframe"] = timeframe
    aggs["period_start"] = aggs["period_start"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    logger.info(
        "Aggregated sentiment: %d periods for %s (%s)",
        len(aggs), asset, timeframe,
    )
    return aggs
