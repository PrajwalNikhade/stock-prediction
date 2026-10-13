"""Hybrid sentiment analysis engine for CrowdWisdomTrading.

Combines VADER (always available) with an optional transformer-based
financial sentiment model (e.g., FinBERT).

Sentiment formula (hybrid mode):
  final_score = vader_weight * vader_compound + transformer_weight * transformer_score
  Default: vader_weight=0.4, transformer_weight=0.6

If transformer is unavailable, uses VADER only and labels method as 'vader'.

Engagement-weighted sentiment:
  weight = log(1 + likes) + 0.5 * log(1 + reposts) + 0.3 * log(1 + views/1000) + 0.2 * log(1 + followers/1000)
  Capped at max_engagement_weight (default 10) to prevent viral-post dominance.
"""

import math
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from src.utils.logging_config import logger
from src.utils.time_utils import format_utc, utc_now
from src.processing.data_cleaning import clean_post_text

# VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    _vader = None
    logger.warning("VADER not available")

# Transformer (optional)
TRANSFORMER_AVAILABLE = False
_transformer_pipeline = None


def _init_transformer():
    """Try to load a financial sentiment transformer model."""
    global TRANSFORMER_AVAILABLE, _transformer_pipeline
    try:
        from transformers import pipeline
        _transformer_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            truncation=True,
            max_length=512,
        )
        TRANSFORMER_AVAILABLE = True
        logger.info("FinBERT transformer loaded successfully")
    except Exception as e:
        TRANSFORMER_AVAILABLE = False
        logger.info("Transformer model not available (will use VADER only): %s", e)


def analyze_sentiment(text: str, use_transformer: bool = True) -> dict:
    """Analyze sentiment of a single text.

    Returns dict with vader scores, optional transformer scores,
    final score, label, and method used.
    """
    cleaned = clean_post_text(text)
    result = {
        "vader_compound": None,
        "vader_pos": None,
        "vader_neg": None,
        "vader_neu": None,
        "transformer_positive": None,
        "transformer_negative": None,
        "transformer_neutral": None,
        "final_sentiment_score": 0.0,
        "sentiment_label": "neutral",
        "processing_method": "none",
    }

    if not cleaned:
        return result

    # VADER
    if VADER_AVAILABLE and _vader is not None:
        vs = _vader.polarity_scores(cleaned)
        result["vader_compound"] = vs["compound"]
        result["vader_pos"] = vs["pos"]
        result["vader_neg"] = vs["neg"]
        result["vader_neu"] = vs["neu"]

    # Transformer
    if use_transformer and TRANSFORMER_AVAILABLE and _transformer_pipeline is not None:
        try:
            outputs = _transformer_pipeline(cleaned[:512])
            if outputs and isinstance(outputs[0], list):
                for item in outputs[0]:
                    label = item["label"].lower()
                    score = item["score"]
                    if label == "positive":
                        result["transformer_positive"] = score
                    elif label == "negative":
                        result["transformer_negative"] = score
                    elif label == "neutral":
                        result["transformer_neutral"] = score
        except Exception as e:
            logger.debug("Transformer error: %s", e)

    # Combine scores
    has_vader = result["vader_compound"] is not None
    has_transformer = result["transformer_positive"] is not None

    if has_vader and has_transformer:
        # Hybrid: normalize transformer to [-1, 1] range
        t_pos = result["transformer_positive"] or 0
        t_neg = result["transformer_negative"] or 0
        transformer_score = t_pos - t_neg  # Range [-1, 1]
        vader_score = result["vader_compound"]  # Range [-1, 1]
        result["final_sentiment_score"] = 0.4 * vader_score + 0.6 * transformer_score
        result["processing_method"] = "hybrid"
    elif has_vader:
        result["final_sentiment_score"] = result["vader_compound"]
        result["processing_method"] = "vader"
    elif has_transformer:
        t_pos = result["transformer_positive"] or 0
        t_neg = result["transformer_negative"] or 0
        result["final_sentiment_score"] = t_pos - t_neg
        result["processing_method"] = "transformer"

    # Label
    score = result["final_sentiment_score"]
    if score >= 0.05:
        result["sentiment_label"] = "positive"
    elif score <= -0.05:
        result["sentiment_label"] = "negative"
    else:
        result["sentiment_label"] = "neutral"

    return result


def process_posts_sentiment(df: pd.DataFrame, use_transformer: bool = True,
                            try_init_transformer: bool = True) -> pd.DataFrame:
    """Process sentiment for all posts in a DataFrame.

    Returns DataFrame with sentiment columns added.
    """
    if df.empty:
        return df

    if try_init_transformer and use_transformer and not TRANSFORMER_AVAILABLE:
        _init_transformer()

    processing_ts = format_utc(utc_now())
    results = []

    for _, row in df.iterrows():
        text = row.get("post_text", "")
        sent = analyze_sentiment(text, use_transformer=use_transformer)
        sent["post_id"] = row.get("post_id")
        sent["asset"] = row.get("asset")
        sent["timestamp_utc"] = row.get("timestamp_utc")
        sent["cleaned_text"] = clean_post_text(text)
        sent["processing_timestamp"] = processing_ts
        sent["is_relevant"] = 1

        # Engagement weight
        sent["engagement_weight"] = calculate_engagement_weight(
            likes=row.get("likes"),
            reposts=row.get("reposts"),
            views=row.get("views"),
            followers=row.get("author_followers"),
        )

        for col in ["author_username", "likes", "reposts", "views", "author_followers", "replies", "source", "search_query"]:
            if col in row:
                sent[col] = row.get(col)

        results.append(sent)

    result_df = pd.DataFrame(results)
    method_counts = result_df["processing_method"].value_counts().to_dict()
    logger.info("Sentiment processed: %d posts, methods: %s", len(result_df), method_counts)
    return result_df


def calculate_engagement_weight(likes=None, reposts=None, views=None,
                                 followers=None, max_weight: float = 10.0) -> float:
    """Calculate engagement weight using log-scaled metrics.

    Formula:
      w = log(1 + likes) + 0.5 * log(1 + reposts) + 0.3 * log(1 + views/1000)
          + 0.2 * log(1 + followers/1000)
    Capped at max_weight.

    Missing values are excluded (not treated as zero).
    """
    w = 0.0
    components = 0
    if likes is not None and not (isinstance(likes, float) and math.isnan(likes)):
        w += math.log(1 + max(0, float(likes)))
        components += 1
    if reposts is not None and not (isinstance(reposts, float) and math.isnan(reposts)):
        w += 0.5 * math.log(1 + max(0, float(reposts)))
        components += 1
    if views is not None and not (isinstance(views, float) and math.isnan(views)):
        w += 0.3 * math.log(1 + max(0, float(views)) / 1000)
        components += 1
    if followers is not None and not (isinstance(followers, float) and math.isnan(followers)):
        w += 0.2 * math.log(1 + max(0, float(followers)) / 1000)
        components += 1

    if components == 0:
        return 1.0  # Default weight when no engagement data available

    return min(w, max_weight)


def analyze_news_sentiment(text: str) -> tuple[float, str]:
    """Simple sentiment for news text. Returns (score, label)."""
    result = analyze_sentiment(text, use_transformer=False)
    return result["final_sentiment_score"], result["sentiment_label"]
