"""Demo/test data generator for CrowdWisdomTrading.

Generates deterministic synthetic data for testing, UI development,
and demonstration when live API credentials are unavailable.

ALL demo data is clearly labeled and must never be mixed with real data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.utils.time_utils import format_utc, utc_now
from src.utils.logging_config import logger

DEMO_SEED = 42


def generate_demo_prices(asset: str = "BTCUSDT", timeframe: str = "1d",
                         days: int = 180, base_price: float = 50000.0) -> pd.DataFrame:
    """Generate deterministic synthetic price data.

    Uses geometric Brownian motion with seed for reproducibility.
    """
    rng = np.random.RandomState(DEMO_SEED)

    if timeframe == "1h":
        n_bars = days * 24
        delta = timedelta(hours=1)
    else:
        n_bars = days
        delta = timedelta(days=1)

    start = utc_now() - timedelta(days=days)
    timestamps = [start + i * delta for i in range(n_bars)]

    # GBM simulation
    mu = 0.0002  # Daily drift
    sigma = 0.02  # Daily volatility
    returns = rng.normal(mu, sigma, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    opens = prices * (1 + rng.normal(0, 0.005, n_bars))
    highs = np.maximum(prices, opens) * (1 + rng.uniform(0, 0.01, n_bars))
    lows = np.minimum(prices, opens) * (1 - rng.uniform(0, 0.01, n_bars))
    volumes = rng.uniform(1e6, 1e8, n_bars)

    df = pd.DataFrame({
        "timestamp_utc": [format_utc(t) for t in timestamps],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "asset": asset,
        "timeframe": timeframe,
        "source": "demo",
    })
    logger.info("Generated %d demo price bars for %s", len(df), asset)
    return df


def generate_demo_posts(asset: str = "BTCUSDT", n_posts: int = 500,
                        days: int = 180) -> pd.DataFrame:
    """Generate deterministic synthetic X posts with sentiment."""
    rng = np.random.RandomState(DEMO_SEED + 1)

    start = utc_now() - timedelta(days=days)
    timestamps = [start + timedelta(seconds=rng.randint(0, days * 86400)) for _ in range(n_posts)]
    timestamps.sort()

    sentiments = ["positive", "negative", "neutral"]
    templates = {
        "positive": [
            f"${asset} looking bullish! 🚀",
            f"Great time to buy {asset}",
            f"{asset} is going to the moon",
            f"Very optimistic about {asset} today",
            f"Strong momentum for {asset}",
        ],
        "negative": [
            f"${asset} is crashing 📉",
            f"Selling all my {asset}",
            f"{asset} looks bearish",
            f"Worried about {asset} direction",
            f"Time to exit {asset} positions",
        ],
        "neutral": [
            f"Watching {asset} closely today",
            f"What do you think about {asset}?",
            f"{asset} trading sideways",
            f"No clear direction for {asset}",
            f"Holding {asset} for now",
        ],
    }

    records = []
    for i, ts in enumerate(timestamps):
        sent = rng.choice(sentiments, p=[0.4, 0.25, 0.35])
        text = rng.choice(templates[sent])
        records.append({
            "post_id": f"demo_{asset}_{i:06d}",
            "asset": asset,
            "timestamp_utc": format_utc(ts),
            "post_text": text,
            "author_id": f"user_{rng.randint(1000, 9999)}",
            "author_username": f"demo_user_{rng.randint(100, 999)}",
            "author_followers": int(rng.lognormal(7, 2)),
            "likes": int(rng.lognormal(3, 2)),
            "reposts": int(rng.lognormal(2, 1.5)),
            "replies": int(rng.lognormal(1, 1)),
            "views": int(rng.lognormal(8, 2)),
            "post_url": f"https://x.com/demo/status/{1000000 + i}",
            "search_query": asset,
            "collection_timestamp_utc": format_utc(utc_now()),
            "source": "demo",
        })

    df = pd.DataFrame(records)
    logger.info("Generated %d demo posts for %s", len(df), asset)
    return df


def generate_demo_news(asset: str = "BTCUSDT", n_articles: int = 20) -> pd.DataFrame:
    """Generate deterministic synthetic news articles."""
    rng = np.random.RandomState(DEMO_SEED + 2)

    titles = [
        f"{asset} Price Analysis: Key Levels to Watch",
        f"Market Update: {asset} Trading Volume Surges",
        f"Expert Opinion: What's Next for {asset}?",
        f"{asset} Technical Analysis Shows Mixed Signals",
        f"Institutional Interest in {asset} Growing",
    ]

    records = []
    for i in range(n_articles):
        pub_time = utc_now() - timedelta(hours=rng.randint(1, 168))
        records.append({
            "news_id": f"demo_news_{i:04d}",
            "asset": asset,
            "title": titles[i % len(titles)],
            "snippet": f"Demo news content about {asset} market analysis and outlook.",
            "source_name": rng.choice(["CoinDesk", "Bloomberg", "Reuters", "CNBC"]),
            "url": f"https://example.com/news/{i}",
            "published_at": format_utc(pub_time),
            "retrieved_at": format_utc(utc_now()),
            "provider": "demo",
        })

    return pd.DataFrame(records)
