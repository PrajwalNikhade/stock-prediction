"""Comprehensive tests for CrowdWisdomTrading.

All tests use deterministic fixtures. No live API calls required.
"""

import sys
import os
from pathlib import Path
import tempfile
import json

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.time_utils import parse_timestamp, to_utc, format_utc, utc_now
from src.processing.data_cleaning import (
    clean_post_text, deduplicate_posts, validate_timestamps,
    filter_empty_text, normalize_price_data, data_quality_report,
)
from src.processing.sentiment import (
    analyze_sentiment, calculate_engagement_weight, process_posts_sentiment,
)
from src.processing.aggregation import aggregate_sentiment
from src.processing.feature_engineering import (
    build_features, create_targets, get_feature_columns, check_feature_leakage,
)
from src.modeling.metrics import (
    mae, rmse, r_squared, mape, smape, directional_accuracy,
    sharpe_ratio, sortino_ratio, max_drawdown, calculate_trading_metrics,
)
from src.modeling.train import walk_forward_validate
from src.modeling.backtesting import run_backtest
from src.database.db import init_db, get_connection
from src.utils.demo_data import generate_demo_prices, generate_demo_posts


# ─── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sample_price_df():
    """Create deterministic price data fixture."""
    rng = np.random.RandomState(42)
    n = 100
    base = 50000
    returns = rng.normal(0, 0.02, n)
    closes = base * np.exp(np.cumsum(returns))
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")

    return pd.DataFrame({
        "timestamp_utc": dates.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": closes * (1 + rng.normal(0, 0.005, n)),
        "high": closes * (1 + rng.uniform(0, 0.01, n)),
        "low": closes * (1 - rng.uniform(0, 0.01, n)),
        "close": closes,
        "volume": rng.uniform(1e6, 1e8, n),
        "asset": "BTCUSDT",
        "timeframe": "1d",
        "source": "test",
    })


@pytest.fixture
def sample_posts_df():
    """Create deterministic posts fixture."""
    return pd.DataFrame({
        "post_id": [f"post_{i}" for i in range(20)],
        "asset": "BTCUSDT",
        "timestamp_utc": pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "post_text": [
            "Bitcoin looking bullish!",
            "$BTC is going to the moon 🚀",
            "BTC crash incoming 📉",
            "Just bought some BTC",
            "Holding Bitcoin for now",
            "BTC price analysis looks good",
            "Bearish on Bitcoin today",
            "BTC volume increasing",
            "Buying more $BTC",
            "Bitcoin resistance at 60k",
            "BTC support holding strong",
            "Selling my Bitcoin position",
            "BTC consolidation phase",
            "Bullish divergence on BTC",
            "Bitcoin dip buying opportunity",
            "BTC breaking out!",
            "Worried about BTC direction",
            "BTC accumulation zone",
            "Bitcoin new ATH soon?",
            "BTC trading sideways",
        ],
        "author_username": [f"user_{i}" for i in range(20)],
        "likes": [10, 50, 3, 100, 5, 20, 8, 15, 200, 1, 30, 12, 0, 45, 7, 60, 2, 25, 80, 4],
        "reposts": [5, 20, 1, 40, 2, 10, 3, 8, 80, 0, 15, 6, 0, 20, 3, 25, 1, 12, 35, 2],
        "views": [1000, 5000, 300, 10000, 500, 2000, 800, 1500, 20000, 100, 3000, 1200, 50, 4500, 700, 6000, 200, 2500, 8000, 400],
        "author_followers": [100, 500, 50, 1000, 200, 300, 80, 400, 5000, 30, 600, 150, 20, 800, 120, 2000, 40, 350, 1500, 70],
        "replies": [2, 10, 0, 20, 1, 5, 1, 3, 30, 0, 8, 3, 0, 12, 2, 15, 0, 6, 18, 1],
        "search_query": ["BTC"] * 20,
        "source": ["test"] * 20,
    })


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as td:
        yield os.path.join(td, "test.db")


# ─── Test 1: Price data normalization ─────────────────────────────────

def test_price_normalization(sample_price_df):
    result = normalize_price_data(sample_price_df)
    assert not result.empty
    assert result["close"].isna().sum() == 0
    assert all(isinstance(v, float) for v in result["close"])


# ─── Test 2: UTC timestamp conversion ─────────────────────────────────

def test_utc_timestamp_conversion():
    # ISO string
    dt = parse_timestamp("2025-01-15T10:30:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.hour == 10

    # Unix timestamp (seconds)
    dt2 = parse_timestamp(1705312200)
    assert dt2 is not None

    # Unix timestamp (milliseconds)
    dt3 = parse_timestamp(1705312200000)
    assert dt3 is not None

    # None
    assert parse_timestamp(None) is None

    # Invalid
    assert parse_timestamp("not-a-date") is None


# ─── Test 3: Post normalization ───────────────────────────────────────

def test_post_text_cleaning():
    text = "Check out https://t.co/abc @user1 $BTC is pumping! 🚀  "
    cleaned = clean_post_text(text)
    assert "https://" not in cleaned
    assert "@user1" not in cleaned
    assert "$BTC" in cleaned
    assert cleaned.strip() == cleaned


# ─── Test 4: Post deduplication ───────────────────────────────────────

def test_deduplication():
    df = pd.DataFrame({
        "post_id": ["a", "b", "a", "c", "b"],
        "text": ["t1", "t2", "t1_dup", "t3", "t2_dup"],
    })
    result, dup_count = deduplicate_posts(df)
    assert len(result) == 3
    assert dup_count == 2


# ─── Test 5: Missing engagement values ────────────────────────────────

def test_missing_engagement():
    weight = calculate_engagement_weight(likes=None, reposts=None, views=None, followers=None)
    assert weight == 1.0  # Default when no engagement data

    weight_with = calculate_engagement_weight(likes=100, reposts=50, views=10000, followers=5000)
    assert weight_with > 1.0
    assert weight_with <= 10.0  # Capped


# ─── Test 6: Sentiment output normalization ───────────────────────────

def test_sentiment_output():
    result = analyze_sentiment("Bitcoin is going to the moon! Very bullish! 🚀")
    assert "vader_compound" in result
    assert "final_sentiment_score" in result
    assert "sentiment_label" in result
    assert result["sentiment_label"] in ("positive", "negative", "neutral")
    assert -1.0 <= result["final_sentiment_score"] <= 1.0
    assert result["processing_method"] in ("vader", "hybrid", "transformer", "none")


def test_empty_text_sentiment():
    result = analyze_sentiment("")
    assert result["processing_method"] == "none"
    assert result["final_sentiment_score"] == 0.0


# ─── Test 7: Hourly aggregation ──────────────────────────────────────

def test_hourly_aggregation():
    posts = pd.DataFrame({
        "post_id": [f"p{i}" for i in range(10)],
        "timestamp_utc": pd.date_range("2025-01-01", periods=10, freq="30min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "final_sentiment_score": np.random.RandomState(42).uniform(-1, 1, 10),
        "sentiment_label": ["positive"] * 5 + ["negative"] * 3 + ["neutral"] * 2,
        "engagement_weight": [1.0] * 10,
        "is_relevant": [1] * 10,
        "likes": [5] * 10,
        "reposts": [2] * 10,
        "views": [100] * 10,
        "author_username": [f"u{i}" for i in range(10)],
    })
    result = aggregate_sentiment(posts, "1h", "BTCUSDT")
    assert not result.empty
    assert "mean_sentiment" in result.columns
    assert "positive_ratio" in result.columns
    assert "engagement_weighted_sentiment" in result.columns
    # Should have fewer rows than posts (aggregated)
    assert len(result) <= len(posts)


# ─── Test 8: Daily aggregation ───────────────────────────────────────

def test_daily_aggregation():
    posts = pd.DataFrame({
        "post_id": [f"p{i}" for i in range(30)],
        "timestamp_utc": pd.date_range("2025-01-01", periods=30, freq="8h", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "final_sentiment_score": np.random.RandomState(42).uniform(-1, 1, 30),
        "sentiment_label": ["positive"] * 15 + ["negative"] * 10 + ["neutral"] * 5,
        "engagement_weight": [1.0] * 30,
        "is_relevant": [1] * 30,
        "likes": [5] * 30,
        "reposts": [2] * 30,
        "views": [100] * 30,
        "author_username": [f"u{i}" for i in range(30)],
    })
    result = aggregate_sentiment(posts, "1d", "BTCUSDT")
    assert not result.empty
    assert len(result) <= 11  # 30 rows at 8h intervals = ~10 days


# ─── Test 9: Future-post leakage prevention ──────────────────────────

def test_no_future_post_leakage(sample_price_df, sample_posts_df):
    """Verify that features built from posts only use backward-looking data."""
    processed = process_posts_sentiment(sample_posts_df, use_transformer=False, try_init_transformer=False)
    agg = aggregate_sentiment(processed, "1d", "BTCUSDT")
    features = build_features(sample_price_df, agg, "1d")

    if not features.empty and "mean_sentiment" in features.columns:
        # The sentiment lag features should be NaN for first rows
        assert pd.isna(features["sentiment_lag_1"].iloc[0]) or features["sentiment_lag_1"].iloc[0] == 0


# ─── Test 10: Target-period close leakage prevention ─────────────────

def test_no_target_leakage(sample_price_df):
    """Verify target generation doesn't leak into features."""
    features = build_features(sample_price_df, pd.DataFrame(), "1d")
    features = create_targets(features, horizons=[1, 2, 3])

    feature_cols = get_feature_columns()
    for col in feature_cols:
        if col in features.columns:
            # No feature should perfectly correlate with future return
            for h in [1, 2, 3]:
                target_col = f"target_return_{h}"
                if target_col in features.columns:
                    valid = features[[col, target_col]].dropna()
                    if len(valid) > 10 and valid[col].std() > 1e-8 and valid[target_col].std() > 1e-8:
                        corr = valid.corr().iloc[0, 1]
                        if not np.isnan(corr):
                            assert abs(corr) < 0.99, f"Possible leakage: {col} corr={corr}"


# ─── Test 11: Future return and target generation ─────────────────────

def test_target_generation(sample_price_df):
    features = build_features(sample_price_df, pd.DataFrame(), "1d")
    features = create_targets(features, horizons=[0, 1, 2, 3, 4, 5])

    for h in range(6):
        col = f"target_return_{h}"
        assert col in features.columns

    # target_return_1 should have last row NaN (no future data)
    assert pd.isna(features["target_return_1"].iloc[-1])

    # target_return_5 should have last 5 rows NaN
    assert features["target_return_5"].iloc[-5:].isna().all()


# ─── Test 12: Forecast horizon ordering ──────────────────────────────

def test_forecast_horizon_ordering():
    from src.modeling.forecasting import generate_forecasts
    # Create minimal features
    df = generate_demo_prices("TEST", "1d", 50, 100)
    features = build_features(df, pd.DataFrame(), "1d")

    forecasts = generate_forecasts(features, {}, {}, "TEST", "1d", 100.0)
    assert len(forecasts) == 6

    horizons = [f["horizon"] for f in forecasts]
    assert horizons == [0, 1, 2, 3, 4, 5]

    # n+0 should have status "current/nowcast"
    assert forecasts[0]["status"] == "current/nowcast"
    assert forecasts[0]["predicted_price"] == 100.0


# ─── Test 13: Chronological train/test separation ────────────────────

def test_chronological_split():
    """Verify walk-forward validation preserves time ordering."""
    from src.modeling.train import prepare_model_data
    demo_prices = generate_demo_prices("TEST", "1d", 100, 100)
    features = build_features(demo_prices, pd.DataFrame(), "1d")
    features = create_targets(features, horizons=[1])

    X, y, timestamps, fcols = prepare_model_data(features, 1)
    assert X is not None
    assert len(X) > 0

    # Timestamps should be monotonically non-decreasing
    if timestamps is not None:
        ts_series = pd.to_datetime(timestamps)
        assert (ts_series.diff().dropna() >= pd.Timedelta(0)).all()


# ─── Test 14: Walk-forward validation ────────────────────────────────

def test_walk_forward_validation():
    demo_prices = generate_demo_prices("TEST", "1d", 100, 100)
    features = build_features(demo_prices, pd.DataFrame(), "1d")
    features = create_targets(features, horizons=[1])

    results = walk_forward_validate(features, horizon=1, n_folds=3)
    if "error" not in results:
        assert "best_model" in results
        assert "validation_metrics" in results
        assert "test_metrics" in results
        assert "naive_test_metrics" in results
        assert results["n_train"] > 0
        assert results["n_test"] > 0


# ─── Test 15: Model baseline comparison ──────────────────────────────

def test_baseline_comparison():
    demo_prices = generate_demo_prices("TEST", "1d", 100, 100)
    features = build_features(demo_prices, pd.DataFrame(), "1d")
    features = create_targets(features, horizons=[1])

    results = walk_forward_validate(features, horizon=1, n_folds=3)
    if "error" not in results:
        vm = results["validation_metrics"]
        assert "Naive" in vm
        # Naive baseline should have non-NaN MAE
        assert not np.isnan(vm["Naive"]["mae"])


# ─── Test 16: Sharpe Ratio calculation ───────────────────────────────

def test_sharpe_ratio():
    returns = np.array([0.01, -0.005, 0.02, 0.003, -0.01, 0.015, 0.005])
    sr = sharpe_ratio(returns, periods_per_year=252)
    assert not np.isnan(sr)
    assert isinstance(sr, float)

    # Zero returns should give NaN (zero std)
    sr_zero = sharpe_ratio(np.zeros(10))
    assert np.isnan(sr_zero)


# ─── Test 17: Sortino Ratio calculation ──────────────────────────────

def test_sortino_ratio():
    returns = np.array([0.01, -0.005, 0.02, 0.003, -0.01, 0.015, 0.005])
    sr = sortino_ratio(returns, periods_per_year=252)
    assert not np.isnan(sr)
    assert isinstance(sr, float)

    # All positive returns → NaN (no downside)
    sr_pos = sortino_ratio(np.array([0.01, 0.02, 0.03]))
    assert np.isnan(sr_pos)


# ─── Test 18: Maximum Drawdown calculation ───────────────────────────

def test_max_drawdown():
    wealth = np.array([1.0, 1.05, 1.10, 1.03, 0.95, 1.02, 1.08])
    mdd = max_drawdown(wealth)
    assert not np.isnan(mdd)
    assert 0 <= mdd <= 1
    # Peak was 1.10, trough 0.95 → MDD ≈ 0.136
    assert abs(mdd - (1.10 - 0.95) / 1.10) < 0.01


# ─── Test 19: Transaction costs ──────────────────────────────────────

def test_transaction_costs():
    predictions = np.array([0.01, -0.01, 0.02, 0.005, -0.005])
    actuals = np.array([0.005, -0.008, 0.015, 0.003, -0.002])

    # With zero costs
    bt_zero = run_backtest(predictions, actuals, threshold=0.0,
                          transaction_cost_bps=0, asset="TEST", timeframe="1d")
    # With costs
    bt_cost = run_backtest(predictions, actuals, threshold=0.0,
                          transaction_cost_bps=50, asset="TEST", timeframe="1d")

    if "error" not in bt_zero and "error" not in bt_cost:
        assert bt_cost["total_transaction_costs"] > 0
        # Strategy with costs should underperform vs zero-cost
        cum_zero = bt_zero["strategy_metrics"]["cumulative_return"]
        cum_cost = bt_cost["strategy_metrics"]["cumulative_return"]
        assert cum_cost <= cum_zero


# ─── Test 20: Database initialization ────────────────────────────────

def test_database_init(temp_db_path):
    init_db(temp_db_path)
    conn = get_connection(temp_db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            "assets", "raw_x_posts", "processed_x_posts", "price_bars",
            "news_articles", "sentiment_aggregates", "engineered_features",
            "forecasts", "validation_results", "backtest_results",
            "ingestion_runs", "model_runs",
        ]
        for t in expected_tables:
            assert t in tables, f"Missing table: {t}"
    finally:
        conn.close()


# ─── Test 21: Demo mode ─────────────────────────────────────────────

def test_demo_data_generation():
    prices = generate_demo_prices("TEST", "1d", 30, 100)
    assert len(prices) == 30
    assert "source" in prices.columns
    assert (prices["source"] == "demo").all()
    assert prices["close"].notna().all()

    posts = generate_demo_posts("TEST", 50, 30)
    assert len(posts) == 50
    assert (posts["source"] == "demo").all()
    assert posts["post_text"].str.len().min() > 0


# ─── Test 22: Missing API credentials ───────────────────────────────

def test_missing_credentials():
    from src.ingestion.apify_client import is_available as apify_avail
    from src.ingestion.tavily_client import is_available as tavily_avail
    from src.ingestion.exa_client import is_available as exa_avail

    # These should not crash, just return availability status
    _ = apify_avail()
    _ = tavily_avail()
    _ = exa_avail()


# ─── Test 23: CSV price ingestion ────────────────────────────────────

def test_csv_import():
    from src.ingestion.price_providers import import_csv
    csv_content = """date,open,high,low,close,volume
2025-01-01,100,105,98,103,1000000
2025-01-02,103,108,101,107,1200000
2025-01-03,107,110,105,108,1100000"""

    result = import_csv(csv_content, "TEST", "1d")
    assert len(result) == 3
    assert "timestamp_utc" in result.columns
    assert "close" in result.columns
    assert (result["asset"] == "TEST").all()
    assert (result["source"] == "csv").all()


# ─── Test 24: Evaluation report generation ───────────────────────────

def test_evaluation_report_generation():
    from src.reporting.evaluation_report import generate_evaluation_report
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        content = generate_evaluation_report(
            asset="TEST",
            timeframe="1d",
            data_source="demo",
            price_coverage={"source": "demo", "count": 100, "date_range": "2025-01-01 to 2025-04-10"},
            post_coverage={"count": 50, "date_range": "2025-01-01 to 2025-04-10"},
            feature_info={"n_features": 30},
            validation_results={
                "best_model": "Ridge",
                "validation_metrics": {
                    "Ridge": {"mae": 0.02, "rmse": 0.03, "r_squared": -0.1, "directional_accuracy": 52.0, "n_folds": 3},
                    "Naive": {"mae": 0.02, "rmse": 0.03, "r_squared": 0.0, "directional_accuracy": 50.0, "n_folds": 3},
                },
                "test_metrics": {"mae": 0.02, "rmse": 0.03, "r_squared": -0.05, "directional_accuracy": 51.0, "smape": 150.0, "prediction_bias": 0.001},
                "naive_test_metrics": {"mae": 0.02, "rmse": 0.03, "r_squared": 0.0, "directional_accuracy": 50.0, "smape": 150.0, "prediction_bias": 0.0},
            },
            backtest_results={
                "strategy_metrics": {"cumulative_return": 0.05, "sharpe_ratio": 0.3, "sortino_ratio": 0.4, "max_drawdown": 0.15, "annualized_volatility": 0.2, "win_rate": 52.0, "num_trades": 30},
                "buyhold_metrics": {"cumulative_return": 0.08, "sharpe_ratio": 0.5, "sortino_ratio": 0.6, "max_drawdown": 0.12, "annualized_volatility": 0.18, "win_rate": 54.0, "num_trades": 1},
                "transaction_cost_bps": 10,
                "n_observations": 50,
                "periods_per_year": 252,
            },
            output_dir=td,
        )
        assert "CrowdWisdomTrading" in content
        assert "DEMO MODE" in content
        assert "Ridge" in content
        # Check file was created
        report_path = Path(td) / "evaluation_report.md"
        assert report_path.exists()


# ─── Additional: Feature column completeness ─────────────────────────

def test_feature_columns():
    cols = get_feature_columns()
    assert len(cols) > 20
    assert "close_price" in cols
    assert "mean_sentiment" in cols
    assert "sentiment_missing" in cols
    assert "return_1" in cols


def test_feature_leakage_check(sample_price_df):
    features = build_features(sample_price_df, pd.DataFrame(), "1d")
    features = create_targets(features, horizons=[1])
    has_leak = check_feature_leakage(features, "target_return_1")
    assert not has_leak


# ─── Additional: Metrics edge cases ──────────────────────────────────

def test_metrics_with_few_observations():
    y = np.array([1.0])
    p = np.array([1.1])
    result = mae(y, p)
    assert abs(result - 0.1) < 1e-10


def test_mape_near_zero():
    y = np.array([0.0001, 0.0, -0.0001])
    p = np.array([0.0002, 0.001, -0.0002])
    result = mape(y, p)
    # Should handle near-zero values
    assert isinstance(result, float)


# ─── Tests 30-34: Provider Normalization, Error Handling & Mode Separation ─

def test_apify_response_normalization_multi_schema():
    from src.ingestion.apify_client import _normalize_posts

    raw_posts = [
        # Schema 1: Apidojo style (camelCase, author dict)
        {
            "id": "184000111222",
            "createdAt": "Sun Oct 12 14:23:01 +0000 2025",
            "full_text": "Bitcoin breaking 85k! $BTC to the moon",
            "author": {
                "id": "999888",
                "userName": "cryptotrader",
                "followers": 15000,
            },
            "likeCount": 120,
            "retweetCount": 45,
            "replyCount": 10,
            "viewCount": 5000,
            "url": "https://x.com/cryptotrader/status/184000111222",
        },
        # Schema 2: Quacker / Legacy Twitter style (snake_case, user dict)
        {
            "tweet_id": "184000333444",
            "created_at": "2025-10-12T15:00:00Z",
            "text": "Bearish divergence spotted on $BTC hourly",
            "user": {
                "id_str": "777666",
                "screen_name": "chartwatcher",
                "followers_count": 2500,
            },
            "favorite_count": 35,
            "retweet_count": 8,
            "reply_count": 2,
            "views": 800,
        },
        # Schema 3: Minimal / partial data (missing engagement numbers)
        {
            "tweetId": "184000555666",
            "timestamp": "2025-10-12T16:00:00Z",
            "tweet_text": "Bitcoin consolidation continues",
            "username": "hodler_123",
        },
    ]

    df = _normalize_posts(raw_posts, "BTCUSDT")
    assert len(df) == 3
    assert set(["post_id", "asset", "timestamp_utc", "post_text", "author_username", "likes", "reposts", "views", "source"]).issubset(df.columns)

    # Check row 1
    r0 = df.iloc[0]
    assert r0["post_id"] == "184000111222"
    assert r0["author_username"] == "cryptotrader"
    assert r0["likes"] == 120
    assert r0["author_followers"] == 15000
    assert r0["timestamp_utc"] == "2025-10-12T14:23:01Z"

    # Check row 2
    r1 = df.iloc[1]
    assert r1["post_id"] == "184000333444"
    assert r1["author_username"] == "chartwatcher"
    assert r1["likes"] == 35

    # Check row 3 (handled missing values safely)
    r2 = df.iloc[2]
    assert r2["post_id"] == "184000555666"
    assert r2["author_username"] == "hodler_123"
    assert r2["likes"] is None or pd.isna(r2["likes"])


def test_apify_actor_input_generation():
    from src.ingestion.apify_client import get_actor_input, format_actor_id

    # Test Apidojo schema
    inp_apidojo = get_actor_input("apidojo/tweet-scraper", "BTC", 50)
    assert "searchTerms" in inp_apidojo
    assert inp_apidojo["searchTerms"] == ["BTC"]
    assert inp_apidojo["maxItems"] == 50

    # Test Quacker schema
    inp_quacker = get_actor_input("quacker/twitter-scraper", "ETH", 25)
    assert "queries" in inp_quacker
    assert inp_quacker["maxTweets"] == 25

    # Test Actor ID formatting
    assert format_actor_id("apidojo/tweet-scraper") == "apidojo~tweet-scraper"


def test_price_provider_source_and_coverage():
    from src.ingestion.price_providers import get_price_coverage, _ensure_schema

    dates = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    raw = pd.DataFrame({
        "timestamp_utc": dates,
        "open": 100.0,
        "high": 105.0,
        "low": 98.0,
        "close": 102.0,
        "volume": 1000.0,
    })
    norm = _ensure_schema(raw, "BTCUSDT", "1d", "binance")
    cov = get_price_coverage(norm)

    assert cov["source"] == "binance"
    assert cov["count"] == 10
    assert "2025-01-01T00:00:00Z" in cov["date_range"]
    assert "2025-01-10T00:00:00Z" in cov["date_range"]


def test_missing_credentials_connection_tests():
    from src.ingestion.tavily_client import test_tavily_connection
    from src.ingestion.exa_client import test_exa_connection

    # Test connection functions return structured dicts without unhandled exceptions
    tav_res = test_tavily_connection(query="Test", max_results=1)
    assert isinstance(tav_res, dict)
    assert "success" in tav_res
    assert "configured" in tav_res

    exa_res = test_exa_connection(query="Test", max_results=1)
    assert isinstance(exa_res, dict)
    assert "success" in exa_res
    assert "configured" in exa_res


def test_live_vs_demo_mode_separation(sample_price_df):
    # Verify that building features without posts (PARTIAL live data) does NOT fabricate fake posts
    features = build_features(sample_price_df, pd.DataFrame(), "1d")
    assert "sentiment_missing" in features.columns
    # When no posts were collected, sentiment_missing should be 1
    assert (features["sentiment_missing"] == 1).all()

    # Demo data generation explicitly labels data as demo
    demo_posts = generate_demo_posts("BTCUSDT", 20)
    assert (demo_posts["source"] == "demo").all()


def test_provenance_and_run_id_persistence(temp_db_path, sample_price_df):
    from src.database.db import init_db
    from src.database.repository import Repository
    init_db(temp_db_path)
    repo = Repository(temp_db_path)

    run_id = "test_run_12345"
    # Test saving engineered features
    features = build_features(sample_price_df, pd.DataFrame(), "1d")
    count_feat = repo.save_engineered_features(features, "TEST", "1d", run_id=run_id)
    assert count_feat == len(features)

    # Test saving forecasts with run_id
    forecast_records = [{
        "asset": "TEST",
        "timeframe": "1d",
        "forecast_origin": "2026-01-01T00:00:00Z",
        "horizon": 1,
        "target_timestamp": "2026-01-02T00:00:00Z",
        "reference_price": 50000.0,
        "predicted_return": 0.01,
        "predicted_price": 50500.0,
        "model_name": "Ridge",
    }]
    count_fc = repo.save_forecasts(forecast_records, run_id=run_id)
    assert count_fc == 1

    # Test saving experiment run
    repo.save_experiment_run({
        "run_id": run_id,
        "asset": "TEST",
        "timeframe": "1d",
        "data_mode": "partial",
        "price_provider": "binance",
        "price_count": 100,
        "price_date_range": "2026-01-01 to 2026-04-10",
        "post_provider": "apify",
        "post_count": 0,
        "post_date_range": "None",
        "news_count": 5,
        "best_model": "Ridge",
        "validation_mae": 0.015,
        "test_mae": 0.018,
        "naive_test_mae": 0.020,
        "strategy_return": 0.05,
        "buyhold_return": 0.02,
        "strategy_sharpe": 1.2,
        "max_drawdown": 0.08,
        "status": "completed",
        "notes": "Unit test verified run",
    })

    runs = repo.get_experiment_runs("TEST")
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == run_id
    assert runs.iloc[0]["data_mode"] == "partial"


def test_equity_trading_session_timestamps(sample_price_df):
    from src.modeling.forecasting import generate_forecasts
    from src.processing.feature_engineering import build_features

    features = build_features(sample_price_df, pd.DataFrame(), "1d")
    # Friday reference timestamp
    friday_ts = "2026-10-09T00:00:00Z"
    ref_price = 150.0

    # Equity forecast (should advance by business days, skipping weekend)
    equity_forecasts = generate_forecasts(
        feature_df=features,
        models={},
        scalers={},
        asset="AAPL",
        timeframe="1d",
        reference_price=ref_price,
        reference_timestamp=friday_ts,
        asset_class="equity",
    )
    assert len(equity_forecasts) == 6
    assert equity_forecasts[0]["horizon"] == 0
    assert "2026-10-09" in equity_forecasts[0]["target_timestamp"]
    # n+1 must be Monday (2026-10-12), not Saturday!
    assert "2026-10-12" in equity_forecasts[1]["target_timestamp"]
    # n+2 must be Tuesday (2026-10-13)
    assert "2026-10-13" in equity_forecasts[2]["target_timestamp"]

    # Crypto forecast (24/7 trading: Friday advances to Saturday)
    crypto_forecasts = generate_forecasts(
        feature_df=features,
        models={},
        scalers={},
        asset="BTCUSDT",
        timeframe="1d",
        reference_price=65000.0,
        reference_timestamp=friday_ts,
        asset_class="crypto",
    )
    assert len(crypto_forecasts) == 6
    assert "2026-10-09" in crypto_forecasts[0]["target_timestamp"]
    # n+1 must be Saturday (2026-10-10)
    assert "2026-10-10" in crypto_forecasts[1]["target_timestamp"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


