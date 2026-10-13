"""SQLite database initialization and access for CrowdWisdomTrading."""

import sqlite3
from pathlib import Path
from src.utils.logging_config import logger


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Get a SQLite connection, creating parent dirs if needed."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_db(conn: sqlite3.Connection) -> None:
    """Migrate schema safely adding missing columns to existing tables."""
    for table in ["forecasts", "validation_results", "backtest_results"]:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cursor.fetchall()]
        if cols and "run_id" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN run_id TEXT")
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_run ON forecasts(run_id)")
    except Exception:
        pass
    conn.commit()


def init_db(db_path: str | Path) -> None:
    """Create all tables if they do not exist and run migrations."""
    conn = get_connection(db_path)
    try:
        _migrate_db(conn)
        conn.executescript(_SCHEMA)
        _migrate_db(conn)
        conn.commit()
        logger.info("Database initialized at %s", db_path)
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    provider TEXT NOT NULL,
    yfinance_ticker TEXT,
    search_terms TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_x_posts (
    post_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    timestamp_utc TEXT,
    post_text TEXT,
    author_id TEXT,
    author_username TEXT,
    author_followers INTEGER,
    likes INTEGER,
    reposts INTEGER,
    replies INTEGER,
    views INTEGER,
    post_url TEXT,
    search_query TEXT,
    collection_timestamp_utc TEXT,
    source TEXT DEFAULT 'apify',
    raw_json TEXT,
    PRIMARY KEY (post_id, asset)
);

CREATE TABLE IF NOT EXISTS processed_x_posts (
    post_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    timestamp_utc TEXT,
    cleaned_text TEXT,
    vader_compound REAL,
    vader_pos REAL,
    vader_neg REAL,
    vader_neu REAL,
    transformer_positive REAL,
    transformer_negative REAL,
    transformer_neutral REAL,
    final_sentiment_score REAL,
    sentiment_label TEXT,
    processing_method TEXT,
    engagement_weight REAL,
    processing_timestamp TEXT,
    is_relevant INTEGER DEFAULT 1,
    PRIMARY KEY (post_id, asset)
);

CREATE TABLE IF NOT EXISTS price_bars (
    asset TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    source TEXT,
    PRIMARY KEY (asset, timestamp_utc, timeframe)
);

CREATE TABLE IF NOT EXISTS news_articles (
    news_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    title TEXT,
    snippet TEXT,
    source_name TEXT,
    url TEXT,
    published_at TEXT,
    retrieved_at TEXT,
    provider TEXT,
    sentiment_score REAL,
    sentiment_label TEXT
);

CREATE TABLE IF NOT EXISTS sentiment_aggregates (
    asset TEXT NOT NULL,
    period_start TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    mean_sentiment REAL,
    median_sentiment REAL,
    positive_ratio REAL,
    negative_ratio REAL,
    neutral_ratio REAL,
    net_sentiment REAL,
    sentiment_std REAL,
    post_count INTEGER,
    relevant_post_count INTEGER,
    unique_author_count INTEGER,
    likes_sum INTEGER,
    reposts_sum INTEGER,
    views_sum INTEGER,
    engagement_weighted_sentiment REAL,
    PRIMARY KEY (asset, period_start, timeframe)
);

CREATE TABLE IF NOT EXISTS engineered_features (
    asset TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    feature_json TEXT NOT NULL,
    run_id TEXT,
    PRIMARY KEY (asset, timestamp_utc, timeframe)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    forecast_origin TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    target_timestamp TEXT,
    reference_price REAL,
    predicted_return REAL,
    predicted_price REAL,
    model_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS validation_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_name TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    fold INTEGER,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    dataset_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    signal TEXT,
    position REAL,
    predicted_return REAL,
    realized_return REAL,
    transaction_cost REAL,
    cumulative_return REAL,
    model_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    provider TEXT NOT NULL,
    run_type TEXT NOT NULL,
    requested_start TEXT,
    requested_end TEXT,
    actual_start TEXT,
    actual_end TEXT,
    record_count INTEGER DEFAULT 0,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    model_name TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    feature_columns TEXT,
    training_start TEXT,
    training_end TEXT,
    validation_start TEXT,
    validation_end TEXT,
    test_start TEXT,
    test_end TEXT,
    config_json TEXT,
    model_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    data_mode TEXT NOT NULL,
    price_provider TEXT NOT NULL,
    price_count INTEGER NOT NULL,
    price_date_range TEXT,
    post_provider TEXT,
    post_count INTEGER NOT NULL,
    post_date_range TEXT,
    news_count INTEGER NOT NULL,
    best_model TEXT,
    validation_mae REAL,
    test_mae REAL,
    naive_test_mae REAL,
    strategy_return REAL,
    buyhold_return REAL,
    strategy_sharpe REAL,
    max_drawdown REAL,
    status TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS experiment_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    experiment TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    asset TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    feature_set TEXT NOT NULL,
    best_model TEXT,
    validation_mae REAL,
    test_mae REAL,
    test_rmse REAL,
    directional_accuracy REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    strategy_return REAL,
    relative_mae_improvement_pct REAL,
    data_mode TEXT NOT NULL,
    n_obs INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_price_bars_asset_tf ON price_bars(asset, timeframe);
CREATE INDEX IF NOT EXISTS idx_raw_posts_asset ON raw_x_posts(asset);
CREATE INDEX IF NOT EXISTS idx_processed_posts_asset ON processed_x_posts(asset);
CREATE INDEX IF NOT EXISTS idx_news_asset ON news_articles(asset);
CREATE INDEX IF NOT EXISTS idx_forecasts_asset ON forecasts(asset, timeframe);
CREATE INDEX IF NOT EXISTS idx_exp_eval_run ON experiment_evaluations(run_id);
CREATE INDEX IF NOT EXISTS idx_exp_eval_asset ON experiment_evaluations(asset, timeframe);
"""
