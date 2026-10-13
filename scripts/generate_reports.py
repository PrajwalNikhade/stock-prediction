"""CLI script to run CrowdWisdomTrading pipeline and generate reports.

Supports:
- Live mode (default): Uses real market prices from Binance/Yahoo, real Apify posts if configured, real news from Tavily/Exa.
  If Apify is unconfigured or authentication fails, processes in PARTIAL DATA mode without silently fabricating fake posts.
- Explicit Demo mode (--demo): Generates labeled synthetic fixtures.
"""

import sys
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from config.settings import get_asset_by_ticker, DATABASE_PATH
from src.database.db import init_db
from src.database.repository import Repository
from src.ingestion.price_providers import fetch_price_data, get_price_coverage
from src.ingestion.apify_client import (
    is_available as apify_available,
    collect_posts,
    verify_credentials as verify_apify,
    test_apify_connection,
)
from src.ingestion.tavily_client import is_available as tavily_available, search_news as tavily_search
from src.ingestion.exa_client import is_available as exa_available, search_news as exa_search
from src.utils.demo_data import generate_demo_prices, generate_demo_posts, generate_demo_news
from src.processing.data_cleaning import deduplicate_posts, validate_timestamps, filter_empty_text
from src.processing.sentiment import process_posts_sentiment, analyze_news_sentiment
from src.processing.aggregation import aggregate_sentiment
from src.processing.feature_engineering import build_features, create_targets, get_feature_columns
from src.modeling.train import walk_forward_validate, train_final_model
from src.modeling.forecasting import generate_forecasts
from src.modeling.backtesting import run_backtest
from src.reporting.sentiment_report import generate_sentiment_outlook
from src.reporting.evaluation_report import generate_evaluation_report
from src.utils.time_utils import format_utc, utc_now
from src.utils.logging_config import logger


def run_pipeline(
    ticker: str = "BTCUSDT",
    timeframe: str = "1d",
    days: int = 180,
    use_demo: bool = False,
    run_id: str = None,
):
    """Execute the end-to-end forecasting pipeline with provenance and audit tracking."""
    run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info("Starting experiment %s for %s (%s)...", run_id, ticker, timeframe)

    logger.info("Initializing database: %s", DATABASE_PATH)
    init_db(DATABASE_PATH)
    repo = Repository(DATABASE_PATH)

    asset_cfg = get_asset_by_ticker(ticker) or {
        "ticker": ticker,
        "name": ticker,
        "asset_class": "crypto",
        "provider": "binance",
        "search_terms": [ticker, "Bitcoin", "#BTC"],
    }
    data_mode = "demo" if use_demo else "live"

    price_df = pd.DataFrame()
    posts_df = pd.DataFrame()
    news_df = pd.DataFrame()
    apify_status_msg = "Not attempted"

    if use_demo:
        logger.info("EXPLICIT DEMO MODE: Generating labeled synthetic data...")
        price_df = generate_demo_prices(ticker, timeframe, n_bars=days, base_price=65000.0)
        posts_df = generate_demo_posts(ticker, n_posts=150)
        news_df = generate_demo_news(ticker, n_articles=10)
        price_source = "demo"
        apify_status_msg = "Demo synthetic fixtures"
    else:
        # 1. Fetch real price data
        logger.info("LIVE MODE: Fetching real market price data for %s (%s, lookback %d days)...", ticker, timeframe, days)
        try:
            price_df = fetch_price_data(asset_cfg, timeframe, days=days)
        except Exception as e:
            logger.error("Price fetch error: %s", e)

        if price_df.empty:
            logger.error("Could not retrieve live price data for %s from Binance.", ticker)
            print("ERROR: Could not retrieve live prices. Pass --demo to explicitly use demo fixtures.")
            return

        price_source = price_df["source"].iloc[0] if "source" in price_df.columns else "binance"
        logger.info("Successfully fetched %d price bars from %s (%s to %s)",
                    len(price_df), price_source, price_df["timestamp_utc"].min(), price_df["timestamp_utc"].max())

        # Log price ingestion
        repo.log_ingestion_run({
            "asset": ticker,
            "provider": price_source,
            "run_type": "prices",
            "requested_start": (utc_now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "requested_end": format_utc(utc_now()),
            "actual_start": str(price_df["timestamp_utc"].min()),
            "actual_end": str(price_df["timestamp_utc"].max()),
            "record_count": len(price_df),
            "status": "success",
            "error_message": None,
        })

        # 2. Collect X posts via Apify (small test run first as required)
        logger.info("Testing Apify Actor connection with a small sample test run...")
        search_terms = asset_cfg.get("search_terms", [ticker])
        test_query = asset_cfg.get("name") or (search_terms[0] if search_terms else ticker)
        test_run = test_apify_connection(query=test_query, max_items=2)

        if test_run["success"]:
            logger.info("Apify test run successful (%d items). Proceeding to collect live posts...", test_run["items_retrieved"])
            # Query top relevant terms for the asset to retrieve live posts
            selected_terms = [t for t in search_terms if len(t) > 2][:3] or [test_query]
            posts_df = collect_posts(ticker, selected_terms, max_items=50)
            apify_status_msg = f"Live retrieved ({len(posts_df)} posts)"
            repo.log_ingestion_run({
                "asset": ticker,
                "provider": "apify",
                "run_type": "x_posts",
                "requested_start": None,
                "requested_end": None,
                "actual_start": str(posts_df["timestamp_utc"].min()) if not posts_df.empty else None,
                "actual_end": str(posts_df["timestamp_utc"].max()) if not posts_df.empty else None,
                "record_count": len(posts_df),
                "status": "success",
                "error_message": None,
            })
        else:
            apify_err = test_run.get("error", "Unknown error")
            apify_status_msg = f"Failed ({apify_err})"
            logger.warning("Apify test run failed: %s", apify_err)
            logger.info("Operating in PARTIAL DATA mode without fabricating synthetic posts.")
            posts_df = pd.DataFrame()
            repo.log_ingestion_run({
                "asset": ticker,
                "provider": "apify",
                "run_type": "x_posts",
                "requested_start": None,
                "requested_end": None,
                "actual_start": None,
                "actual_end": None,
                "record_count": 0,
                "status": "failed",
                "error_message": apify_err,
            })

        # 3. Collect News via Tavily and Exa
        news_articles = []
        if tavily_available():
            try:
                tdf = tavily_search(ticker, search_terms, max_results=10)
                if not tdf.empty:
                    news_articles.append(tdf)
                    logger.info("Tavily retrieved %d live news articles", len(tdf))
                    repo.log_ingestion_run({
                        "asset": ticker,
                        "provider": "tavily",
                        "run_type": "news",
                        "requested_start": None,
                        "requested_end": None,
                        "actual_start": str(tdf["published_at"].min()) if "published_at" in tdf.columns else None,
                        "actual_end": str(tdf["published_at"].max()) if "published_at" in tdf.columns else None,
                        "record_count": len(tdf),
                        "status": "success",
                        "error_message": None,
                    })
            except Exception as e:
                logger.warning("Tavily news error: %s", e)

        if exa_available():
            try:
                edf = exa_search(ticker, search_terms, max_results=10)
                if not edf.empty:
                    news_articles.append(edf)
                    logger.info("Exa retrieved %d live news articles", len(edf))
                    repo.log_ingestion_run({
                        "asset": ticker,
                        "provider": "exa",
                        "run_type": "news",
                        "requested_start": None,
                        "requested_end": None,
                        "actual_start": str(edf["published_at"].min()) if "published_at" in edf.columns else None,
                        "actual_end": str(edf["published_at"].max()) if "published_at" in edf.columns else None,
                        "record_count": len(edf),
                        "status": "success",
                        "error_message": None,
                    })
            except Exception as e:
                logger.warning("Exa news error: %s", e)

        if news_articles:
            news_df = pd.concat(news_articles, ignore_index=True).drop_duplicates(subset=["url"]).reset_index(drop=True)
            for idx, row in news_df.iterrows():
                text = f"{row.get('title', '')} {row.get('snippet', '')}"
                score, label = analyze_news_sentiment(text)
                news_df.at[idx, "sentiment_score"] = score
                news_df.at[idx, "sentiment_label"] = label

        # Determine overall mode
        if not posts_df.empty and not price_df.empty:
            data_mode = "live"
        elif not price_df.empty:
            data_mode = "partial"
        else:
            data_mode = "none"

    # Save prices to SQLite
    repo.save_price_bars(price_df)

    # Process and save posts if present
    sentiment_agg = pd.DataFrame()
    if not posts_df.empty:
        posts_df, _ = deduplicate_posts(posts_df)
        posts_df = validate_timestamps(posts_df)
        posts_df = filter_empty_text(posts_df)
        processed_posts = process_posts_sentiment(posts_df, use_transformer=False, try_init_transformer=False)
        sentiment_agg = aggregate_sentiment(processed_posts, timeframe, ticker)
        repo.save_raw_posts(posts_df)
        repo.save_processed_posts(processed_posts)
        if not sentiment_agg.empty:
            repo.save_sentiment_aggregates(sentiment_agg)

    # Save news if present
    if not news_df.empty:
        repo.save_news(news_df)

    # Feature engineering (preserves missing-sentiment flags without synthetic leakage)
    logger.info("Building features (data mode: %s)...", data_mode)
    features_df = build_features(price_df, sentiment_agg, timeframe)
    features_df = create_targets(features_df, horizons=[1, 2, 3, 4, 5])
    repo.save_engineered_features(features_df, ticker, timeframe, run_id=run_id)

    # Walk-forward validation
    logger.info("Running expanding-window walk-forward validation (folds=3, gap=horizon+1)...")
    val_results = walk_forward_validate(features_df, horizon=1, n_folds=3)

    # Persist validation metrics with run_id
    val_records = []
    best_model_name = val_results.get("best_model", "Ridge")
    for m_name, m_dict in val_results.get("validation_metrics", {}).items():
        for k, v in m_dict.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                val_records.append({
                    "asset": ticker,
                    "timeframe": timeframe,
                    "model_name": m_name,
                    "horizon": 1,
                    "fold": 0,
                    "metric_name": k,
                    "metric_value": float(v),
                    "dataset_type": "val_avg",
                    "run_id": run_id,
                })
    for k, v in val_results.get("test_metrics", {}).items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            val_records.append({
                "asset": ticker,
                "timeframe": timeframe,
                "model_name": best_model_name,
                "horizon": 1,
                "fold": -1,
                "metric_name": k,
                "metric_value": float(v),
                "dataset_type": "test",
                "run_id": run_id,
            })
    for k, v in val_results.get("naive_test_metrics", {}).items():
        if isinstance(v, (int, float)) and not np.isnan(v):
            val_records.append({
                "asset": ticker,
                "timeframe": timeframe,
                "model_name": "Naive",
                "horizon": 1,
                "fold": -1,
                "metric_name": k,
                "metric_value": float(v),
                "dataset_type": "test_naive",
                "run_id": run_id,
            })
    if val_records:
        repo.save_validation_results(val_records, run_id=run_id)

    # Train direct models for multi-horizon forecast matrix
    feature_cols = get_feature_columns()
    trained_models = {"name": best_model_name}
    trained_scalers = {}
    for h in range(1, 6):
        m, s = train_final_model(features_df, h, model_name=best_model_name, feature_cols=feature_cols)
        if m is not None:
            trained_models[h] = m
            trained_scalers[h] = s

    last_close = float(price_df["close"].iloc[-1])
    last_ts = str(price_df["timestamp_utc"].iloc[-1])
    forecasts = generate_forecasts(features_df, trained_models, trained_scalers, ticker, timeframe, last_close, last_ts)

    # Persist forecasts with run_id
    if forecasts:
        repo.save_forecasts(forecasts, run_id=run_id)

    # Strategy Backtest (10 bps transaction costs)
    logger.info("Running out-of-sample directional strategy backtest (cost: 10 bps)...")
    y_test = val_results.get("y_test")
    y_pred = val_results.get("y_test_pred")
    test_ts = val_results.get("test_timestamps")
    if y_test is not None and y_pred is not None:
        backtest_results = run_backtest(
            predictions=y_pred,
            actual_returns=y_test,
            timestamps=test_ts,
            threshold=0.0,
            transaction_cost_bps=10,
            asset=ticker,
            timeframe=timeframe,
            model_name=best_model_name,
        )
        if "results_df" in backtest_results:
            repo.save_backtest_results(backtest_results["results_df"], run_id=run_id)
    else:
        backtest_results = {}

    # Experiment run summary persistence
    cov = get_price_coverage(price_df)
    sm = backtest_results.get("strategy_metrics", {})
    bm = backtest_results.get("buyhold_metrics", {})
    tm = val_results.get("test_metrics", {})
    nm = val_results.get("naive_test_metrics", {})
    vm = val_results.get("validation_metrics", {}).get(best_model_name, {})

    post_coverage = {
        "count": len(posts_df),
        "date_range": f"{posts_df['timestamp_utc'].min()} to {posts_df['timestamp_utc'].max()}" if not posts_df.empty else "N/A (No posts collected)",
    }
    price_coverage = {
        "source": cov["source"],
        "count": cov["count"],
        "date_range": cov["date_range"],
    }
    feature_info = {"n_features": len(get_feature_columns())}

    provenance_metadata = {
        "run_id": run_id,
        "price_source": price_source,
        "price_count": len(price_df),
        "price_date_range": cov["date_range"],
        "apify_status": apify_status_msg,
        "post_count": len(posts_df),
        "post_date_range": post_coverage["date_range"],
        "news_count": len(news_df),
        "synthetic_posts_used": False if not use_demo else True,
        "database": str(DATABASE_PATH),
    }

    repo.save_experiment_run({
        "run_id": run_id,
        "asset": ticker,
        "timeframe": timeframe,
        "data_mode": data_mode,
        "price_provider": price_source,
        "price_count": len(price_df),
        "price_date_range": cov["date_range"],
        "post_provider": "apify",
        "post_count": len(posts_df),
        "post_date_range": post_coverage["date_range"],
        "news_count": len(news_df),
        "best_model": best_model_name,
        "validation_mae": vm.get("mae"),
        "test_mae": tm.get("mae"),
        "naive_test_mae": nm.get("mae"),
        "strategy_return": sm.get("cumulative_return"),
        "buyhold_return": bm.get("cumulative_return"),
        "strategy_sharpe": sm.get("sharpe_ratio"),
        "max_drawdown": sm.get("max_drawdown"),
        "status": "completed",
        "notes": f"Provenance: {price_source} ({cov['date_range']}), Apify: {apify_status_msg}, News: {len(news_df)} articles",
    })

    # Generate Reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating sentiment outlook report...")
    generate_sentiment_outlook(
        asset=ticker,
        price_df=price_df,
        sentiment_df=sentiment_agg,
        news_df=news_df,
        forecasts=forecasts,
        data_source=data_mode,
        output_dir=str(reports_dir),
        run_id=run_id,
        provenance=provenance_metadata,
    )

    logger.info("Generating evaluation report...")
    generate_evaluation_report(
        asset=ticker,
        timeframe=timeframe,
        data_source=data_mode,
        price_coverage=price_coverage,
        post_coverage=post_coverage,
        feature_info=feature_info,
        validation_results=val_results,
        backtest_results=backtest_results,
        output_dir=str(reports_dir),
        run_id=run_id,
        provenance=provenance_metadata,
    )

    logger.info("Pipeline completed. Run ID: %s. Data Mode: %s. Reports saved in %s", run_id, data_mode.upper(), reports_dir)
    print("\n" + "=" * 70)
    print(f" PIPELINE COMPLETE — RUN ID: {run_id} | MODE: {data_mode.upper()}")
    print("=" * 70)
    print(f"* Asset: {ticker} ({timeframe})")
    print(f"* Price Observations: {len(price_df)} ({price_coverage['date_range']}) [Source: {price_source}]")
    print(f"* X Posts Collected: {len(posts_df)} [{apify_status_msg}]")
    print(f"* News Articles Collected: {len(news_df)} (Tavily + Exa)")
    print(f"* Selected Model: {best_model_name} (Val MAE: {vm.get('mae', 'N/A')})")
    print(f"* Out-of-Sample Test MAE: {tm.get('mae', 'N/A')} (Naive Baseline: {nm.get('mae', 'N/A')})")
    print(f"* Strategy Return: {sm.get('cumulative_return', 'N/A')} vs Buy & Hold: {bm.get('cumulative_return', 'N/A')}")
    print(f"* Strategy Sharpe: {sm.get('sharpe_ratio', 'N/A')}, Max Drawdown: {sm.get('max_drawdown', 'N/A')}")
    print(f"* Reports Generated:")
    print(f"  - Outlook:    {reports_dir / 'current_sentiment_outlook.md'}")
    print(f"  - Evaluation: {reports_dir / 'evaluation_report.md'}")
    print("=" * 70)

    return {
        "run_id": run_id,
        "data_mode": data_mode,
        "price_df": price_df,
        "posts_df": posts_df,
        "news_df": news_df,
        "forecasts": forecasts,
        "val_results": val_results,
        "backtest_results": backtest_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CrowdWisdomTrading pipeline and generate reports.")
    parser.add_argument("--asset", default="BTCUSDT", help="Asset ticker (default: BTCUSDT)")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h"], help="Timeframe (default: 1d)")
    parser.add_argument("--days", type=int, default=180, help="Lookback days (default: 180)")
    parser.add_argument("--demo", action="store_true", help="Explicitly run in synthetic demo mode")
    parser.add_argument("--run-id", default=None, help="Custom run identifier")
    args = parser.parse_args()

    run_pipeline(ticker=args.asset, timeframe=args.timeframe, days=args.days, use_demo=args.demo, run_id=args.run_id)
