"""CLI script to run CrowdWisdomTrading pipeline and generate reports.

Supports:
- Live mode (default): Uses real market prices from Binance/Yahoo, checks Apify Actor for live posts, retrieves news from Tavily/Exa.
  If Apify exceeds its monthly usage limit, enters PARTIAL DATA mode with explicit missing-sentiment flags without fabricating data.
- Historical X mode (--use-historical-x or --csv-posts): Imports genuine historical X posts (e.g. from StephanAkkerman/financial-tweets or user CSV)
  and aligns prices to the exact historical period for rigorous empirical validation.
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
from src.ingestion.dataset_importer import (
    import_historical_x_csv,
    extract_huggingface_posts,
    validate_date_coverage,
)
from src.ingestion.tavily_client import is_available as tavily_available, search_news as tavily_search
from src.ingestion.exa_client import is_available as exa_available, search_news as exa_search
from src.utils.demo_data import generate_demo_prices, generate_demo_posts, generate_demo_news
from src.processing.data_cleaning import deduplicate_posts, validate_timestamps, filter_empty_text
from src.processing.sentiment import process_posts_sentiment, analyze_news_sentiment
from src.processing.aggregation import aggregate_sentiment
from src.processing.feature_engineering import build_features, create_targets, get_feature_columns
from src.modeling.train import walk_forward_validate, train_final_model, run_experiment_suite
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
    csv_posts: str = None,
    use_historical_x: bool = False,
    run_id: str = None,
):
    """Execute the end-to-end forecasting pipeline with provenance, fair comparison, and audit tracking."""
    run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info("Starting experiment %s for %s (%s)...", run_id, ticker, timeframe)

    init_db(DATABASE_PATH)
    repo = Repository(DATABASE_PATH)

    asset_cfg = get_asset_by_ticker(ticker) or {
        "ticker": ticker,
        "name": ticker,
        "asset_class": "crypto" if "BTC" in ticker or "USDT" in ticker else "us_equities",
        "provider": "binance" if "BTC" in ticker else "yfinance",
        "search_terms": [ticker],
    }

    price_df = pd.DataFrame()
    posts_df = pd.DataFrame()
    news_df = pd.DataFrame()
    post_source = "apify"
    apify_status_msg = "Not attempted"
    coverage_diagnostics = {}

    if use_demo:
        logger.info("EXPLICIT DEMO MODE: Generating labeled synthetic data...")
        data_mode = "demo"
        price_df = generate_demo_prices(ticker, timeframe, n_bars=days, base_price=65000.0 if "BTC" in ticker else 120.0)
        posts_df = generate_demo_posts(ticker, n_posts=150)
        news_df = generate_demo_news(ticker, n_articles=10)
        price_source = "demo"
        post_source = "demo"
        apify_status_msg = "Demo synthetic fixtures"

    elif csv_posts:
        logger.info("HISTORICAL CSV MODE: Importing X posts from %s...", csv_posts)
        posts_df = import_historical_x_csv(csv_posts, asset=ticker, source_label="csv_import")
        post_source = "csv_import"
        apify_status_msg = f"Imported CSV ({len(posts_df)} posts)"

        # Fetch matching price history if possible or standard lookback
        price_df = fetch_price_data(asset_cfg, timeframe, days=days)
        price_source = price_df["source"].iloc[0] if "source" in price_df.columns else "binance"

    elif use_historical_x:
        logger.info("HISTORICAL X DATASET MODE: Using StephanAkkerman/financial-tweets (MIT)...")
        hf_parquet = PROJECT_ROOT / "data" / "hf_sample.parquet"
        if hf_parquet.exists():
            posts_df = extract_huggingface_posts(hf_parquet, asset=ticker, max_posts=5000)
            post_source = "huggingface_stephanakkerman"
            apify_status_msg = f"HuggingFace dataset ({len(posts_df)} posts)"
        else:
            logger.warning("Local parquet %s not found. Falling back to live mode.", hf_parquet)
            posts_df = pd.DataFrame()

        # Align prices to the historical dataset date window (Nov 2023 - May 2024)
        if not posts_df.empty:
            t_min = str(posts_df["timestamp_utc"].min())[:10]
            t_max = str(posts_df["timestamp_utc"].max())[:10]
            logger.info("Fetching price data aligned to historical posts: %s to %s...", t_min, t_max)
            price_df = fetch_price_data(asset_cfg, timeframe, start_date=t_min, end_date=t_max)
        else:
            price_df = fetch_price_data(asset_cfg, timeframe, days=days)

        price_source = price_df["source"].iloc[0] if "source" in price_df.columns else "binance"

    else:
        # Standard Live Mode: Fetch market prices
        logger.info("LIVE MODE: Fetching real market price data for %s (%s, lookback %d days)...", ticker, timeframe, days)
        try:
            price_df = fetch_price_data(asset_cfg, timeframe, days=days)
        except Exception as e:
            logger.error("Price fetch error: %s", e)

        if price_df.empty:
            logger.error("Could not retrieve live price data for %s.", ticker)
            print("ERROR: Could not retrieve live prices. Pass --demo to explicitly use demo fixtures.")
            return None

        price_source = price_df["source"].iloc[0] if "source" in price_df.columns else "binance"
        logger.info("Successfully fetched %d price bars from %s (%s to %s)",
                    len(price_df), price_source, price_df["timestamp_utc"].min(), price_df["timestamp_utc"].max())

        # Collect live X posts via Apify (small test run first)
        logger.info("Testing Apify Actor availability with a small sample test run...")
        search_terms = asset_cfg.get("search_terms", [ticker])
        test_query = asset_cfg.get("name") or (search_terms[0] if search_terms else ticker)
        test_run = test_apify_connection(query=test_query, max_items=2)

        if test_run["success"]:
            logger.info("Apify test run successful (%d items). Collecting live posts...", test_run["items_retrieved"])
            selected_terms = [t for t in search_terms if len(t) > 2][:3] or [test_query]
            posts_df = collect_posts(ticker, selected_terms, max_items=50)
            apify_status_msg = f"Live retrieved ({len(posts_df)} posts)"
            post_source = "apify"
        else:
            apify_err = test_run.get("error", "Unknown error")
            apify_status_msg = f"Quota/Access restriction: {apify_err}"
            logger.warning("Apify Actor test failed: %s", apify_err)
            logger.info("Operating in PARTIAL DATA mode without fabricating synthetic posts.")
            posts_df = pd.DataFrame()
            post_source = "none (quota exceeded)"

        # Collect News via Tavily and Exa
        news_articles = []
        if tavily_available():
            try:
                tdf = tavily_search(ticker, search_terms, max_results=10)
                if not tdf.empty:
                    news_articles.append(tdf)
            except Exception as e:
                logger.warning("Tavily news error: %s", e)

        if exa_available():
            try:
                edf = exa_search(ticker, search_terms, max_results=10)
                if not edf.empty:
                    news_articles.append(edf)
            except Exception as e:
                logger.warning("Exa news error: %s", e)

        if news_articles:
            news_df = pd.concat(news_articles, ignore_index=True).drop_duplicates(subset=["url"]).reset_index(drop=True)
            for idx, row in news_df.iterrows():
                text = f"{row.get('title', '')} {row.get('snippet', '')}"
                score, label = analyze_news_sentiment(text)
                news_df.at[idx, "sentiment_score"] = score
                news_df.at[idx, "sentiment_label"] = label

    # Validate temporal coverage and date overlap
    coverage_diagnostics = validate_date_coverage(price_df, posts_df)
    if coverage_diagnostics.get("warning"):
        logger.warning("Coverage check: %s", coverage_diagnostics["warning"])

    # Determine overall data mode
    if use_demo:
        data_mode = "demo"
    elif not posts_df.empty and not price_df.empty and coverage_diagnostics.get("has_adequate_overlap", False):
        data_mode = "live"
    elif not price_df.empty:
        data_mode = "partial"
    else:
        data_mode = "none"

    # Persist price bars to SQLite
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

    # Feature engineering
    logger.info("Building features (data mode: %s)...", data_mode)
    features_df = build_features(price_df, sentiment_agg, timeframe, news_df=news_df)
    features_df = create_targets(features_df, horizons=[1, 2, 3, 4, 5])
    repo.save_engineered_features(features_df, ticker, timeframe, run_id=run_id)

    # Run multi-variant experiment comparison suite across all horizons (Priority 2)
    has_real_sentiment = not sentiment_agg.empty and coverage_diagnostics.get("has_adequate_overlap", False)
    has_eng = bool(has_real_sentiment and "engagement_weighted_sentiment" in sentiment_agg.columns and sentiment_agg["engagement_weighted_sentiment"].notna().any())
    has_nws = bool(not news_df.empty and "news_mean_sentiment" in features_df.columns and features_df["news_mean_sentiment"].notna().any())

    logger.info("Executing fair model comparison suite across horizons 1..5...")
    exp_suite = run_experiment_suite(
        features_df,
        horizons=[1, 2, 3, 4, 5],
        n_folds=3,
        test_ratio=0.2,
        asset=ticker,
        timeframe=timeframe,
        transaction_cost_bps=10.0,
        has_real_sentiment=has_real_sentiment,
        has_engagement=has_eng,
        has_news=has_nws,
    )
    comparison_df = exp_suite["comparison_df"]
    if not comparison_df.empty:
        repo.save_experiment_evaluations(comparison_df, run_id=run_id)

    # Extract primary model results for horizon 1
    h1_results = exp_suite["horizon_results"].get(1, {})
    val_results = h1_results.get("res_c") or walk_forward_validate(features_df, horizon=1, n_folds=3)
    best_model_name = val_results.get("best_model", "Ridge")

    # Train direct production models for multi-horizon forecast matrix
    feature_cols = get_feature_columns(mode="price_sentiment" if has_real_sentiment else "price_only")
    trained_models = {"name": best_model_name}
    trained_scalers = {}
    model_names_map = {}
    for h in range(1, 6):
        m, s = train_final_model(features_df, h, model_name=best_model_name, feature_cols=feature_cols)
        if m is not None:
            trained_models[h] = m
            trained_scalers[h] = s
            model_names_map[h] = best_model_name

    last_close = float(price_df["close"].iloc[-1])
    last_ts = str(price_df["timestamp_utc"].iloc[-1])
    forecasts = generate_forecasts(
        features_df,
        trained_models,
        trained_scalers,
        ticker,
        timeframe,
        last_close,
        last_ts,
        asset_class=asset_cfg.get("asset_class"),
        data_mode=data_mode,
        model_names=model_names_map,
    )
    if forecasts:
        repo.save_forecasts(forecasts, run_id=run_id)

    # Strategy Backtest for primary model (horizon 1)
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

    # Persist summary experiment run record
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
        "post_source": post_source,
        "apify_status": apify_status_msg,
        "post_count": len(posts_df),
        "post_date_range": post_coverage["date_range"],
        "news_count": len(news_df),
        "synthetic_posts_used": bool(use_demo),
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
        "post_provider": post_source,
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
        "notes": f"Provenance: {price_source} ({cov['date_range']}), Post provider: {post_source} ({apify_status_msg})",
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
        experiment_comparison_df=comparison_df,
        coverage_diagnostics=coverage_diagnostics,
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
    print(f"* Multi-Horizon Experiment Suite: Evaluated {len(comparison_df)} model variants across horizons 1..5")
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
        "comparison_df": comparison_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CrowdWisdomTrading pipeline and generate reports.")
    parser.add_argument("--asset", default="BTCUSDT", help="Asset ticker (default: BTCUSDT)")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h"], help="Timeframe (default: 1d)")
    parser.add_argument("--days", type=int, default=180, help="Lookback days (default: 180)")
    parser.add_argument("--demo", action="store_true", help="Explicitly run in synthetic demo mode")
    parser.add_argument("--csv-posts", default=None, help="Path to historical X posts CSV file")
    parser.add_argument("--use-historical-x", action="store_true", help="Use genuine historical X dataset (StephanAkkerman/financial-tweets)")
    parser.add_argument("--run-id", default=None, help="Custom run identifier")
    args = parser.parse_args()

    run_pipeline(
        ticker=args.asset,
        timeframe=args.timeframe,
        days=args.days,
        use_demo=args.demo,
        csv_posts=args.csv_posts,
        use_historical_x=args.use_historical_x,
        run_id=args.run_id,
    )
