"""CrowdWisdomTrading — Streamlit Dashboard Application.

A quantitative research tool for predicting asset prices using
X (Twitter) sentiment and historical price data.
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone

from config.settings import (
    load_asset_config, get_all_assets, get_asset_by_ticker,
    api_status, DATABASE_PATH, DEFAULT_ASSET, DEFAULT_TIMEFRAME,
    HISTORY_DAYS, TRANSACTION_COST_BPS, RANDOM_STATE,
)
from src.database.db import init_db
from src.database.repository import Repository
from src.utils.logging_config import logger
from src.utils.time_utils import utc_now, format_utc
from src.utils.demo_data import generate_demo_prices, generate_demo_posts, generate_demo_news
from src.processing.data_cleaning import (
    deduplicate_posts, validate_timestamps, filter_empty_text,
    normalize_price_data, data_quality_report,
)
from src.processing.sentiment import process_posts_sentiment, analyze_news_sentiment
from src.processing.aggregation import aggregate_sentiment
from src.processing.feature_engineering import build_features, create_targets, get_feature_columns
from src.modeling.train import walk_forward_validate, train_final_model
from src.modeling.forecasting import generate_forecasts
from src.modeling.backtesting import run_backtest
from src.modeling.metrics import calculate_all_metrics
from src.reporting.sentiment_report import generate_sentiment_outlook
from src.reporting.evaluation_report import generate_evaluation_report

# ─── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrowdWisdomTrading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Initialize ──────────────────────────────────────────────────────────
init_db(DATABASE_PATH)
repo = Repository(DATABASE_PATH)

# ─── Session State ───────────────────────────────────────────────────────
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}
if "price_df" not in st.session_state:
    st.session_state.price_df = pd.DataFrame()
if "posts_df" not in st.session_state:
    st.session_state.posts_df = pd.DataFrame()
if "processed_posts" not in st.session_state:
    st.session_state.processed_posts = pd.DataFrame()
if "sentiment_agg" not in st.session_state:
    st.session_state.sentiment_agg = pd.DataFrame()
if "features_df" not in st.session_state:
    st.session_state.features_df = pd.DataFrame()
if "validation_results" not in st.session_state:
    st.session_state.validation_results = {}
if "forecasts" not in st.session_state:
    st.session_state.forecasts = []
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = {}
if "news_df" not in st.session_state:
    st.session_state.news_df = pd.DataFrame()
if "data_source" not in st.session_state:
    st.session_state.data_source = ""
if "ingestion_errors" not in st.session_state:
    st.session_state.ingestion_errors = {}


def get_data_mode() -> str:
    """Return explicit data mode label."""
    if st.session_state.get("demo_mode", False):
        return "DEMO DATA"
    has_price = not st.session_state.price_df.empty
    has_posts = not st.session_state.posts_df.empty
    if has_price and has_posts:
        return "LIVE DATA"
    elif has_price or has_posts:
        return "PARTIAL DATA"
    return "NO DATA"

# ─── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 CrowdWisdomTrading")
    st.caption("X Sentiment Price Prediction")

    # Asset selection
    assets = get_all_assets()
    asset_classes = {}
    for a in assets:
        cls = a["asset_class"].replace("_", " ").title()
        if cls not in asset_classes:
            asset_classes[cls] = []
        asset_classes[cls].append(a)

    selected_class = st.selectbox("Asset Class", list(asset_classes.keys()))
    class_assets = asset_classes[selected_class]
    asset_options = {f"{a['name']} ({a['ticker']})": a['ticker'] for a in class_assets}
    selected_label = st.selectbox("Asset", list(asset_options.keys()))
    selected_ticker = asset_options[selected_label]
    asset_config = get_asset_by_ticker(selected_ticker)

    # Timeframe
    timeframe = st.selectbox("Timeframe", ["1d", "1h"], format_func=lambda x: "Daily" if x == "1d" else "Hourly")

    # Settings
    st.divider()
    history_days = st.number_input("History (days)", 30, 365, HISTORY_DAYS)
    transaction_cost = st.number_input("Transaction cost (bps)", 0, 100, TRANSACTION_COST_BPS)

    # API & Provider Status
    st.divider()
    st.subheader("Provider & Auth Status")
    st.caption("🟢 Binance: Public Price Feed")
    st.caption("🟡 Yahoo Finance: Public Price Feed")

    from src.ingestion.apify_client import verify_credentials as verify_apify
    ap_stat = verify_apify()
    if ap_stat["valid"]:
        st.caption(f"✅ Apify X: Active (@{ap_stat.get('username')})")
    elif ap_stat["configured"]:
        st.caption(f"❌ Apify X: Auth Failed (401)")
    else:
        st.caption("⚠️ Apify X: Token Missing (.env)")

    status = api_status()
    st.caption(f"{'✅' if status['tavily'] else '⚠️'} Tavily News: {'Active' if status['tavily'] else 'Key Missing'}")
    st.caption(f"{'✅' if status['exa'] else '⚠️'} Exa Neural: {'Active' if status['exa'] else 'Key Missing'}")

    st.divider()
    # Data Mode Badge
    current_mode = get_data_mode()
    if current_mode == "LIVE DATA":
        st.success("Mode: 🟢 LIVE DATA")
    elif current_mode == "PARTIAL DATA":
        st.warning("Mode: 🟡 PARTIAL DATA")
    elif current_mode == "DEMO DATA":
        st.info("Mode: 🔵 DEMO DATA")
    else:
        st.caption("Mode: ⚪ NO DATA LOADED")

    st.divider()

    # Action buttons
    if st.button("🔄 Start Ingestion", use_container_width=True):
        _run_ingestion(selected_ticker, asset_config, timeframe, history_days)

    if st.button("🧪 Use Demo Data", use_container_width=True):
        _run_demo(selected_ticker, timeframe, history_days)

    if st.button("🤖 Train & Validate", use_container_width=True):
        _run_training(selected_ticker, timeframe, transaction_cost)

    if st.button("📈 Generate Forecasts", use_container_width=True):
        _run_forecasting(selected_ticker, timeframe)

    if st.button("📄 Generate Reports", use_container_width=True):
        _run_reporting(selected_ticker, timeframe)


# ─── Helper Functions ────────────────────────────────────────────────────
def _run_ingestion(ticker, config, timeframe, days):
    """Run data ingestion pipeline."""
    st.session_state.ingestion_errors = {}

    with st.spinner("Fetching price data..."):
        from src.ingestion.price_providers import fetch_price_data
        try:
            price_df = fetch_price_data(config, timeframe, days)
            if price_df.empty:
                st.session_state.ingestion_errors["price"] = "Price provider returned 0 bars. Existing saved price data preserved."
            else:
                price_df = normalize_price_data(price_df)
                repo.save_price_bars(price_df)
                repo.log_ingestion_run({
                    "asset": ticker, "provider": config.get("provider", "yfinance"),
                    "run_type": "price", "record_count": len(price_df),
                    "actual_start": price_df["timestamp_utc"].min(),
                    "actual_end": price_df["timestamp_utc"].max(),
                    "status": "success",
                })
                st.session_state.price_df = price_df
                st.session_state.demo_mode = False
        except Exception as e:
            st.session_state.ingestion_errors["price"] = f"Price fetch failed: {str(e)}. Existing price data preserved."

    # X posts
    with st.spinner("Fetching X posts..."):
        from src.ingestion.apify_client import is_available as apify_available, collect_posts, verify_credentials as verify_apify
        if apify_available():
            ap_cred = verify_apify()
            if not ap_cred["valid"]:
                st.session_state.ingestion_errors["apify"] = f"Apify auth failed: {ap_cred['error']}. Social posts not collected."
            else:
                try:
                    posts_df = collect_posts(ticker, config.get("search_terms", [ticker]))
                    if not posts_df.empty:
                        posts_df, dup_count = deduplicate_posts(posts_df)
                        posts_df = validate_timestamps(posts_df)
                        posts_df = filter_empty_text(posts_df)
                        repo.save_raw_posts(posts_df)
                        st.session_state.posts_df = posts_df
                    else:
                        st.session_state.ingestion_errors["apify"] = "Apify returned 0 posts. Existing post data preserved."
                except Exception as e:
                    st.session_state.ingestion_errors["apify"] = f"Apify collection error: {str(e)}"
        else:
            st.session_state.ingestion_errors["apify"] = "Apify API token not configured in .env. Live X posts skipped (No synthetic posts fabricated)."

    # News
    with st.spinner("Fetching news..."):
        news_dfs = []
        from src.ingestion.tavily_client import is_available as tavily_avail, search_news as tavily_search
        from src.ingestion.exa_client import is_available as exa_avail, search_news as exa_search

        search_terms = config.get("search_terms", [ticker])
        if tavily_avail():
            try:
                tav_df = tavily_search(ticker, search_terms)
                if not tav_df.empty:
                    news_dfs.append(tav_df)
            except Exception as e:
                st.session_state.ingestion_errors["tavily"] = f"Tavily error: {str(e)}"
        if exa_avail():
            try:
                exa_df = exa_search(ticker, search_terms)
                if not exa_df.empty:
                    news_dfs.append(exa_df)
            except Exception as e:
                st.session_state.ingestion_errors["exa"] = f"Exa error: {str(e)}"

        if news_dfs:
            news_df = pd.concat(news_dfs, ignore_index=True).drop_duplicates(subset=["url"])
            for idx, row in news_df.iterrows():
                text = f"{row.get('title', '')} {row.get('snippet', '')}"
                score, label = analyze_news_sentiment(text)
                news_df.at[idx, "sentiment_score"] = score
                news_df.at[idx, "sentiment_label"] = label
            repo.save_news(news_df)
            st.session_state.news_df = news_df

    # Process sentiment
    if not st.session_state.posts_df.empty:
        with st.spinner("Processing sentiment..."):
            processed = process_posts_sentiment(st.session_state.posts_df, use_transformer=False)
            repo.save_processed_posts(processed)
            st.session_state.processed_posts = processed

            agg = aggregate_sentiment(processed, timeframe, ticker)
            if not agg.empty:
                repo.save_sentiment_aggregates(agg)
                st.session_state.sentiment_agg = agg

    st.session_state.data_source = "live" if (not st.session_state.price_df.empty and not st.session_state.posts_df.empty) else "partial"
    st.rerun()


def _run_demo(ticker, timeframe, days):
    """Load demo data."""
    # Determine base price by asset class
    config = get_asset_by_ticker(ticker) or {}
    base_prices = {
        "BTCUSDT": 50000, "ETHUSDT": 3000, "SOLUSDT": 100,
        "NVDA": 500, "TSLA": 250, "AAPL": 180, "MSFT": 380,
        "RELIANCE.NS": 2500, "TCS.NS": 3800, "INFY.NS": 1500,
        "EURUSD=X": 1.08, "GBPUSD=X": 1.27,
        "GC=F": 2000, "CL=F": 75,
    }
    bp = base_prices.get(ticker, 100)

    price_df = generate_demo_prices(ticker, timeframe, days, bp)
    posts_df = generate_demo_posts(ticker, 500, days)
    news_df = generate_demo_news(ticker)

    # Process
    processed = process_posts_sentiment(posts_df, use_transformer=False)
    agg = aggregate_sentiment(processed, timeframe, ticker)

    # Store
    st.session_state.price_df = price_df
    st.session_state.posts_df = posts_df
    st.session_state.processed_posts = processed
    st.session_state.sentiment_agg = agg
    st.session_state.news_df = news_df
    st.session_state.demo_mode = True
    st.session_state.data_source = "demo"

    # Persist to DB
    repo.save_price_bars(price_df)
    if not processed.empty:
        repo.save_processed_posts(processed)
    if not agg.empty:
        repo.save_sentiment_aggregates(agg)

    st.success("Demo data loaded successfully")
    st.rerun()


def _run_training(ticker, timeframe, tc_bps):
    """Run training and validation pipeline."""
    price_df = st.session_state.price_df
    sentiment_agg = st.session_state.sentiment_agg

    if price_df.empty:
        st.error("No price data. Run ingestion or load demo data first.")
        return

    with st.spinner("Building features..."):
        features = build_features(price_df, sentiment_agg, timeframe)
        features = create_targets(features, horizons=[0, 1, 2, 3, 4, 5])
        st.session_state.features_df = features

    # Validate for horizon 1 (primary)
    with st.spinner("Running walk-forward validation..."):
        results = walk_forward_validate(features, horizon=1, n_folds=5)

        if "error" in results:
            st.error(f"Validation error: {results['error']} ({results.get('n_obs', 0)} observations)")
            return

        st.session_state.validation_results = results

    # Run backtest
    with st.spinner("Running backtest..."):
        if results.get("y_test") is not None and len(results["y_test"]) > 5:
            backtest = run_backtest(
                predictions=results["y_test_pred"],
                actual_returns=results["y_test"],
                timestamps=results.get("test_timestamps"),
                threshold=0.0,
                transaction_cost_bps=tc_bps,
                asset=ticker,
                timeframe=timeframe,
                model_name=results.get("best_model", ""),
            )
            st.session_state.backtest_results = backtest
        else:
            st.warning("Insufficient test data for backtest")

    st.success(f"Training complete. Best model: {results.get('best_model', 'N/A')}")
    st.rerun()


def _run_forecasting(ticker, timeframe):
    """Generate latest forecasts."""
    features = st.session_state.features_df
    vr = st.session_state.validation_results

    if features.empty:
        st.error("No features. Run training first.")
        return

    best_name = vr.get("best_model", "Ridge")
    feature_cols = vr.get("feature_cols", get_feature_columns())

    with st.spinner("Generating forecasts..."):
        models = {}
        scalers = {}
        for h in range(1, 6):
            model, scaler = train_final_model(features, h, best_name, feature_cols)
            if model is not None:
                models[h] = model
                scalers[h] = scaler
        models["name"] = best_name

        ref_price = float(features["close"].iloc[-1])
        ref_ts = str(features["timestamp_utc"].iloc[-1])

        forecasts = generate_forecasts(features, models, scalers, ticker, timeframe,
                                        ref_price, ref_ts)
        st.session_state.forecasts = forecasts

        # Save to DB
        if forecasts:
            repo.save_forecasts(forecasts)

    st.success(f"Generated {len(forecasts)} forecasts")
    st.rerun()


def _run_reporting(ticker, timeframe):
    """Generate reports."""
    with st.spinner("Generating reports..."):
        price_df = st.session_state.price_df
        sentiment_agg = st.session_state.sentiment_agg
        news_df = st.session_state.news_df
        forecasts = st.session_state.forecasts
        vr = st.session_state.validation_results
        bt = st.session_state.backtest_results
        data_source = st.session_state.data_source or "demo"

        # Sentiment outlook
        generate_sentiment_outlook(
            ticker, price_df, sentiment_agg, news_df, forecasts,
            data_source=data_source,
        )

        # Evaluation report
        price_coverage = {
            "source": price_df["source"].iloc[0] if not price_df.empty and "source" in price_df.columns else "N/A",
            "count": len(price_df),
            "date_range": f"{price_df['timestamp_utc'].min()} to {price_df['timestamp_utc'].max()}" if not price_df.empty else "N/A",
        }
        post_coverage = {
            "count": len(st.session_state.posts_df),
            "date_range": (
                f"{st.session_state.posts_df['timestamp_utc'].min()} to {st.session_state.posts_df['timestamp_utc'].max()}"
                if not st.session_state.posts_df.empty else "N/A"
            ),
        }
        feature_info = {
            "n_features": len(get_feature_columns()),
        }

        generate_evaluation_report(
            asset=ticker,
            timeframe=timeframe,
            data_source=data_source,
            price_coverage=price_coverage,
            post_coverage=post_coverage,
            feature_info=feature_info,
            validation_results=vr,
            backtest_results=bt,
        )

    st.success("Reports generated in reports/ directory")


# ─── Data Status Banner ──────────────────────────────────────────────────
current_mode = get_data_mode()
if current_mode == "LIVE DATA":
    st.success("🟢 **LIVE DATA MODE ACTIVE** — Verified live market prices and verified live X post collections.")
elif current_mode == "PARTIAL DATA":
    source_name = st.session_state.price_df.get("source", pd.Series(["live"])).iloc[0] if not st.session_state.price_df.empty else "live"
    st.warning(
        f"🟡 **PARTIAL DATA MODE ACTIVE** — Market prices are LIVE from **{source_name.upper()}** "
        f"({len(st.session_state.price_df)} observations), but live social sentiment is pending/unconfigured. "
        "Features are computed using missing-sentiment flags. Synthetic fixtures are **NOT** silently fabricated."
    )
elif current_mode == "DEMO DATA":
    st.info("🔵 **DEMO MODE ACTIVE** — Displaying synthetic demonstration data for evaluation. All records are fixtures.")

if st.session_state.get("ingestion_errors"):
    for prov_key, err_val in st.session_state.ingestion_errors.items():
        st.caption(f"⚠️ **{prov_key.upper()} Provider Notice**: {err_val}")

# ─── Main Content ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "📈 Sentiment & Market", "🎯 Forecast Matrix",
    "🔬 Model Evaluation", "💰 Trading Strategy", "📰 News & Sentiment",
    "📋 Data & Logs",
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Executive Overview
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.demo_mode:
        st.warning("⚠️ DEMO MODE — Displaying synthetic data. Not verified live data.")

    st.header(f"Executive Overview — {selected_label}")

    price_df = st.session_state.price_df
    if price_df.empty:
        st.info("No data loaded. Click **Start Ingestion** or **Use Demo Data** in the sidebar.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        last_close = float(price_df["close"].iloc[-1])
        prev_close = float(price_df["close"].iloc[-2]) if len(price_df) > 1 else last_close
        change_pct = (last_close - prev_close) / prev_close * 100

        with col1:
            st.metric("Last Price", f"${last_close:,.2f}", f"{change_pct:+.2f}%")
        with col2:
            ts = price_df["timestamp_utc"].iloc[-1]
            st.metric("Data Timestamp", str(ts)[:16])
        with col3:
            sent_agg = st.session_state.sentiment_agg
            if not sent_agg.empty and "mean_sentiment" in sent_agg.columns:
                latest_sent = sent_agg["mean_sentiment"].iloc[-1]
                label = "Bullish" if latest_sent > 0.05 else ("Bearish" if latest_sent < -0.05 else "Neutral")
                st.metric("Sentiment", label, f"{latest_sent:.4f}")
            else:
                st.metric("Sentiment", "N/A")
        with col4:
            n_posts = len(st.session_state.processed_posts)
            st.metric("X Posts", f"{n_posts:,}")

        # Sentiment breakdown
        if not sent_agg.empty:
            col5, col6, col7 = st.columns(3)
            latest = sent_agg.iloc[-1]
            with col5:
                st.metric("Positive", f"{latest.get('positive_ratio', 0)*100:.1f}%")
            with col6:
                st.metric("Negative", f"{latest.get('negative_ratio', 0)*100:.1f}%")
            with col7:
                st.metric("Neutral", f"{latest.get('neutral_ratio', 0)*100:.1f}%")

        # Quick price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(price_df["timestamp_utc"]),
            y=price_df["close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#2196F3", width=2),
        ))
        fig.update_layout(
            title=f"{selected_ticker} Price History",
            xaxis_title="Date", yaxis_title="Price",
            template="plotly_white", height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Model info
        vr = st.session_state.validation_results
        if vr and "best_model" in vr:
            st.info(f"**Selected Model:** {vr['best_model']} | "
                    f"**Test MAE:** {vr.get('test_metrics', {}).get('mae', 'N/A'):.6f}")

        # Forecasts summary
        if st.session_state.forecasts:
            st.subheader("Latest Forecast")
            fc = st.session_state.forecasts
            for f in fc[:3]:
                h = f.get("horizon", "?")
                pp = f.get("predicted_price")
                if pp:
                    st.caption(f"n+{h}: ${pp:,.2f}")

        # News
        if not st.session_state.news_df.empty:
            st.subheader("Latest News")
            for _, row in st.session_state.news_df.head(3).iterrows():
                url = row.get("url", "")
                title = row.get("title", "")
                if url:
                    st.caption(f"📰 [{title}]({url})")
                else:
                    st.caption(f"📰 {title}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Sentiment & Market Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Sentiment & Market Analysis")
    price_df = st.session_state.price_df
    sent_agg = st.session_state.sentiment_agg

    if price_df.empty:
        st.info("Load data first.")
    else:
        # Main dual-axis chart
        fig = go.Figure()
        price_dates = pd.to_datetime(price_df["timestamp_utc"])
        fig.add_trace(go.Scatter(
            x=price_dates, y=price_df["close"],
            name="Close Price", line=dict(color="#2196F3", width=2),
            yaxis="y1",
        ))

        if not sent_agg.empty and "mean_sentiment" in sent_agg.columns:
            sent_dates = pd.to_datetime(sent_agg["period_start"])
            fig.add_trace(go.Bar(
                x=sent_dates, y=sent_agg["mean_sentiment"],
                name="Mean Sentiment", opacity=0.5,
                marker_color=sent_agg["mean_sentiment"].apply(
                    lambda x: "#4CAF50" if x > 0 else "#F44336"
                ),
                yaxis="y2",
            ))

        fig.update_layout(
            title="Price vs X Sentiment",
            xaxis_title="Date",
            yaxis=dict(title="Price", side="left"),
            yaxis2=dict(title="Sentiment", side="right", overlaying="y"),
            template="plotly_white", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Supplementary charts
        if not sent_agg.empty:
            col1, col2 = st.columns(2)

            with col1:
                # Post volume
                if "post_count" in sent_agg.columns:
                    fig_vol = px.bar(
                        sent_agg, x="period_start", y="post_count",
                        title="Post Volume Over Time",
                    )
                    fig_vol.update_layout(template="plotly_white", height=300)
                    st.plotly_chart(fig_vol, use_container_width=True)

            with col2:
                # Sentiment distribution
                processed = st.session_state.processed_posts
                if not processed.empty and "final_sentiment_score" in processed.columns:
                    fig_dist = px.histogram(
                        processed, x="final_sentiment_score", nbins=50,
                        title="Sentiment Score Distribution",
                        color_discrete_sequence=["#2196F3"],
                    )
                    fig_dist.update_layout(template="plotly_white", height=300)
                    st.plotly_chart(fig_dist, use_container_width=True)

            # Engagement-weighted vs unweighted
            if "engagement_weighted_sentiment" in sent_agg.columns:
                fig_ew = go.Figure()
                sent_dates = pd.to_datetime(sent_agg["period_start"])
                fig_ew.add_trace(go.Scatter(
                    x=sent_dates, y=sent_agg["mean_sentiment"],
                    name="Unweighted", line=dict(color="#2196F3"),
                ))
                fig_ew.add_trace(go.Scatter(
                    x=sent_dates, y=sent_agg["engagement_weighted_sentiment"],
                    name="Engagement-Weighted", line=dict(color="#FF9800"),
                ))
                fig_ew.update_layout(
                    title="Unweighted vs Engagement-Weighted Sentiment",
                    template="plotly_white", height=350,
                )
                st.plotly_chart(fig_ew, use_container_width=True)

        # Returns and volatility
        if len(price_df) > 20:
            pdf = price_df.copy()
            pdf["close"] = pd.to_numeric(pdf["close"])
            pdf["return"] = np.log(pdf["close"] / pdf["close"].shift(1))
            pdf["rolling_vol"] = pdf["return"].rolling(20).std()

            col1, col2 = st.columns(2)
            with col1:
                fig_ret = px.line(
                    pdf, x="timestamp_utc", y="return",
                    title="Log Returns",
                )
                fig_ret.update_layout(template="plotly_white", height=300)
                st.plotly_chart(fig_ret, use_container_width=True)
            with col2:
                fig_vol = px.line(
                    pdf.dropna(subset=["rolling_vol"]),
                    x="timestamp_utc", y="rolling_vol",
                    title="20-Period Rolling Volatility",
                )
                fig_vol.update_layout(template="plotly_white", height=300)
                st.plotly_chart(fig_vol, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Forecast Matrix
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Forecast Matrix")
    forecasts = st.session_state.forecasts

    if not forecasts:
        st.info("No forecasts available. Run **Train & Validate** then **Generate Forecasts**.")
    else:
        # Forecast table
        fc_data = []
        for f in forecasts:
            fc_data.append({
                "Horizon": f"n+{f['horizon']}",
                "Target Timestamp": f.get("target_timestamp", ""),
                "Reference Price": f.get("reference_price"),
                "Predicted Price": f.get("predicted_price"),
                "Predicted Return": f.get("predicted_return"),
                "Model": f.get("model_name", ""),
                "Status": f.get("status", ""),
            })

        fc_df = pd.DataFrame(fc_data)
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

        # Forecast trajectory chart
        valid_fc = [f for f in forecasts if f.get("predicted_price") is not None]
        if valid_fc:
            fig = go.Figure()
            horizons = [f["horizon"] for f in valid_fc]
            prices = [f["predicted_price"] for f in valid_fc]
            ref_price = valid_fc[0].get("reference_price", prices[0])

            fig.add_trace(go.Scatter(
                x=horizons, y=prices,
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color="#2196F3", width=3),
                marker=dict(size=10),
            ))
            fig.add_hline(
                y=ref_price, line_dash="dash",
                line_color="gray", annotation_text="Reference Price",
            )
            fig.update_layout(
                title="Predicted Price Trajectory",
                xaxis_title="Horizon (periods ahead)",
                yaxis_title="Price",
                template="plotly_white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption("Note: Prediction intervals are not displayed as they cannot be "
                   "statistically validated with the current methodology.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Model Evaluation
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Evaluation")
    vr = st.session_state.validation_results

    if not vr or "validation_metrics" not in vr:
        st.info("No validation results. Run **Train & Validate** first.")
    else:
        st.subheader(f"Best Model: {vr.get('best_model', 'N/A')}")
        st.caption(f"Training: {vr.get('n_train', 0)} obs | Test: {vr.get('n_test', 0)} obs")

        # Validation metrics table
        vm = vr.get("validation_metrics", {})
        if vm:
            st.subheader("Walk-Forward Validation (avg across folds)")
            vm_rows = []
            for model_name, metrics in vm.items():
                row = {"Model": model_name}
                row.update({k: f"{v:.6f}" if isinstance(v, float) and k != "n_folds" else v
                           for k, v in metrics.items()})
                vm_rows.append(row)
            st.dataframe(pd.DataFrame(vm_rows), use_container_width=True, hide_index=True)

        # Test metrics
        tm = vr.get("test_metrics", {})
        nm = vr.get("naive_test_metrics", {})
        if tm:
            st.subheader("Final Test Set")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MAE", f"{tm.get('mae', 0):.6f}",
                          f"vs Naive: {nm.get('mae', 0):.6f}" if nm else "")
            with col2:
                st.metric("RMSE", f"{tm.get('rmse', 0):.6f}")
            with col3:
                st.metric("Dir. Accuracy", f"{tm.get('directional_accuracy', 0):.1f}%")
            with col4:
                st.metric("R²", f"{tm.get('r_squared', 0):.4f}")

        # Actual vs predicted scatter
        y_test = vr.get("y_test")
        y_pred = vr.get("y_test_pred")
        if y_test is not None and y_pred is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test, y=y_pred, mode="markers",
                name="Predictions", marker=dict(color="#2196F3", size=6, opacity=0.6),
            ))
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", name="Perfect",
                line=dict(color="gray", dash="dash"),
            ))
            fig.update_layout(
                title="Actual vs Predicted Returns (Test Set)",
                xaxis_title="Actual Return",
                yaxis_title="Predicted Return",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 5: Trading Strategy
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    st.header("Trading Strategy Backtest")
    bt = st.session_state.backtest_results

    if not bt or "strategy_metrics" not in bt:
        st.info("No backtest results. Run **Train & Validate** first.")
    else:
        sm = bt["strategy_metrics"]
        bm = bt.get("buyhold_metrics", {})

        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sharpe Ratio", f"{sm.get('sharpe_ratio', 0):.4f}")
        with col2:
            st.metric("Sortino Ratio", f"{sm.get('sortino_ratio', 0):.4f}")
        with col3:
            st.metric("Max Drawdown", f"{sm.get('max_drawdown', 0)*100:.2f}%")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Cumulative Return", f"{sm.get('cumulative_return', 0)*100:.2f}%")
        with col5:
            st.metric("Win Rate", f"{sm.get('win_rate', 0):.1f}%")
        with col6:
            st.metric("Num Trades", f"{sm.get('num_trades', 0)}")

        # Equity curve
        rdf = bt.get("results_df")
        if rdf is not None and not rdf.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(rdf))), y=rdf["cumulative_return"],
                name="Strategy", line=dict(color="#2196F3", width=2),
            ))
            fig.add_trace(go.Scatter(
                x=list(range(len(rdf))), y=rdf["bh_cumulative"],
                name="Buy & Hold", line=dict(color="#FF9800", width=2, dash="dash"),
            ))
            fig.update_layout(
                title="Cumulative Returns: Strategy vs Buy & Hold",
                xaxis_title="Period", yaxis_title="Wealth",
                template="plotly_white", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown chart
            wealth = rdf["cumulative_return"].values
            peak = np.maximum.accumulate(wealth)
            drawdown = (peak - wealth) / np.where(peak > 0, peak, 1)

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=list(range(len(drawdown))), y=-drawdown * 100,
                fill="tozeroy", name="Drawdown",
                line=dict(color="#F44336"),
            ))
            fig_dd.update_layout(
                title="Strategy Drawdown",
                xaxis_title="Period", yaxis_title="Drawdown (%)",
                template="plotly_white", height=300,
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        # Strategy details
        with st.expander("Strategy Assumptions"):
            st.write(f"- **Threshold:** {bt.get('threshold', 0)}")
            st.write(f"- **Transaction cost:** {bt.get('transaction_cost_bps', 10)} bps")
            st.write(f"- **Allow short:** {bt.get('allow_short', False)}")
            st.write(f"- **Annualization:** {bt.get('periods_per_year', 252)} periods/year")
            st.write(f"- **Risk-free rate:** 0 (assumption)")
            st.write(f"- **Sortino target:** 0 (minimum acceptable return)")
            st.write(f"- **Total transaction costs:** {bt.get('total_transaction_costs', 0):.6f}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 6: News & Sentiment Summary
# ═══════════════════════════════════════════════════════════════════════
with tab6:
    st.header("News & Sentiment Summary")
    news_df = st.session_state.news_df

    if news_df.empty:
        st.info("No news data. Configure TAVILY_API_KEY or EXA_API_KEY, or use demo data.")
    else:
        for _, row in news_df.iterrows():
            with st.container():
                url = row.get("url", "")
                title = row.get("title", "Untitled")
                snippet = row.get("snippet", "")
                source = row.get("source_name", "")
                pub = row.get("published_at", "")
                sent_label = row.get("sentiment_label", "")
                provider = row.get("provider", "")

                sent_color = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(sent_label, "⚪")

                if url:
                    st.markdown(f"### [{title}]({url})")
                else:
                    st.markdown(f"### {title}")
                st.caption(f"{sent_color} {sent_label} | {source} | {pub} | via {provider}")
                st.write(snippet[:300])
                st.divider()

    # Download outlook
    if st.button("📥 Download Sentiment Outlook"):
        outlook_path = Path("reports/current_sentiment_outlook.md")
        if outlook_path.exists():
            content = outlook_path.read_text(encoding="utf-8")
            st.download_button(
                "Download Outlook (Markdown)", content,
                file_name="current_sentiment_outlook.md",
                mime="text/markdown",
            )
        else:
            st.warning("Generate reports first using the sidebar button.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 7: Data & Run Logs
# ═══════════════════════════════════════════════════════════════════════
with tab7:
    st.header("Data & Run Logs")

    price_df = st.session_state.price_df
    posts_df = st.session_state.posts_df

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Price Observations", len(price_df))
    with col2:
        st.metric("X Posts", len(posts_df))
    with col3:
        st.metric("News Articles", len(st.session_state.news_df))

    if not price_df.empty:
        st.subheader("Price Data Coverage")
        source_name = price_df["source"].iloc[0] if "source" in price_df.columns else "N/A"
        st.write(f"- **Provider:** {source_name}")
        st.write(f"- **Earliest:** {price_df['timestamp_utc'].min()}")
        st.write(f"- **Latest:** {price_df['timestamp_utc'].max()}")
        st.write(f"- **Timeframe:** {price_df['timeframe'].iloc[0] if 'timeframe' in price_df.columns else 'N/A'}")

    if not posts_df.empty:
        st.subheader("X Post Coverage")
        st.write(f"- **Earliest:** {posts_df['timestamp_utc'].min()}")
        st.write(f"- **Latest:** {posts_df['timestamp_utc'].max()}")
        dq = data_quality_report(price_df, posts_df)
        st.write(f"- **Missing text:** {dq['post_missing_text']}")
        st.write(f"- **Missing timestamps:** {dq['post_missing_timestamp']}")

    # Ingestion history
    st.subheader("Ingestion History")
    runs = repo.get_ingestion_runs()
    if not runs.empty:
        st.dataframe(runs, use_container_width=True, hide_index=True)
    else:
        st.caption("No ingestion runs recorded.")

    # API status
    st.subheader("API Configuration")
    status = api_status()
    for name, avail in status.items():
        if avail:
            st.success(f"✅ {name.title()} — Configured")
        else:
            st.error(f"❌ {name.title()} — Not configured")

    # CSV download
    st.subheader("Data Export")
    if not price_df.empty:
        csv_data = price_df.to_csv(index=False)
        st.download_button("📥 Download Price Data (CSV)", csv_data,
                          file_name=f"{selected_ticker}_prices.csv", mime="text/csv")
    if not st.session_state.processed_posts.empty:
        csv_posts = st.session_state.processed_posts.to_csv(index=False)
        st.download_button("📥 Download Processed Posts (CSV)", csv_posts,
                          file_name=f"{selected_ticker}_posts.csv", mime="text/csv")
