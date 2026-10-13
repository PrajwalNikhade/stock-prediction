"""Sentiment outlook report generator for CrowdWisdomTrading.

Aggregation definitions (for consistency with aggregation.py):
  - Overall sentiment: arithmetic mean of mean_sentiment across ALL periods.
  - Positive/negative/neutral %: average of per-period ratios across ALL periods.
  - Net sentiment score: (positive_ratio - negative_ratio) averaged across ALL periods.
  - Engagement-weighted sentiment: arithmetic mean of engagement_weighted_sentiment across ALL periods.
  - Trend: comparison of last 3 periods vs the prior 3 periods.
"""

from datetime import datetime
from pathlib import Path
from src.utils.time_utils import format_utc, utc_now
from src.utils.logging_config import logger


def generate_sentiment_outlook(asset: str, price_df, sentiment_df, news_df,
                                forecasts: list, data_source: str = "live",
                                output_dir: str = "reports",
                                run_id: str = None,
                                provenance: dict = None) -> str:
    """Generate a one-page current sentiment outlook in Markdown.

    Returns the Markdown content string and saves to file.
    """
    now_str = format_utc(utc_now())
    is_demo = data_source == "demo"
    prov = provenance or {}

    is_historical = data_source in ("historical", "historical_benchmark")

    lines = []
    if is_historical:
        lines.append(f"# Historical Sentiment Archive Report: {asset}")
    else:
        lines.append(f"# Current Sentiment Outlook: {asset}")
    lines.append("")
    if is_demo:
        lines.append("> ⚠️ **DEMO MODE — NO VERIFIED LIVE DATA (Synthetic Fixtures)**")
        lines.append("")
    elif data_source == "partial":
        lines.append("> ℹ️ **PARTIAL LIVE DATA — Live market prices and news active; social X scraper quota reached (0 live posts).**")
        lines.append("> *No synthetic data substituted. All metrics strictly reflect actual observations.*")
        lines.append("")
    elif is_historical:
        lines.append("> 🟡 **HISTORICAL ARCHIVE BENCHMARK — Evaluated on historical price and X sentiment archive (Dec 2021 – Sep 2024).**")
        lines.append("> *This report presents historical archive metrics, NOT a live nowcast of current real-time market prices.*")
        lines.append("> *The historical dataset ending September 2024 does NOT satisfy the most recent 180-day X collection requirement.*")
        lines.append("")
    else:
        lines.append("> 🟢 **LIVE DATA — VERIFIED CURRENT MARKET & RECENTLY SCRAPED X POSTS**")
        lines.append("")
    lines.append(f"**Run Identifier (run_id):** `{run_id or 'N/A'}`  ")
    lines.append(f"**Report Generated:** {now_str}  ")
    lines.append("")

    # Data coverage
    lines.append("## Data Coverage")
    if price_df is not None and not price_df.empty:
        price_min = price_df['timestamp_utc'].min()
        price_max = price_df['timestamp_utc'].max()
        lines.append(f"- **Price observations:** {len(price_df)}")
        lines.append(f"- **Date range:** {price_min} to {price_max}")
        last_close = price_df['close'].iloc[-1]
        if is_historical:
            lines.append(f"- **Final archive close price:** {last_close:,.2f} *(historical, not current market price)*")
        else:
            lines.append(f"- **Last close price:** {last_close:,.2f}")
    else:
        lines.append("- Price data: Not available")
        last_close = None
    lines.append("")

    # Sentiment summary
    if is_historical:
        lines.append("## Historical X Sentiment Summary")
    else:
        lines.append("## X Sentiment Summary")
    if sentiment_df is not None and not sentiment_df.empty:
        n_posts = int(sentiment_df['post_count'].sum()) if 'post_count' in sentiment_df.columns else len(sentiment_df)
        n_periods = len(sentiment_df)
        lines.append(f"- **Total relevant posts:** {n_posts} across {n_periods} aggregation periods")

        if 'mean_sentiment' in sentiment_df.columns:
            # Overall sentiment: arithmetic mean across ALL periods (not a tail subset)
            overall_sent = sentiment_df['mean_sentiment'].mean()
            if overall_sent > 0.05:
                classification = "**Bullish** 📈"
            elif overall_sent < -0.05:
                classification = "**Bearish** 📉"
            else:
                classification = "**Neutral** ➡️"
            lines.append(f"- **Overall sentiment (all-period mean):** {classification} ({overall_sent:.4f})")

        # Positive/negative/neutral ratios: mean across ALL periods
        if 'positive_ratio' in sentiment_df.columns:
            avg_pos = sentiment_df['positive_ratio'].mean()
            avg_neg = sentiment_df['negative_ratio'].mean()
            avg_neu = sentiment_df['neutral_ratio'].mean()
            lines.append(f"- **Positive (period-average):** {avg_pos*100:.1f}%")
            lines.append(f"- **Negative (period-average):** {avg_neg*100:.1f}%")
            lines.append(f"- **Neutral (period-average):** {avg_neu*100:.1f}%")

        # Net sentiment score: mean of (positive_ratio - negative_ratio) across all periods
        if 'net_sentiment' in sentiment_df.columns:
            avg_net = sentiment_df['net_sentiment'].mean()
            lines.append(f"- **Net sentiment score (all-period mean):** {avg_net:.4f}")

        # Engagement-weighted sentiment: mean across ALL periods
        if 'engagement_weighted_sentiment' in sentiment_df.columns:
            ews_mean = sentiment_df['engagement_weighted_sentiment'].mean()
            lines.append(f"- **Engagement-weighted sentiment (all-period mean):** {ews_mean:.4f}")

        # Trend: compare last 3 periods vs prior 3 periods (6 periods required)
        if len(sentiment_df) >= 6:
            recent_avg = sentiment_df['mean_sentiment'].tail(3).mean()
            prior_avg = sentiment_df['mean_sentiment'].iloc[-6:-3].mean()
            if recent_avg > prior_avg + 0.02:
                lines.append("- **Trend (last 3 vs prior 3 periods):** Sentiment improving ↗️")
            elif recent_avg < prior_avg - 0.02:
                lines.append("- **Trend (last 3 vs prior 3 periods):** Sentiment declining ↘️")
            else:
                lines.append("- **Trend (last 3 vs prior 3 periods):** Sentiment stable →")
    else:
        lines.append("- No sentiment data available")
        if is_historical:
            lines.append("- *Historical archive did not contain sufficient posts for this asset.*")
    lines.append("")

    # Current X data status (Task 4 — never substitute old data as current)
    if is_historical:
        lines.append("## Current X Sentiment Status")
        lines.append("")
        lines.append("> ⚠️ **Current X sentiment cannot be determined from the available dataset.**")
        lines.append("> The historical archive covers December 2021 through September 2024. No authorized, free API is available for continuous recent X post collection (X API v2 read access requires $100/month minimum since July 2023).")
        lines.append("> The metrics above reflect *archived historical sentiment only* and must not be interpreted as current market sentiment.")
        lines.append("")

    # News
    lines.append("## Recent News")
    if news_df is not None and not news_df.empty:
        for _, row in news_df.head(5).iterrows():
            title = row.get('title', 'Untitled')
            url = row.get('url', '')
            source = row.get('source_name', '') or row.get('provider', '')
            pub = row.get('published_at', '')
            label = row.get('sentiment_label', '')
            details = []
            if source:
                details.append(f"*{source}*")
            if pub:
                details.append(f"({pub})")
            if label:
                details.append(f"[{label.capitalize()}]")
            suffix = f" — {' '.join(details)}" if details else ""
            if url:
                lines.append(f"- [{title}]({url}){suffix}")
            else:
                lines.append(f"- {title}{suffix}")
    else:
        lines.append("- No recent news available")
    lines.append("")

    # Forecast
    if is_historical:
        lines.append("## Historical Forecast Summary")
        lines.append("")
        lines.append("> *These forecasts are based on the final historical observation in the archive. They do not predict current or future prices.*")
        lines.append("")
    else:
        lines.append("## Forecast Summary")
    if forecasts:
        lines.append("| Horizon | Predicted Price | Predicted Return | Status |")
        lines.append("|---------|---------------:|------------------:|--------|")
        for f in forecasts:
            h = f.get("horizon", "?")
            pp = f.get("predicted_price")
            pr = f.get("predicted_return")
            st = f.get("status", "")
            pp_str = f"{pp:,.2f}" if pp is not None else "N/A"
            pr_str = f"{pr*100:.3f}%" if pr is not None else "N/A"
            lines.append(f"| n+{h} | {pp_str} | {pr_str} | {st} |")
    else:
        lines.append("- No forecasts available. Run the training pipeline first.")
    lines.append("")

    # Limitations
    lines.append("## Limitations and Disclaimers")
    lines.append("- This report is generated from available data and model outputs.")
    lines.append("- Sentiment analysis reflects observed X posts which may not represent broader market views.")
    lines.append("- Historical engagement counts may not reflect values available at original posting time.")
    lines.append("- Correlation between sentiment and price does not establish causation.")
    lines.append("- **This is not financial advice.** Past performance does not guarantee future results.")
    if is_demo:
        lines.append("- **This report uses synthetic demo data and does not reflect real market conditions.**")
    if is_historical:
        lines.append("- **Historical archive limitation:** The dataset ends September 2024 and does NOT satisfy a rolling 180-day recent collection requirement.")
        lines.append("- **No current X sentiment available:** Prices and sentiment shown are archival, not real-time.")
    lines.append("")

    content = "\n".join(lines)

    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "current_sentiment_outlook.md"
    filepath.write_text(content, encoding="utf-8")
    logger.info("Sentiment outlook saved to %s", filepath)

    return content
