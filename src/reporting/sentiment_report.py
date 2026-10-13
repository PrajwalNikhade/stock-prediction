"""Sentiment outlook report generator for CrowdWisdomTrading."""

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

    lines = []
    lines.append(f"# Current Sentiment Outlook: {asset}")
    lines.append("")
    if is_demo:
        lines.append("> ⚠️ **DEMO MODE — NO VERIFIED LIVE DATA (Synthetic Fixtures)**")
        lines.append("")
    elif data_source == "partial":
        lines.append("> ℹ️ **PARTIAL DATA — LIVE PRICE FEED & NEWS ACTIVE, LIVE POSTS UNCONFIGURED/FAILED (Apify 401)**")
        lines.append("> *No synthetic data substituted. All metrics strictly reflect actual observations.*")
        lines.append("")
    else:
        lines.append("> 🟢 **LIVE DATA — VERIFIED LIVE MARKET & SENTIMENT FEEDS**")
        lines.append("")
    lines.append(f"**Run Identifier (run_id):** `{run_id or 'N/A'}`  ")
    lines.append(f"**Report Generated:** {now_str}  ")
    lines.append("")

    # Data coverage
    lines.append("## Data Coverage")
    if price_df is not None and not price_df.empty:
        lines.append(f"- **Price observations:** {len(price_df)}")
        lines.append(f"- **Date range:** {price_df['timestamp_utc'].min()} to {price_df['timestamp_utc'].max()}")
        last_close = price_df['close'].iloc[-1]
        lines.append(f"- **Last close price:** {last_close:,.2f}")
    else:
        lines.append("- Price data: Not available")
        last_close = None
    lines.append("")

    # Sentiment summary
    lines.append("## X Sentiment Summary")
    if sentiment_df is not None and not sentiment_df.empty:
        n_posts = int(sentiment_df['post_count'].sum()) if 'post_count' in sentiment_df.columns else len(sentiment_df)
        lines.append(f"- **Total relevant posts:** {n_posts}")

        if 'mean_sentiment' in sentiment_df.columns:
            recent = sentiment_df.tail(5)
            avg_sent = recent['mean_sentiment'].mean()
            if avg_sent > 0.05:
                classification = "**Bullish** 📈"
            elif avg_sent < -0.05:
                classification = "**Bearish** 📉"
            else:
                classification = "**Neutral** ➡️"
            lines.append(f"- **Overall sentiment:** {classification}")
            lines.append(f"- **Recent net sentiment score:** {avg_sent:.4f}")

        if 'positive_ratio' in sentiment_df.columns:
            last = sentiment_df.iloc[-1]
            lines.append(f"- **Positive:** {last.get('positive_ratio', 0)*100:.1f}%")
            lines.append(f"- **Negative:** {last.get('negative_ratio', 0)*100:.1f}%")
            lines.append(f"- **Neutral:** {last.get('neutral_ratio', 0)*100:.1f}%")

        if 'engagement_weighted_sentiment' in sentiment_df.columns:
            ews = sentiment_df['engagement_weighted_sentiment'].iloc[-1]
            lines.append(f"- **Engagement-weighted sentiment:** {ews:.4f}")

        # Trend
        if len(sentiment_df) >= 5:
            recent_avg = sentiment_df['mean_sentiment'].tail(3).mean()
            prior_avg = sentiment_df['mean_sentiment'].iloc[-6:-3].mean()
            if recent_avg > prior_avg + 0.02:
                lines.append("- **Trend:** Sentiment improving ↗️")
            elif recent_avg < prior_avg - 0.02:
                lines.append("- **Trend:** Sentiment declining ↘️")
            else:
                lines.append("- **Trend:** Sentiment stable →")
    else:
        lines.append("- No sentiment data available")
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
    lines.append("")

    content = "\n".join(lines)

    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "current_sentiment_outlook.md"
    filepath.write_text(content, encoding="utf-8")
    logger.info("Sentiment outlook saved to %s", filepath)

    return content
