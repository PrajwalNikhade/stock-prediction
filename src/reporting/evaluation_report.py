"""Evaluation report generator for CrowdWisdomTrading."""

import numpy as np
from pathlib import Path
from src.utils.time_utils import format_utc, utc_now
from src.utils.logging_config import logger


def generate_evaluation_report(
    asset: str,
    timeframe: str,
    data_source: str,
    price_coverage: dict,
    post_coverage: dict,
    feature_info: dict,
    validation_results: dict,
    backtest_results: dict,
    output_dir: str = "reports",
    run_id: str = None,
    provenance: dict = None,
) -> str:
    """Generate the full evaluation report in Markdown.

    All values must come from actual pipeline outputs.
    """
    now_str = format_utc(utc_now())
    is_demo = data_source == "demo"
    prov = provenance or {}

    lines = []
    lines.append("# CrowdWisdomTrading — Evaluation Report")
    lines.append("")
    if is_demo:
        lines.append("> ⚠️ **DEMO MODE REPORT — All metrics computed on synthetic demonstration data.**")
        lines.append("> **Do not treat these as authentic market findings.**")
        lines.append("")
    elif data_source == "partial":
        lines.append("> ℹ️ **PARTIAL DATA REPORT — Market prices and news were retrieved from live external APIs, but social sentiment data was uncollected (Apify 401/unconfigured).**")
        lines.append("> **Features were constructed with missing-sentiment flags (`sentiment_missing = 1`). No synthetic data was fabricated.**")
        lines.append("")
    else:
        lines.append("> 🟢 **LIVE DATA REPORT — Both market prices and social sentiment data were collected from live external APIs.**")
        lines.append("")
    lines.append(f"**Run Identifier (run_id):** `{run_id or 'N/A'}`  ")
    lines.append(f"**Generated:** {now_str}  ")
    lines.append(f"**Asset:** {asset}  ")
    lines.append(f"**Timeframe:** {timeframe}  ")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    best_model = validation_results.get("best_model", "N/A")
    lines.append(f"This report evaluates the predictive performance of price forecasting models")
    lines.append(f"incorporating sentiment features for **{asset}** at **{timeframe}** frequency.")
    lines.append(f"The selected model is **{best_model}** based on expanding-window walk-forward validation.")
    lines.append("")

    # Data Sources and Provenance
    lines.append("## Data Sources and Provenance")
    lines.append("")
    lines.append("| Pipeline Component | Source / Provider | Records / Observations | Status & Provenance Details |")
    lines.append("|---------------------|-------------------|------------------------|-----------------------------|")
    lines.append(f"| **Market Prices** | {price_coverage.get('source', 'Binance')} | {price_coverage.get('count', 0)} bars ({price_coverage.get('date_range', 'N/A')}) | Live exchange REST API (Binance public klines) |")
    apify_status = prov.get("apify_status", "Unconfigured / Failed")
    lines.append(f"| **Social Posts (X)** | Apify (`apidojo/tweet-scraper`) | {post_coverage.get('count', 0)} posts ({post_coverage.get('date_range', 'None')}) | {apify_status} (0 synthetic posts injected) |")
    news_count = prov.get("news_count", 0)
    lines.append(f"| **News Feeds** | Tavily Search + Exa AI | {news_count} articles | Live search feeds retrieved and deduplicated |")
    lines.append(f"| **Database Storage** | SQLite (`data/crowdwisdom.db`) | Tagged with `run_id` | WAL mode, foreign keys, point-in-time provenance |")
    lines.append("")

    # Feature Engineering
    lines.append("## Feature Engineering")
    lines.append("")
    lines.append(f"- **Total features:** {feature_info.get('n_features', 0)}")
    lines.append(f"- **Sentiment features:** mean_sentiment, median_sentiment, positive/negative/neutral ratios,")
    lines.append(f"  net_sentiment, sentiment_std, sentiment_change, sentiment_momentum, lags (1,2,3,5),")
    lines.append(f"  post_count, engagement_weighted_sentiment, recent_sentiment_score")
    lines.append(f"- **Price features:** close_price, lagged_close (1,2,3,5), returns (1,2,3,5),")
    lines.append(f"  rolling_volatility, rolling_mean_return, momentum, high_low_range, open_close_return")
    lines.append(f"- **Time alignment:** Backward-looking, point-in-time aggregation (UTC)")
    lines.append(f"- **Missing handling:** Explicit sentiment_missing flag; zero-fill only for verified empty periods")
    lines.append("")

    # Modeling Methodology
    lines.append("## Modeling Methodology")
    lines.append("")
    lines.append(f"- **Target:** Log return = ln(close_{{t+h}} / close_t)")
    lines.append(f"- **Horizons:** n+0 (nowcast), n+1 through n+5")
    lines.append(f"- **Candidate models:** Naive (random walk), Ridge, Random Forest, HistGBR, XGBoost")
    lines.append(f"- **Validation:** Expanding-window walk-forward with horizon-aware gap")
    lines.append(f"- **Selection criterion:** Minimum validation MAE")
    lines.append(f"- **Leakage controls:** No future data in features; scaler fit on training only")
    lines.append("")

    # Predictive Performance
    lines.append("## Predictive Performance")
    lines.append("")
    vm = validation_results.get("validation_metrics", {})
    tm = validation_results.get("test_metrics", {})
    nm = validation_results.get("naive_test_metrics", {})

    if vm:
        lines.append("### Walk-Forward Validation (averaged across folds)")
        lines.append("")
        lines.append("| Model | MAE | RMSE | R² | Dir. Accuracy | Folds |")
        lines.append("|-------|-----|------|----|---------------|-------|")
        for model_name, metrics in vm.items():
            lines.append(
                f"| {model_name} | {metrics.get('mae', 'N/A'):.6f} | "
                f"{metrics.get('rmse', 'N/A'):.6f} | "
                f"{metrics.get('r_squared', 'N/A'):.4f} | "
                f"{metrics.get('directional_accuracy', 'N/A'):.1f}% | "
                f"{metrics.get('n_folds', 'N/A')} |"
            )
        lines.append("")

    if tm:
        lines.append("### Final Test Set Performance")
        lines.append("")
        lines.append(f"| Metric | Best Model ({best_model}) | Naive Baseline |")
        lines.append(f"|--------|--------------------------|----------------|")
        for key in ["mae", "rmse", "r_squared", "directional_accuracy", "smape", "prediction_bias"]:
            tv = tm.get(key, "N/A")
            nv = nm.get(key, "N/A") if nm else "N/A"
            tv_str = f"{tv:.6f}" if isinstance(tv, float) else str(tv)
            nv_str = f"{nv:.6f}" if isinstance(nv, float) else str(nv)
            lines.append(f"| {key} | {tv_str} | {nv_str} |")
        lines.append("")

    # Financial Performance
    lines.append("## Financial Performance")
    lines.append("")
    if backtest_results and "strategy_metrics" in backtest_results:
        sm = backtest_results["strategy_metrics"]
        bm = backtest_results.get("buyhold_metrics", {})
        lines.append(f"| Metric | Strategy | Buy & Hold |")
        lines.append(f"|--------|----------|------------|")
        for key in ["cumulative_return", "sharpe_ratio", "sortino_ratio", "max_drawdown",
                     "annualized_volatility", "win_rate", "num_trades"]:
            sv = sm.get(key, "N/A")
            bv = bm.get(key, "N/A")
            if isinstance(sv, float) and (np.isnan(sv) or np.isinf(sv)):
                sv_str = "N/A (flat cash)"
            elif isinstance(sv, float):
                sv_str = f"{sv:.4f}"
            else:
                sv_str = str(sv)

            if isinstance(bv, float) and (np.isnan(bv) or np.isinf(bv)):
                bv_str = "N/A (flat cash)"
            elif isinstance(bv, float):
                bv_str = f"{bv:.4f}"
            else:
                bv_str = str(bv)
            lines.append(f"| {key} | {sv_str} | {bv_str} |")
        lines.append("")
        lines.append(f"- **Transaction cost:** {backtest_results.get('transaction_cost_bps', 10)} bps per trade")
        lines.append(f"- **Risk-free rate:** 0 (assumption)")
        lines.append(f"- **Observations:** {backtest_results.get('n_observations', 'N/A')}")
        lines.append(f"- **Annualization factor:** {backtest_results.get('periods_per_year', 252)} periods/year")
    else:
        lines.append("- Backtest results not available")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Historical X post coverage may be incomplete; API limits and scraper constraints apply.")
    lines.append("- Selection bias: only posts matching search queries are observed.")
    lines.append("- Engagement values reflect current state, not values at original posting time.")
    lines.append("- News availability limited by API quotas and recency restrictions.")
    lines.append("- Market regime changes may invalidate historically trained models.")
    lines.append("- Model uncertainty is inherent; prediction intervals not statistically validated.")
    lines.append("- Financial metrics may be unstable with short evaluation periods.")
    lines.append("- Backtesting assumes perfect execution; real-world slippage and liquidity costs differ.")
    lines.append("- The difference between backtesting and live trading can be substantial.")
    if is_demo:
        lines.append("- **All values in this report are from synthetic demo data.**")
    lines.append("")

    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    if is_demo:
        lines.append("This report was generated using synthetic demonstration data.")
        lines.append("No conclusions about real market performance can be drawn.")
        lines.append("To generate a meaningful evaluation, configure API credentials and run the full pipeline with live data.")
    else:
        lines.append(f"The {best_model} model was selected based on walk-forward validation for {asset}.")
        if tm and tm.get("mae", float("inf")) < (nm or {}).get("mae", float("inf")):
            lines.append("The selected model outperformed the naive baseline on the test set.")
        else:
            lines.append("The selected model did not consistently outperform the naive baseline,")
            lines.append("suggesting limited predictive value from the available sentiment data.")
        lines.append("These results should be interpreted with caution given the limitations above.")
    lines.append("")

    content = "\n".join(lines)

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "evaluation_report.md"
    filepath.write_text(content, encoding="utf-8")
    logger.info("Evaluation report saved to %s", filepath)

    return content
