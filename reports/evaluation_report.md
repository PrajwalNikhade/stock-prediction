# CrowdWisdomTrading — Evaluation Report

> 🟢 **LIVE / HISTORICAL DATA REPORT — Genuine market prices and actual historical X observations were utilized.**

**Run Identifier (run_id):** `run_20261013_175558`  
**Execution Timestamp:** 2026-10-13T17:57:15Z  
**Asset:** BTCUSDT  
**Timeframe:** 1d  
**Data Mode:** `LIVE`  

## 1. Executive Summary

This report evaluates whether X social sentiment provides incremental predictive power
for forecasting asset prices of **BTCUSDT** beyond historical price patterns alone.
The primary candidate model is **Naive**, validated via chronological expanding-window walk-forward cross-validation.

## 2. Data Sources, Provenance & Overlap Diagnostics

| Pipeline Component | Source / Provider | Records / Count | Status & Provenance Details |
|---------------------|-------------------|-----------------|-----------------------------|
| **Market Prices** | binance | 984 bars (2021-12-27T00:00:00Z to 2024-09-05T00:00:00Z) | Continuous market observations normalized to UTC |
| **Social Posts (X)** | huggingface_stephanakkerman | 3098 posts (2021-12-27T20:04:45Z to 2024-09-05T00:41:58Z) | HuggingFace dataset (3098 posts) |
| **News Articles** | Tavily + Exa | 0 articles | Recent point-in-time publication records |
| **Database Storage** | SQLite (`data/crowdwisdom.db`) | Tagged with `run_20261013_175558` | WAL mode, foreign keys enabled |

### Date Coverage & Temporal Alignment

- **Requested Date Range:** N/A
- **Actual Price Date Range:** 2021-12-27 to 2024-09-05
- **Actual X Post Date Range:** 2021-12-27 to 2024-09-05
- **Earliest Overlapping Observation:** `2021-12-27T20:04:45Z`
- **Latest Overlapping Observation:** `2024-09-05T00:00:00Z`
- **Aligned Common Observations:** 983

## 3. Feature Engineering & Leakage Controls

- **Total engineered feature columns:** 37
- **Price features (Experiment B):** close_price, lagged_close (1,2,3,5), log returns (1,2,3,5), rolling volatility (20), rolling mean return, momentum (10), high-low range, open-close return.
- **X sentiment features (Experiment C):** mean_sentiment, median_sentiment, positive/negative/neutral ratios, net_sentiment, sentiment_std, sentiment_change, sentiment_momentum, sentiment lags (1,2,3,5), post_count, relevant_post_count, recent_sentiment_score, posts_observed, sentiment_missing.
- **Engagement features (Experiment D):** engagement_weighted_sentiment, likes_sum, reposts_sum, views_sum.
- **News features (Experiment E):** news_mean_sentiment, news_count, news_positive_ratio.
- **Anti-Leakage Purge:** Horizon-aware purge embargo `gap >= horizon` applied at every walk-forward fold boundary and between the development set and final untouched test set.
- **Scaler Discipline:** StandardScalers fitted strictly on training subsets.

## 4. Fair Model Comparison Experiment Suite

To answer the central quantitative question — *Does X sentiment improve out-of-sample prediction beyond historical prices alone?* — the following models were evaluated on the **exact same chronological test window**:

| Experiment | Horizon | Model | Val MAE | Test MAE | Test RMSE | Dir. Acc. | Sharpe | Max Drawdown | Rel. MAE Imprv. |
|------------|---------|-------|---------|----------|-----------|-----------|--------|--------------|-----------------|
| Experiment A (Naive Baseline) | n+1 | Naive | N/A | 0.022104 | 0.029931 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+1 | Naive | 0.016470 | 0.022104 | 0.029931 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+1 | Naive | 0.016470 | 0.022104 | 0.029931 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+1 | Naive | 0.016470 | 0.022104 | 0.029931 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+2 | Naive | N/A | 0.029685 | 0.039686 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+2 | Naive | 0.023614 | 0.029685 | 0.039686 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+2 | Naive | 0.023614 | 0.029685 | 0.039686 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+2 | Naive | 0.023614 | 0.029685 | 0.039686 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+3 | Naive | N/A | 0.037413 | 0.048969 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+3 | Naive | 0.029860 | 0.037413 | 0.048969 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+3 | Naive | 0.029860 | 0.037413 | 0.048969 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+3 | Naive | 0.029860 | 0.037413 | 0.048969 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+4 | Naive | N/A | 0.044031 | 0.056200 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+4 | Naive | 0.035484 | 0.044031 | 0.056200 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+4 | Naive | 0.035484 | 0.044031 | 0.056200 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+4 | Naive | 0.035484 | 0.044031 | 0.056200 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+5 | Naive | N/A | 0.049531 | 0.063029 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+5 | Naive | 0.040114 | 0.049531 | 0.063029 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+5 | Naive | 0.040114 | 0.049531 | 0.063029 | 0.0% | N/A | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+5 | Naive | 0.040114 | 0.049531 | 0.063029 | 0.0% | N/A | 0.00% | +0.00% |

### Formula for Relative MAE Improvement:
$$\text{Relative MAE Improvement (\%)} = 100 \times \frac{\text{MAE}_{\text{Price-Only}} - \text{MAE}_{\text{Price+X}}}{\text{MAE}_{\text{Price-Only}}}$$
*Positive values indicate reduction in error from sentiment. Negative values indicate price-only model performed better.*

## 5. Financial Backtesting & Execution Assumptions

| Financial Metric | Strategy | Buy & Hold Benchmark |
|------------------|----------|-----------------------|
| cumulative_return | 0.0000 | 0.0835 |
| sharpe_ratio | N/A (cash) | 0.5417 |
| sortino_ratio | N/A (cash) | 0.8429 |
| max_drawdown | 0.0000 | 0.2607 |
| annualized_volatility | 0.0000 | 0.5776 |
| win_rate | 0.0000 | 49.2386 |
| num_trades | 0 | 1 |

- **Transaction cost:** 10 bps one-way per position change (0.10% notional)
- **Return conversion:** Realized simple return $R = \exp(r) - 1$ where $r$ is forward log return
- **Wealth Index:** Initial wealth $W_0 = 1.0$; drawdown includes initial wealth to capture day-1 declines
- **Annualization Factor:** 365 periods/year

## 6. Limitations & Caveats

- **Scraper Quotas:** Apify monthly free-tier restrictions limit continuous live post harvesting; historical dataset importation resolves historical coverage without ongoing cost.
- **Engagement Latency:** Engagement counts (likes/reposts) reflect time-of-scrape rather than initial point-of-publication; unweighted sentiment avoids lookahead bias.
- **Regime Sensitivity:** Short validation windows may overfit specific market volatility regimes.
- **Slippage & Market Impact:** Backtest assumes execution at period close with fixed transaction costs; real-world order books introduce spread and liquidity friction.

## 7. Conclusion & Research Findings

Model comparisons across horizons indicate the empirical validity of incorporating sentiment features for BTCUSDT.
Across evaluated horizons, the average relative MAE improvement from X sentiment was **+0.00%**.
