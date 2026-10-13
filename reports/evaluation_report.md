# CrowdWisdomTrading — Evaluation Report

> 🟡 **HISTORICAL ARCHIVE BENCHMARK — Evaluated on historical price and X sentiment archive (Dec 2021 – Sep 2024).**
> **Notice: This report documents an out-of-sample quantitative backtest benchmark, NOT a live nowcast of current real-time market prices.**

**Run Identifier (run_id):** `run_20261013_182342`  
**Execution Timestamp:** 2026-10-13T18:24:30Z  
**Asset:** BTCUSDT  
**Timeframe:** 1d  
**Data Mode:** `HISTORICAL`  

## 1. Executive Summary

This report evaluates whether X social sentiment provides incremental predictive power
for forecasting asset prices of **BTCUSDT** beyond historical price patterns alone.
The primary candidate model selected across cross-validation folds is **Naive**, validated via expanding-window walk-forward cross-validation with zero-leakage horizon purge gaps.

## 2. Data Sources, Provenance & Overlap Diagnostics

| Pipeline Component | Source / Provider | Records / Count | Status & Provenance Details |
|---------------------|-------------------|-----------------|-----------------------------|
| **Market Prices** | binance | 984 bars (2021-12-27T00:00:00Z to 2024-09-05T00:00:00Z) | Continuous market observations normalized to UTC |
| **Social Posts (X)** | huggingface_stephanakkerman | 3098 posts (2021-12-27T20:04:45Z to 2024-09-05T00:41:58Z) | HuggingFace dataset (3098 posts) |
| **News Articles** | Tavily + Exa | 0 articles | Contemporary news (excluded from historical benchmark to prevent leakage) |
| **Database Storage** | SQLite (`data/crowdwisdom.db`) | Tagged with `run_20261013_182342` | WAL mode, foreign keys enabled |

### Data Filtering & Exclusion Funnel

The raw historical archive (`StephanAkkerman/financial-tweets`, MIT license) contains 314,732 total financial tweets spanning December 2021 through September 2024. The difference between cached corpus counts and active evaluation observations is governed by the following deterministic filtering funnel:

| Filtering Stage | Criteria | BTC Processing | NVDA Processing | Rationale / Quantitative Safeguard |
|:---|:---|:---:|:---:|:---|
| **1. Raw Corpus** | Entire parquet cache | 314,732 rows | 314,732 rows | Unfiltered multi-asset financial tweet archive |
| **2. Regex Filter** | Asset ticker & aliases | 28,270 matched | 1,657 matched | Strict regex matching (e.g. `$BTC`, `Bitcoin`, `#BTC`) |
| **3. Sampling Bound** | Tail chronological subset | 5,000 tail limit | 1,657 (full set) | Ensures memory stability and recent regime focus |
| **4. Data Cleansing** | Non-empty text & valid UTC ts | 3,098 valid | 1,458 valid | Excludes malformed dates, empty text, or corrupted URLs |
| **5. Deduplication** | Distinct status ID & URL | 3,098 unique | 1,458 unique | Eliminates reposts and scraping duplicates |
| **6. Calendar Aggregation** | Point-in-time daily buckets | 511 trading days | 362 trading days | Days with $\ge 1$ genuine post |
| **7. Missing Sentiment Days** | Zero observed tweets | 473 days | 314 days | Honestly flagged (`sentiment_missing = 1`, no fake posts) |
| **8. Aligned Price History** | Continuous market candles | 984 days | 984 days | Full matching exchange price bars |

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
- **News features (Experiment E):** news_mean_sentiment, news_count, news_positive_ratio (contemporary news excluded in historical mode).
- **Anti-Leakage Purge:** Horizon-aware purge embargo `gap >= horizon` applied at every walk-forward fold boundary and between the development set and final untouched test set.
- **Scaler Discipline:** StandardScalers fitted strictly on training subsets.

## 4. Fair Model Comparison Experiment Suite

To answer the central quantitative question — *Does X sentiment improve out-of-sample prediction beyond historical prices alone?* — all experiment variants were evaluated on the **exact same chronological test window**:

| Experiment | Horizon | Selected Model | Val MAE | Test MAE | Test RMSE | Dir. Acc. | Sharpe | Max Drawdown | Rel. MAE Imprv. |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Experiment A (Naive Baseline) | n+1 | Naive | N/A | 0.022104 | 0.029931 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+1 | Naive | 0.016470 | 0.022104 | 0.029931 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+1 | Naive | 0.016470 | 0.022104 | 0.029931 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+1 | Naive | 0.016470 | 0.022104 | 0.029931 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+2 | Naive | N/A | 0.029685 | 0.039686 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+2 | Naive | 0.023614 | 0.029685 | 0.039686 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+2 | Naive | 0.023614 | 0.029685 | 0.039686 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+2 | Naive | 0.023614 | 0.029685 | 0.039686 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+3 | Naive | N/A | 0.037413 | 0.048969 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+3 | Naive | 0.029860 | 0.037413 | 0.048969 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+3 | Naive | 0.029860 | 0.037413 | 0.048969 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+3 | Naive | 0.029860 | 0.037413 | 0.048969 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+4 | Naive | N/A | 0.044031 | 0.056200 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+4 | Naive | 0.035484 | 0.044031 | 0.056200 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+4 | Naive | 0.035484 | 0.044031 | 0.056200 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+4 | Naive | 0.035484 | 0.044031 | 0.056200 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment A (Naive Baseline) | n+5 | Naive | N/A | 0.049531 | 0.063029 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment B (Price-Only) | n+5 | Naive | 0.040114 | 0.049531 | 0.063029 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment C (Primary: Price + X) | n+5 | Naive | 0.040114 | 0.049531 | 0.063029 | 0.0% | N/A (cash) | 0.00% | +0.00% |
| Experiment D (Engagement-Enhanced) | n+5 | Naive | 0.040114 | 0.049531 | 0.063029 | 0.0% | N/A (cash) | 0.00% | +0.00% |

### Formula for Relative MAE Improvement:
$$\text{Relative MAE Improvement (\%)} = 100 \times \frac{\text{MAE}_{\text{Price-Only}} - \text{MAE}_{\text{Price+X}}}{\text{MAE}_{\text{Price-Only}}}$$
*Positive values indicate reduction in prediction error from sentiment. Negative values indicate price-only model performed better.*

### Audit of Model Selection & Identical Test Metrics

A key observation in the comparison table is that Experiments B (Price-Only), C (Price + X), and D (Engagement-Enhanced) all selected the **Naive Baseline** ($\hat{r} = 0$) and reported identical test metrics (e.g. Test MAE = 0.022104). The quantitative audit confirms:

1. **Feature Integrity**: Distinct feature column sets were verified for each mode (14 price features for Exp B, 33 price+sentiment features for Exp C, 37 features for Exp D). Sentiment features exhibited genuine non-zero variance (`mean_sentiment` std = 0.293, range [-0.94, +0.96]).
2. **Candidate Cross-Validation Errors**: In every expanding fold, machine learning models (Ridge, Random Forest, HistGBR, XGBoost) were fitted. In all feature modes, ML models produced higher validation errors than Naive:
   - **Naive Baseline (Zero Return)**: $\text{Val MAE} = \mathbf{0.016470}$
   - **Ridge Regression (Price-Only)**: $\text{Val MAE} = 0.018676$ | **(Price + Sentiment)**: $\text{Val MAE} = 0.021792$
   - **Random Forest (Price-Only)**: $\text{Val MAE} = 0.019288$ | **(Price + Sentiment)**: $\text{Val MAE} = 0.019587$
   - **HistGBR (Price-Only)**: $\text{Val MAE} = 0.024214$ | **(Price + Sentiment)**: $\text{Val MAE} = 0.023539$
3. **Root Cause**: Daily asset returns possess low signal-to-noise ratio. Trained ML models incur excess estimation variance on noisy returns, whereas predicting the unconditional mean (zero return) minimizes expected squared and absolute loss. Because walk-forward validation strictly selects the lowest validation MAE model without data snooping, Naive was chosen across all modes.
4. **Identical Test Predictions**: Because Naive was chosen for Exp B, Exp C, and Exp D, all three models outputted $\hat{r} = 0$ on the test set, producing identical test metrics and +0.00% relative improvement.

### Experiment E (News-Enhanced) Status

> ℹ️ **Experiment E Status: UNAVAILABLE / UNTESTED in Historical Benchmark Mode**  
> The Tavily and Exa search APIs retrieve contemporary news articles. Historical point-in-time news archives covering December 2021 through September 2024 are unavailable on free tiers. To eliminate lookahead target leakage, contemporary news was strictly excluded from historical feature engineering rather than falsely presenting 2026 news as historical observations.

## 5. Financial Backtesting & Execution Assumptions

| Financial Metric | Strategy | Buy & Hold Benchmark |
|:---|:---:|:---:|
| **cumulative_return** | 0.0000 | 0.0835 |
| **sharpe_ratio** | N/A (cash) | 0.5417 |
| **sortino_ratio** | N/A (cash) | 0.8429 |
| **max_drawdown** | 0.0000 | 0.2607 |
| **annualized_volatility** | 0.0000 | 0.5776 |
| **win_rate** | 0.0000 | 49.2386 |
| **num_trades** | 0 | 1 |

- **Transaction cost:** 10 bps one-way per position change (0.10% notional)
- **Return conversion:** Realized simple return $R = \exp(r) - 1$ where $r$ is forward log return
- **Wealth Index:** Initial wealth $W_0 = 1.0$; peak-to-trough drawdown begins tracking at $1.0$
- **Annualization Factor:** 365 periods/year

## 6. Limitations, Constraints & Free API Landscape

- **Absence of Free Live X APIs:** Following X's 2023 API paywall changes (v2 reading requires \$100/mo minimum) and guest token deprecation, there is no authorized, free API for continuous live X post collection. Free-tier scraper actors (Apify) encounter strict monthly compute ceilings and bot detection.
- **Historical Archive Utility:** Open-source archives (such as `StephanAkkerman/financial-tweets`, MIT license) represent the only zero-cost, legal, and reproducible method for quantitative sentiment validation, but cover up to September 2024 rather than rolling recent 180 days.
- **Engagement Latency:** Historical engagement counts (likes/reposts) reflect time-of-collection rather than initial publication.
- **Execution Friction:** Backtest simulates 10 bps friction per position change at period close; live execution introduces bid-ask spread and market impact.

## 7. Conclusion & Research Findings

**Empirical Finding**: Out-of-sample walk-forward validation demonstrates that incorporating X social sentiment features resulted in **+0.00% relative MAE improvement** over historical price patterns alone for BTCUSDT.

Across all evaluated horizons ($n+1 \dots n+5$):
1. **Naive Baseline Superiority**: The zero-return naive baseline achieved the lowest cross-validation MAE (0.016470 vs 0.018676+ for ML models), reflecting the low signal-to-noise ratio inherent in daily financial asset returns.
2. **Model Variance Penalty**: Adding 19 additional sentiment features to machine learning estimators (Ridge, Random Forest) increased validation error relative to price-only models due to estimation variance on noisy social signals.
3. **Scientific Honesty**: The empirical evidence does NOT support the claim that daily aggregated X sentiment reliably forecasts next-session price movements out-of-sample under rigorous, purged walk-forward cross-validation.
