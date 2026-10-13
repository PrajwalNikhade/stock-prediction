# CrowdWisdomTrading — Evaluation Report

> ℹ️ **PARTIAL DATA REPORT — Market prices and news were retrieved from live external APIs, but social sentiment data was uncollected (Apify 401/unconfigured).**
> **Features were constructed with missing-sentiment flags (`sentiment_missing = 1`). No synthetic data was fabricated.**

**Run Identifier (run_id):** `run_20261013_164832`  
**Generated:** 2026-10-13T16:49:02Z  
**Asset:** BTCUSDT  
**Timeframe:** 1d  

## Executive Summary

This report evaluates the predictive performance of price forecasting models
incorporating sentiment features for **BTCUSDT** at **1d** frequency.
The selected model is **Naive** based on expanding-window walk-forward validation.

## Data Sources and Provenance

| Pipeline Component | Source / Provider | Records / Observations | Status & Provenance Details |
|---------------------|-------------------|------------------------|-----------------------------|
| **Market Prices** | binance | 180 bars (2026-04-17T00:00:00Z to 2026-10-13T00:00:00Z) | Live exchange REST API (Binance public klines) |
| **Social Posts (X)** | Apify (`apidojo/tweet-scraper`) | 0 posts (N/A (No posts collected)) | Live retrieved (0 posts) (0 synthetic posts injected) |
| **News Feeds** | Tavily Search + Exa AI | 20 articles | Live search feeds retrieved and deduplicated |
| **Database Storage** | SQLite (`data/crowdwisdom.db`) | Tagged with `run_id` | WAL mode, foreign keys, point-in-time provenance |

## Feature Engineering

- **Total features:** 34
- **Sentiment features:** mean_sentiment, median_sentiment, positive/negative/neutral ratios,
  net_sentiment, sentiment_std, sentiment_change, sentiment_momentum, lags (1,2,3,5),
  post_count, engagement_weighted_sentiment, recent_sentiment_score
- **Price features:** close_price, lagged_close (1,2,3,5), returns (1,2,3,5),
  rolling_volatility, rolling_mean_return, momentum, high_low_range, open_close_return
- **Time alignment:** Backward-looking, point-in-time aggregation (UTC)
- **Missing handling:** Explicit sentiment_missing flag; zero-fill only for verified empty periods

## Modeling Methodology

- **Target:** Log return = ln(close_{t+h} / close_t)
- **Horizons:** n+0 (nowcast), n+1 through n+5
- **Candidate models:** Naive (random walk), Ridge, Random Forest, HistGBR, XGBoost
- **Validation:** Expanding-window walk-forward with horizon-aware gap
- **Selection criterion:** Minimum validation MAE
- **Leakage controls:** No future data in features; scaler fit on training only

## Predictive Performance

### Walk-Forward Validation (averaged across folds)

| Model | MAE | RMSE | R² | Dir. Accuracy | Folds |
|-------|-----|------|----|---------------|-------|
| Ridge | 0.024878 | 0.031937 | -1.8975 | 44.8% | 3 |
| RandomForest | 0.016896 | 0.022781 | -0.3244 | 43.8% | 3 |
| HistGBR | 0.016920 | 0.022937 | -0.3843 | 56.2% | 3 |
| XGBoost | 0.017406 | 0.023720 | -0.4276 | 51.4% | 3 |
| Naive | 0.014713 | 0.020516 | -0.0760 | 0.0% | 3 |

### Final Test Set Performance

| Metric | Best Model (Naive) | Naive Baseline |
|--------|--------------------------|----------------|
| mae | 0.012510 | 0.012510 |
| rmse | 0.019157 | 0.019157 |
| r_squared | -0.006112 | -0.006112 |
| directional_accuracy | 0.000000 | 0.000000 |
| smape | 200.000000 | 200.000000 |
| prediction_bias | -0.001493 | -0.001493 |

## Financial Performance

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| cumulative_return | 0.0000 | 0.0484 |
| sharpe_ratio | N/A (flat cash) | 1.2237 |
| sortino_ratio | N/A (flat cash) | 2.2416 |
| max_drawdown | 0.0000 | 0.0573 |
| annualized_volatility | 0.0000 | 0.3075 |
| win_rate | 0.0000 | 52.7778 |
| num_trades | 0 | 0 |

- **Transaction cost:** 10 bps per trade
- **Risk-free rate:** 0 (assumption)
- **Observations:** 36
- **Annualization factor:** 252 periods/year

## Limitations

- Historical X post coverage may be incomplete; API limits and scraper constraints apply.
- Selection bias: only posts matching search queries are observed.
- Engagement values reflect current state, not values at original posting time.
- News availability limited by API quotas and recency restrictions.
- Market regime changes may invalidate historically trained models.
- Model uncertainty is inherent; prediction intervals not statistically validated.
- Financial metrics may be unstable with short evaluation periods.
- Backtesting assumes perfect execution; real-world slippage and liquidity costs differ.
- The difference between backtesting and live trading can be substantial.

## Conclusion

The Naive model was selected based on walk-forward validation for BTCUSDT.
The selected model did not consistently outperform the naive baseline,
suggesting limited predictive value from the available sentiment data.
These results should be interpreted with caution given the limitations above.
