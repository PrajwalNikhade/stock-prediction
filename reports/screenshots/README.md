# CrowdWisdomTrading Dashboard Screenshots Index

This directory contains real visual captures from the running Streamlit application (`http://localhost:8501`) operating on the genuine historical benchmark dataset (`BTCUSDT`, Daily timeframe, Dec 2021 – Sep 2024).

All screenshots are captured at a consistent desktop resolution (1440 × 1000 viewport) without artificial mockups, cropped elements, or exposed secrets.

---

### Screenshot Catalog

| File | Dashboard View | Description | Key Visible Metrics / Elements |
| :--- | :--- | :--- | :--- |
| [`01_dashboard_overview.png`](./01_dashboard_overview.png) | **Executive Overview (Tab 1)** | Asset summary, data mode banner, price history, and key metrics. | `HISTORICAL ARCHIVE BENCHMARK ACTIVE` banner, Last Price `$83,015.98`, 3,098 genuine historical posts, 100% neutral/aggregate breakdown, interactive Plotly price history line, and selected baseline model summary. |
| [`02_sentiment_analysis.png`](./02_sentiment_analysis.png) | **Sentiment & Market Analysis (Tab 2)** | Dual-axis market price and social sentiment alignment. | Primary y-axis (Close Price) overlaid with secondary y-axis (Daily Mean Sentiment bars in green/red), post volume trajectory, and sentiment distribution. |
| [`03_multi_horizon_forecasts.png`](./03_multi_horizon_forecasts.png) | **Forecast Matrix (Tab 3)** | Multi-step forecast generation spanning $n+0$ through $n+5$. | $n+0$ current reference price nowcast (`$83,015.98`), forward predictions for $n+1$ through $n+5$, reference price horizontal benchmark, and multi-horizon price trajectory chart. |
| [`04_model_comparison.png`](./04_model_comparison.png) | **Model Evaluation & Fair Experiment Suite (Tab 4)** | Rigorous side-by-side experiment comparison table across all horizons. | Experiments A (Naive Baseline), B (Price-Only), C (Price + X Sentiment), and D (Engagement-Enhanced); out-of-sample Test MAE, Test RMSE, Directional Accuracy (%), cleanly formatted `N/A` Sharpe/Sortino ratios, and relative MAE improvement metrics. |
| [`05_financial_backtesting.png`](./05_financial_backtesting.png) | **Trading Strategy Backtest (Tab 5)** | Financial simulation incorporating transaction costs (10 bps). | Key metrics (Sharpe Ratio displayed cleanly as `N/A (Cash)` for inactive/cash regimes, Max Drawdown `0.00%`, Cumulative Return `0.00%`, 0 trades), Strategy vs. Buy & Hold wealth accumulation curve, and strategy drawdown chart. |

---

### Verification and Data Mode Notice
- **Data Mode**: All captures reflect `HISTORICAL ARCHIVE` mode.
- **Security Audit**: No API keys, tokens, or environment credentials are displayed in the interface or terminal logs.
- **Reproducibility**: The exact views can be reproduced by launching `streamlit run app.py` and clicking `Load Historical Benchmark` in the sidebar.
