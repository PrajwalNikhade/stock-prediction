# CrowdWisdomTrading — X Sentiment Price Prediction Agent

A quantitative time-series research tool and machine learning application that predicts asset prices using historical X (Twitter) sentiment and financial price data. Built for the CrowdWisdomTrading Quantitative Data Scientist assessment.

---

## 📌 Project Overview

CrowdWisdomTrading combines:
* **Alternative Data**: Historical X posts via Apify, recent financial news via Tavily and Exa.
* **Financial Price Data**: High-resolution OHLCV price histories from Binance (Crypto) and Yahoo Finance (Equities/ETFs), plus custom CSV ingestion.
* **Hybrid Sentiment Engine**: VADER rule-based sentiment + engagement-weighted scoring (log-scaled likes, reposts, views, follower reach) with optional financial FinBERT transformers.
* **Rigorous Quantitative Features**: Point-in-time, backward-looking features with strict anti-leakage controls.
* **Walk-Forward Validation**: Chronological expanding-window validation with horizon-aware embargo/gap to eliminate target overlap leakage.
* **Direct Multi-Horizon Forecasting**: 6 forecast output slots ($n+0$ nowcast through $n+1 \dots n+5$ forward horizons) with $\exp(\hat{r}_h)$ conversion.
* **Strategy Backtesting**: Out-of-sample directional trading with realistic transaction costs (10 bps default) compared against Buy-and-Hold.
* **Interactive Research Dashboard**: Full-featured 7-tab Streamlit dashboard with Plotly charts, executive KPIs, model inspections, and logs.

---

## 📁 Repository Structure

```
stock/
├── app.py                      # Main Streamlit dashboard application (7 interactive tabs)
├── requirements.txt            # Project dependencies
├── .env.example                # Sample environment configuration template
├── README.md                   # Complete documentation and user guide
├── config/
│   ├── assets.yaml             # Asset registry (Crypto, US Equities, Commodities, Indices)
│   └── settings.py             # Central application configuration & environment loader
├── src/
│   ├── ingestion/              # Data ingestion adapters
│   │   ├── apify_client.py     # Apify actor integration for X posts
│   │   ├── price_providers.py  # Binance public API & Yahoo Finance & CSV loaders
│   │   ├── tavily_client.py    # Tavily news discovery
│   │   └── exa_client.py       # Exa neural search client
│   ├── processing/             # Data cleaning, sentiment, & feature engineering
│   │   ├── data_cleaning.py    # Deduplication, text normalization, timestamp validation
│   │   ├── sentiment.py        # Hybrid VADER + engagement weighting
│   │   ├── aggregation.py      # Hourly/daily UTC sentiment period aggregations
│   │   └── feature_engineering.py # Point-in-time features & target generation
│   ├── modeling/               # Machine learning & backtesting
│   │   ├── train.py            # Chronological walk-forward validation & candidate models
│   │   ├── forecasting.py      # Direct multi-horizon forecast matrix (n+0 to n+5)
│   │   ├── metrics.py          # MAE, RMSE, R², Dir. Accuracy, Sharpe, Sortino, Drawdown
│   │   └── backtesting.py      # Out-of-sample directional trading backtest
│   ├── database/               # Local persistence
│   │   ├── db.py               # SQLite schema initialization
│   │   └── repository.py       # CRUD operations for prices, posts, models, and forecasts
│   ├── reporting/              # Markdown report generators
│   │   ├── sentiment_report.py # One-page current sentiment outlook generator
│   │   └── evaluation_report.py# Comprehensive model evaluation report generator
│   └── utils/
│       ├── demo_data.py        # Synthetic deterministic demo data generator
│       ├── logging_config.py   # Standardized logging configuration
│       └── time_utils.py       # Strict UTC datetime parsers and converters
├── scripts/
│   ├── run_live_workflow.py    # Main reproducible live workflow entrypoint
│   ├── generate_reports.py     # Pipeline runner and report generator
│   └── check_live_apis.py      # Diagnostic tool to verify external API credentials
├── reports/                    # Generated Markdown reports
│   ├── current_sentiment_outlook.md
│   └── evaluation_report.md
└── tests/
    └── test_all.py             # Comprehensive test suite (35 tests covering all pipeline stages)
```

---

## 🚀 Quickstart

### 1. Installation

Ensure Python 3.10+ (tested on Python 3.13) is installed:

```bash
git clone <repo-url>
cd stock
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to supply API tokens (optional for demo/testing mode):
```ini
APIFY_API_TOKEN=your_apify_token_here
TAVILY_API_KEY=your_tavily_api_key_here
EXA_API_KEY=your_exa_api_key_here
DATABASE_PATH=data/crowdwisdom.db
TRANSACTION_COST_BPS=10
```

> **Note on Graceful Degradation**: If external API credentials are not set, the platform operates seamlessly in **Demo Mode**, generating clearly-labeled synthetic data for testing and interactive dashboard demonstration. Public crypto price feeds (Binance) work live with zero credentials!

### 3. Running the Streamlit Dashboard

Launch the interactive web application:

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

### 4. Running the CLI Pipeline & Generating Reports

Run the end-to-end pipeline and generate fresh markdown reports:

```bash
python scripts/generate_reports.py
```

Outputs will be saved in:
* `reports/current_sentiment_outlook.md`
* `reports/evaluation_report.md`

### 5. Running the Test Suite

Execute all 29 automated tests:

```bash
pytest tests/test_all.py -v
```

---

## 🔬 Methodology & Architecture

### Hybrid Sentiment Analysis
Post sentiment is calculated via a dual-layer approach:
1. **Rule-Based Engine**: VADER sentiment provides robust text polarity scores (`compound`, `pos`, `neg`, `neu`) calibrated for social media informal language and financial cashtags.
2. **Engagement Weighting**: High-impact tweets from influential accounts receive higher weight in period aggregations using a bounded logarithmic formula:
   $$\text{weight} = \ln(1 + \text{likes}) + 0.5 \ln(1 + \text{reposts}) + 0.3 \ln(1 + \frac{\text{views}}{1000}) + 0.2 \ln(1 + \frac{\text{followers}}{1000})$$
   (capped at 10.0 to prevent outlier distortion).

### Point-in-Time Feature Engineering & Leakage Prevention
To ensure institutional-grade validity:
* All features use **backward-looking lags only** ($t-1, t-2, t-3, t-5$).
* Missing sentiment periods are explicitly flagged (`sentiment_missing = 1`) and differentiated from 0.0 scores.
* Targets are computed as future log returns:
  $$r_{t, h} = \ln\left(\frac{P_{t+h}}{P_t}\right)$$
* Feature standard scalers are fitted **only** on the training split of each fold and applied to validation/test without lookahead.

### Walk-Forward Cross-Validation
* Uses chronological expanding windows across $K$ folds.
* Incorporates a **horizon-aware embargo gap** ($\text{gap} = h + 1$) between the training window and validation set to ensure multi-step forward return targets never overlap between splits.
* Compares candidate estimators (**Ridge Regression, HistGradientBoosting, Random Forest, XGBoost**) against a strict **Naive Random-Walk Baseline** (zero return prediction).

### Forecast Matrix
Generates 6 distinct slots:
* **$n+0$**: Current-period reference/nowcast (observed close price).
* **$n+1$ to $n+5$**: Forward multi-step predictions generated by dedicated direct models, converting predicted log return to price:
  $$\hat{P}_{n+h} = P_n \times \exp(\hat{r}_h)$$

---

## 🧪 Verification & Test Coverage

The test suite in `tests/test_all.py` includes 29 deterministic test cases:
* Data cleaning: text normalization, deduplication, UTC timestamp parsing, price normalization.
* Missing values: engagement weight handling, empty text fallback.
* Sentiment: output ranges $[-1, 1]$, neutral text handling.
* Aggregation: hourly and daily UTC boundaries.
* Temporal integrity: anti-leakage checks, target correlation tests, monotonic time ordering.
* Validation: expanding-window walk-forward execution, naive baseline comparison.
* Financial metrics: Sharpe ratio, Sortino ratio, Max Drawdown, transaction cost deduction.
* Database & utilities: SQLite table creation, repository roundtrips, demo data generation.

---

## ⚠️ Limitations & Disclaimers

1. **Market Efficiency**: Sentiment signals exhibit regime-dependent predictive decay. Out-of-sample models should always be compared against naive zero-return baselines.
2. **Engagement Lag**: Scraping APIs capture engagement metrics at collection time rather than publication time.
3. **Execution Assumptions**: Backtest calculations assume fill at bar close and fixed 10 bps transaction costs without market impact or slippage modeling.
4. **Educational Use**: This software is intended strictly for quantitative research and evaluation. It does not constitute investment advice.
