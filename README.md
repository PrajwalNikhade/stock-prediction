# CrowdWisdomTrading — X Sentiment Price Prediction Agent

A quantitative time-series research tool and machine learning application that evaluates whether social sentiment from X (Twitter) provides predictive power for financial asset prices beyond historical market data alone. Built for the CrowdWisdomTrading Quantitative Data Scientist assessment.

---

## 📌 Project Overview

CrowdWisdomTrading combines:
* **Alternative Data**: Historical X posts via open-source financial tweet archives (`StephanAkkerman/financial-tweets`, MIT license), live X scraping via Apify, and financial news via Tavily and Exa.
* **Financial Price Data**: High-resolution OHLCV price histories from Binance (Crypto) and Yahoo Finance (Equities/ETFs), with calendar-aware trading sessions.
* **Hybrid Sentiment Engine**: VADER rule-based sentiment + engagement-weighted scoring (log-scaled likes, reposts, views, follower reach) with optional financial FinBERT transformers.
* **Rigorous Quantitative Features**: Point-in-time, backward-looking features with strict anti-leakage controls.
* **Walk-Forward Validation**: Chronological expanding-window validation with horizon-aware embargo/gap to eliminate target overlap leakage.
* **Direct Multi-Horizon Forecasting**: 6 forecast output slots ($n+0$ nowcast through $n+1 \dots n+5$ forward horizons) with $\exp(\hat{r}_h)$ conversion.
* **Fair Multi-Variant Experiment Suite**: Side-by-side evaluation of Experiment A (Naive), B (Price-Only), C (Price + X Sentiment), D (Engagement-Enhanced), and E (News-Enhanced).
* **Strategy Backtesting**: Out-of-sample directional trading with realistic transaction costs (10 bps default) compared against Buy-and-Hold.
* **Interactive Research Dashboard**: Full-featured 7-tab Streamlit dashboard with Plotly charts, executive KPIs, model inspections, and logs.
* **Automated Test Suite**: 45 unit, integration, and regression tests ensuring quantitative soundness and software reliability.

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
├── data/
│   ├── crowdwisdom.db          # SQLite database (WAL mode, schema versioned, gitignored)
│   └── hf_sample.parquet       # Historical financial tweet archive (StephanAkkerman/financial-tweets)
├── src/
│   ├── ingestion/              # Data ingestion adapters
│   │   ├── apify_client.py     # Apify actor integration for live X posts
│   │   ├── price_providers.py  # Binance public API & Yahoo Finance & CSV loaders
│   │   ├── dataset_importer.py # Historical CSV & Parquet importer with date overlap diagnostics
│   │   ├── tavily_client.py    # Tavily news discovery
│   │   └── exa_client.py       # Exa neural search client
│   ├── processing/             # Data cleaning, sentiment, & feature engineering
│   │   ├── data_cleaning.py    # Deduplication, text normalization, timestamp validation
│   │   ├── sentiment.py        # Hybrid VADER + engagement weighting
│   │   ├── aggregation.py      # Hourly/daily UTC sentiment period aggregations
│   │   └── feature_engineering.py # Point-in-time features & target generation
│   ├── modeling/               # Machine learning & backtesting
│   │   ├── train.py            # Chronological walk-forward validation & multi-variant experiment suite
│   │   ├── forecasting.py      # Direct multi-horizon forecast matrix (n+0 to n+5)
│   │   ├── metrics.py          # MAE, RMSE, R², Dir. Accuracy, Sharpe, Sortino, Drawdown
│   │   └── backtesting.py      # Out-of-sample directional trading backtest
│   ├── database/               # Local persistence
│   │   ├── db.py               # SQLite schema initialization
│   │   └── repository.py       # CRUD operations for prices, posts, models, and forecasts
│   ├── reporting/              # Markdown report generators
│   │   ├── sentiment_report.py # Sentiment outlook report generator
│   │   └── evaluation_report.py# Comprehensive multi-asset evaluation report generator
│   └── utils/
│       ├── demo_data.py        # Synthetic deterministic demo data generator
│       ├── logging_config.py   # Standardized logging configuration
│       └── time_utils.py       # Strict UTC datetime parsers and converters
├── scripts/
│   ├── run_live_workflow.py    # Main reproducible live workflow entrypoint
│   ├── generate_reports.py     # End-to-end pipeline runner and report generator
│   └── check_live_apis.py      # Diagnostic tool to verify external API credentials
├── reports/                    # Generated Markdown reports
│   ├── current_sentiment_outlook.md
│   └── evaluation_report.md
└── tests/
    └── test_all.py             # Comprehensive 45-test suite
```

---

## 🚀 Installation & Setup

### 1. Requirements

* **Python 3.10+** (Tested and verified on Python 3.13.1).
* Git for version control.

### 2. Installation

Clone the repository and install required Python packages:

```bash
git clone https://github.com/PrajwalNikhade/stock-prediction.git
cd stock-prediction
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure API tokens in `.env` (optional for historical benchmark and demo modes):

```ini
APIFY_API_TOKEN=your_apify_token_here
TAVILY_API_KEY=your_tavily_api_key_here
EXA_API_KEY=your_exa_api_key_here
DATABASE_PATH=data/crowdwisdom.db
TRANSACTION_COST_BPS=10
```

> **Zero-Credential Operation**: Public market data feeds (Binance for crypto, Yahoo Finance for equities) and the historical X archive require **no paid API keys** and operate out-of-the-box. If Apify or news keys are missing or exceed usage limits, the system automatically enters `PARTIAL` data mode without fabricating synthetic posts.

---

## 📊 Data Modes & Limitations

To ensure absolute scientific integrity, the platform strictly segregates execution modes:

| Mode | Banner / Tag | Description | Data Integrity Safeguards |
|:---|:---|:---|:---|
| **HISTORICAL** | `🟡 HISTORICAL ARCHIVE BENCHMARK` | Evaluates genuine historical X posts (`StephanAkkerman/financial-tweets`) aligned with historical exchange prices (Dec 2021 – Sep 2024). | Never labeled as live data. Prices and sentiment explicitly dated. Contemporary news excluded to prevent lookahead leakage. |
| **LIVE** | `🟢 LIVE DATA` | Current real-time market prices and verified live X posts newly retrieved via Apify. | Requires active, non-quota-exhausted API scraper. Verified via test ping. |
| **PARTIAL** | `ℹ️ PARTIAL LIVE DATA` | Live market prices and news active, but social X scraper quota reached (0 live posts). | Features flagged with `sentiment_missing = 1`. No synthetic posts fabricated. |
| **DEMO** | `⚠️ DEMO MODE` | Explicitly requested synthetic fixtures for UI walkthrough. | Banners on all charts and reports clearly declare demonstration status. |

### Data Source Limitations & Free API Landscape

1. **Absence of Free Live X APIs**:
   * Following X's July 2023 API pricing changes, standard read access requires the **Pro tier ($100/month minimum)**, and guest tokens were deprecated.
   * Free-tier scraping actors (e.g. Apify `apidojo/tweet-scraper`) operate on monthly compute credit allowances ($5/month free). Once exhausted, no new live posts are returned. The platform transparently detects quota exhaustion and enters `PARTIAL` mode.
2. **Historical Archive Date Bounds**:
   * The genuine historical X dataset (`StephanAkkerman/financial-tweets`, MIT license) spans **December 27, 2021 through September 5, 2024**.
   * **Explicit Limitation**: This historical dataset ends in September 2024 and does **NOT** satisfy a rolling, recent 180-day X collection requirement. Old historical tweets are never labeled as current real-time sentiment.
3. **Historical News Availability (Experiment E)**:
   * Free news search APIs (Tavily, Exa) retrieve current contemporary news articles. Historical point-in-time financial news archives spanning 2021–2024 are unavailable on free tiers.
   * To prevent lookahead leakage, contemporary news is strictly excluded from historical experiments. **Experiment E is explicitly designated as UNAVAILABLE / UNTESTED** in historical benchmarks.

---

## 🏃 Running Pipelines & Generating Reports

### 1. Historical Benchmark Pipeline (Recommended for Quantitative Audit)

Run the full walk-forward validation and multi-horizon experiment comparison suite on genuine historical X sentiment:

**For Bitcoin (BTCUSDT):**
```bash
python scripts/generate_reports.py --asset BTCUSDT --timeframe 1d --use-historical-x
```

**For NVIDIA (NVDA US Equities):**
```bash
python scripts/generate_reports.py --asset NVDA --timeframe 1d --use-historical-x
```

### 2. Live Market Pipeline

Run with live market price feeds and check for newly scraped X posts:

```bash
python scripts/generate_reports.py --asset BTCUSDT --timeframe 1d
```

### 3. Explicit Demo Mode

Run deterministic demonstration fixtures for dashboard testing:

```bash
python scripts/generate_reports.py --demo
```

### 4. Interactive Streamlit Dashboard

Launch the web dashboard:

```bash
streamlit run app.py
```

The application provides 7 comprehensive tabs:
1. **Executive Summary**: Core metrics, data mode banner, and recent model performance.
2. **Live Data Ingestion**: Real-time price chart, social post feed, and collection logs.
3. **Sentiment Analysis**: Sentiment distributions, engagement-weighted vs. raw scores, and trend diagnostics.
4. **Model Performance**: Walk-forward validation curves, residual distributions, and feature importance.
5. **Multi-Horizon Forecasts**: Direct forecasts from $n+0$ through $n+5$ with prediction intervals.
6. **Financial Backtest**: Equity curves, drawdown charts, monthly returns heatmap, and trade logs.
7. **Multi-Variant Experiment Suite**: Side-by-side comparison of Experiments A, B, C, D, and E across all horizons.

---

## 🧪 Automated Test Suite (45 Passing Tests)

The test suite enforces quantitative correctness, data integrity, and strict leakage prevention.

Run the complete test suite:

```bash
python -m pytest tests/test_all.py -v
```

### Test Suite Structure (45 Tests):

| Category | Test Cases | Scope Verified |
|:---|:---:|:---|
| **Data Normalization & Cleaning** | 5 | Price OHLCV schema, strict UTC timestamp formatting, post text normalization, SHA-256 deduplication, missing engagement preservation. |
| **Sentiment Analysis** | 2 | VADER polarity bounds $[-1, 1]$, empty/whitespace text handling, neutral classifications. |
| **Temporal Aggregation** | 2 | UTC hourly and daily period boundaries, point-in-time calculation. |
| **Leakage & Feature Integrity** | 4 | No future post leakage, target generation non-overlap, feature column completeness, zero leakage across features. |
| **Walk-Forward Validation** | 3 | Chronological split ordering, expanding window execution, zero-leakage horizon purge embargo gap ($gap \ge h$). |
| **Baseline & Financial Metrics** | 7 | Naive baseline benchmark, Sharpe ratio, Sortino ratio, Max Drawdown with $W_0=1.0$, transaction cost deduction (10 bps), degenerate strategy handling (NaN Sharpe for all-cash), log-to-simple return conversion. |
| **Database & Persistence** | 3 | SQLite schema creation, CRUD operations, run provenance and `run_id` tracking, experiment evaluation table persistence. |
| **Credentials & Mode Separation** | 3 | Missing API credentials graceful handling, live vs demo mode isolation, connection test diagnostics without secret exposure. |
| **Trading Calendar Logic** | 1 | Equity trading session timestamps (skipping weekend non-trading days) vs continuous 24/7 crypto. |
| **Multi-Schema Ingestion** | 4 | Apify multi-schema normalization, Actor input generation, historical CSV importer, date coverage and overlap diagnostics. |
| **Multi-Variant Experiment Suite** | 2 | End-to-end execution of Experiments A, B, C, D across horizons 1..5, evaluation persistence. |
| **Report Generation & Edge Cases** | 4 | Markdown report creation, MAPE near-zero handling, low observation safeguards. |

---

## 🔬 Key Quantitative Findings

### 0.00% Relative MAE Improvement over Naive Baseline

Empirical out-of-sample walk-forward validation across both **BTCUSDT** and **NVDA** demonstrates that incorporating daily X sentiment features resulted in **+0.00% relative MAE improvement** over historical price patterns alone:

$$\text{Relative MAE Improvement (\%)} = 100 \times \frac{\text{MAE}_{\text{Price-Only}} - \text{MAE}_{\text{Price+X}}}{\text{MAE}_{\text{Price-Only}}} = +0.00\%$$

### Why the Naive Baseline Dominates:
1. **Low Signal-to-Noise Ratio**: Daily financial asset returns closely follow a martingalian random walk. Predicting the unconditional mean ($\hat{r} = 0$) minimizes expected squared and absolute prediction loss.
2. **Model Variance Penalty**: Supervised ML models (Ridge, Random Forest, HistGBR, XGBoost) fitted on noisy daily sentiment features incurred excess out-of-sample estimation variance, producing higher cross-validation errors than the Naive baseline:
   * **BTCUSDT Horizon 1**: Naive Val MAE = **0.016470** vs. ML Models Val MAE $\in [0.018676, 0.024214]$.
   * **NVDA Horizon 1**: Naive Val MAE = **0.023215** vs. ML Models Val MAE $\in [0.024890, 0.031450]$.
3. **Strict Zero-Snooping Selection**: Because expanding-window walk-forward validation strictly selects the model with the lowest cross-validation MAE, the Naive baseline was selected across Experiments B (Price-Only), C (Price + X), and D (Engagement-Enhanced).
4. **Identical Out-of-Sample Predictions**: All three variants outputted $\hat{r} = 0$ on the test set, producing identical test MAE (0.022104 on BTC, 0.029152 on NVDA) and exactly +0.00% relative improvement.

---

## ⚠️ Disclaimers

* **Not Financial Advice**: This project is developed strictly for quantitative research, methodology validation, and educational assessment. Past performance and backtests do not guarantee future returns.
* **Execution Assumptions**: Backtests simulate 10 bps friction per position change at period close; live market execution entails additional bid-ask spreads, execution delay, and market impact.
