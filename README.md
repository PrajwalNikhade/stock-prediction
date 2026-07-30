# SmartStock AI — Explainable Indian Stock Market Analytics & Prediction Platform

A production-quality, portfolio-grade full-stack application combining real-time Indian stock market analytics with explainable AI predictions.

![Tech Stack](https://img.shields.io/badge/Next.js-15-black?logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue?logo=typescript)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

### 📊 Real-Time Dashboard
- Live market status (IST-aware open/close detection)
- Nifty 50 & Sensex indices with change tracking
- Top Gainers & Losers (calculated from NSE stocks)
- Sector Performance Heatmap (10 sectors)
- Market News with sentiment indicators
- Watchlist widget

### 🔍 Stock Search & Analysis
- Autocomplete search across 50 curated NSE stocks
- Recent search history tracking
- Comprehensive stock info: OHLCV, 52-week range, market cap, P/E, EPS, beta

### 📈 Interactive Charts
- Candlestick charts with volume overlay (Plotly.js)
- Multiple time periods: 1M, 3M, 6M, 1Y, 2Y, 5Y
- Feature importance horizontal bar charts
- Technical indicator visualizations

### 🤖 ML Predictions (XGBoost)
- **1-Day, 7-Day, 30-Day** price forecasts
- 15+ engineered features: SMA, EMA, MACD, RSI, ADX, ATR, Bollinger Bands, VWAP, OBV, CCI, Momentum, ROC, Williams %R, Stochastic, lag features, rolling statistics
- TimeSeriesSplit cross-validation (5 folds)
- Model metrics: MAE, RMSE, R² with standard deviations
- Prediction confidence scores

### 🧠 Explainable AI
- Feature importance visualization (top 15 features)
- Top contributing factors for each prediction
- Detailed signal breakdown for recommendations
- No black-box LLMs — fully transparent rule engine

### 🕯️ Candlestick Pattern Detection
- 8 patterns: Hammer, Doji, Morning Star, Evening Star, Bullish/Bearish Engulfing, Harami, Shooting Star
- Bullish/bearish bias summary
- Pattern descriptions and significance ratings

### 📰 News & Sentiment
- Google News RSS feed integration
- TextBlob sentiment analysis (polarity scoring)
- Positive/Neutral/Negative classification
- Sentiment summary per stock
- Sentiment scores feed into ML predictions

### 💡 AI Recommendations
- Buy / Hold / Sell based on:
  - RSI (oversold/overbought zones)
  - MACD crossover signals
  - ADX trend strength
  - Bollinger Band positioning
  - Stochastic Oscillator
  - ML prediction direction & confidence
  - News sentiment score
- Detailed signal-by-signal breakdown
- Confidence percentage with composite scoring

### ⭐ Watchlist & 🔔 Alerts
- Full CRUD watchlist with pin/favorite support
- Price alerts (above/below)
- RSI alerts (above/below)
- Persistent storage in MySQL/SQLite

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 15, TypeScript, Tailwind CSS, React Query, Plotly.js, Framer Motion, Lucide Icons |
| **Backend** | Flask, Flask-CORS, SQLAlchemy, Flask-Migrate |
| **Database** | MySQL (production) / SQLite (development) |
| **ML** | XGBoost, Scikit-learn, Pandas, NumPy |
| **Technical Analysis** | ta library (30+ indicators) |
| **NLP** | TextBlob, feedparser |
| **Data** | Yahoo Finance (yfinance) |

---

## Project Structure

```
Stock/
├── frontend/                  # Next.js 15 App Router
│   ├── src/
│   │   ├── app/               # Pages (/, /stocks, /stocks/[symbol], /watchlist, /alerts, /about)
│   │   ├── components/        # Reusable components (charts, layout, ui)
│   │   ├── hooks/             # React Query hooks
│   │   ├── lib/               # API client, types, utilities
│   │   └── providers/         # Query & Theme providers
│   └── package.json
│
├── backend/                   # Flask REST API
│   ├── app/
│   │   ├── models/            # SQLAlchemy ORM models (6 tables)
│   │   ├── routes/            # API blueprints (market, stocks, prediction, news, watchlist, alerts)
│   │   ├── services/          # Business logic (stock, prediction, news, recommendation, candlestick, market)
│   │   ├── ml/                # XGBoost model, feature engineering
│   │   └── utils/             # NSE stocks list
│   ├── requirements.txt
│   └── run.py
│
├── database/
│   └── schema.sql             # MySQL schema
│
└── README.md
```

---

## Installation & Setup

### Prerequisites
- **Node.js** 20+ and npm
- **Python** 3.10+
- **MySQL** (optional — SQLite is used by default)

### 1. Clone & Setup Backend

```bash
cd Stock/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Copy environment config
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# Run the server
python run.py
```

The Flask API starts at **http://localhost:5000**.

### 2. Setup Frontend

```bash
cd Stock/frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

The Next.js app starts at **http://localhost:3000**.

### 3. (Optional) MySQL Setup

```bash
# Create database and run schema
mysql -u root -p < database/schema.sql

# Update backend/.env
# DATABASE_URL=mysql+pymysql://root:password@localhost:3306/smartstock
```

---

## API Documentation

### Market Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/market/status` | Market open/close status |
| GET | `/api/market/indices` | Nifty 50 & Sensex data |
| GET | `/api/market/gainers` | Top gaining stocks |
| GET | `/api/market/losers` | Top losing stocks |
| GET | `/api/market/sectors` | Sector performance |

### Stock Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks/search?q=` | Autocomplete search |
| GET | `/api/stocks/{symbol}` | Stock info |
| GET | `/api/stocks/{symbol}/history?period=1y` | Historical OHLCV |
| GET | `/api/stocks/{symbol}/indicators` | Technical indicators |
| GET | `/api/stocks/{symbol}/candlestick-patterns` | Pattern detection |

### Prediction Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/predict/{symbol}` | ML predictions (1d, 7d, 30d) |
| GET | `/api/predict/{symbol}/importance` | Feature importance |
| GET | `/api/predict/{symbol}/metrics` | Model evaluation metrics |
| GET | `/api/predict/{symbol}/recommendation` | AI Buy/Hold/Sell |

### News Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/news/market` | Market news |
| GET | `/api/news/{symbol}` | Stock news + sentiment |

### Watchlist Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/watchlist` | Get watchlist |
| POST | `/api/watchlist` | Add to watchlist |
| PUT | `/api/watchlist/{id}` | Update pin/favorite |
| DELETE | `/api/watchlist/{id}` | Remove from watchlist |

### Alert Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/alerts` | Get alerts |
| POST | `/api/alerts` | Create alert |
| PUT | `/api/alerts/{id}` | Update alert |
| DELETE | `/api/alerts/{id}` | Delete alert |

---

## Database Schema (ER Diagram)

```
┌─────────────┐     ┌─────────────┐     ┌──────────────┐
│   USERS     │────<│  WATCHLIST   │     │PREDICTIONS   │
│─────────────│     │─────────────│     │   CACHE      │
│ id (PK)     │     │ id (PK)     │     │──────────────│
│ username    │     │ user_id(FK) │     │ id (PK)      │
│ email       │     │ symbol      │     │ symbol       │
│ password    │     │ company_name│     │ pred_1d      │
│ created_at  │     │ is_pinned   │     │ pred_7d      │
│ updated_at  │     │ is_favorite │     │ pred_30d     │
└─────────────┘     │ added_at    │     │ confidence   │
       │            └─────────────┘     │ feature_imp  │
       │                                │ metrics      │
       │            ┌─────────────┐     │ recommendation│
       └───────────<│   ALERTS    │     └──────────────┘
                    │─────────────│
                    │ id (PK)     │     ┌──────────────┐
                    │ user_id(FK) │     │ NEWS_CACHE   │
                    │ symbol      │     │──────────────│
                    │ alert_type  │     │ id (PK)      │
                    │ target_value│     │ symbol       │
                    │ is_active   │     │ title        │
                    │ is_triggered│     │ sentiment    │
                    └─────────────┘     │ published_at │
       │                                └──────────────┘
       │            ┌──────────────┐
       └───────────<│RECENT_SEARCH │
                    │──────────────│
                    │ id (PK)      │
                    │ user_id (FK) │
                    │ symbol       │
                    │ searched_at  │
                    └──────────────┘
```

---

## Deployment

### Frontend → Vercel

```bash
cd frontend
npx vercel --prod
```

Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`

### Backend → Render

1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn run:app`
4. Add environment variables: `DATABASE_URL`, `SECRET_KEY`, `FRONTEND_URL`

---

## Key Design Decisions

1. **XGBoost only** — focused model with proven performance on tabular financial data
2. **Rule-based recommendations** — fully transparent, no LLM/ChatGPT dependency
3. **Yahoo Finance** — free, reliable, no API key needed (.NS suffix for NSE)
4. **TextBlob** — lightweight sentiment analysis, no external API calls
5. **SQLite default** — zero-config development, MySQL for production
6. **Prediction caching** — avoids re-training on every request (24h TTL)
7. **TimeSeriesSplit** — proper time-series cross-validation to prevent lookahead bias

---

## License

MIT — Educational portfolio project. Not financial advice.
