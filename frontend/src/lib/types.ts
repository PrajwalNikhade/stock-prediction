/* ========================================
   SmartStock AI TypeScript Types
   ======================================== */

// --- Market ---
export interface MarketStatus {
  is_open: boolean;
  status: string;
  current_time: string;
  market_hours: string;
  next_open: string | null;
}

export interface IndexData {
  value: number;
  change: number;
  change_percent: number;
  high: number;
  low: number;
  open: number;
  error?: string;
}

export interface MarketIndices {
  nifty50: IndexData;
  sensex: IndexData;
}

export interface SectorData {
  sector: string;
  change_percent: number;
  stocks_count: number;
  direction: "up" | "down";
}

// --- Stock ---
export interface Stock {
  symbol: string;
  name: string;
  sector: string;
}

export interface StockInfo {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  current_price: number;
  previous_close: number;
  change: number;
  change_percent: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  fifty_two_week_high: number;
  fifty_two_week_low: number;
  market_cap: number;
  pe_ratio: number | null;
  eps: number | null;
  dividend_yield: number | null;
  beta: number | null;
  description: string;
  website: string;
  currency: string;
  error?: string;
}

export interface HistoricalData {
  symbol: string;
  period: string;
  interval: string;
  data: {
    dates: string[];
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume: number[];
  };
  count: number;
  error?: string;
}

export interface StockMover {
  symbol: string;
  name: string;
  sector: string;
  price: number;
  change: number;
  change_percent: number;
}

// --- Prediction ---
export interface PredictionHorizon {
  predicted_price: number;
  change: number;
  change_percent: number;
  direction: "up" | "down";
  confidence: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface ModelMetrics {
  mae: number;
  rmse: number;
  r2: number;
  mae_std: number;
  rmse_std: number;
  r2_std: number;
}

export interface PredictionResult {
  symbol: string;
  current_price: number;
  prediction_date: string;
  predictions: {
    "1d"?: PredictionHorizon;
    "7d"?: PredictionHorizon;
    "30d"?: PredictionHorizon;
  };
  metrics: {
    "1d"?: ModelMetrics;
    "7d"?: ModelMetrics;
    "30d"?: ModelMetrics;
  };
  feature_importance: {
    "1d"?: FeatureImportance[];
    "7d"?: FeatureImportance[];
    "30d"?: FeatureImportance[];
  };
  top_factors: FeatureImportance[];
  error?: string;
}

// --- Recommendation ---
export interface Signal {
  factor: string;
  signal: string;
  detail: string;
  weight: "strong" | "moderate" | "weak";
}

export interface Recommendation {
  symbol: string;
  recommendation: "Buy" | "Hold" | "Sell";
  score: number;
  confidence: number;
  reason: string;
  signals: Signal[];
  bullish_count: number;
  bearish_count: number;
}

// --- News ---
export interface NewsArticle {
  title: string;
  url: string;
  source: string;
  published_at: string | null;
  sentiment_score: number;
  sentiment_label: "positive" | "neutral" | "negative";
}

export interface StockNews {
  symbol: string;
  articles: NewsArticle[];
  sentiment_summary: {
    average_score: number;
    overall_label: string;
    counts: { positive: number; neutral: number; negative: number };
    total: number;
  };
}

// --- Technical Indicators ---
export interface TechnicalIndicators {
  symbol: string;
  indicators: {
    rsi: number;
    macd: number;
    macd_signal: number;
    macd_histogram: number;
    adx: number;
    atr: number;
    cci: number;
    williams_r: number;
    stochastic_k: number;
    stochastic_d: number;
    momentum: number;
    roc: number;
    obv: number;
    sma_10: number;
    sma_20: number;
    sma_50: number;
    ema_12: number;
    ema_20: number;
    ema_26: number;
    bb_upper: number;
    bb_middle: number;
    bb_lower: number;
    bb_width: number;
    vwap: number;
    daily_return: number;
    rsi_signal: string;
    macd_signal_text: string;
    trend: string;
  };
}

// --- Candlestick ---
export interface CandlestickPattern {
  pattern: string;
  date: string;
  type: "bullish" | "bearish" | "neutral";
  significance: "very_high" | "high" | "moderate";
  description: string;
  price: number;
}

export interface CandlestickResult {
  symbol: string;
  total: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  bias: string;
  patterns: CandlestickPattern[];
}

// --- Watchlist ---
export interface WatchlistItem {
  id: number;
  user_id: number;
  symbol: string;
  company_name: string;
  is_pinned: boolean;
  is_favorite: boolean;
  added_at: string;
}

// --- Portfolio ---
export interface PortfolioHolding {
  id: number;
  symbol: string;
  company_name: string;
  quantity: number;
  buy_price: number;
  current_price: number;
  total_invested: number;
  current_value: number;
  pnl: number;
  pnl_percent: number;
  sector: string;
  bought_at: string;
}

export interface SectorAllocation {
  sector: string;
  value: number;
  percentage: number;
}

export interface PortfolioData {
  summary: {
    total_invested: number;
    total_current_value: number;
    total_pnl: number;
    total_roi_percent: number;
    holdings_count: number;
  };
  holdings: PortfolioHolding[];
  sector_allocation: SectorAllocation[];
}

// --- Analytics & Insights ---
export interface BusinessInsight {
  id: number;
  type: "sector" | "volatility" | "prediction" | "indicator" | "system";
  category: string;
  title: string;
  description: string;
  impact: "positive" | "warning" | "neutral";
  metric: string;
}

export interface CorrelationData {
  symbols: string[];
  matrix: number[][];
  error?: string;
}

export interface MarketAnalyticsData {
  best_performing: { symbol: string; name: string; sector: string; return_6m: number };
  worst_performing: { symbol: string; name: string; sector: string; return_6m: number };
  most_volatile: { symbol: string; name: string; sector: string; volatility_annual: number };
  highest_volume: { symbol: string; name: string; sector: string; avg_volume: number };
  stocks: Array<{
    symbol: string;
    name: string;
    sector: string;
    return_6m: number;
    volatility_annual: number;
    avg_volume: number;
  }>;
  avg_market_return: number;
  avg_market_volatility: number;
}

// --- Alert ---
export interface Alert {
  id: number;
  user_id: number;
  symbol: string;
  alert_type: "price_above" | "price_below" | "rsi_above" | "rsi_below";
  target_value: number;
  is_active: boolean;
  is_triggered: boolean;
  created_at: string;
  triggered_at: string | null;
}

// --- Recent Search ---
export interface RecentSearch {
  id: number;
  symbol: string;
  company_name: string;
  searched_at: string;
}
