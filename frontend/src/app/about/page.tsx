"use client";

import { motion } from "framer-motion";
import {
  BarChart3, Brain, TrendingUp, Newspaper, Shield,
  Layers, Database, Code, Cpu, GitBranch,
} from "lucide-react";
import { Badge } from "@/components/ui/shared";

const fadeIn = { initial: { opacity: 0, y: 15 }, animate: { opacity: 1, y: 0 } };

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-6 space-y-8">
      {/* Hero */}
      <motion.div {...fadeIn} className="text-center space-y-4 py-8">
        <div className="w-16 h-16 rounded-2xl gradient-brand flex items-center justify-center mx-auto">
          <BarChart3 className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold">
          SmartStock <span className="text-[var(--color-brand)]">AI</span>
        </h1>
        <p className="text-lg text-[var(--muted-foreground)] max-w-2xl mx-auto">
          An Explainable Indian Stock Market Analytics & Prediction Platform powered by
          Machine Learning, Technical Analysis, and Natural Language Processing.
        </p>
      </motion.div>

      {/* Features Grid */}
      <motion.div {...fadeIn} className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {[
          {
            icon: Brain,
            title: "XGBoost Predictions",
            description: "1-day, 7-day, and 30-day price predictions using gradient-boosted decision trees trained on 15+ technical indicators with TimeSeriesSplit cross-validation.",
          },
          {
            icon: TrendingUp,
            title: "Technical Analysis",
            description: "Complete suite of indicators: RSI, MACD, ADX, ATR, Bollinger Bands, VWAP, OBV, CCI, Stochastic, Williams %R, Momentum, ROC, and multiple moving averages.",
          },
          {
            icon: Newspaper,
            title: "Sentiment Analysis",
            description: "Real-time news fetching from Google News RSS with TextBlob-based sentiment analysis. Sentiment scores feed directly into the ML prediction pipeline.",
          },
          {
            icon: Shield,
            title: "Explainable AI",
            description: "Full transparency with feature importance charts, model metrics (MAE, RMSE, R²), prediction confidence scores, and detailed reasoning for every recommendation.",
          },
          {
            icon: Layers,
            title: "Candlestick Patterns",
            description: "Automated detection of 8 candlestick patterns: Hammer, Doji, Morning Star, Evening Star, Bullish/Bearish Engulfing, Harami, and Shooting Star.",
          },
          {
            icon: Cpu,
            title: "Rule-Based Recommendations",
            description: "Buy/Hold/Sell recommendations generated from RSI, MACD, ADX, Bollinger Bands, Stochastic, ML predictions, and news sentiment — no LLM dependency.",
          },
        ].map(({ icon: Icon, title, description }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 * i }}
            className="glass-card p-6"
          >
            <Icon className="w-8 h-8 text-[var(--color-brand)] mb-3" />
            <h3 className="font-semibold mb-2">{title}</h3>
            <p className="text-sm text-[var(--muted-foreground)] leading-relaxed">{description}</p>
          </motion.div>
        ))}
      </motion.div>

      {/* Tech Stack */}
      <motion.div {...fadeIn} className="glass-card p-6">
        <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Code className="w-5 h-5 text-[var(--color-brand)]" /> Technology Stack
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <h3 className="text-sm font-semibold mb-2 text-[var(--color-brand)]">Frontend</h3>
            <div className="flex flex-wrap gap-1.5">
              {["Next.js 15", "TypeScript", "Tailwind CSS", "React Query", "Plotly.js", "Framer Motion", "Lucide Icons"].map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-sm font-semibold mb-2 text-[var(--color-brand)]">Backend</h3>
            <div className="flex flex-wrap gap-1.5">
              {["Flask", "SQLAlchemy", "MySQL/SQLite", "Flask-CORS", "Gunicorn"].map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
          </div>
          <div>
            <h3 className="text-sm font-semibold mb-2 text-[var(--color-brand)]">ML / Data</h3>
            <div className="flex flex-wrap gap-1.5">
              {["XGBoost", "Scikit-learn", "Pandas", "NumPy", "ta", "TextBlob", "yfinance", "feedparser"].map((t) => (
                <Badge key={t}>{t}</Badge>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Architecture */}
      <motion.div {...fadeIn} className="glass-card p-6">
        <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-[var(--color-brand)]" /> ML Pipeline
        </h2>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {[
            "Yahoo Finance Data",
            "→",
            "Feature Engineering (15+ indicators)",
            "→",
            "StandardScaler",
            "→",
            "XGBoost Regressor",
            "→",
            "TimeSeriesSplit CV",
            "→",
            "Predictions + Feature Importance",
            "→",
            "Rule-Based Recommendation",
          ].map((step, i) =>
            step === "→" ? (
              <span key={i} className="text-[var(--color-brand)] font-bold">→</span>
            ) : (
              <Badge key={i} variant="info">{step}</Badge>
            )
          )}
        </div>
      </motion.div>

      {/* Data Sources */}
      <motion.div {...fadeIn} className="glass-card p-6">
        <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-[var(--color-brand)]" /> Data Sources
        </h2>
        <div className="space-y-2 text-sm text-[var(--muted-foreground)]">
          <p>📊 <strong>Stock Data:</strong> Yahoo Finance API (yfinance) — real-time and historical OHLCV data for NSE-listed stocks</p>
          <p>📰 <strong>News:</strong> Google News RSS feeds — parsed with feedparser, analyzed with TextBlob</p>
          <p>📈 <strong>Technical Indicators:</strong> ta library — battle-tested implementation of 30+ technical indicators</p>
          <p>🏪 <strong>Market Indices:</strong> Nifty 50 (^NSEI) and Sensex (^BSESN) from Yahoo Finance</p>
        </div>
      </motion.div>

      {/* Disclaimer */}
      <motion.div {...fadeIn} className="rounded-2xl border border-amber-500/30 bg-amber-500/5 p-6">
        <h3 className="text-sm font-bold text-amber-500 mb-2">⚠️ Important Disclaimer</h3>
        <p className="text-sm text-[var(--muted-foreground)] leading-relaxed">
          This is an educational portfolio project built to demonstrate full-stack engineering, machine learning,
          and data science skills. The predictions and recommendations generated by this platform are for
          demonstration purposes only and should NOT be used for actual trading or investment decisions.
          Always consult a qualified financial advisor before making investment decisions.
        </p>
      </motion.div>
    </div>
  );
}
