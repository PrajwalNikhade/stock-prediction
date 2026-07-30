"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import { motion } from "framer-motion";
import {
  TrendingUp, TrendingDown, ArrowUpRight, ArrowDownRight,
  Star, Bell, Brain, BarChart3, Newspaper, Activity,
  Target, Gauge, Info, Layers, ExternalLink,
} from "lucide-react";
import Link from "next/link";
import {
  useStockInfo, useStockHistory, usePrediction,
  useRecommendation, useStockNews, useTechnicalIndicators,
  useCandlestickPatterns, useAddToWatchlist,
} from "@/hooks/use-api";
import { formatCurrency, formatPercent, formatMarketCap, formatNumber, getChangeColor, getChangeBg, cn, timeAgo } from "@/lib/utils";
import { CardSkeleton, Skeleton, Badge, ErrorDisplay } from "@/components/ui/shared";
import { CandlestickChart, FeatureImportanceChart, StockChart } from "@/components/charts/stock-charts";
import type { Data } from "plotly.js";

const fadeIn = { initial: { opacity: 0, y: 15 }, animate: { opacity: 1, y: 0 } };

export default function StockDetailPage() {
  const params = useParams();
  const symbol = (params.symbol as string)?.toUpperCase() || "";
  const [chartPeriod, setChartPeriod] = useState("1y");
  const [activeTab, setActiveTab] = useState<"overview" | "prediction" | "indicators" | "news">("overview");

  const { data: stockInfo, isLoading: infoLoading, error: infoError } = useStockInfo(symbol);
  const { data: history, isLoading: histLoading } = useStockHistory(symbol, chartPeriod);
  const { data: prediction, isLoading: predLoading } = usePrediction(symbol);
  const { data: recommendation, isLoading: recLoading } = useRecommendation(symbol);
  const { data: news, isLoading: newsLoading } = useStockNews(symbol);
  const { data: indicators } = useTechnicalIndicators(symbol);
  const { data: patterns } = useCandlestickPatterns(symbol);
  const addToWatchlist = useAddToWatchlist();

  if (infoError) return <div className="max-w-7xl mx-auto px-4 py-8"><ErrorDisplay message={`Failed to load data for ${symbol}`} /></div>;

  const periods = ["1mo", "3mo", "6mo", "1y", "2y", "5y"];

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <motion.div {...fadeIn} className="glass-card p-6">
        {infoLoading ? (
          <div className="space-y-3">
            <Skeleton className="h-8 w-64" />
            <Skeleton className="h-12 w-40" />
            <Skeleton className="h-4 w-48" />
          </div>
        ) : stockInfo && !stockInfo.error ? (
          <div className="flex flex-col sm:flex-row justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="w-12 h-12 rounded-xl gradient-brand flex items-center justify-center text-white font-bold text-lg">
                  {symbol.slice(0, 2)}
                </div>
                <div>
                  <h1 className="text-2xl font-bold">{symbol}</h1>
                  <p className="text-sm text-[var(--muted-foreground)]">{stockInfo.name}</p>
                </div>
                <Badge>{stockInfo.sector}</Badge>
              </div>
              <div className="flex items-baseline gap-3 mt-3">
                <span className="text-3xl font-bold">{formatCurrency(stockInfo.current_price)}</span>
                <span className={cn("text-lg font-semibold flex items-center gap-1", getChangeColor(stockInfo.change))}>
                  {stockInfo.change > 0 ? <ArrowUpRight className="w-5 h-5" /> : <ArrowDownRight className="w-5 h-5" />}
                  {stockInfo.change > 0 ? "+" : ""}{stockInfo.change.toFixed(2)} ({formatPercent(stockInfo.change_percent)})
                </span>
              </div>
            </div>
            <div className="flex gap-2 items-start">
              <button
                onClick={() => addToWatchlist.mutate({ symbol, company_name: stockInfo.name || symbol })}
                className="px-4 py-2 rounded-xl border border-[var(--border)] hover:border-amber-500 text-sm flex items-center gap-2 transition-colors"
              >
                <Star className="w-4 h-4" /> Watchlist
              </button>
              <Link
                href={`/alerts?symbol=${symbol}`}
                className="px-4 py-2 rounded-xl border border-[var(--border)] hover:border-[var(--color-brand)] text-sm flex items-center gap-2 transition-colors"
              >
                <Bell className="w-4 h-4" /> Alert
              </Link>
            </div>
          </div>
        ) : (
          <ErrorDisplay message={stockInfo?.error || `No data for ${symbol}`} />
        )}
      </motion.div>

      {/* Tab Navigation */}
      <div className="flex gap-1 p-1 rounded-xl bg-[var(--muted)] w-fit">
        {(["overview", "prediction", "indicators", "news"] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              "px-4 py-2 rounded-lg text-sm font-medium transition-all capitalize",
              activeTab === tab
                ? "bg-[var(--card)] shadow-sm text-[var(--foreground)]"
                : "text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            )}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Price Chart */}
          <motion.div {...fadeIn} className="glass-card p-5">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-[var(--color-brand)]" /> Price Chart
              </h2>
              <div className="flex gap-1">
                {periods.map((p) => (
                  <button
                    key={p}
                    onClick={() => setChartPeriod(p)}
                    className={cn(
                      "px-3 py-1 rounded-lg text-xs font-medium transition-colors",
                      chartPeriod === p ? "bg-[var(--color-brand)] text-white" : "text-[var(--muted-foreground)] hover:bg-[var(--muted)]"
                    )}
                  >
                    {p.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            {histLoading ? (
              <Skeleton className="h-[500px] w-full" />
            ) : history?.data ? (
              <CandlestickChart
                dates={history.data.dates}
                open={history.data.open}
                high={history.data.high}
                low={history.data.low}
                close={history.data.close}
                volume={history.data.volume}
                title={`${symbol} — ${chartPeriod.toUpperCase()}`}
              />
            ) : null}
          </motion.div>

          {/* Stock Info Grid */}
          {stockInfo && !stockInfo.error && (
            <motion.div {...fadeIn} className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
              {[
                { label: "Open", value: formatCurrency(stockInfo.open) },
                { label: "High", value: formatCurrency(stockInfo.high) },
                { label: "Low", value: formatCurrency(stockInfo.low) },
                { label: "Volume", value: formatNumber(stockInfo.volume) },
                { label: "52W High", value: formatCurrency(stockInfo.fifty_two_week_high) },
                { label: "52W Low", value: formatCurrency(stockInfo.fifty_two_week_low) },
                { label: "Market Cap", value: formatMarketCap(stockInfo.market_cap) },
                { label: "P/E Ratio", value: stockInfo.pe_ratio?.toFixed(2) || "N/A" },
                { label: "EPS", value: stockInfo.eps ? formatCurrency(stockInfo.eps) : "N/A" },
                { label: "Dividend Yield", value: stockInfo.dividend_yield ? `${stockInfo.dividend_yield}%` : "N/A" },
                { label: "Beta", value: stockInfo.beta?.toFixed(2) || "N/A" },
                { label: "Prev Close", value: formatCurrency(stockInfo.previous_close) },
              ].map(({ label, value }) => (
                <div key={label} className="glass-card p-4">
                  <p className="text-xs text-[var(--muted-foreground)]">{label}</p>
                  <p className="text-sm font-semibold mt-1">{value}</p>
                </div>
              ))}
            </motion.div>
          )}

          {/* AI Recommendation */}
          {recLoading ? <CardSkeleton /> : recommendation && (
            <motion.div {...fadeIn} className="glass-card p-6">
              <div className="flex items-center gap-2 mb-4">
                <Brain className="w-5 h-5 text-[var(--color-brand)]" />
                <h2 className="text-sm font-semibold">AI Recommendation</h2>
              </div>
              <div className="flex flex-col sm:flex-row items-start gap-6">
                <div className={cn(
                  "px-6 py-4 rounded-2xl text-center min-w-[120px]",
                  recommendation.recommendation === "Buy" ? "bg-emerald-500/15" :
                  recommendation.recommendation === "Sell" ? "bg-red-500/15" : "bg-amber-500/15"
                )}>
                  <p className={cn("text-3xl font-bold",
                    recommendation.recommendation === "Buy" ? "text-emerald-500" :
                    recommendation.recommendation === "Sell" ? "text-red-500" : "text-amber-500"
                  )}>
                    {recommendation.recommendation}
                  </p>
                  <p className="text-xs text-[var(--muted-foreground)] mt-1">
                    Confidence: {(recommendation.confidence * 100).toFixed(0)}%
                  </p>
                </div>
                <div className="flex-1 space-y-2">
                  <p className="text-sm">{recommendation.reason}</p>
                  <div className="flex flex-wrap gap-2">
                    {recommendation.signals?.slice(0, 6).map((signal, i) => (
                      <Badge
                        key={i}
                        variant={signal.signal.includes("Bullish") ? "success" : signal.signal.includes("Bearish") ? "danger" : "default"}
                      >
                        {signal.factor}: {signal.signal}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Candlestick Patterns */}
          {patterns && patterns.patterns && patterns.patterns.length > 0 && (
            <motion.div {...fadeIn} className="glass-card p-5">
              <div className="flex items-center gap-2 mb-4">
                <Layers className="w-5 h-5 text-[var(--color-brand)]" />
                <h2 className="text-sm font-semibold">Candlestick Patterns Detected</h2>
                <Badge variant={patterns.bias === "Bullish" ? "success" : patterns.bias === "Bearish" ? "danger" : "default"}>
                  {patterns.bias} Bias
                </Badge>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {patterns.patterns.slice(0, 6).map((p, i) => (
                  <div key={i} className={cn(
                    "p-3 rounded-xl border border-[var(--border)]",
                    p.type === "bullish" ? "bg-emerald-500/5" : p.type === "bearish" ? "bg-red-500/5" : "bg-[var(--muted)]"
                  )}>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">{p.pattern}</span>
                      <Badge variant={p.type === "bullish" ? "success" : p.type === "bearish" ? "danger" : "default"}>
                        {p.type}
                      </Badge>
                    </div>
                    <p className="text-xs text-[var(--muted-foreground)] mt-1">{p.description}</p>
                    <p className="text-xs text-[var(--muted-foreground)]">{p.date} — ₹{p.price.toFixed(2)}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          )}
        </div>
      )}

      {activeTab === "prediction" && (
        <div className="space-y-6">
          {predLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <CardSkeleton /><CardSkeleton /><CardSkeleton />
            </div>
          ) : prediction && !prediction.error ? (
            <>
              {/* Prediction Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {(["1d", "7d", "30d"] as const).map((horizon) => {
                  const pred = prediction.predictions[horizon];
                  if (!pred) return null;
                  const isUp = pred.direction === "up";
                  return (
                    <motion.div key={horizon} {...fadeIn} className="glass-card p-6">
                      <p className="text-xs text-[var(--muted-foreground)] mb-2">
                        {horizon === "1d" ? "1 Day" : horizon === "7d" ? "7 Day" : "30 Day"} Prediction
                      </p>
                      <p className="text-2xl font-bold">{formatCurrency(pred.predicted_price)}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <Badge variant={isUp ? "success" : "danger"}>
                          {isUp ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                          {formatPercent(pred.change_percent)}
                        </Badge>
                        <span className="text-xs text-[var(--muted-foreground)]">
                          Confidence: {(pred.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      {/* Confidence bar */}
                      <div className="mt-3 h-1.5 rounded-full bg-[var(--muted)] overflow-hidden">
                        <div
                          className={cn("h-full rounded-full transition-all", isUp ? "bg-emerald-500" : "bg-red-500")}
                          style={{ width: `${pred.confidence * 100}%` }}
                        />
                      </div>
                    </motion.div>
                  );
                })}
              </div>

              {/* Model Metrics */}
              <motion.div {...fadeIn} className="glass-card p-5">
                <h2 className="text-sm font-semibold flex items-center gap-2 mb-4">
                  <Gauge className="w-4 h-4 text-[var(--color-brand)]" /> Model Performance
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {(["1d", "7d", "30d"] as const).map((h) => {
                    const m = prediction.metrics[h];
                    if (!m) return null;
                    return (
                      <div key={h} className="rounded-xl border border-[var(--border)] p-4">
                        <p className="text-xs text-[var(--muted-foreground)] mb-2">
                          {h === "1d" ? "1 Day" : h === "7d" ? "7 Day" : "30 Day"} Model
                        </p>
                        <div className="space-y-1">
                          <div className="flex justify-between text-sm"><span>MAE</span><span className="font-mono">{m.mae.toFixed(2)}</span></div>
                          <div className="flex justify-between text-sm"><span>RMSE</span><span className="font-mono">{m.rmse.toFixed(2)}</span></div>
                          <div className="flex justify-between text-sm"><span>R²</span><span className="font-mono">{m.r2.toFixed(4)}</span></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </motion.div>

              {/* Feature Importance */}
              <motion.div {...fadeIn} className="glass-card p-5">
                <h2 className="text-sm font-semibold flex items-center gap-2 mb-4">
                  <Target className="w-4 h-4 text-[var(--color-brand)]" /> Feature Importance — Top Contributing Indicators
                </h2>
                {prediction.feature_importance?.["1d"] && (
                  <FeatureImportanceChart features={prediction.feature_importance["1d"]} title="1-Day Prediction Feature Importance" />
                )}
                {prediction.top_factors && prediction.top_factors.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    <span className="text-xs text-[var(--muted-foreground)]">Top Factors:</span>
                    {prediction.top_factors.map((f) => (
                      <Badge key={f.feature} variant="info">{f.feature}</Badge>
                    ))}
                  </div>
                )}
              </motion.div>
            </>
          ) : (
            <ErrorDisplay message={prediction?.error || "Failed to generate prediction"} />
          )}
        </div>
      )}

      {activeTab === "indicators" && indicators?.indicators && (
        <div className="space-y-6">
          <motion.div {...fadeIn} className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {[
              { label: "RSI (14)", value: indicators.indicators.rsi.toFixed(2), signal: indicators.indicators.rsi_signal },
              { label: "MACD", value: indicators.indicators.macd.toFixed(4), signal: indicators.indicators.macd_signal_text },
              { label: "ADX", value: indicators.indicators.adx.toFixed(2), signal: indicators.indicators.trend },
              { label: "ATR", value: indicators.indicators.atr.toFixed(2) },
              { label: "CCI", value: indicators.indicators.cci.toFixed(2) },
              { label: "Williams %R", value: indicators.indicators.williams_r.toFixed(2) },
              { label: "Stochastic K", value: indicators.indicators.stochastic_k.toFixed(2) },
              { label: "Stochastic D", value: indicators.indicators.stochastic_d.toFixed(2) },
              { label: "Momentum", value: indicators.indicators.momentum.toFixed(2) },
              { label: "ROC", value: indicators.indicators.roc.toFixed(2) },
              { label: "OBV", value: formatNumber(indicators.indicators.obv) },
              { label: "VWAP", value: formatCurrency(indicators.indicators.vwap) },
              { label: "SMA 10", value: formatCurrency(indicators.indicators.sma_10) },
              { label: "SMA 20", value: formatCurrency(indicators.indicators.sma_20) },
              { label: "SMA 50", value: formatCurrency(indicators.indicators.sma_50) },
              { label: "EMA 12", value: formatCurrency(indicators.indicators.ema_12) },
              { label: "EMA 20", value: formatCurrency(indicators.indicators.ema_20) },
              { label: "EMA 26", value: formatCurrency(indicators.indicators.ema_26) },
              { label: "BB Upper", value: formatCurrency(indicators.indicators.bb_upper) },
              { label: "BB Lower", value: formatCurrency(indicators.indicators.bb_lower) },
            ].map(({ label, value, signal }) => (
              <div key={label} className="glass-card p-4">
                <p className="text-xs text-[var(--muted-foreground)]">{label}</p>
                <p className="text-sm font-semibold mt-1">{value}</p>
                {signal && (
                  <Badge className="mt-1" variant={
                    signal === "Bullish" || signal === "Strong" ? "success" :
                    signal === "Bearish" ? "danger" :
                    signal === "Oversold" ? "success" :
                    signal === "Overbought" ? "danger" : "default"
                  }>
                    {signal}
                  </Badge>
                )}
              </div>
            ))}
          </motion.div>
        </div>
      )}

      {activeTab === "news" && (
        <div className="space-y-4">
          {newsLoading ? (
            <div className="space-y-3">{[1, 2, 3, 4, 5].map((i) => <Skeleton key={i} className="h-20 w-full" />)}</div>
          ) : news?.articles && news.articles.length > 0 ? (
            <>
              {/* Sentiment Summary */}
              <motion.div {...fadeIn} className="glass-card p-5">
                <h2 className="text-sm font-semibold mb-3">Sentiment Analysis</h2>
                <div className="flex flex-wrap gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold">{news.sentiment_summary.total}</p>
                    <p className="text-xs text-[var(--muted-foreground)]">Articles</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-emerald-500">{news.sentiment_summary.counts.positive}</p>
                    <p className="text-xs text-[var(--muted-foreground)]">Positive</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-slate-400">{news.sentiment_summary.counts.neutral}</p>
                    <p className="text-xs text-[var(--muted-foreground)]">Neutral</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-red-500">{news.sentiment_summary.counts.negative}</p>
                    <p className="text-xs text-[var(--muted-foreground)]">Negative</p>
                  </div>
                  <Badge variant={
                    news.sentiment_summary.overall_label === "positive" ? "success" :
                    news.sentiment_summary.overall_label === "negative" ? "danger" : "default"
                  }>
                    Overall: {news.sentiment_summary.overall_label} ({news.sentiment_summary.average_score.toFixed(2)})
                  </Badge>
                </div>
              </motion.div>

              {/* News List */}
              <div className="space-y-2">
                {news.articles.map((article, i) => (
                  <motion.a
                    key={i}
                    href={article.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.05 * i }}
                    className="glass-card p-4 block hover:border-[var(--color-brand)]/30 transition-all"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-medium leading-snug">{article.title}</p>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant={
                            article.sentiment_label === "positive" ? "success" :
                            article.sentiment_label === "negative" ? "danger" : "default"
                          }>
                            {article.sentiment_label} ({article.sentiment_score.toFixed(2)})
                          </Badge>
                          <span className="text-xs text-[var(--muted-foreground)]">{article.source}</span>
                          {article.published_at && (
                            <span className="text-xs text-[var(--muted-foreground)]">{timeAgo(article.published_at)}</span>
                          )}
                        </div>
                      </div>
                      <ExternalLink className="w-4 h-4 shrink-0 text-[var(--muted-foreground)]" />
                    </div>
                  </motion.a>
                ))}
              </div>
            </>
          ) : (
            <ErrorDisplay message="No news available" />
          )}
        </div>
      )}
    </div>
  );
}
