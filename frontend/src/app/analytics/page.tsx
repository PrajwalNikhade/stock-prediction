"use client";

import { motion } from "framer-motion";
import {
  LineChart, Sparkles, TrendingUp, TrendingDown, Activity,
  Zap, AlertTriangle, ShieldCheck, BarChart3, Info, Flame,
} from "lucide-react";
import {
  useBusinessInsights, useCorrelationMatrix, useMarketAnalytics,
} from "@/hooks/use-api";
import { formatPercent, getChangeColor, cn } from "@/lib/utils";
import { CardSkeleton, Skeleton, Badge, ErrorDisplay } from "@/components/ui/shared";
import { StockChart } from "@/components/charts/stock-charts";

const fadeIn = { initial: { opacity: 0, y: 15 }, animate: { opacity: 1, y: 0 } };

export default function AnalyticsPage() {
  const { data: insights, isLoading: insLoading } = useBusinessInsights();
  const { data: correlation, isLoading: corrLoading } = useCorrelationMatrix();
  const { data: marketSummary, isLoading: summaryLoading } = useMarketAnalytics();

  // Prepare Correlation Matrix Heatmap
  const corrHeatmapData = correlation?.matrix ? [
    {
      type: "heatmap" as const,
      z: correlation.matrix,
      x: correlation.symbols,
      y: correlation.symbols,
      colorscale: "Viridis",
      reversescale: true,
      showscale: true,
    },
  ] : [];

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 space-y-8">
      {/* Hero Header */}
      <motion.div {...fadeIn} className="glass-card p-6 sm:p-8 gradient-mesh">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center">
            <LineChart className="w-5 h-5 text-white" />
          </div>
          <Badge variant="info">Institutional Analytics</Badge>
        </div>
        <h1 className="text-3xl font-bold tracking-tight">
          Executive <span className="text-[var(--color-brand)]">Business Intelligence</span> & Risk Analytics
        </h1>
        <p className="text-[var(--muted-foreground)] max-w-2xl mt-1">
          Automated insights engine, cross-asset correlation matrices, market volatility metrics, and sector dispersion analytics.
        </p>
      </motion.div>

      {/* Market Leaders / Risk Cards */}
      {summaryLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => <CardSkeleton key={i} />)}
        </div>
      ) : marketSummary ? (
        <motion.div {...fadeIn} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass-card p-5 border-emerald-500/20 bg-emerald-500/5">
            <div className="flex items-center justify-between text-xs text-[var(--muted-foreground)]">
              <span>Best Performer (6M)</span>
              <TrendingUp className="w-4 h-4 text-emerald-500" />
            </div>
            <p className="text-xl font-bold mt-2">{marketSummary.best_performing.symbol}</p>
            <p className="text-xs text-[var(--muted-foreground)]">{marketSummary.best_performing.name}</p>
            <p className="text-sm font-bold text-emerald-500 mt-1">+{marketSummary.best_performing.return_6m}%</p>
          </div>

          <div className="glass-card p-5 border-red-500/20 bg-red-500/5">
            <div className="flex items-center justify-between text-xs text-[var(--muted-foreground)]">
              <span>Worst Performer (6M)</span>
              <TrendingDown className="w-4 h-4 text-red-500" />
            </div>
            <p className="text-xl font-bold mt-2">{marketSummary.worst_performing.symbol}</p>
            <p className="text-xs text-[var(--muted-foreground)]">{marketSummary.worst_performing.name}</p>
            <p className="text-sm font-bold text-red-500 mt-1">{marketSummary.worst_performing.return_6m}%</p>
          </div>

          <div className="glass-card p-5 border-amber-500/20 bg-amber-500/5">
            <div className="flex items-center justify-between text-xs text-[var(--muted-foreground)]">
              <span>Highest Volatility</span>
              <Flame className="w-4 h-4 text-amber-500" />
            </div>
            <p className="text-xl font-bold mt-2">{marketSummary.most_volatile.symbol}</p>
            <p className="text-xs text-[var(--muted-foreground)]">{marketSummary.most_volatile.sector}</p>
            <p className="text-sm font-bold text-amber-500 mt-1">{marketSummary.most_volatile.volatility_annual}% Ann. Vol</p>
          </div>

          <div className="glass-card p-5 border-blue-500/20 bg-blue-500/5">
            <div className="flex items-center justify-between text-xs text-[var(--muted-foreground)]">
              <span>Avg Market Return</span>
              <Activity className="w-4 h-4 text-blue-500" />
            </div>
            <p className="text-xl font-bold mt-2">{formatPercent(marketSummary.avg_market_return)}</p>
            <p className="text-xs text-[var(--muted-foreground)]">Across tracking universe</p>
            <p className="text-xs text-[var(--muted-foreground)] mt-1">Avg Vol: {marketSummary.avg_market_volatility}%</p>
          </div>
        </motion.div>
      ) : null}

      {/* Business Insights Grid */}
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-[var(--color-brand)]" />
          <h2 className="text-lg font-bold">Automated Business Insights</h2>
        </div>

        {insLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[1, 2, 3, 4].map((i) => <CardSkeleton key={i} />)}
          </div>
        ) : insights && insights.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {insights.map((insight) => (
              <motion.div
                key={insight.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card p-6 border-l-4 border-l-[var(--color-brand)] space-y-3"
              >
                <div className="flex items-center justify-between">
                  <Badge variant={insight.impact === "positive" ? "success" : insight.impact === "warning" ? "warning" : "info"}>
                    {insight.category}
                  </Badge>
                  <span className="text-xs font-mono font-bold text-[var(--color-brand)]">{insight.metric}</span>
                </div>
                <h3 className="text-base font-semibold leading-snug">{insight.title}</h3>
                <p className="text-sm text-[var(--muted-foreground)] leading-relaxed">{insight.description}</p>
              </motion.div>
            ))}
          </div>
        ) : (
          <ErrorDisplay message="No business insights available" />
        )}
      </div>

      {/* Correlation Matrix & Performance Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Correlation Matrix Heatmap */}
        <motion.div {...fadeIn} className="glass-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold flex items-center gap-2">
              <Zap className="w-4 h-4 text-[var(--color-brand)]" /> Cross-Asset Correlation Matrix
            </h2>
            <span className="text-xs text-[var(--muted-foreground)]">Daily Returns (6M)</span>
          </div>

          {corrLoading ? (
            <Skeleton className="h-[350px] w-full" />
          ) : correlation?.matrix ? (
            <StockChart
              data={corrHeatmapData}
              height={360}
              layout={{
                xaxis: { automargin: true },
                yaxis: { automargin: true },
                margin: { l: 80, b: 80, t: 10, r: 10 },
              }}
            />
          ) : (
            <p className="text-xs text-[var(--muted-foreground)] text-center py-16">Correlation matrix unavailable</p>
          )}
        </motion.div>

        {/* Market Constituents Risk & Return Table */}
        <motion.div {...fadeIn} className="glass-card p-5">
          <h2 className="text-sm font-semibold flex items-center gap-2 mb-4">
            <BarChart3 className="w-4 h-4 text-[var(--color-brand)]" /> Constituents Performance & Volatility
          </h2>

          {marketSummary?.stocks ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-[var(--border)] text-[var(--muted-foreground)]">
                    <th className="pb-3 font-semibold">Stock</th>
                    <th className="pb-3 font-semibold">Sector</th>
                    <th className="pb-3 font-semibold text-right">6M Return</th>
                    <th className="pb-3 font-semibold text-right">Ann. Volatility</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {marketSummary.stocks.map((s) => (
                    <tr key={s.symbol} className="hover:bg-[var(--muted)]/50 transition-colors">
                      <td className="py-2.5 font-medium">{s.symbol}</td>
                      <td className="py-2.5 text-[var(--muted-foreground)]">{s.sector}</td>
                      <td className={cn("py-2.5 font-bold text-right", getChangeColor(s.return_6m))}>
                        {s.return_6m > 0 ? "+" : ""}{s.return_6m}%
                      </td>
                      <td className="py-2.5 font-mono text-right">{s.volatility_annual}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Skeleton className="h-[350px] w-full" />
          )}
        </motion.div>
      </div>
    </div>
  );
}
