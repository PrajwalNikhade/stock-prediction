"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  PieChart as PieChartIcon, Plus, Trash2, TrendingUp, TrendingDown,
  ArrowUpRight, ArrowDownRight, DollarSign, Briefcase, Layers,
} from "lucide-react";
import Link from "next/link";
import {
  usePortfolio, useAddHolding, useDeleteHolding, useStockSearch,
} from "@/hooks/use-api";
import { formatCurrency, formatPercent, getChangeColor, getChangeBg, cn } from "@/lib/utils";
import { Skeleton, Badge, EmptyState, ErrorDisplay, CardSkeleton } from "@/components/ui/shared";
import { StockChart } from "@/components/charts/stock-charts";

const fadeIn = { initial: { opacity: 0, y: 15 }, animate: { opacity: 1, y: 0 } };

export default function PortfolioPage() {
  const { data: portfolio, isLoading, error } = usePortfolio();
  const addMutation = useAddHolding();
  const deleteMutation = useDeleteHolding();

  const [showAddForm, setShowAddForm] = useState(false);
  const [symbol, setSymbol] = useState("");
  const [quantity, setQuantity] = useState("");
  const [buyPrice, setBuyPrice] = useState("");

  const { data: searchResults } = useStockSearch(symbol);

  const handleAdd = () => {
    if (!symbol || !quantity || !buyPrice) return;
    addMutation.mutate({
      symbol: symbol.toUpperCase(),
      quantity: parseInt(quantity, 10),
      buy_price: parseFloat(buyPrice),
    });
    setSymbol("");
    setQuantity("");
    setBuyPrice("");
    setShowAddForm(false);
  };

  if (error) return <div className="max-w-6xl mx-auto px-4 py-8"><ErrorDisplay message="Failed to load portfolio" /></div>;

  const summary = portfolio?.summary;
  const isPositivePnl = summary ? summary.total_pnl >= 0 : true;

  // Prepare sector pie chart data
  const sectorData = portfolio?.sector_allocation || [];
  const pieChartData = [
    {
      type: "pie" as const,
      labels: sectorData.map((s) => s.sector),
      values: sectorData.map((s) => s.value),
      textinfo: "label+percent" as const,
      insidetextorientation: "radial" as const,
      hole: 0.4,
      marker: {
        colors: [
          "#6366f1", "#10b981", "#3b82f6", "#f59e0b", "#8b5cf6",
          "#ec4899", "#14b8a6", "#f97316", "#a855f7", "#64748b",
        ],
      },
    },
  ];

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 space-y-8">
      {/* Header */}
      <motion.div {...fadeIn} className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Portfolio <span className="text-[var(--color-brand)]">Manager</span>
          </h1>
          <p className="text-[var(--muted-foreground)]">
            Track holdings, real-time ROI, and sector allocation
          </p>
        </div>
        <button
          onClick={() => setShowAddForm(!showAddForm)}
          className="px-4 py-2.5 rounded-xl gradient-brand text-white text-sm font-medium flex items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <Plus className="w-4 h-4" /> Add Holding
        </button>
      </motion.div>

      {/* Add Holding Form */}
      <AnimatePresence>
        {showAddForm && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="glass-card p-5 overflow-hidden"
          >
            <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
              <Plus className="w-4 h-4 text-[var(--color-brand)]" /> Add Stock to Portfolio
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Stock Symbol (e.g. RELIANCE)"
                  value={symbol}
                  onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                  className="w-full px-4 py-2.5 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)]"
                />
              </div>
              <input
                type="number"
                placeholder="Quantity (e.g. 10)"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                className="px-4 py-2.5 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)]"
              />
              <input
                type="number"
                step="0.01"
                placeholder="Average Buy Price (₹)"
                value={buyPrice}
                onChange={(e) => setBuyPrice(e.target.value)}
                className="px-4 py-2.5 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)]"
              />
              <button
                onClick={handleAdd}
                disabled={!symbol || !quantity || !buyPrice}
                className="px-4 py-2.5 rounded-xl gradient-brand text-white text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
              >
                Save Holding
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Portfolio Summary Cards */}
      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => <CardSkeleton key={i} />)}
        </div>
      ) : summary ? (
        <motion.div {...fadeIn} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass-card p-6">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--muted-foreground)]">Total Invested</span>
              <Briefcase className="w-4 h-4 text-[var(--color-brand)]" />
            </div>
            <p className="text-2xl font-bold mt-2">{formatCurrency(summary.total_invested)}</p>
            <p className="text-xs text-[var(--muted-foreground)] mt-1">{summary.holdings_count} Holdings</p>
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--muted-foreground)]">Current Value</span>
              <DollarSign className="w-4 h-4 text-blue-500" />
            </div>
            <p className="text-2xl font-bold mt-2">{formatCurrency(summary.total_current_value)}</p>
            <p className="text-xs text-[var(--muted-foreground)] mt-1">Real-time valuation</p>
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--muted-foreground)]">Total Profit / Loss</span>
              {isPositivePnl ? (
                <TrendingUp className="w-4 h-4 text-emerald-500" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-500" />
              )}
            </div>
            <p className={cn("text-2xl font-bold mt-2", getChangeColor(summary.total_pnl))}>
              {summary.total_pnl > 0 ? "+" : ""}{formatCurrency(summary.total_pnl)}
            </p>
            <Badge variant={isPositivePnl ? "success" : "danger"} className="mt-1">
              {formatPercent(summary.total_roi_percent)} ROI
            </Badge>
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center justify-between">
              <span className="text-xs text-[var(--muted-foreground)]">Overall ROI</span>
              <PieChartIcon className="w-4 h-4 text-purple-500" />
            </div>
            <p className={cn("text-2xl font-bold mt-2", getChangeColor(summary.total_roi_percent))}>
              {formatPercent(summary.total_roi_percent)}
            </p>
            <p className="text-xs text-[var(--muted-foreground)] mt-1">Return on Investment</p>
          </div>
        </motion.div>
      ) : null}

      {/* Holdings & Sector Chart Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Holdings Table */}
        <div className="lg:col-span-2 space-y-4">
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <Briefcase className="w-4 h-4 text-[var(--color-brand)]" /> Your Holdings
          </h2>

          {isLoading ? (
            <div className="space-y-3">{[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}</div>
          ) : !portfolio?.holdings || portfolio.holdings.length === 0 ? (
            <EmptyState
              icon="💼"
              title="No holdings in portfolio"
              description="Add your stock positions to track ROI and sector exposure"
              action={
                <button onClick={() => setShowAddForm(true)} className="px-4 py-2 rounded-xl gradient-brand text-white text-sm">
                  Add Your First Holding
                </button>
              }
            />
          ) : (
            <div className="glass-card overflow-hidden divide-y divide-[var(--border)]">
              {portfolio.holdings.map((h) => {
                const isGain = h.pnl >= 0;
                return (
                  <motion.div
                    key={h.id}
                    layout
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-[var(--muted)]/50 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-[var(--color-brand)]/10 flex items-center justify-center text-sm font-bold text-[var(--color-brand)]">
                        {h.symbol.slice(0, 2)}
                      </div>
                      <div>
                        <Link href={`/stocks/${h.symbol}`} className="font-semibold hover:text-[var(--color-brand)] text-sm flex items-center gap-2">
                          {h.symbol} <Badge>{h.sector}</Badge>
                        </Link>
                        <p className="text-xs text-[var(--muted-foreground)]">{h.company_name} • Qty: {h.quantity}</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 text-right">
                      <div>
                        <p className="text-xs text-[var(--muted-foreground)]">Buy Price</p>
                        <p className="text-sm font-medium">{formatCurrency(h.buy_price)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-[var(--muted-foreground)]">Current Price</p>
                        <p className="text-sm font-medium">{formatCurrency(h.current_price)}</p>
                      </div>
                      <div className="col-span-2 sm:col-span-1">
                        <p className="text-xs text-[var(--muted-foreground)]">P&L</p>
                        <p className={cn("text-sm font-bold flex items-center justify-end gap-0.5", getChangeColor(h.pnl))}>
                          {isGain ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
                          {formatCurrency(h.pnl)} ({formatPercent(h.pnl_percent)})
                        </p>
                      </div>
                    </div>

                    <button
                      onClick={() => deleteMutation.mutate(h.id)}
                      className="p-2 rounded-lg hover:bg-red-500/10 text-[var(--muted-foreground)] hover:text-red-500 transition-colors self-end sm:self-center"
                      title="Remove Holding"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>

        {/* Sector Allocation Breakdown */}
        <div className="space-y-4">
          <h2 className="text-sm font-semibold flex items-center gap-2">
            <Layers className="w-4 h-4 text-[var(--color-brand)]" /> Sector Allocation
          </h2>
          <div className="glass-card p-5">
            {sectorData.length > 0 ? (
              <>
                <StockChart
                  data={pieChartData}
                  height={280}
                  layout={{
                    showlegend: true,
                    margin: { t: 10, b: 10, l: 10, r: 10 },
                  }}
                />
                <div className="mt-4 space-y-2">
                  {sectorData.map((s) => (
                    <div key={s.sector} className="flex justify-between items-center text-xs">
                      <span className="text-[var(--muted-foreground)]">{s.sector}</span>
                      <span className="font-semibold">{formatCurrency(s.value)} ({s.percentage}%)</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-xs text-[var(--muted-foreground)] text-center py-12">
                Add holdings to visualize sector allocation
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
