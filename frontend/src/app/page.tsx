"use client";

import { motion } from "framer-motion";
import {
  TrendingUp, TrendingDown, Activity, BarChart3,
  ArrowUpRight, ArrowDownRight, Clock, Newspaper,
  Brain, Eye,
} from "lucide-react";
import Link from "next/link";
import {
  useMarketStatus, useMarketIndices, useTopMovers,
  useSectors, useMarketNews, useWatchlist,
} from "@/hooks/use-api";
import { formatCurrency, formatPercent, getChangeColor, getChangeBg, timeAgo, cn } from "@/lib/utils";
import { CardSkeleton, Skeleton, Badge, ErrorDisplay } from "@/components/ui/shared";
import type { StockMover, SectorData, NewsArticle } from "@/lib/types";

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
};

const stagger = {
  animate: { transition: { staggerChildren: 0.08 } },
};

export default function DashboardPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Hero */}
      <motion.div {...fadeInUp}>
        <div className="glass-card p-6 sm:p-8 gradient-mesh">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">
                SmartStock <span className="text-[var(--color-brand)]">AI</span> Dashboard
              </h1>
              <p className="text-[var(--muted-foreground)] mt-1">
                Explainable Indian Stock Market Analytics & Predictions
              </p>
            </div>
            <MarketStatusBadge />
          </div>
        </div>
      </motion.div>

      {/* Index Cards */}
      <IndexCards />

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <MoversSection />
          <SectorHeatmap />
        </div>
        <div className="space-y-6">
          <WatchlistWidget />
          <NewsWidget />
        </div>
      </div>
    </div>
  );
}

// --- Market Status Badge ---
function MarketStatusBadge() {
  const { data, isLoading } = useMarketStatus();
  if (isLoading) return <Skeleton className="h-8 w-28" />;
  if (!data) return null;

  return (
    <div className="flex items-center gap-2">
      <div className={cn(
        "w-2.5 h-2.5 rounded-full",
        data.is_open ? "bg-emerald-500 animate-pulse" : "bg-red-500"
      )} />
      <span className="text-sm font-medium">{data.status}</span>
      <span className="text-xs text-[var(--muted-foreground)]">
        <Clock className="w-3 h-3 inline mr-1" />
        {data.market_hours}
      </span>
    </div>
  );
}

// --- Index Cards ---
function IndexCards() {
  const { data, isLoading, error } = useMarketIndices();

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <CardSkeleton />
        <CardSkeleton />
      </div>
    );
  }

  if (error || !data) return <ErrorDisplay message="Failed to load market indices" />;

  const indices = [
    { name: "NIFTY 50", data: data.nifty50, icon: TrendingUp },
    { name: "SENSEX", data: data.sensex, icon: BarChart3 },
  ];

  return (
    <motion.div variants={stagger} initial="initial" animate="animate" className="grid grid-cols-1 sm:grid-cols-2 gap-4">
      {indices.map(({ name, data: idx, icon: Icon }) => (
        <motion.div key={name} variants={fadeInUp} className="glass-card p-6">
          {idx.error ? (
            <p className="text-sm text-[var(--muted-foreground)]">{idx.error}</p>
          ) : (
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <Icon className="w-5 h-5 text-[var(--color-brand)]" />
                  <span className="text-sm font-medium text-[var(--muted-foreground)]">{name}</span>
                </div>
                <p className="text-2xl font-bold tracking-tight">
                  {idx.value?.toLocaleString("en-IN", { maximumFractionDigits: 2 })}
                </p>
                <div className="flex items-center gap-2 mt-1">
                  <span className={cn("text-sm font-semibold", getChangeColor(idx.change))}>
                    {idx.change > 0 ? "+" : ""}{idx.change?.toFixed(2)}
                  </span>
                  <Badge variant={idx.change_percent >= 0 ? "success" : "danger"}>
                    {idx.change_percent >= 0 ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                    {formatPercent(idx.change_percent)}
                  </Badge>
                </div>
              </div>
              <div className={cn("p-2 rounded-xl", getChangeBg(idx.change))}>
                {idx.change >= 0 ? (
                  <TrendingUp className={cn("w-6 h-6", getChangeColor(idx.change))} />
                ) : (
                  <TrendingDown className={cn("w-6 h-6", getChangeColor(idx.change))} />
                )}
              </div>
            </div>
          )}
        </motion.div>
      ))}
    </motion.div>
  );
}

// --- Top Movers ---
function MoversSection() {
  const { data: gainers, isLoading: gLoading } = useTopMovers("gainers");
  const { data: losers, isLoading: lLoading } = useTopMovers("losers");

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <MoverCard title="Top Gainers" icon={TrendingUp} items={gainers} isLoading={gLoading} positive />
      <MoverCard title="Top Losers" icon={TrendingDown} items={losers} isLoading={lLoading} positive={false} />
    </div>
  );
}

function MoverCard({
  title, icon: Icon, items, isLoading, positive,
}: {
  title: string;
  icon: typeof TrendingUp;
  items?: StockMover[];
  isLoading: boolean;
  positive: boolean;
}) {
  if (isLoading) return <CardSkeleton />;

  return (
    <motion.div variants={fadeInUp} className="glass-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <Icon className={cn("w-5 h-5", positive ? "text-emerald-500" : "text-red-500")} />
        <h3 className="text-sm font-semibold">{title}</h3>
      </div>
      <div className="space-y-2">
        {items?.slice(0, 5).map((stock) => (
          <Link
            key={stock.symbol}
            href={`/stocks/${stock.symbol}`}
            className="flex items-center justify-between py-2 px-3 rounded-xl hover:bg-[var(--muted)] transition-colors"
          >
            <div>
              <p className="text-sm font-medium">{stock.symbol}</p>
              <p className="text-xs text-[var(--muted-foreground)] truncate max-w-[140px]">{stock.name}</p>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium">{formatCurrency(stock.price)}</p>
              <p className={cn("text-xs font-semibold", getChangeColor(stock.change_percent))}>
                {formatPercent(stock.change_percent)}
              </p>
            </div>
          </Link>
        ))}
        {(!items || items.length === 0) && (
          <p className="text-xs text-[var(--muted-foreground)] text-center py-4">No data available</p>
        )}
      </div>
    </motion.div>
  );
}

// --- Sector Heatmap ---
function SectorHeatmap() {
  const { data, isLoading } = useSectors();

  if (isLoading) return <CardSkeleton />;

  return (
    <motion.div variants={fadeInUp} className="glass-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-5 h-5 text-[var(--color-brand)]" />
        <h3 className="text-sm font-semibold">Sector Performance</h3>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
        {data?.map((sector: SectorData) => (
          <div
            key={sector.sector}
            className={cn(
              "rounded-xl p-3 text-center transition-all hover:scale-105",
              sector.change_percent >= 0 ? "bg-emerald-500/10" : "bg-red-500/10"
            )}
          >
            <p className="text-xs font-medium truncate">{sector.sector}</p>
            <p className={cn("text-sm font-bold mt-1", getChangeColor(sector.change_percent))}>
              {formatPercent(sector.change_percent)}
            </p>
          </div>
        ))}
        {(!data || data.length === 0) && (
          <p className="text-xs text-[var(--muted-foreground)] col-span-full text-center py-4">No sector data</p>
        )}
      </div>
    </motion.div>
  );
}

// --- Watchlist Widget ---
function WatchlistWidget() {
  const { data, isLoading } = useWatchlist();

  return (
    <motion.div variants={fadeInUp} className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Eye className="w-5 h-5 text-[var(--color-brand)]" />
          <h3 className="text-sm font-semibold">Watchlist</h3>
        </div>
        <Link href="/watchlist" className="text-xs text-[var(--color-brand)] hover:underline">
          View All
        </Link>
      </div>
      {isLoading ? (
        <div className="space-y-2">
          {[1, 2, 3].map((i) => <Skeleton key={i} className="h-10 w-full" />)}
        </div>
      ) : data && data.length > 0 ? (
        <div className="space-y-2">
          {data.slice(0, 5).map((item) => (
            <Link
              key={item.id}
              href={`/stocks/${item.symbol}`}
              className="flex items-center justify-between py-2 px-3 rounded-xl hover:bg-[var(--muted)] transition-colors"
            >
              <div>
                <p className="text-sm font-medium">{item.symbol}</p>
                <p className="text-xs text-[var(--muted-foreground)] truncate max-w-[140px]">{item.company_name}</p>
              </div>
              {item.is_pinned && <Badge variant="info">Pinned</Badge>}
            </Link>
          ))}
        </div>
      ) : (
        <p className="text-xs text-[var(--muted-foreground)] text-center py-4">
          No stocks in watchlist. <Link href="/stocks" className="text-[var(--color-brand)]">Add some →</Link>
        </p>
      )}
    </motion.div>
  );
}

// --- News Widget ---
function NewsWidget() {
  const { data, isLoading } = useMarketNews();

  return (
    <motion.div variants={fadeInUp} className="glass-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <Newspaper className="w-5 h-5 text-[var(--color-brand)]" />
        <h3 className="text-sm font-semibold">Latest News</h3>
      </div>
      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-14 w-full" />)}
        </div>
      ) : (
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {data?.slice(0, 8).map((article: NewsArticle, i: number) => (
            <a
              key={i}
              href={article.url}
              target="_blank"
              rel="noopener noreferrer"
              className="block py-2 px-3 rounded-xl hover:bg-[var(--muted)] transition-colors"
            >
              <p className="text-sm leading-snug line-clamp-2">{article.title}</p>
              <div className="flex items-center gap-2 mt-1">
                <Badge
                  variant={
                    article.sentiment_label === "positive" ? "success"
                    : article.sentiment_label === "negative" ? "danger"
                    : "default"
                  }
                >
                  {article.sentiment_label}
                </Badge>
                <span className="text-xs text-[var(--muted-foreground)]">{article.source}</span>
                {article.published_at && (
                  <span className="text-xs text-[var(--muted-foreground)]">{timeAgo(article.published_at)}</span>
                )}
              </div>
            </a>
          ))}
          {(!data || data.length === 0) && (
            <p className="text-xs text-[var(--muted-foreground)] text-center py-4">No news available</p>
          )}
        </div>
      )}
    </motion.div>
  );
}
