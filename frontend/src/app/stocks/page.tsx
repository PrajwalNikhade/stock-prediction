"use client";

import { useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Search, Clock, Star, TrendingUp, ArrowRight } from "lucide-react";
import Link from "next/link";
import { useStockSearch, useRecentSearches, useWatchlist } from "@/hooks/use-api";
import { cn } from "@/lib/utils";
import { Skeleton, Badge } from "@/components/ui/shared";

const NSE_TOP_STOCKS = [
  { symbol: "RELIANCE", name: "Reliance Industries", sector: "Energy" },
  { symbol: "TCS", name: "Tata Consultancy Services", sector: "IT" },
  { symbol: "HDFCBANK", name: "HDFC Bank", sector: "Banking" },
  { symbol: "INFY", name: "Infosys", sector: "IT" },
  { symbol: "ICICIBANK", name: "ICICI Bank", sector: "Banking" },
  { symbol: "SBIN", name: "State Bank of India", sector: "Banking" },
  { symbol: "TATAMOTORS", name: "Tata Motors", sector: "Auto" },
  { symbol: "WIPRO", name: "Wipro", sector: "IT" },
  { symbol: "BAJFINANCE", name: "Bajaj Finance", sector: "Finance" },
  { symbol: "SUNPHARMA", name: "Sun Pharma", sector: "Pharma" },
  { symbol: "ITC", name: "ITC Limited", sector: "FMCG" },
  { symbol: "MARUTI", name: "Maruti Suzuki", sector: "Auto" },
];

export default function StocksPage() {
  const [query, setQuery] = useState("");
  const { data: searchResults, isLoading: searchLoading } = useStockSearch(query);
  const { data: recentSearches } = useRecentSearches();
  const { data: watchlist } = useWatchlist();

  const watchlistSymbols = new Set(watchlist?.map((w) => w.symbol) || []);

  const handleSearch = useCallback((val: string) => {
    setQuery(val);
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center space-y-2"
      >
        <h1 className="text-3xl font-bold tracking-tight">
          Explore <span className="text-[var(--color-brand)]">Indian Stocks</span>
        </h1>
        <p className="text-[var(--muted-foreground)]">
          Search, analyze, and get AI predictions for NSE stocks
        </p>
      </motion.div>

      {/* Search Bar */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="max-w-2xl mx-auto"
      >
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-[var(--muted-foreground)]" />
          <input
            type="text"
            placeholder="Search by symbol or company name... (e.g. RELIANCE, TCS, HDFC)"
            value={query}
            onChange={(e) => handleSearch(e.target.value)}
            className="w-full pl-12 pr-4 py-4 rounded-2xl bg-[var(--card)] border border-[var(--border)] text-base
              focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)] focus:border-transparent
              placeholder:text-[var(--muted-foreground)] transition-all shadow-sm"
          />
        </div>
      </motion.div>

      {/* Search Results */}
      {query.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="max-w-2xl mx-auto"
        >
          {searchLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
            </div>
          ) : searchResults && searchResults.length > 0 ? (
            <div className="glass-card overflow-hidden divide-y divide-[var(--border)]">
              {searchResults.map((stock) => (
                <Link
                  key={stock.symbol}
                  href={`/stocks/${stock.symbol}`}
                  className="flex items-center justify-between p-4 hover:bg-[var(--muted)] transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl gradient-brand flex items-center justify-center text-white text-sm font-bold">
                      {stock.symbol.slice(0, 2)}
                    </div>
                    <div>
                      <p className="font-medium">{stock.symbol}</p>
                      <p className="text-sm text-[var(--muted-foreground)]">{stock.name}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge>{stock.sector}</Badge>
                    {watchlistSymbols.has(stock.symbol) && (
                      <Star className="w-4 h-4 text-amber-500 fill-amber-500" />
                    )}
                    <ArrowRight className="w-4 h-4 text-[var(--muted-foreground)]" />
                  </div>
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-center text-[var(--muted-foreground)]">No results for &quot;{query}&quot;</p>
          )}
        </motion.div>
      )}

      {/* Recent Searches */}
      {query.length === 0 && recentSearches && recentSearches.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-4 h-4 text-[var(--muted-foreground)]" />
            <h2 className="text-sm font-semibold text-[var(--muted-foreground)]">Recent Searches</h2>
          </div>
          <div className="flex flex-wrap gap-2">
            {recentSearches.slice(0, 8).map((s) => (
              <Link
                key={s.symbol + s.searched_at}
                href={`/stocks/${s.symbol}`}
                className="px-4 py-2 rounded-xl bg-[var(--card)] border border-[var(--border)] text-sm hover:border-[var(--color-brand)] transition-colors"
              >
                {s.symbol}
              </Link>
            ))}
          </div>
        </motion.div>
      )}

      {/* Popular Stocks Grid */}
      {query.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-4 h-4 text-[var(--color-brand)]" />
            <h2 className="text-sm font-semibold">Popular Stocks</h2>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {NSE_TOP_STOCKS.map((stock, i) => (
              <motion.div
                key={stock.symbol}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 * i }}
              >
                <Link
                  href={`/stocks/${stock.symbol}`}
                  className="glass-card p-4 block hover:border-[var(--color-brand)]/30 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-[var(--color-brand)]/10 flex items-center justify-center text-xs font-bold text-[var(--color-brand)] group-hover:gradient-brand group-hover:text-white transition-all">
                      {stock.symbol.slice(0, 2)}
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-semibold truncate">{stock.symbol}</p>
                      <p className="text-xs text-[var(--muted-foreground)] truncate">{stock.name}</p>
                    </div>
                  </div>
                  <Badge className="mt-2">{stock.sector}</Badge>
                </Link>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}
