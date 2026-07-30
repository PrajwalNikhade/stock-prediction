"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Star, Pin, Trash2, Plus, Search, ExternalLink } from "lucide-react";
import Link from "next/link";
import {
  useWatchlist, useAddToWatchlist, useUpdateWatchlist,
  useRemoveFromWatchlist, useStockSearch,
} from "@/hooks/use-api";
import { cn } from "@/lib/utils";
import { Skeleton, Badge, EmptyState, ErrorDisplay } from "@/components/ui/shared";

export default function WatchlistPage() {
  const { data: watchlist, isLoading, error } = useWatchlist();
  const addMutation = useAddToWatchlist();
  const updateMutation = useUpdateWatchlist();
  const removeMutation = useRemoveFromWatchlist();

  const [showAdd, setShowAdd] = useState(false);
  const [addQuery, setAddQuery] = useState("");
  const { data: searchResults } = useStockSearch(addQuery);

  if (error) return <div className="max-w-4xl mx-auto px-4 py-8"><ErrorDisplay message="Failed to load watchlist" /></div>;

  const pinnedItems = watchlist?.filter((w) => w.is_pinned) || [];
  const favoriteItems = watchlist?.filter((w) => w.is_favorite && !w.is_pinned) || [];
  const otherItems = watchlist?.filter((w) => !w.is_pinned && !w.is_favorite) || [];

  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-bold">Watchlist</h1>
          <p className="text-sm text-[var(--muted-foreground)]">
            Track and manage your favorite stocks
          </p>
        </div>
        <button
          onClick={() => setShowAdd(!showAdd)}
          className="px-4 py-2 rounded-xl gradient-brand text-white text-sm font-medium flex items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <Plus className="w-4 h-4" /> Add Stock
        </button>
      </motion.div>

      {/* Add Stock Panel */}
      <AnimatePresence>
        {showAdd && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="glass-card p-5 overflow-hidden"
          >
            <div className="relative mb-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--muted-foreground)]" />
              <input
                type="text"
                placeholder="Search stocks to add..."
                value={addQuery}
                onChange={(e) => setAddQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm
                  focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)]"
              />
            </div>
            {searchResults && searchResults.length > 0 && (
              <div className="space-y-1 max-h-60 overflow-y-auto">
                {searchResults.map((stock) => (
                  <button
                    key={stock.symbol}
                    onClick={() => {
                      addMutation.mutate({ symbol: stock.symbol, company_name: stock.name });
                      setAddQuery("");
                    }}
                    className="w-full flex items-center justify-between p-3 rounded-xl hover:bg-[var(--muted)] transition-colors text-left"
                  >
                    <div>
                      <p className="text-sm font-medium">{stock.symbol}</p>
                      <p className="text-xs text-[var(--muted-foreground)]">{stock.name}</p>
                    </div>
                    <Plus className="w-4 h-4 text-[var(--color-brand)]" />
                  </button>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3, 4, 5].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
        </div>
      ) : !watchlist || watchlist.length === 0 ? (
        <EmptyState
          icon="⭐"
          title="Your watchlist is empty"
          description="Add stocks to track their performance"
          action={
            <button
              onClick={() => setShowAdd(true)}
              className="px-4 py-2 rounded-xl gradient-brand text-white text-sm"
            >
              Add Your First Stock
            </button>
          }
        />
      ) : (
        <div className="space-y-6">
          {/* Pinned */}
          {pinnedItems.length > 0 && (
            <Section title="📌 Pinned" items={pinnedItems} updateMutation={updateMutation} removeMutation={removeMutation} />
          )}
          {/* Favorites */}
          {favoriteItems.length > 0 && (
            <Section title="⭐ Favorites" items={favoriteItems} updateMutation={updateMutation} removeMutation={removeMutation} />
          )}
          {/* Other */}
          {otherItems.length > 0 && (
            <Section title="All Stocks" items={otherItems} updateMutation={updateMutation} removeMutation={removeMutation} />
          )}
        </div>
      )}
    </div>
  );
}

function Section({
  title, items, updateMutation, removeMutation,
}: {
  title: string;
  items: { id: number; symbol: string; company_name: string; is_pinned: boolean; is_favorite: boolean }[];
  updateMutation: ReturnType<typeof useUpdateWatchlist>;
  removeMutation: ReturnType<typeof useRemoveFromWatchlist>;
}) {
  return (
    <div>
      <h2 className="text-sm font-semibold text-[var(--muted-foreground)] mb-2">{title}</h2>
      <div className="space-y-2">
        {items.map((item) => (
          <motion.div
            key={item.id}
            layout
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass-card p-4 flex items-center justify-between"
          >
            <Link href={`/stocks/${item.symbol}`} className="flex items-center gap-3 flex-1 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-[var(--color-brand)]/10 flex items-center justify-center text-sm font-bold text-[var(--color-brand)]">
                {item.symbol.slice(0, 2)}
              </div>
              <div className="min-w-0">
                <p className="font-medium text-sm">{item.symbol}</p>
                <p className="text-xs text-[var(--muted-foreground)] truncate">{item.company_name}</p>
              </div>
            </Link>
            <div className="flex items-center gap-1">
              <button
                onClick={() => updateMutation.mutate({ id: item.id, is_pinned: !item.is_pinned })}
                className={cn("p-2 rounded-lg hover:bg-[var(--muted)] transition-colors",
                  item.is_pinned && "text-[var(--color-brand)]")}
                title="Pin"
              >
                <Pin className="w-4 h-4" />
              </button>
              <button
                onClick={() => updateMutation.mutate({ id: item.id, is_favorite: !item.is_favorite })}
                className={cn("p-2 rounded-lg hover:bg-[var(--muted)] transition-colors",
                  item.is_favorite && "text-amber-500")}
                title="Favorite"
              >
                <Star className={cn("w-4 h-4", item.is_favorite && "fill-current")} />
              </button>
              <button
                onClick={() => removeMutation.mutate(item.id)}
                className="p-2 rounded-lg hover:bg-red-500/10 text-[var(--muted-foreground)] hover:text-red-500 transition-colors"
                title="Remove"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
