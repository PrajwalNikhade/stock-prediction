"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "@/lib/api";
import type {
  MarketStatus, MarketIndices, SectorData, StockInfo, Stock,
  HistoricalData, StockMover, PredictionResult, Recommendation,
  StockNews, NewsArticle, TechnicalIndicators, CandlestickResult,
  WatchlistItem, Alert as AlertType, RecentSearch,
  PortfolioData, BusinessInsight, CorrelationData, MarketAnalyticsData,
} from "@/lib/types";

// ========================
// Market Hooks
// ========================

export function useMarketStatus() {
  return useQuery<MarketStatus>({
    queryKey: ["market", "status"],
    queryFn: () => api.get("/market/status").then((r) => r.data),
    refetchInterval: 60000,
  });
}

export function useMarketIndices() {
  return useQuery<MarketIndices>({
    queryKey: ["market", "indices"],
    queryFn: () => api.get("/market/indices").then((r) => r.data),
    staleTime: 30000,
  });
}

export function useTopMovers(direction: "gainers" | "losers") {
  return useQuery<StockMover[]>({
    queryKey: ["market", direction],
    queryFn: () => api.get(`/market/${direction}`).then((r) => r.data),
    staleTime: 60000,
  });
}

export function useSectors() {
  return useQuery<SectorData[]>({
    queryKey: ["market", "sectors"],
    queryFn: () => api.get("/market/sectors").then((r) => r.data),
    staleTime: 120000,
  });
}

// ========================
// Stock Hooks
// ========================

export function useStockSearch(query: string) {
  return useQuery<Stock[]>({
    queryKey: ["stocks", "search", query],
    queryFn: () => api.get(`/stocks/search?q=${query}`).then((r) => r.data),
    enabled: query.length >= 1,
    staleTime: 300000,
  });
}

export function useStockInfo(symbol: string) {
  return useQuery<StockInfo>({
    queryKey: ["stocks", symbol, "info"],
    queryFn: () => api.get(`/stocks/${symbol}`).then((r) => r.data),
    enabled: !!symbol,
    staleTime: 30000,
  });
}

export function useStockHistory(symbol: string, period: string = "1y", interval: string = "1d") {
  return useQuery<HistoricalData>({
    queryKey: ["stocks", symbol, "history", period, interval],
    queryFn: () =>
      api.get(`/stocks/${symbol}/history?period=${period}&interval=${interval}`).then((r) => r.data),
    enabled: !!symbol,
    staleTime: 60000,
  });
}

export function useRecentSearches() {
  return useQuery<RecentSearch[]>({
    queryKey: ["stocks", "recent-searches"],
    queryFn: () => api.get("/stocks/recent-searches").then((r) => r.data),
  });
}

// ========================
// Technical Indicators
// ========================

export function useTechnicalIndicators(symbol: string) {
  return useQuery<TechnicalIndicators>({
    queryKey: ["stocks", symbol, "indicators"],
    queryFn: () => api.get(`/stocks/${symbol}/indicators`).then((r) => r.data),
    enabled: !!symbol,
    staleTime: 60000,
  });
}

export function useCandlestickPatterns(symbol: string) {
  return useQuery<CandlestickResult>({
    queryKey: ["stocks", symbol, "candlestick"],
    queryFn: () => api.get(`/stocks/${symbol}/candlestick-patterns`).then((r) => r.data),
    enabled: !!symbol,
    staleTime: 300000,
  });
}

// ========================
// Prediction Hooks
// ========================

export function usePrediction(symbol: string) {
  return useQuery<PredictionResult>({
    queryKey: ["prediction", symbol],
    queryFn: () => api.get(`/predict/${symbol}`).then((r) => r.data),
    enabled: !!symbol,
    staleTime: 300000,
    gcTime: 600000,
  });
}

export function useRecommendation(symbol: string) {
  return useQuery<Recommendation>({
    queryKey: ["recommendation", symbol],
    queryFn: () => api.get(`/predict/${symbol}/recommendation`).then((r) => r.data),
    enabled: !!symbol,
    staleTime: 300000,
  });
}

// ========================
// News Hooks
// ========================

export function useMarketNews() {
  return useQuery<NewsArticle[]>({
    queryKey: ["news", "market"],
    queryFn: () => api.get("/news/market").then((r) => r.data),
    staleTime: 300000,
  });
}

export function useStockNews(symbol: string) {
  return useQuery<StockNews>({
    queryKey: ["news", symbol],
    queryFn: () => api.get(`/news/${symbol}`).then((r) => r.data),
    enabled: !!symbol,
    staleTime: 300000,
  });
}

// ========================
// Watchlist Hooks
// ========================

export function useWatchlist() {
  return useQuery<WatchlistItem[]>({
    queryKey: ["watchlist"],
    queryFn: () => api.get("/watchlist").then((r) => r.data),
  });
}

export function useAddToWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { symbol: string; company_name: string; is_favorite?: boolean }) =>
      api.post("/watchlist", data).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlist"] }),
  });
}

export function useUpdateWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, ...data }: { id: number; is_pinned?: boolean; is_favorite?: boolean }) =>
      api.put(`/watchlist/${id}`, data).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlist"] }),
  });
}

export function useRemoveFromWatchlist() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.delete(`/watchlist/${id}`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["watchlist"] }),
  });
}

// ========================
// Portfolio Hooks
// ========================

export function usePortfolio() {
  return useQuery<PortfolioData>({
    queryKey: ["portfolio"],
    queryFn: () => api.get("/portfolio").then((r) => r.data),
  });
}

export function useAddHolding() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { symbol: string; quantity: number; buy_price: number; company_name?: string }) =>
      api.post("/portfolio", data).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["portfolio"] }),
  });
}

export function useDeleteHolding() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.delete(`/portfolio/${id}`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["portfolio"] }),
  });
}

// ========================
// Analytics & Business Insights Hooks
// ========================

export function useBusinessInsights() {
  return useQuery<BusinessInsight[]>({
    queryKey: ["analytics", "insights"],
    queryFn: () => api.get("/analytics/insights").then((r) => r.data),
    staleTime: 120000,
  });
}

export function useCorrelationMatrix(symbols?: string) {
  return useQuery<CorrelationData>({
    queryKey: ["analytics", "correlation", symbols],
    queryFn: () => api.get(`/analytics/correlation${symbols ? `?symbols=${symbols}` : ""}`).then((r) => r.data),
    staleTime: 300000,
  });
}

export function useMarketAnalytics() {
  return useQuery<MarketAnalyticsData>({
    queryKey: ["analytics", "market-summary"],
    queryFn: () => api.get("/analytics/market-summary").then((r) => r.data),
    staleTime: 120000,
  });
}

// ========================
// Alert Hooks
// ========================

export function useAlerts() {
  return useQuery<AlertType[]>({
    queryKey: ["alerts"],
    queryFn: () => api.get("/alerts").then((r) => r.data),
  });
}

export function useCreateAlert() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: { symbol: string; alert_type: string; target_value: number }) =>
      api.post("/alerts", data).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }),
  });
}

export function useDeleteAlert() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.delete(`/alerts/${id}`).then((r) => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }),
  });
}
