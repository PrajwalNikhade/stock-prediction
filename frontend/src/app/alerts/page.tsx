"use client";

import { useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, Plus, Trash2, Check, AlertCircle } from "lucide-react";
import { useAlerts, useCreateAlert, useDeleteAlert } from "@/hooks/use-api";
import { cn } from "@/lib/utils";
import { Skeleton, Badge, EmptyState, ErrorDisplay } from "@/components/ui/shared";

function AlertsContent() {
  const searchParams = useSearchParams();
  const prefillSymbol = searchParams.get("symbol") || "";

  const { data: alerts, isLoading, error } = useAlerts();
  const createMutation = useCreateAlert();
  const deleteMutation = useDeleteAlert();

  const [showForm, setShowForm] = useState(!!prefillSymbol);
  const [symbol, setSymbol] = useState(prefillSymbol);
  const [alertType, setAlertType] = useState<string>("price_above");
  const [targetValue, setTargetValue] = useState("");

  const handleCreate = () => {
    if (!symbol || !targetValue) return;
    createMutation.mutate({
      symbol: symbol.toUpperCase(),
      alert_type: alertType,
      target_value: parseFloat(targetValue),
    });
    setSymbol("");
    setTargetValue("");
    setShowForm(false);
  };

  if (error) return <ErrorDisplay message="Failed to load alerts" />;

  const alertTypeLabels: Record<string, string> = {
    price_above: "Price Above",
    price_below: "Price Below",
    rsi_above: "RSI Above",
    rsi_below: "RSI Below",
  };

  return (
    <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-bold">Alerts</h1>
          <p className="text-sm text-[var(--muted-foreground)]">
            Set price and RSI alerts for your stocks
          </p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="px-4 py-2 rounded-xl gradient-brand text-white text-sm font-medium flex items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <Plus className="w-4 h-4" /> New Alert
        </button>
      </motion.div>

      {/* Create Alert Form */}
      <AnimatePresence>
        {showForm && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="glass-card p-5 overflow-hidden"
          >
            <h3 className="text-sm font-semibold mb-4">Create New Alert</h3>
            <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
              <input
                type="text"
                placeholder="Stock Symbol (e.g. RELIANCE)"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="px-4 py-2.5 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm
                  focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)]"
              />
              <select
                value={alertType}
                onChange={(e) => setAlertType(e.target.value)}
                className="px-4 py-2.5 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm
                  focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)]"
              >
                <option value="price_above">Price Above</option>
                <option value="price_below">Price Below</option>
                <option value="rsi_above">RSI Above</option>
                <option value="rsi_below">RSI Below</option>
              </select>
              <input
                type="number"
                placeholder="Target Value"
                value={targetValue}
                onChange={(e) => setTargetValue(e.target.value)}
                className="px-4 py-2.5 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm
                  focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)]"
              />
              <button
                onClick={handleCreate}
                disabled={!symbol || !targetValue}
                className="px-4 py-2.5 rounded-xl gradient-brand text-white text-sm font-medium
                  disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
              >
                Create Alert
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Alerts List */}
      {isLoading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
        </div>
      ) : !alerts || alerts.length === 0 ? (
        <EmptyState
          icon="🔔"
          title="No alerts set"
          description="Create alerts to get notified when stocks hit your targets"
          action={
            <button onClick={() => setShowForm(true)} className="px-4 py-2 rounded-xl gradient-brand text-white text-sm">
              Create Your First Alert
            </button>
          }
        />
      ) : (
        <div className="space-y-2">
          {alerts.map((alert) => (
            <motion.div
              key={alert.id}
              layout
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="glass-card p-4 flex items-center justify-between"
            >
              <div className="flex items-center gap-4">
                <div className={cn(
                  "w-10 h-10 rounded-xl flex items-center justify-center",
                  alert.is_triggered ? "bg-emerald-500/15" : alert.is_active ? "bg-[var(--color-brand)]/10" : "bg-[var(--muted)]"
                )}>
                  {alert.is_triggered ? (
                    <Check className="w-5 h-5 text-emerald-500" />
                  ) : (
                    <Bell className={cn("w-5 h-5", alert.is_active ? "text-[var(--color-brand)]" : "text-[var(--muted-foreground)]")} />
                  )}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold">{alert.symbol}</p>
                    <Badge variant={alert.is_triggered ? "success" : alert.is_active ? "info" : "default"}>
                      {alert.is_triggered ? "Triggered" : alert.is_active ? "Active" : "Inactive"}
                    </Badge>
                  </div>
                  <p className="text-xs text-[var(--muted-foreground)]">
                    {alertTypeLabels[alert.alert_type] || alert.alert_type}: {alert.target_value}
                  </p>
                </div>
              </div>
              <button
                onClick={() => deleteMutation.mutate(alert.id)}
                className="p-2 rounded-lg hover:bg-red-500/10 text-[var(--muted-foreground)] hover:text-red-500 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function AlertsPage() {
  return (
    <Suspense fallback={<div className="mx-auto max-w-4xl px-4 py-8"><Skeleton className="h-64 w-full" /></div>}>
      <AlertsContent />
    </Suspense>
  );
}
