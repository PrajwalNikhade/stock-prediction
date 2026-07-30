"use client";

import { motion } from "framer-motion";
import { Sparkles, TrendingUp, AlertTriangle, ShieldCheck, Info } from "lucide-react";
import { useBusinessInsights } from "@/hooks/use-api";
import { CardSkeleton, Badge } from "@/components/ui/shared";

export function BusinessInsightsWidget() {
  const { data: insights, isLoading } = useBusinessInsights();

  if (isLoading) return <CardSkeleton />;
  if (!insights || insights.length === 0) return null;

  return (
    <div className="glass-card p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Sparkles className="w-5 h-5 text-[var(--color-brand)]" />
        <h3 className="text-sm font-semibold">Automated Business Insights</h3>
      </div>
      <div className="space-y-3">
        {insights.slice(0, 3).map((insight) => (
          <motion.div
            key={insight.id}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-3.5 rounded-xl border border-[var(--border)] bg-[var(--muted)]/40 space-y-1.5"
          >
            <div className="flex items-center justify-between">
              <Badge variant={insight.impact === "positive" ? "success" : insight.impact === "warning" ? "warning" : "info"}>
                {insight.category}
              </Badge>
              <span className="text-xs font-mono font-semibold text-[var(--color-brand)]">{insight.metric}</span>
            </div>
            <p className="text-xs font-semibold">{insight.title}</p>
            <p className="text-xs text-[var(--muted-foreground)] leading-relaxed">{insight.description}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
