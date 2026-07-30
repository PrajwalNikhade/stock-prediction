"use client";

import { cn } from "@/lib/utils";
import { type ReactNode } from "react";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return <div className={cn("skeleton", className)} />;
}

export function CardSkeleton() {
  return (
    <div className="glass-card p-6 space-y-4">
      <Skeleton className="h-4 w-1/3" />
      <Skeleton className="h-8 w-2/3" />
      <Skeleton className="h-4 w-1/2" />
    </div>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4 items-center">
          <Skeleton className="h-10 w-16" />
          <Skeleton className="h-10 flex-1" />
          <Skeleton className="h-10 w-20" />
          <Skeleton className="h-10 w-24" />
        </div>
      ))}
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="glass-card p-6 space-y-4">
      <Skeleton className="h-4 w-1/4" />
      <Skeleton className="h-[300px] w-full" />
    </div>
  );
}

// --- Error Boundary ---
interface ErrorDisplayProps {
  message?: string;
  onRetry?: () => void;
}

export function ErrorDisplay({ message = "Something went wrong", onRetry }: ErrorDisplayProps) {
  return (
    <div className="glass-card p-8 text-center space-y-4">
      <div className="text-4xl">⚠️</div>
      <p className="text-[var(--muted-foreground)]">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-4 py-2 rounded-xl bg-[var(--color-brand)] text-white hover:bg-[var(--color-brand-dark)] transition-colors"
        >
          Try Again
        </button>
      )}
    </div>
  );
}

// --- Badge ---
interface BadgeProps {
  children: ReactNode;
  variant?: "default" | "success" | "danger" | "warning" | "info";
  className?: string;
}

const badgeStyles = {
  default: "bg-[var(--muted)] text-[var(--muted-foreground)]",
  success: "bg-emerald-500/15 text-emerald-500",
  danger: "bg-red-500/15 text-red-500",
  warning: "bg-amber-500/15 text-amber-500",
  info: "bg-blue-500/15 text-blue-500",
};

export function Badge({ children, variant = "default", className }: BadgeProps) {
  return (
    <span className={cn(
      "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium",
      badgeStyles[variant],
      className
    )}>
      {children}
    </span>
  );
}

// --- Empty State ---
interface EmptyStateProps {
  icon?: string;
  title: string;
  description?: string;
  action?: ReactNode;
}

export function EmptyState({ icon = "📭", title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-16 space-y-4">
      <span className="text-5xl">{icon}</span>
      <h3 className="text-lg font-semibold">{title}</h3>
      {description && <p className="text-[var(--muted-foreground)] text-sm">{description}</p>}
      {action}
    </div>
  );
}
