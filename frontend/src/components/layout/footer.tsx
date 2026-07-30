"use client";

import { BarChart3, Heart } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-[var(--border)] mt-auto">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md gradient-brand flex items-center justify-center">
              <BarChart3 className="w-3 h-3 text-white" />
            </div>
            <span className="text-sm font-semibold">
              SmartStock<span className="text-[var(--color-brand)]"> AI</span>
            </span>
          </div>

          <p className="text-xs text-[var(--muted-foreground)] text-center">
            Explainable Indian Stock Market Analytics & Prediction Platform.
            Built with Next.js, Flask, XGBoost.
          </p>

          <div className="flex items-center gap-4 text-[var(--muted-foreground)]">
            <span className="text-xs flex items-center gap-1">
              Made with <Heart className="w-3 h-3 text-red-500 fill-red-500" /> for learning
            </span>
          </div>
        </div>

        <p className="text-xs text-center text-[var(--muted-foreground)] mt-4">
          ⚠️ Disclaimer: This is an educational project. Predictions are not financial advice. Always consult a qualified financial advisor.
        </p>
      </div>
    </footer>
  );
}
