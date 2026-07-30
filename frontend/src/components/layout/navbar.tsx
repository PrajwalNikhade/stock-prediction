"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard, Search, TrendingUp, Star, Bell, Info,
  Sun, Moon, Menu, X, BarChart3, PieChart, LineChart,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useStockSearch } from "@/hooks/use-api";

const navLinks = [
  { href: "/", label: "Dashboard", icon: LayoutDashboard },
  { href: "/stocks", label: "Stocks", icon: TrendingUp },
  { href: "/portfolio", label: "Portfolio", icon: PieChart },
  { href: "/analytics", label: "Analytics", icon: LineChart },
  { href: "/watchlist", label: "Watchlist", icon: Star },
  { href: "/alerts", label: "Alerts", icon: Bell },
  { href: "/about", label: "About", icon: Info },
];

export function Navbar() {
  const pathname = usePathname();
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);

  const { data: searchResults } = useStockSearch(searchQuery);

  useEffect(() => setMounted(true), []);

  // Close search dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setSearchOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const handleSearchInput = useCallback((val: string) => {
    setSearchQuery(val);
    setSearchOpen(val.length > 0);
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-[var(--border)] bg-[var(--glass-bg)] backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between gap-4">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2.5 shrink-0">
            <div className="w-8 h-8 rounded-lg gradient-brand flex items-center justify-center">
              <BarChart3 className="w-4.5 h-4.5 text-white" />
            </div>
            <span className="text-lg font-bold tracking-tight hidden sm:block">
              SmartStock<span className="text-[var(--color-brand)]"> AI</span>
            </span>
          </Link>

          {/* Search Bar */}
          <div ref={searchRef} className="relative flex-1 max-w-md hidden md:block">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--muted-foreground)]" />
              <input
                type="text"
                placeholder="Search stocks... (e.g. RELIANCE, TCS)"
                value={searchQuery}
                onChange={(e) => handleSearchInput(e.target.value)}
                className="w-full pl-10 pr-4 py-2 rounded-xl bg-[var(--muted)] border border-[var(--border)] text-sm
                  focus:outline-none focus:ring-2 focus:ring-[var(--color-brand)] focus:border-transparent
                  placeholder:text-[var(--muted-foreground)] transition-all"
              />
            </div>

            <AnimatePresence>
              {searchOpen && searchResults && searchResults.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  className="absolute top-full left-0 right-0 mt-2 rounded-xl border border-[var(--border)] bg-[var(--card)] shadow-xl overflow-hidden z-50"
                >
                  {searchResults.slice(0, 6).map((stock) => (
                    <Link
                      key={stock.symbol}
                      href={`/stocks/${stock.symbol}`}
                      onClick={() => { setSearchOpen(false); setSearchQuery(""); }}
                      className="flex items-center gap-3 px-4 py-3 hover:bg-[var(--muted)] transition-colors"
                    >
                      <div className="w-8 h-8 rounded-lg bg-[var(--color-brand)]/10 flex items-center justify-center text-xs font-bold text-[var(--color-brand)]">
                        {stock.symbol.slice(0, 2)}
                      </div>
                      <div>
                        <p className="text-sm font-medium">{stock.symbol}</p>
                        <p className="text-xs text-[var(--muted-foreground)]">{stock.name}</p>
                      </div>
                    </Link>
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Nav Links */}
          <nav className="hidden lg:flex items-center gap-1">
            {navLinks.map(({ href, label, icon: Icon }) => {
              const isActive = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={cn(
                    "flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm font-medium transition-all",
                    isActive
                      ? "bg-[var(--color-brand)]/10 text-[var(--color-brand)]"
                      : "text-[var(--muted-foreground)] hover:text-[var(--foreground)] hover:bg-[var(--muted)]"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {label}
                </Link>
              );
            })}
          </nav>

          {/* Right side */}
          <div className="flex items-center gap-2">
            {mounted && (
              <button
                onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
                className="p-2 rounded-xl hover:bg-[var(--muted)] transition-colors"
                aria-label="Toggle theme"
              >
                {theme === "dark" ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
            )}

            {/* Mobile menu */}
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="p-2 rounded-xl hover:bg-[var(--muted)] transition-colors lg:hidden"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile nav */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="lg:hidden border-t border-[var(--border)] overflow-hidden"
          >
            <nav className="p-4 space-y-1">
              {navLinks.map(({ href, label, icon: Icon }) => {
                const isActive = pathname === href;
                return (
                  <Link
                    key={href}
                    href={href}
                    onClick={() => setMobileOpen(false)}
                    className={cn(
                      "flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all",
                      isActive
                        ? "bg-[var(--color-brand)]/10 text-[var(--color-brand)]"
                        : "text-[var(--muted-foreground)] hover:bg-[var(--muted)]"
                    )}
                  >
                    <Icon className="w-5 h-5" />
                    {label}
                  </Link>
                );
              })}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
