import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "@/providers/providers";
import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "SmartStock AI — Explainable Indian Stock Market Analytics",
  description:
    "AI-powered Indian stock market analytics platform with XGBoost predictions, technical indicators, candlestick pattern detection, sentiment analysis, and explainable recommendations.",
  keywords: [
    "Indian stocks",
    "NSE",
    "Nifty 50",
    "Sensex",
    "stock prediction",
    "XGBoost",
    "technical analysis",
    "machine learning",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} font-sans min-h-screen flex flex-col antialiased`}
      >
        <Providers>
          <Navbar />
          <main className="flex-1">{children}</main>
          <Footer />
        </Providers>
      </body>
    </html>
  );
}
