"use client";

import dynamic from "next/dynamic";
import type { Data, Layout, Config } from "plotly.js";
import { ChartSkeleton } from "@/components/ui/shared";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <ChartSkeleton />,
});

interface ChartProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  height?: number;
}

const darkLayout: Partial<Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#94a3b8", family: "Inter, system-ui, sans-serif", size: 12 },
  xaxis: {
    gridcolor: "rgba(148, 163, 184, 0.08)",
    zerolinecolor: "rgba(148, 163, 184, 0.1)",
    tickfont: { size: 10 },
  },
  yaxis: {
    gridcolor: "rgba(148, 163, 184, 0.08)",
    zerolinecolor: "rgba(148, 163, 184, 0.1)",
    tickfont: { size: 10 },
  },
  margin: { t: 30, r: 20, b: 40, l: 60 },
  showlegend: true,
  legend: {
    bgcolor: "transparent",
    font: { size: 11 },
    orientation: "h",
    y: -0.15,
  },
  hovermode: "x unified" as const,
};

const defaultConfig: Partial<Config> = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};

export function StockChart({ data, layout, config, className, height = 400 }: ChartProps) {
  const mergedLayout: Partial<Layout> = {
    ...darkLayout,
    height,
    ...layout,
  };

  return (
    <div className={`chart-container ${className || ""}`}>
      <Plot
        data={data}
        layout={mergedLayout}
        config={{ ...defaultConfig, ...config }}
        style={{ width: "100%", height }}
        useResizeHandler
      />
    </div>
  );
}

// --- Candlestick Chart ---
export function CandlestickChart({
  dates, open, high, low, close, volume, title = "Price History",
}: {
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
  title?: string;
}) {
  const candlestickData: Data[] = [
    {
      type: "candlestick",
      x: dates,
      open,
      high,
      low,
      close,
      increasing: { line: { color: "#10b981" }, fillcolor: "#10b98133" },
      decreasing: { line: { color: "#ef4444" }, fillcolor: "#ef444433" },
      name: "OHLC",
    } as unknown as Data,
  ];

  const volumeData: Data[] = [
    {
      type: "bar",
      x: dates,
      y: volume,
      marker: {
        color: close.map((c, i) => (c >= open[i] ? "#10b98144" : "#ef444444")),
      },
      name: "Volume",
      yaxis: "y2",
    },
  ];

  return (
    <StockChart
      data={[...candlestickData, ...volumeData]}
      height={500}
      layout={{
        title: { text: title, font: { size: 14 } },
        xaxis: { rangeslider: { visible: false }, type: "date" },
        yaxis: { title: { text: "Price (₹)" }, side: "right", domain: [0.25, 1] },
        yaxis2: { title: { text: "Volume" }, domain: [0, 0.2], anchor: "x" },
        grid: { rows: 2, columns: 1, pattern: "independent" },
      } as any}
    />
  );
}

// --- Feature Importance Chart ---
export function FeatureImportanceChart({
  features, title = "Feature Importance",
}: {
  features: { feature: string; importance: number }[];
  title?: string;
}) {
  const sorted = [...features].sort((a, b) => a.importance - b.importance);
  const data: Data[] = [
    {
      type: "bar",
      x: sorted.map((f) => f.importance),
      y: sorted.map((f) => f.feature),
      orientation: "h",
      marker: {
        color: sorted.map((_, i) => {
          const t = i / sorted.length;
          return `hsl(${230 + t * 60}, 80%, ${50 + t * 15}%)`;
        }),
        line: { width: 0 },
      },
    },
  ];

  return (
    <StockChart
      data={data}
      height={Math.max(300, sorted.length * 28)}
      layout={{
        title: { text: title, font: { size: 14 } },
        xaxis: { title: { text: "Importance Score" } },
        yaxis: { automargin: true, tickfont: { size: 11 } },
        showlegend: false,
        margin: { l: 140 },
      } as any}
    />
  );
}

// --- Indicator Chart (RSI, MACD) ---
export function IndicatorLineChart({
  dates, values, title, thresholds, color = "#6366f1",
}: {
  dates: string[];
  values: number[];
  title: string;
  thresholds?: { value: number; label: string; color: string }[];
  color?: string;
}) {
  const data: Data[] = [
    {
      type: "scatter",
      mode: "lines",
      x: dates,
      y: values,
      line: { color, width: 2 },
      name: title,
      fill: "tozeroy",
      fillcolor: `${color}11`,
    },
  ];

  const shapes = thresholds?.map((t) => ({
    type: "line" as const,
    x0: dates[0],
    x1: dates[dates.length - 1],
    y0: t.value,
    y1: t.value,
    line: { color: t.color, width: 1, dash: "dash" as const },
  }));

  return (
    <StockChart
      data={data}
      height={250}
      layout={{
        title: { text: title, font: { size: 13 } },
        shapes,
        showlegend: false,
      }}
    />
  );
}
