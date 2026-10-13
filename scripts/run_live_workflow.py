"""Reproducible Live-Data Workflow for CrowdWisdomTrading Assessment.

Executes the complete live-data pipeline for BTCUSDT (1d, 180 days lookback):
1. Fetches up to 180 days of actual BTCUSDT daily candles from Binance.
2. Performs a small live test against configured Apify actor before full attempt.
3. Retrieves recent, relevant news via Tavily and Exa.
4. Deduplicates and normalizes records with UTC timestamps.
5. Saves all observations and experiment provenance into SQLite.
6. Builds point-in-time features with explicit missing sentiment flags.
7. Conducts expanding-window walk-forward validation across candidate models and naive baseline.
8. Produces n+0 through n+5 forecast matrix.
9. Runs trading strategy backtest with 10 bps transaction costs.
10. Generates reports/evaluation_report.md and reports/current_sentiment_outlook.md.
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.generate_reports import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Execute reproducible live-data workflow.")
    parser.add_argument("--asset", default="BTCUSDT", help="Asset ticker (default: BTCUSDT)")
    parser.add_argument("--timeframe", default="1d", choices=["1d", "1h"], help="Timeframe (default: 1d)")
    parser.add_argument("--days", type=int, default=180, help="Lookback days (default: 180)")
    parser.add_argument("--run-id", default=None, help="Optional run identifier override")
    args = parser.parse_args()

    print("========================================================================")
    print(" CROWDWISDOMTRADING — REPRODUCIBLE LIVE DATA WORKFLOW")
    print(f" Asset: {args.asset} | Timeframe: {args.timeframe} | Lookback: {args.days} days")
    print("========================================================================")

    res = run_pipeline(
        ticker=args.asset,
        timeframe=args.timeframe,
        days=args.days,
        use_demo=False,
        run_id=args.run_id,
    )

    if res:
        print("\nWorkflow completed successfully.")
        print(f"Run ID: {res['run_id']}")
        print(f"Data Mode: {res['data_mode'].upper()}")


if __name__ == "__main__":
    main()
