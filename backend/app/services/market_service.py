"""Market data service for indices and sector data."""
import yfinance as yf
from datetime import datetime
import pytz


class MarketService:
    """Service for market-level data: indices, status, sectors."""

    # NSE index symbols on Yahoo Finance
    NIFTY_SYMBOL = '^NSEI'
    SENSEX_SYMBOL = '^BSESN'

    # Sector ETFs / representative stocks for sector performance
    SECTOR_REPRESENTATIVES = {
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
        'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'],
        'Energy': ['RELIANCE.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS', 'BPCL.NS'],
        'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
        'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'M&M.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS'],
        'FMCG': ['HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS'],
        'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS', 'VEDL.NS'],
        'Consumer': ['TITAN.NS', 'ASIANPAINT.NS'],
        'Infrastructure': ['LT.NS', 'ADANIPORTS.NS'],
        'Finance': ['BAJFINANCE.NS', 'BAJAJFINSV.NS'],
    }

    @staticmethod
    def get_market_status() -> dict:
        """Check if Indian stock market is open."""
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)

        is_weekday = now.weekday() < 5  # Mon-Fri
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        is_open = is_weekday and market_open <= now <= market_close

        return {
            'is_open': is_open,
            'status': 'Open' if is_open else 'Closed',
            'current_time': now.strftime('%Y-%m-%d %H:%M:%S IST'),
            'market_hours': '09:15 AM - 03:30 PM IST',
            'next_open': 'Next trading day at 09:15 AM IST' if not is_open else None,
        }

    @staticmethod
    def get_indices() -> dict:
        """Get Nifty 50 and Sensex data."""
        indices = {}

        for name, symbol in [('nifty50', MarketService.NIFTY_SYMBOL),
                              ('sensex', MarketService.SENSEX_SYMBOL)]:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')

                if len(hist) >= 2:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2])
                    change = current - prev
                    change_pct = (change / prev) * 100

                    indices[name] = {
                        'value': round(current, 2),
                        'change': round(change, 2),
                        'change_percent': round(change_pct, 2),
                        'high': round(float(hist['High'].iloc[-1]), 2),
                        'low': round(float(hist['Low'].iloc[-1]), 2),
                        'open': round(float(hist['Open'].iloc[-1]), 2),
                    }
                else:
                    indices[name] = {'error': 'Insufficient data'}

            except Exception as e:
                indices[name] = {'error': str(e)}

        return indices

    @staticmethod
    def get_sector_performance() -> list[dict]:
        """Calculate sector performance from representative stocks."""
        sectors = []

        for sector_name, symbols in MarketService.SECTOR_REPRESENTATIVES.items():
            changes = []
            for sym in symbols:
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period='2d')
                    if len(hist) >= 2:
                        current = float(hist['Close'].iloc[-1])
                        prev = float(hist['Close'].iloc[-2])
                        change_pct = ((current - prev) / prev) * 100
                        changes.append(change_pct)
                except Exception:
                    continue

            if changes:
                avg_change = sum(changes) / len(changes)
                sectors.append({
                    'sector': sector_name,
                    'change_percent': round(avg_change, 2),
                    'stocks_count': len(symbols),
                    'direction': 'up' if avg_change > 0 else 'down',
                })

        # Sort by change percent
        sectors.sort(key=lambda x: x['change_percent'], reverse=True)
        return sectors
