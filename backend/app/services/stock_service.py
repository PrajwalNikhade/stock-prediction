"""Stock data service using Yahoo Finance."""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from app.utils.nse_stocks import NSE_STOCKS, NSE_STOCKS_MAP


class StockService:
    """Service for fetching stock data from Yahoo Finance."""

    @staticmethod
    def _nse_symbol(symbol: str) -> str:
        """Convert symbol to NSE Yahoo Finance format."""
        symbol = symbol.upper().replace('.NS', '')
        return f"{symbol}.NS"

    @staticmethod
    def search_stocks(query: str, limit: int = 10) -> list[dict]:
        """Search stocks by symbol or company name."""
        query = query.upper().strip()
        if not query:
            return []

        results = []
        for stock in NSE_STOCKS:
            if (query in stock['symbol'].upper() or
                    query in stock['name'].upper()):
                results.append(stock)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def get_stock_info(symbol: str) -> dict:
        """Get comprehensive stock information."""
        try:
            ticker = yf.Ticker(StockService._nse_symbol(symbol))
            info = ticker.info
            hist = ticker.history(period='5d')

            if hist.empty:
                return {'error': f'No data found for {symbol}'}

            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else 0

            # Get stock details from curated list
            stock_meta = NSE_STOCKS_MAP.get(symbol.upper(), {})

            return {
                'symbol': symbol.upper(),
                'name': info.get('longName', stock_meta.get('name', symbol)),
                'sector': info.get('sector', stock_meta.get('sector', 'N/A')),
                'industry': info.get('industry', 'N/A'),
                'current_price': round(float(current_price), 2),
                'previous_close': round(float(prev_close), 2),
                'change': round(float(change), 2),
                'change_percent': round(float(change_pct), 2),
                'open': round(float(hist['Open'].iloc[-1]), 2),
                'high': round(float(hist['High'].iloc[-1]), 2),
                'low': round(float(hist['Low'].iloc[-1]), 2),
                'volume': int(hist['Volume'].iloc[-1]),
                'fifty_two_week_high': round(float(info.get('fiftyTwoWeekHigh', 0)), 2),
                'fifty_two_week_low': round(float(info.get('fiftyTwoWeekLow', 0)), 2),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': round(float(info.get('trailingPE', 0)), 2) if info.get('trailingPE') else None,
                'eps': round(float(info.get('trailingEps', 0)), 2) if info.get('trailingEps') else None,
                'dividend_yield': round(float(info.get('dividendYield', 0)) * 100, 2) if info.get('dividendYield') else None,
                'beta': round(float(info.get('beta', 0)), 2) if info.get('beta') else None,
                'description': info.get('longBusinessSummary', ''),
                'website': info.get('website', ''),
                'currency': info.get('currency', 'INR'),
            }
        except Exception as e:
            return {'error': str(e), 'symbol': symbol}

    @staticmethod
    def get_historical_data(symbol: str, period: str = '1y',
                            interval: str = '1d') -> dict:
        """Get historical OHLCV data."""
        try:
            ticker = yf.Ticker(StockService._nse_symbol(symbol))
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                return {'error': f'No historical data for {symbol}'}

            hist = hist.reset_index()
            # Convert datetime to string for JSON serialization
            if 'Date' in hist.columns:
                hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
            elif 'Datetime' in hist.columns:
                hist['Date'] = hist['Datetime'].dt.strftime('%Y-%m-%d %H:%M')

            return {
                'symbol': symbol.upper(),
                'period': period,
                'interval': interval,
                'data': {
                    'dates': hist['Date'].tolist(),
                    'open': [round(x, 2) for x in hist['Open'].tolist()],
                    'high': [round(x, 2) for x in hist['High'].tolist()],
                    'low': [round(x, 2) for x in hist['Low'].tolist()],
                    'close': [round(x, 2) for x in hist['Close'].tolist()],
                    'volume': [int(x) for x in hist['Volume'].tolist()],
                },
                'count': len(hist),
            }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def get_top_movers(direction: str = 'gainers', limit: int = 10) -> list[dict]:
        """Get top gainers or losers from curated NSE stocks."""
        movers = []
        symbols = [s['symbol'] for s in NSE_STOCKS[:30]]  # Check top 30

        for symbol in symbols:
            try:
                ticker = yf.Ticker(StockService._nse_symbol(symbol))
                hist = ticker.history(period='2d')
                if len(hist) < 2:
                    continue

                current = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2])
                change_pct = ((current - prev) / prev) * 100

                stock_meta = NSE_STOCKS_MAP.get(symbol, {})
                movers.append({
                    'symbol': symbol,
                    'name': stock_meta.get('name', symbol),
                    'sector': stock_meta.get('sector', 'N/A'),
                    'price': round(current, 2),
                    'change': round(current - prev, 2),
                    'change_percent': round(change_pct, 2),
                })
            except Exception:
                continue

        reverse = direction == 'gainers'
        movers.sort(key=lambda x: x['change_percent'], reverse=reverse)
        return movers[:limit]

    @staticmethod
    def get_raw_dataframe(symbol: str, period: str = '2y') -> pd.DataFrame:
        """Get raw pandas DataFrame for ML processing."""
        ticker = yf.Ticker(StockService._nse_symbol(symbol))
        df = ticker.history(period=period)
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        return df
