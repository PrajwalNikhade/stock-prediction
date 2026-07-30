"""Stock API routes."""
from flask import Blueprint, jsonify, request
from app.services.stock_service import StockService
from app.services.candlestick_service import CandlestickService
from app.services.prediction_service import PredictionService
from app import db
from app.models.recent_search import RecentSearch
from datetime import datetime

stocks_bp = Blueprint('stocks', __name__)


@stocks_bp.route('/search')
def search_stocks():
    """Search stocks by symbol or company name."""
    query = request.args.get('q', '').strip()
    limit = int(request.args.get('limit', 10))

    if not query:
        return jsonify([])

    results = StockService.search_stocks(query, limit=limit)

    # Record search for the first result
    if results:
        try:
            search = RecentSearch(
                user_id=1,  # Default demo user
                symbol=results[0]['symbol'],
                company_name=results[0]['name'],
                searched_at=datetime.utcnow(),
            )
            db.session.add(search)
            db.session.commit()
        except Exception:
            db.session.rollback()

    return jsonify(results)


@stocks_bp.route('/recent-searches')
def recent_searches():
    """Get recent search history."""
    limit = int(request.args.get('limit', 10))
    searches = RecentSearch.query.filter_by(user_id=1).order_by(
        RecentSearch.searched_at.desc()
    ).limit(limit).all()

    # Deduplicate by symbol
    seen = set()
    unique = []
    for s in searches:
        if s.symbol not in seen:
            seen.add(s.symbol)
            unique.append(s.to_dict())

    return jsonify(unique)


@stocks_bp.route('/<symbol>')
def stock_info(symbol: str):
    """Get comprehensive stock information."""
    data = StockService.get_stock_info(symbol)
    return jsonify(data)


@stocks_bp.route('/<symbol>/history')
def stock_history(symbol: str):
    """Get historical OHLCV data."""
    period = request.args.get('period', '1y')
    interval = request.args.get('interval', '1d')
    data = StockService.get_historical_data(symbol, period=period, interval=interval)
    return jsonify(data)


@stocks_bp.route('/<symbol>/indicators')
def stock_indicators(symbol: str):
    """Get technical indicators."""
    data = PredictionService.get_technical_indicators(symbol)
    return jsonify(data)


@stocks_bp.route('/<symbol>/candlestick-patterns')
def candlestick_patterns(symbol: str):
    """Detect candlestick patterns."""
    try:
        df = StockService.get_raw_dataframe(symbol, period='3mo')
        patterns = CandlestickService.detect_patterns(df, lookback=30)
        summary = CandlestickService.get_pattern_summary(patterns)
        return jsonify({'symbol': symbol.upper(), **summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
