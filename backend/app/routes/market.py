"""Market API routes."""
from flask import Blueprint, jsonify
from app.services.market_service import MarketService

market_bp = Blueprint('market', __name__)


@market_bp.route('/status')
def market_status():
    """Get current market status (open/closed)."""
    data = MarketService.get_market_status()
    return jsonify(data)


@market_bp.route('/indices')
def market_indices():
    """Get Nifty 50 and Sensex data."""
    data = MarketService.get_indices()
    return jsonify(data)


@market_bp.route('/gainers')
def top_gainers():
    """Get top gaining stocks."""
    from app.services.stock_service import StockService
    data = StockService.get_top_movers(direction='gainers', limit=10)
    return jsonify(data)


@market_bp.route('/losers')
def top_losers():
    """Get top losing stocks."""
    from app.services.stock_service import StockService
    data = StockService.get_top_movers(direction='losers', limit=10)
    return jsonify(data)


@market_bp.route('/sectors')
def sector_performance():
    """Get sector-wise performance for heatmap."""
    data = MarketService.get_sector_performance()
    return jsonify(data)
