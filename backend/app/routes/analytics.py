"""Analytics API routes."""
from flask import Blueprint, jsonify, request
from app.services.analytics_service import AnalyticsService

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/insights')
def business_insights():
    """Get automated natural-language business insights."""
    insights = AnalyticsService.generate_business_insights()
    return jsonify(insights)


@analytics_bp.route('/correlation')
def correlation_matrix():
    """Get correlation matrix for selected stocks."""
    symbols_param = request.args.get('symbols', '')
    symbols = [s.strip().upper() for s in symbols_param.split(',') if s.strip()] if symbols_param else None

    data = AnalyticsService.get_correlation_matrix(symbols)
    return jsonify(data)


@analytics_bp.route('/market-summary')
def market_summary():
    """Get overall market analytics, best/worst stocks, volatility metrics."""
    data = AnalyticsService.get_market_analytics()
    return jsonify(data)
