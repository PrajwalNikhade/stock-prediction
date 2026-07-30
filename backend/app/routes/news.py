"""News API routes."""
from flask import Blueprint, jsonify
from app.services.news_service import NewsService

news_bp = Blueprint('news', __name__)


@news_bp.route('/market')
def market_news():
    """Get general Indian stock market news."""
    articles = NewsService.get_market_news(limit=20)
    return jsonify(articles)


@news_bp.route('/<symbol>')
def stock_news(symbol: str):
    """Get news for a specific stock with sentiment analysis."""
    from app.utils.nse_stocks import NSE_STOCKS_MAP
    stock_meta = NSE_STOCKS_MAP.get(symbol.upper(), {})
    company_name = stock_meta.get('name', '')

    data = NewsService.get_stock_news(symbol, company_name=company_name, limit=15)
    return jsonify(data)
