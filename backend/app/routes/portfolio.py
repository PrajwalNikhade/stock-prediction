"""Portfolio API routes."""
from flask import Blueprint, jsonify, request
from app import db
from app.models.portfolio import Portfolio
from app.services.stock_service import StockService
from app.utils.nse_stocks import NSE_STOCKS_MAP

portfolio_bp = Blueprint('portfolio', __name__)
DEFAULT_USER_ID = 1


@portfolio_bp.route('', methods=['GET'])
def get_portfolio():
    """Get user portfolio holdings with live P&L, ROI%, and sector allocation."""
    holdings = Portfolio.query.filter_by(user_id=DEFAULT_USER_ID).all()

    total_invested = 0.0
    total_current_value = 0.0
    detailed_holdings = []
    sector_allocation = {}

    for item in holdings:
        # Fetch current price from yfinance / service
        info = StockService.get_stock_info(item.symbol)
        current_price = info.get('current_price', item.buy_price) if 'error' not in info else item.buy_price
        sector = info.get('sector', NSE_STOCKS_MAP.get(item.symbol, {}).get('sector', 'Other'))

        invested = item.quantity * item.buy_price
        current_val = item.quantity * current_price
        pnl = current_val - invested
        pnl_percent = (pnl / invested * 100) if invested > 0 else 0.0

        total_invested += invested
        total_current_value += current_val

        # Aggregate sector allocation
        sector_allocation[sector] = sector_allocation.get(sector, 0.0) + current_val

        detailed_holdings.append({
            'id': item.id,
            'symbol': item.symbol,
            'company_name': item.company_name,
            'quantity': item.quantity,
            'buy_price': item.buy_price,
            'current_price': current_price,
            'total_invested': round(invested, 2),
            'current_value': round(current_val, 2),
            'pnl': round(pnl, 2),
            'pnl_percent': round(pnl_percent, 2),
            'sector': sector,
            'bought_at': item.bought_at.isoformat() if item.bought_at else None,
        })

    total_pnl = total_current_value - total_invested
    total_roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0

    # Sector allocation percentages
    sector_chart = []
    if total_current_value > 0:
        for sec, val in sector_allocation.items():
            sector_chart.append({
                'sector': sec,
                'value': round(val, 2),
                'percentage': round((val / total_current_value) * 100, 2)
            })

    return jsonify({
        'summary': {
            'total_invested': round(total_invested, 2),
            'total_current_value': round(total_current_value, 2),
            'total_pnl': round(total_pnl, 2),
            'total_roi_percent': round(total_roi, 2),
            'holdings_count': len(detailed_holdings),
        },
        'holdings': detailed_holdings,
        'sector_allocation': sector_chart,
    })


@portfolio_bp.route('', methods=['POST'])
def add_holding():
    """Add a stock holding to user portfolio."""
    data = request.get_json()
    if not data or not data.get('symbol') or not data.get('quantity') or not data.get('buy_price'):
        return jsonify({'error': 'symbol, quantity, and buy_price are required'}), 400

    symbol = data['symbol'].upper()
    stock_meta = NSE_STOCKS_MAP.get(symbol, {})
    company_name = data.get('company_name', stock_meta.get('name', symbol))

    holding = Portfolio(
        user_id=DEFAULT_USER_ID,
        symbol=symbol,
        company_name=company_name,
        quantity=int(data['quantity']),
        buy_price=float(data['buy_price']),
    )
    db.session.add(holding)
    db.session.commit()

    return jsonify(holding.to_dict()), 201


@portfolio_bp.route('/<int:holding_id>', methods=['PUT'])
def update_holding(holding_id: int):
    """Update portfolio holding quantity or buy price."""
    holding = Portfolio.query.filter_by(id=holding_id, user_id=DEFAULT_USER_ID).first()
    if not holding:
        return jsonify({'error': 'Holding not found'}), 404

    data = request.get_json()
    if 'quantity' in data:
        holding.quantity = int(data['quantity'])
    if 'buy_price' in data:
        holding.buy_price = float(data['buy_price'])

    db.session.commit()
    return jsonify(holding.to_dict())


@portfolio_bp.route('/<int:holding_id>', methods=['DELETE'])
def delete_holding(holding_id: int):
    """Delete holding from portfolio."""
    holding = Portfolio.query.filter_by(id=holding_id, user_id=DEFAULT_USER_ID).first()
    if not holding:
        return jsonify({'error': 'Holding not found'}), 404

    db.session.delete(holding)
    db.session.commit()
    return jsonify({'message': 'Holding deleted', 'id': holding_id})
