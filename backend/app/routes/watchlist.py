"""Watchlist API routes."""
from flask import Blueprint, jsonify, request
from app import db
from app.models.watchlist import Watchlist

watchlist_bp = Blueprint('watchlist', __name__)
DEFAULT_USER_ID = 1


@watchlist_bp.route('', methods=['GET'])
def get_watchlist():
    """Get user's watchlist."""
    items = Watchlist.query.filter_by(user_id=DEFAULT_USER_ID).order_by(
        Watchlist.is_pinned.desc(),
        Watchlist.added_at.desc()
    ).all()
    return jsonify([item.to_dict() for item in items])


@watchlist_bp.route('', methods=['POST'])
def add_to_watchlist():
    """Add a stock to watchlist."""
    data = request.get_json()
    if not data or not data.get('symbol'):
        return jsonify({'error': 'Symbol is required'}), 400

    # Check for duplicates
    existing = Watchlist.query.filter_by(
        user_id=DEFAULT_USER_ID,
        symbol=data['symbol'].upper()
    ).first()

    if existing:
        return jsonify({'error': 'Stock already in watchlist', 'item': existing.to_dict()}), 409

    item = Watchlist(
        user_id=DEFAULT_USER_ID,
        symbol=data['symbol'].upper(),
        company_name=data.get('company_name', data['symbol']),
        is_pinned=data.get('is_pinned', False),
        is_favorite=data.get('is_favorite', False),
    )
    db.session.add(item)
    db.session.commit()

    return jsonify(item.to_dict()), 201


@watchlist_bp.route('/<int:item_id>', methods=['PUT'])
def update_watchlist_item(item_id: int):
    """Update watchlist item (pin/favorite)."""
    item = Watchlist.query.filter_by(id=item_id, user_id=DEFAULT_USER_ID).first()
    if not item:
        return jsonify({'error': 'Item not found'}), 404

    data = request.get_json()
    if 'is_pinned' in data:
        item.is_pinned = data['is_pinned']
    if 'is_favorite' in data:
        item.is_favorite = data['is_favorite']

    db.session.commit()
    return jsonify(item.to_dict())


@watchlist_bp.route('/<int:item_id>', methods=['DELETE'])
def remove_from_watchlist(item_id: int):
    """Remove a stock from watchlist."""
    item = Watchlist.query.filter_by(id=item_id, user_id=DEFAULT_USER_ID).first()
    if not item:
        return jsonify({'error': 'Item not found'}), 404

    db.session.delete(item)
    db.session.commit()
    return jsonify({'message': 'Removed from watchlist', 'id': item_id})
