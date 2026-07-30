"""Alerts API routes."""
from flask import Blueprint, jsonify, request
from app import db
from app.models.alert import Alert

alerts_bp = Blueprint('alerts', __name__)
DEFAULT_USER_ID = 1


@alerts_bp.route('', methods=['GET'])
def get_alerts():
    """Get user's alerts."""
    active_only = request.args.get('active', 'false').lower() == 'true'

    query = Alert.query.filter_by(user_id=DEFAULT_USER_ID)
    if active_only:
        query = query.filter_by(is_active=True)

    alerts = query.order_by(Alert.created_at.desc()).all()
    return jsonify([alert.to_dict() for alert in alerts])


@alerts_bp.route('', methods=['POST'])
def create_alert():
    """Create a new price or RSI alert."""
    data = request.get_json()

    if not data or not data.get('symbol') or not data.get('alert_type'):
        return jsonify({'error': 'symbol, alert_type, and target_value are required'}), 400

    valid_types = ['price_above', 'price_below', 'rsi_above', 'rsi_below']
    if data['alert_type'] not in valid_types:
        return jsonify({'error': f'alert_type must be one of: {valid_types}'}), 400

    alert = Alert(
        user_id=DEFAULT_USER_ID,
        symbol=data['symbol'].upper(),
        alert_type=data['alert_type'],
        target_value=float(data.get('target_value', 0)),
        is_active=True,
    )
    db.session.add(alert)
    db.session.commit()

    return jsonify(alert.to_dict()), 201


@alerts_bp.route('/<int:alert_id>', methods=['PUT'])
def update_alert(alert_id: int):
    """Update an alert."""
    alert = Alert.query.filter_by(id=alert_id, user_id=DEFAULT_USER_ID).first()
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404

    data = request.get_json()
    if 'is_active' in data:
        alert.is_active = data['is_active']
    if 'target_value' in data:
        alert.target_value = float(data['target_value'])

    db.session.commit()
    return jsonify(alert.to_dict())


@alerts_bp.route('/<int:alert_id>', methods=['DELETE'])
def delete_alert(alert_id: int):
    """Delete an alert."""
    alert = Alert.query.filter_by(id=alert_id, user_id=DEFAULT_USER_ID).first()
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404

    db.session.delete(alert)
    db.session.commit()
    return jsonify({'message': 'Alert deleted', 'id': alert_id})
