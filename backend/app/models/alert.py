"""Alert model."""
from app import db
from datetime import datetime


class Alert(db.Model):
    """Price and indicator alert model."""
    __tablename__ = 'alerts'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    alert_type = db.Column(db.String(20), nullable=False)  # price_above, price_below, rsi_above, rsi_below
    target_value = db.Column(db.Float, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_triggered = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    triggered_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'alert_type': self.alert_type,
            'target_value': self.target_value,
            'is_active': self.is_active,
            'is_triggered': self.is_triggered,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
        }
