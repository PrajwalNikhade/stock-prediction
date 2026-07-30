"""Portfolio holding model."""
from app import db
from datetime import datetime


class Portfolio(db.Model):
    """User portfolio holding model."""
    __tablename__ = 'portfolios'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    company_name = db.Column(db.String(200), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=1)
    buy_price = db.Column(db.Float, nullable=False)
    bought_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'quantity': self.quantity,
            'buy_price': self.buy_price,
            'total_invested': round(self.quantity * self.buy_price, 2),
            'bought_at': self.bought_at.isoformat() if self.bought_at else None,
        }
