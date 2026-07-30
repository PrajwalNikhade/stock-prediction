"""Watchlist model."""
from app import db
from datetime import datetime


class Watchlist(db.Model):
    """User watchlist item model."""
    __tablename__ = 'watchlist'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    company_name = db.Column(db.String(200), nullable=False)
    is_pinned = db.Column(db.Boolean, default=False)
    is_favorite = db.Column(db.Boolean, default=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'symbol', name='uq_user_symbol'),
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'is_pinned': self.is_pinned,
            'is_favorite': self.is_favorite,
            'added_at': self.added_at.isoformat() if self.added_at else None,
        }
