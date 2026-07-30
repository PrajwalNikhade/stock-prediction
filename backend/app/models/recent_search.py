"""Recent search model."""
from app import db
from datetime import datetime


class RecentSearch(db.Model):
    """User recent search history model."""
    __tablename__ = 'recent_searches'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    company_name = db.Column(db.String(200))
    searched_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'searched_at': self.searched_at.isoformat() if self.searched_at else None,
        }
