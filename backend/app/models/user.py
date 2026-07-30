"""User model."""
from app import db
from datetime import datetime


class User(db.Model):
    """User account model."""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    watchlist_items = db.relationship('Watchlist', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    alerts = db.relationship('Alert', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    recent_searches = db.relationship('RecentSearch', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
