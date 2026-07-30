"""News cache model."""
from app import db
from datetime import datetime


class NewsCache(db.Model):
    """Cached news articles with sentiment model."""
    __tablename__ = 'news_cache'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    symbol = db.Column(db.String(20), nullable=True)
    title = db.Column(db.String(500), nullable=False)
    url = db.Column(db.String(1000), nullable=False)
    source = db.Column(db.String(200))
    sentiment_score = db.Column(db.Float, default=0.0)
    sentiment_label = db.Column(db.String(10), default='neutral')
    published_at = db.Column(db.DateTime)
    cached_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'title': self.title,
            'url': self.url,
            'source': self.source,
            'sentiment_score': self.sentiment_score,
            'sentiment_label': self.sentiment_label,
            'published_at': self.published_at.isoformat() if self.published_at else None,
            'cached_at': self.cached_at.isoformat() if self.cached_at else None,
        }
