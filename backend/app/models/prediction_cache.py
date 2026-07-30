"""Prediction cache model."""
from app import db
from datetime import datetime
import json


class PredictionCache(db.Model):
    """Cached prediction results model."""
    __tablename__ = 'predictions_cache'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    symbol = db.Column(db.String(20), nullable=False)
    prediction_date = db.Column(db.Date, nullable=False)
    current_price = db.Column(db.Float)
    pred_1d = db.Column(db.Float)
    pred_7d = db.Column(db.Float)
    pred_30d = db.Column(db.Float)
    confidence = db.Column(db.Float)
    feature_importance = db.Column(db.Text)  # JSON string
    metrics = db.Column(db.Text)  # JSON string
    recommendation = db.Column(db.String(10))
    recommendation_reasons = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ttl_hours = db.Column(db.Integer, default=24)

    __table_args__ = (
        db.UniqueConstraint('symbol', 'prediction_date', name='uq_pred_symbol_date'),
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'current_price': self.current_price,
            'pred_1d': self.pred_1d,
            'pred_7d': self.pred_7d,
            'pred_30d': self.pred_30d,
            'confidence': self.confidence,
            'feature_importance': json.loads(self.feature_importance) if self.feature_importance else {},
            'metrics': json.loads(self.metrics) if self.metrics else {},
            'recommendation': self.recommendation,
            'recommendation_reasons': json.loads(self.recommendation_reasons) if self.recommendation_reasons else [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
