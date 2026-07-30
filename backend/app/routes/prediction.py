"""Prediction API routes."""
from flask import Blueprint, jsonify
from app.services.prediction_service import PredictionService
from app.services.recommendation_service import RecommendationService
from app.services.news_service import NewsService

prediction_bp = Blueprint('prediction', __name__)


@prediction_bp.route('/<symbol>')
def get_prediction(symbol: str):
    """Get ML predictions for 1d, 7d, 30d."""
    data = PredictionService.get_prediction(symbol)
    if 'error' in data:
        return jsonify(data), 400
    return jsonify(data)


@prediction_bp.route('/<symbol>/importance')
def feature_importance(symbol: str):
    """Get feature importance from the trained model."""
    data = PredictionService.get_prediction(symbol)
    if 'error' in data:
        return jsonify(data), 400

    return jsonify({
        'symbol': symbol.upper(),
        'feature_importance': data.get('feature_importance', {}),
        'top_factors': data.get('top_factors', []),
    })


@prediction_bp.route('/<symbol>/metrics')
def model_metrics(symbol: str):
    """Get model evaluation metrics (MAE, RMSE, R²)."""
    data = PredictionService.get_prediction(symbol)
    if 'error' in data:
        return jsonify(data), 400

    return jsonify({
        'symbol': symbol.upper(),
        'metrics': data.get('metrics', {}),
    })


@prediction_bp.route('/<symbol>/recommendation')
def ai_recommendation(symbol: str):
    """Get AI-powered Buy/Hold/Sell recommendation."""
    # Get prediction data
    prediction_data = PredictionService.get_prediction(symbol)
    if 'error' in prediction_data:
        return jsonify(prediction_data), 400

    # Get technical indicators
    indicators = PredictionService.get_technical_indicators(symbol)
    if 'error' in indicators:
        return jsonify(indicators), 400

    # Get sentiment score
    sentiment_score = NewsService.get_sentiment_score_for_prediction(symbol)

    # Generate recommendation
    recommendation = RecommendationService.generate_recommendation(
        prediction_data=prediction_data,
        indicators=indicators,
        sentiment_score=sentiment_score,
    )

    return jsonify({
        'symbol': symbol.upper(),
        **recommendation,
    })
