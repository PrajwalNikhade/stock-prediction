"""Prediction orchestration service."""
import json
from datetime import datetime, date, timedelta
from app import db
from app.services.stock_service import StockService
from app.ml.feature_engineering import FeatureEngineer
from app.ml.model import StockPredictor
from app.models.prediction_cache import PredictionCache


class PredictionService:
    """Orchestrates the full prediction pipeline."""

    @staticmethod
    def get_prediction(symbol: str, use_cache: bool = True) -> dict:
        """
        Get stock prediction with caching support.

        Pipeline: fetch data → engineer features → train XGBoost → predict → cache
        """
        symbol = symbol.upper()

        # Check cache first
        if use_cache:
            cached = PredictionService._get_cached(symbol)
            if cached:
                return cached

        try:
            # Step 1: Fetch raw data
            df = StockService.get_raw_dataframe(symbol, period='2y')

            if len(df) < 100:
                return {'error': f'Insufficient data for {symbol}. Need at least 100 trading days.'}

            current_price = round(float(df['Close'].iloc[-1]), 2)

            # Step 2: Feature engineering
            fe = FeatureEngineer()
            df_features = fe.generate_features(df)

            # Step 3: Train and predict for each horizon
            predictor = StockPredictor()
            predictions = {}
            all_metrics = {}
            all_importance = {}

            for horizon_days, horizon_key in [(1, '1d'), (7, '7d'), (30, '30d')]:
                X, y, feature_names, scaler = fe.prepare_features(
                    df_features, target_horizon=horizon_days
                )

                if len(X) < 50:
                    continue

                metrics = predictor.train(X, y, feature_names, horizon_key)
                all_metrics[horizon_key] = metrics

                # Get latest features for prediction
                latest_X = fe.get_latest_features(df_features)
                pred_price = predictor.predict(latest_X, horizon_key)
                confidence = predictor.get_confidence(horizon_key)

                change = pred_price - current_price
                change_pct = (change / current_price) * 100 if current_price else 0

                predictions[horizon_key] = {
                    'predicted_price': pred_price,
                    'change': round(change, 2),
                    'change_percent': round(change_pct, 2),
                    'direction': 'up' if change > 0 else 'down',
                    'confidence': confidence,
                }

                all_importance[horizon_key] = predictor.get_top_features(horizon_key, top_n=15)

            if not predictions:
                return {'error': 'Failed to generate predictions'}

            # Step 4: Build result
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'prediction_date': date.today().isoformat(),
                'predictions': predictions,
                'metrics': all_metrics,
                'feature_importance': all_importance,
                'top_factors': predictor.get_top_features('1d', top_n=5),
            }

            # Step 5: Cache results
            PredictionService._cache_result(symbol, result)

            return result

        except Exception as e:
            return {'error': f'Prediction failed for {symbol}: {str(e)}'}

    @staticmethod
    def _get_cached(symbol: str) -> dict | None:
        """Retrieve cached prediction if still valid."""
        cached = PredictionCache.query.filter_by(
            symbol=symbol,
            prediction_date=date.today()
        ).first()

        if cached:
            ttl = timedelta(hours=cached.ttl_hours)
            if datetime.utcnow() - cached.created_at < ttl:
                return cached.to_dict()

        return None

    @staticmethod
    def _cache_result(symbol: str, result: dict) -> None:
        """Cache prediction result to database."""
        try:
            preds = result.get('predictions', {})

            # Delete existing cache for today
            PredictionCache.query.filter_by(
                symbol=symbol,
                prediction_date=date.today()
            ).delete()

            cache_entry = PredictionCache(
                symbol=symbol,
                prediction_date=date.today(),
                current_price=result['current_price'],
                pred_1d=preds.get('1d', {}).get('predicted_price'),
                pred_7d=preds.get('7d', {}).get('predicted_price'),
                pred_30d=preds.get('30d', {}).get('predicted_price'),
                confidence=preds.get('1d', {}).get('confidence', 0),
                feature_importance=json.dumps(result.get('feature_importance', {})),
                metrics=json.dumps(result.get('metrics', {})),
                recommendation=result.get('recommendation', ''),
                recommendation_reasons=json.dumps(result.get('recommendation_reasons', [])),
            )
            db.session.add(cache_entry)
            db.session.commit()
        except Exception:
            db.session.rollback()

    @staticmethod
    def get_technical_indicators(symbol: str) -> dict:
        """Get current technical indicator values for a stock."""
        try:
            df = StockService.get_raw_dataframe(symbol, period='1y')
            fe = FeatureEngineer()
            df_feat = fe.generate_features(df)

            latest = df_feat.iloc[-1]

            indicators = {
                'rsi': round(float(latest.get('RSI', 0)), 2),
                'macd': round(float(latest.get('MACD', 0)), 4),
                'macd_signal': round(float(latest.get('MACD_Signal', 0)), 4),
                'macd_histogram': round(float(latest.get('MACD_Hist', 0)), 4),
                'adx': round(float(latest.get('ADX', 0)), 2),
                'atr': round(float(latest.get('ATR', 0)), 2),
                'cci': round(float(latest.get('CCI', 0)), 2),
                'williams_r': round(float(latest.get('Williams_R', 0)), 2),
                'stochastic_k': round(float(latest.get('Stochastic_K', 0)), 2),
                'stochastic_d': round(float(latest.get('Stochastic_D', 0)), 2),
                'momentum': round(float(latest.get('Momentum', 0)), 2),
                'roc': round(float(latest.get('ROC', 0)), 2),
                'obv': int(latest.get('OBV', 0)),
                'sma_10': round(float(latest.get('SMA_10', 0)), 2),
                'sma_20': round(float(latest.get('SMA_20', 0)), 2),
                'sma_50': round(float(latest.get('SMA_50', 0)), 2),
                'ema_12': round(float(latest.get('EMA_12', 0)), 2),
                'ema_20': round(float(latest.get('EMA_20', 0)), 2),
                'ema_26': round(float(latest.get('EMA_26', 0)), 2),
                'bb_upper': round(float(latest.get('BB_Upper', 0)), 2),
                'bb_middle': round(float(latest.get('BB_Middle', 0)), 2),
                'bb_lower': round(float(latest.get('BB_Lower', 0)), 2),
                'bb_width': round(float(latest.get('BB_Width', 0)), 4),
                'vwap': round(float(latest.get('VWAP', 0)), 2),
                'daily_return': round(float(latest.get('Daily_Return', 0)), 4),
            }

            # Add interpretation
            indicators['rsi_signal'] = (
                'Overbought' if indicators['rsi'] > 70
                else 'Oversold' if indicators['rsi'] < 30
                else 'Neutral'
            )
            indicators['macd_signal_text'] = (
                'Bullish' if indicators['macd'] > indicators['macd_signal']
                else 'Bearish'
            )
            indicators['trend'] = (
                'Strong' if indicators['adx'] > 25 else 'Weak'
            )

            return {'symbol': symbol.upper(), 'indicators': indicators}

        except Exception as e:
            return {'error': str(e)}
