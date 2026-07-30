"""Rule-based AI recommendation service."""


class RecommendationService:
    """
    Generate Buy/Hold/Sell recommendations using a pure rule-based engine.
    No LLM or ChatGPT - only technical indicators, prediction, and sentiment.
    """

    @staticmethod
    def generate_recommendation(
        prediction_data: dict,
        indicators: dict,
        sentiment_score: float = 0.0,
    ) -> dict:
        """
        Generate a trading recommendation.

        Args:
            prediction_data: Prediction results with predicted prices
            indicators: Technical indicator values
            sentiment_score: Average news sentiment (-1 to 1)

        Returns:
            Dict with recommendation, confidence, and reasons
        """
        signals = []
        score = 0  # -100 to +100 scale

        ind = indicators.get('indicators', indicators)

        # --- RSI Signal ---
        rsi = ind.get('rsi', 50)
        if rsi < 30:
            signals.append({'factor': 'RSI', 'signal': 'Bullish',
                           'detail': f'RSI at {rsi:.1f} (oversold below 30)',
                           'weight': 'strong'})
            score += 20
        elif rsi > 70:
            signals.append({'factor': 'RSI', 'signal': 'Bearish',
                           'detail': f'RSI at {rsi:.1f} (overbought above 70)',
                           'weight': 'strong'})
            score -= 20
        elif rsi < 45:
            signals.append({'factor': 'RSI', 'signal': 'Slightly Bullish',
                           'detail': f'RSI at {rsi:.1f} (below midpoint)',
                           'weight': 'moderate'})
            score += 10
        elif rsi > 55:
            signals.append({'factor': 'RSI', 'signal': 'Slightly Bearish',
                           'detail': f'RSI at {rsi:.1f} (above midpoint)',
                           'weight': 'moderate'})
            score -= 10
        else:
            signals.append({'factor': 'RSI', 'signal': 'Neutral',
                           'detail': f'RSI at {rsi:.1f} (neutral zone)',
                           'weight': 'weak'})

        # --- MACD Signal ---
        macd = ind.get('macd', 0)
        macd_signal = ind.get('macd_signal', 0)
        if macd > macd_signal:
            signals.append({'factor': 'MACD', 'signal': 'Bullish',
                           'detail': 'MACD above signal line (bullish crossover)',
                           'weight': 'strong'})
            score += 15
        else:
            signals.append({'factor': 'MACD', 'signal': 'Bearish',
                           'detail': 'MACD below signal line (bearish crossover)',
                           'weight': 'strong'})
            score -= 15

        # --- Trend (ADX) ---
        adx = ind.get('adx', 20)
        if adx > 25:
            signals.append({'factor': 'Trend Strength', 'signal': 'Strong',
                           'detail': f'ADX at {adx:.1f} indicates strong trend',
                           'weight': 'moderate'})
            # Strong trend amplifies existing direction
            if score > 0:
                score += 10
            else:
                score -= 10
        else:
            signals.append({'factor': 'Trend Strength', 'signal': 'Weak',
                           'detail': f'ADX at {adx:.1f} indicates weak/no trend',
                           'weight': 'weak'})

        # --- Prediction Direction ---
        preds = prediction_data.get('predictions', {})
        pred_1d = preds.get('1d', {})
        pred_7d = preds.get('7d', {})

        if pred_1d.get('direction') == 'up':
            score += 15
            signals.append({'factor': 'ML Prediction (1D)', 'signal': 'Bullish',
                           'detail': f'Predicted {pred_1d.get("change_percent", 0):.2f}% rise',
                           'weight': 'strong'})
        elif pred_1d.get('direction') == 'down':
            score -= 15
            signals.append({'factor': 'ML Prediction (1D)', 'signal': 'Bearish',
                           'detail': f'Predicted {abs(pred_1d.get("change_percent", 0)):.2f}% decline',
                           'weight': 'strong'})

        if pred_7d.get('direction') == 'up':
            score += 10
            signals.append({'factor': 'ML Prediction (7D)', 'signal': 'Bullish',
                           'detail': f'Weekly trend predicted up {pred_7d.get("change_percent", 0):.2f}%',
                           'weight': 'moderate'})
        elif pred_7d.get('direction') == 'down':
            score -= 10
            signals.append({'factor': 'ML Prediction (7D)', 'signal': 'Bearish',
                           'detail': f'Weekly trend predicted down {abs(pred_7d.get("change_percent", 0)):.2f}%',
                           'weight': 'moderate'})

        # --- Bollinger Bands ---
        bb_pct = ind.get('bb_pct', 0.5) if isinstance(ind.get('bb_pct'), (int, float)) else 0.5
        if bb_pct < 0.2:
            signals.append({'factor': 'Bollinger Bands', 'signal': 'Bullish',
                           'detail': 'Price near lower band (potential bounce)',
                           'weight': 'moderate'})
            score += 10
        elif bb_pct > 0.8:
            signals.append({'factor': 'Bollinger Bands', 'signal': 'Bearish',
                           'detail': 'Price near upper band (potential pullback)',
                           'weight': 'moderate'})
            score -= 10

        # --- Sentiment ---
        if sentiment_score > 0.2:
            signals.append({'factor': 'News Sentiment', 'signal': 'Bullish',
                           'detail': f'Positive news sentiment ({sentiment_score:.2f})',
                           'weight': 'moderate'})
            score += 10
        elif sentiment_score < -0.2:
            signals.append({'factor': 'News Sentiment', 'signal': 'Bearish',
                           'detail': f'Negative news sentiment ({sentiment_score:.2f})',
                           'weight': 'moderate'})
            score -= 10
        else:
            signals.append({'factor': 'News Sentiment', 'signal': 'Neutral',
                           'detail': f'Mixed/neutral news sentiment ({sentiment_score:.2f})',
                           'weight': 'weak'})

        # --- Stochastic ---
        stoch_k = ind.get('stochastic_k', 50)
        if stoch_k < 20:
            score += 8
            signals.append({'factor': 'Stochastic', 'signal': 'Bullish',
                           'detail': f'Stochastic K at {stoch_k:.1f} (oversold)',
                           'weight': 'moderate'})
        elif stoch_k > 80:
            score -= 8
            signals.append({'factor': 'Stochastic', 'signal': 'Bearish',
                           'detail': f'Stochastic K at {stoch_k:.1f} (overbought)',
                           'weight': 'moderate'})

        # --- Generate final recommendation ---
        score = max(-100, min(100, score))

        if score >= 30:
            recommendation = 'Buy'
        elif score <= -30:
            recommendation = 'Sell'
        else:
            recommendation = 'Hold'

        confidence = min(abs(score) / 100, 0.95)

        # Generate human-readable reason
        bullish_factors = [s['factor'] for s in signals if 'Bullish' in s['signal']]
        bearish_factors = [s['factor'] for s in signals if 'Bearish' in s['signal']]

        if recommendation == 'Buy':
            reason = f"Bullish signals from {', '.join(bullish_factors[:3])}. Score: {score}/100"
        elif recommendation == 'Sell':
            reason = f"Bearish signals from {', '.join(bearish_factors[:3])}. Score: {score}/100"
        else:
            reason = f"Mixed signals. Bullish: {len(bullish_factors)}, Bearish: {len(bearish_factors)}. Score: {score}/100"

        return {
            'recommendation': recommendation,
            'score': score,
            'confidence': round(confidence, 4),
            'reason': reason,
            'signals': signals,
            'bullish_count': len(bullish_factors),
            'bearish_count': len(bearish_factors),
        }
