"""Candlestick pattern detection service."""
import pandas as pd
import numpy as np


class CandlestickService:
    """Detect common candlestick patterns from OHLCV data."""

    @staticmethod
    def detect_patterns(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
        """
        Detect candlestick patterns in the last N candles.

        Args:
            df: DataFrame with OHLC columns
            lookback: Number of recent candles to check

        Returns:
            List of detected patterns with date and significance
        """
        df = df.tail(lookback).copy()
        df = df.reset_index()

        if 'Date' not in df.columns and 'Datetime' not in df.columns:
            df['Date'] = df.index

        patterns = []

        for i in range(2, len(df)):
            date_val = df.iloc[i].get('Date', df.iloc[i].name)
            date_str = str(date_val)[:10] if date_val is not None else ''

            o, h, l, c = (float(df.iloc[i]['Open']), float(df.iloc[i]['High']),
                          float(df.iloc[i]['Low']), float(df.iloc[i]['Close']))
            body = abs(c - o)
            candle_range = h - l
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l

            if candle_range == 0:
                continue

            body_ratio = body / candle_range

            # Previous candle
            po, ph, pl, pc = (float(df.iloc[i-1]['Open']), float(df.iloc[i-1]['High']),
                              float(df.iloc[i-1]['Low']), float(df.iloc[i-1]['Close']))
            p_body = abs(pc - po)

            # Two candles ago
            ppo, pph, ppl, ppc = (float(df.iloc[i-2]['Open']), float(df.iloc[i-2]['High']),
                                   float(df.iloc[i-2]['Low']), float(df.iloc[i-2]['Close']))

            # --- Doji ---
            if body_ratio < 0.1:
                patterns.append({
                    'pattern': 'Doji',
                    'date': date_str,
                    'type': 'neutral',
                    'significance': 'high',
                    'description': 'Indecision candle — open and close nearly equal. Potential reversal signal.',
                    'price': c,
                })

            # --- Hammer ---
            if (body_ratio < 0.35 and
                    lower_shadow > body * 2 and
                    upper_shadow < body * 0.5 and
                    c >= o):  # Bullish
                patterns.append({
                    'pattern': 'Hammer',
                    'date': date_str,
                    'type': 'bullish',
                    'significance': 'high',
                    'description': 'Bullish reversal pattern. Long lower shadow shows buying pressure.',
                    'price': c,
                })

            # --- Shooting Star ---
            if (body_ratio < 0.35 and
                    upper_shadow > body * 2 and
                    lower_shadow < body * 0.5 and
                    c <= o):  # Bearish
                patterns.append({
                    'pattern': 'Shooting Star',
                    'date': date_str,
                    'type': 'bearish',
                    'significance': 'high',
                    'description': 'Bearish reversal pattern. Long upper shadow shows selling pressure.',
                    'price': c,
                })

            # --- Bullish Engulfing ---
            if (pc < po and  # Previous was bearish
                    c > o and  # Current is bullish
                    o <= pc and c >= po and  # Current engulfs previous
                    body > p_body):
                patterns.append({
                    'pattern': 'Bullish Engulfing',
                    'date': date_str,
                    'type': 'bullish',
                    'significance': 'very_high',
                    'description': 'Strong bullish reversal. Current candle completely engulfs the previous bearish candle.',
                    'price': c,
                })

            # --- Bearish Engulfing ---
            if (pc > po and  # Previous was bullish
                    c < o and  # Current is bearish
                    o >= pc and c <= po and  # Current engulfs previous
                    body > p_body):
                patterns.append({
                    'pattern': 'Bearish Engulfing',
                    'date': date_str,
                    'type': 'bearish',
                    'significance': 'very_high',
                    'description': 'Strong bearish reversal. Current candle completely engulfs the previous bullish candle.',
                    'price': c,
                })

            # --- Morning Star (3-candle pattern) ---
            if (ppc > ppo and  # First: bearish
                    abs(pc - po) < (pph - ppl) * 0.3 and  # Second: small body (star)
                    c > o and  # Third: bullish
                    c > (ppo + ppc) / 2):  # Third closes above midpoint of first
                patterns.append({
                    'pattern': 'Morning Star',
                    'date': date_str,
                    'type': 'bullish',
                    'significance': 'very_high',
                    'description': 'Three-candle bullish reversal pattern. Strong buy signal.',
                    'price': c,
                })

            # --- Evening Star (3-candle pattern) ---
            if (ppc < ppo and  # First: bullish
                    abs(pc - po) < (pph - ppl) * 0.3 and  # Second: small body (star)
                    c < o and  # Third: bearish
                    c < (ppo + ppc) / 2):  # Third closes below midpoint of first
                patterns.append({
                    'pattern': 'Evening Star',
                    'date': date_str,
                    'type': 'bearish',
                    'significance': 'very_high',
                    'description': 'Three-candle bearish reversal pattern. Strong sell signal.',
                    'price': c,
                })

            # --- Bullish Harami ---
            if (pc < po and  # Previous was bearish
                    c > o and  # Current is bullish
                    o > pc and c < po and  # Current body inside previous
                    body < p_body * 0.6):
                patterns.append({
                    'pattern': 'Bullish Harami',
                    'date': date_str,
                    'type': 'bullish',
                    'significance': 'moderate',
                    'description': 'Bullish reversal. Small bullish candle contained within previous bearish candle.',
                    'price': c,
                })

            # --- Bearish Harami ---
            if (pc > po and  # Previous was bullish
                    c < o and  # Current is bearish
                    o < pc and c > po and  # Current body inside previous
                    body < p_body * 0.6):
                patterns.append({
                    'pattern': 'Bearish Harami',
                    'date': date_str,
                    'type': 'bearish',
                    'significance': 'moderate',
                    'description': 'Bearish reversal. Small bearish candle contained within previous bullish candle.',
                    'price': c,
                })

        # Return unique patterns sorted by date
        return patterns

    @staticmethod
    def get_pattern_summary(patterns: list[dict]) -> dict:
        """Summarize detected patterns."""
        bullish = [p for p in patterns if p['type'] == 'bullish']
        bearish = [p for p in patterns if p['type'] == 'bearish']
        neutral = [p for p in patterns if p['type'] == 'neutral']

        return {
            'total': len(patterns),
            'bullish_count': len(bullish),
            'bearish_count': len(bearish),
            'neutral_count': len(neutral),
            'bias': 'Bullish' if len(bullish) > len(bearish) else
                    'Bearish' if len(bearish) > len(bullish) else 'Neutral',
            'patterns': patterns,
        }
