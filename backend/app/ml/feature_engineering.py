"""Feature engineering for stock prediction using technical indicators."""
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Generate features from OHLCV data for ML models."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all technical indicator features from OHLCV data."""
        df = df.copy()

        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # --- Moving Averages ---
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)

        # --- MACD ---
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # --- RSI ---
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

        # --- ADX ---
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['ADX'] = adx.adx()
        df['ADX_Pos'] = adx.adx_pos()
        df['ADX_Neg'] = adx.adx_neg()

        # --- ATR ---
        df['ATR'] = ta.volatility.average_true_range(
            df['High'], df['Low'], df['Close'], window=14
        )

        # --- Bollinger Bands ---
        bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Middle'] = bb.bollinger_mavg()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Pct'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # --- VWAP (Volume Weighted Average Price) ---
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

        # --- OBV (On Balance Volume) ---
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

        # --- CCI (Commodity Channel Index) ---
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)

        # --- Momentum ---
        df['Momentum'] = ta.momentum.roc(df['Close'], window=10)

        # --- ROC (Rate of Change) ---
        df['ROC'] = ta.momentum.roc(df['Close'], window=12)

        # --- Williams %R ---
        df['Williams_R'] = ta.momentum.williams_r(
            df['High'], df['Low'], df['Close'], lbp=14
        )

        # --- Stochastic Oscillator ---
        stoch = ta.momentum.StochasticOscillator(
            df['High'], df['Low'], df['Close'], window=14, smooth_window=3
        )
        df['Stochastic_K'] = stoch.stoch()
        df['Stochastic_D'] = stoch.stoch_signal()

        # --- Lag Features ---
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)

        # --- Rolling Statistics ---
        for window in [5, 10, 20]:
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
            df[f'Rolling_Vol_Mean_{window}'] = df['Volume'].rolling(window=window).mean()

        # --- Price-based features ---
        df['Daily_Return'] = df['Close'].pct_change()
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Spread'] = (df['Close'] - df['Open']) / df['Open']

        # Drop rows with NaN values (from indicator calculations)
        df = df.dropna()

        return df

    def prepare_features(self, df: pd.DataFrame,
                         target_horizon: int = 1) -> tuple:
        """
        Prepare features and target for model training.

        Args:
            df: DataFrame with generated features
            target_horizon: Number of days ahead to predict (1, 7, 30)

        Returns:
            Tuple of (X_scaled, y, feature_names, scaler)
        """
        df = df.copy()

        # Create target: future close price
        df['Target'] = df['Close'].shift(-target_horizon)
        df = df.dropna()

        # Define feature columns (exclude OHLCV and target)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'Target', 'Dividends', 'Stock Splits']
        self.feature_names = [c for c in df.columns if c not in exclude_cols]

        X = df[self.feature_names].values
        y = df['Target'].values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, self.feature_names, self.scaler

    def get_latest_features(self, df: pd.DataFrame) -> np.ndarray:
        """Get the most recent feature vector for prediction."""
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'Target', 'Dividends', 'Stock Splits']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        latest = df[feature_cols].iloc[-1:].values
        return self.scaler.transform(latest)
