"""XGBoost prediction model for stock price forecasting."""
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class StockPredictor:
    """XGBoost-based stock price predictor."""

    def __init__(self):
        self.models: dict[str, XGBRegressor] = {}
        self.feature_importances: dict[str, dict] = {}
        self.metrics: dict[str, dict] = {}

    def _create_model(self) -> XGBRegressor:
        """Create an XGBoost model with tuned hyperparameters."""
        return XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror',
        )

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: list[str], horizon: str = '1d') -> dict:
        """
        Train XGBoost model for a specific prediction horizon.

        Args:
            X: Scaled feature matrix
            y: Target values
            feature_names: List of feature names
            horizon: Prediction horizon key ('1d', '7d', '30d')

        Returns:
            Dictionary of evaluation metrics
        """
        model = self._create_model()

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores, rmse_scores, r2_scores = [], [], []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            r2_scores.append(r2_score(y_val, y_pred))

        # Final training on all data
        model.fit(X, y, verbose=False)
        self.models[horizon] = model

        # Feature importance
        importances = model.feature_importances_
        importance_dict = {}
        for name, imp in zip(feature_names, importances):
            importance_dict[name] = round(float(imp), 6)

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        self.feature_importances[horizon] = importance_dict

        # Store metrics
        metrics = {
            'mae': round(float(np.mean(mae_scores)), 4),
            'rmse': round(float(np.mean(rmse_scores)), 4),
            'r2': round(float(np.mean(r2_scores)), 4),
            'mae_std': round(float(np.std(mae_scores)), 4),
            'rmse_std': round(float(np.std(rmse_scores)), 4),
            'r2_std': round(float(np.std(r2_scores)), 4),
        }
        self.metrics[horizon] = metrics

        return metrics

    def predict(self, X: np.ndarray, horizon: str = '1d') -> float:
        """Make a prediction for the given features."""
        if horizon not in self.models:
            raise ValueError(f"Model not trained for horizon: {horizon}")

        prediction = self.models[horizon].predict(X)
        return round(float(prediction[0]), 2)

    def get_confidence(self, horizon: str = '1d') -> float:
        """Calculate prediction confidence based on R² score."""
        if horizon not in self.metrics:
            return 0.0

        r2 = self.metrics[horizon]['r2']
        # Normalize R² to a 0-1 confidence score
        confidence = max(0.0, min(1.0, (r2 + 1) / 2))
        return round(confidence, 4)

    def get_top_features(self, horizon: str = '1d',
                         top_n: int = 10) -> list[dict]:
        """Get top N most important features."""
        if horizon not in self.feature_importances:
            return []

        features = self.feature_importances[horizon]
        top = list(features.items())[:top_n]
        return [{'feature': name, 'importance': imp} for name, imp in top]
