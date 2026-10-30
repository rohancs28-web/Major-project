import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class MultimodalXGBoostFusion:
    """
    Multimodal Late-Fusion model using XGBoost.
    Combines CNN spatial vectors, LSTM temporal vectors, and static agronomic variables.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=5):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective='reg:squarederror',
            random_state=42
        )

    def fit(self, spatial_features, temporal_features, static_features, y_target):
        """
        Concatenates multimodal features and trains the XGBoost regressor.
        """
        X_combined = np.hstack((spatial_features, temporal_features, static_features))
        self.model.fit(X_combined, y_target)

    def predict(self, spatial_features, temporal_features, static_features):
        """
        Generates final crop yield predictions (Tons/Hectare).
        """
        X_combined = np.hstack((spatial_features, temporal_features, static_features))
        return self.model.predict(X_combined)

    def evaluate(self, spatial_features, temporal_features, static_features, y_true):
        """
        Evaluates model performance using Root Mean Square Error (RMSE).
        """
        predictions = self.predict(spatial_features, temporal_features, static_features)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        return rmse