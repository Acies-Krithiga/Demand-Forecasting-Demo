import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

# Example Config (replace with your actual ML_MODELS_CONFIG import)
from config.config import ML_MODELS_CONFIG

logger = logging.getLogger(__name__)

class UpliftDemandForecaster:
    def __init__(self, feature_df: pd.DataFrame):
        self.feature_df = feature_df.copy()
        self.level_columns = ML_MODELS_CONFIG["level_columns"]
        self.target_column = ML_MODELS_CONFIG["target_column"]
        self.date_column = ML_MODELS_CONFIG["date_column"]
        self.model_params = ML_MODELS_CONFIG["model_params"]

        # Feature columns: exclude levels, date, and target
        self.feature_columns = [
            col for col in self.feature_df.columns
            if col not in self.level_columns + [self.target_column, self.date_column]
        ]

        logger.info(f"Initialized UpliftDemandForecaster with {len(self.feature_columns)} features")
        logger.info(f"Feature columns: {self.feature_columns}")

    # ================================================================
    # Step 1: Prepare Data
    # ================================================================
    def prepare_data(self, intersection_values: dict) -> Tuple[pd.DataFrame, pd.Series]:
        filter_cond = np.ones(len(self.feature_df), dtype=bool)
        for col, val in intersection_values.items():
            filter_cond &= (self.feature_df[col] == val)

        subset = self.feature_df[filter_cond].copy()
        if subset.empty:
            logger.warning(f"No data found for intersection: {intersection_values}")
            return None, None

        X = subset[self.feature_columns]
        y = subset[self.target_column]
        return X, y

    # ================================================================
    # Step 2: Train Base Model
    # ================================================================
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> object:
        from sklearn.ensemble import RandomForestRegressor
        model_params = self.model_params["random_forest"].copy()
        model = RandomForestRegressor(**model_params)
        model.fit(X, y)
        logger.info("Random Forest model trained successfully.")
        return model

    # ================================================================
    # Step 3: Compute Uplift Effects
    # ================================================================
    def compute_uplift_effects(self, X: pd.DataFrame, model: object) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        X_with_all = X.copy()
        y_pred_with_all = model.predict(X_with_all)
        factor_effects = {}

        for factor in self.feature_columns:
            X_without = X_with_all.copy()
            X_without[factor] = 0
            y_pred_without = model.predict(X_without)

            # Compute uplift ratio (with safeguard)
            # effect_ratio = np.divide(y_pred_with_all, np.maximum(y_pred_without, 1e-6))
            effect_ratio=((y_pred_with_all-y_pred_without) /np.maximum(y_pred_without, 1e-6))*100

            factor_effects[factor] = effect_ratio

            logger.info(f"Computed uplift effect for {factor}")

        return y_pred_with_all, factor_effects

    # ================================================================
    # Step 4: Forecast per Intersection
    # ================================================================
    def run_forecast_for_intersection(self, intersection_values: dict) -> Dict[str, float]:
        X, y = self.prepare_data(intersection_values)
        if X is None or y is None:
            return {}

        model = self.train_model(X, y)
        _, factor_effects = self.compute_uplift_effects(X, model)

        result = { **intersection_values        }

        # Add mean uplift per factor
        for factor, ratios in factor_effects.items():
            result[f"{factor}_uplift"] = np.mean(ratios).round(2)

        logger.info(f"Uplift forecast completed for {intersection_values}: {result}")
        return result
    
    # ================================================================
    # Step 5: Run for All Intersections
    # ================================================================
    def run_all_forecasts(self) -> pd.DataFrame:
        unique_intersections = self.feature_df[self.level_columns].drop_duplicates()
        all_results = []

        for _, row in unique_intersections.iterrows():
            intersection_values = {col: row[col] for col in self.level_columns}
            forecast_result = self.run_forecast_for_intersection(intersection_values)
            if forecast_result:
                all_results.append(forecast_result)

        logger.info("All uplift forecasts completed successfully.")
        return pd.DataFrame(all_results)
