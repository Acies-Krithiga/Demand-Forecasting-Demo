import pandas as pd
import numpy as np
import sys
import os
import logging
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import ML_MODELS_CONFIG
from utils.ml import MLForecastingEngine

logger = logging.getLogger(__name__)


class BestFitMLForecaster:

    
    def __init__(self, best_fit_df: pd.DataFrame, feature_df: pd.DataFrame, future_features_df: pd.DataFrame):
        self.best_fit_df = best_fit_df.copy()
        self.feature_df = feature_df.copy()
        self.future_features_df = future_features_df.copy()
        self.level_columns = ML_MODELS_CONFIG["level_columns"]
        self.target_column = ML_MODELS_CONFIG["target_column"]
        self.date_column = ML_MODELS_CONFIG["date_column"]
        self.available_algorithms = ML_MODELS_CONFIG["available_algorithms"]
        self.model_params = ML_MODELS_CONFIG["model_params"]
        
        # Convert date column to datetime if it exists
        if self.date_column in self.feature_df.columns:
            self.feature_df[self.date_column] = pd.to_datetime(self.feature_df[self.date_column])
        
        if self.date_column in self.future_features_df.columns:
            self.future_features_df[self.date_column] = pd.to_datetime(self.future_features_df[self.date_column])
        
        # Get feature columns (exclude level columns, date, and target)
        self.feature_columns = [col for col in self.feature_df.columns 
                               if col not in self.level_columns + [self.date_column, self.target_column]]
        
        # Verify that future_features_df has the same feature columns
        missing_features = [col for col in self.feature_columns if col not in self.future_features_df.columns]
        if missing_features:
            logger.warning(f"Missing feature columns in future_features_df: {missing_features}")
        
        logger.info(f"Initialized BestFitMLForecaster with {len(self.feature_columns)} features")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        # Model storage
        self.trained_models = {}
        
    def prepare_training_data(self, intersection_values: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Build dynamic filter conditions
        filter_conditions = []
        for col_name, col_value in intersection_values.items():
            filter_conditions.append(self.feature_df[col_name] == col_value)
        
        # Apply all filter conditions
        if len(filter_conditions) == 1:
            intersection_data = self.feature_df[filter_conditions[0]]
        else:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            intersection_data = self.feature_df[combined_filter]
        
        if intersection_data.empty:
            return None, None
        
        # Sort by date to ensure proper time series order
        if self.date_column in intersection_data.columns:
            intersection_data = intersection_data.sort_values(self.date_column)
        
        # Use ALL available data for training (no exclusion of last cycle)
        training_data = intersection_data.copy()
        
        if training_data.empty:
            return None, None
        
        # Prepare features and targets
        X = training_data[self.feature_columns].copy()
        y = training_data[self.target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        return X, y
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, algorithm: str) -> object:
        try:
            # Create minimal dummy validation set (required by MLForecastingEngine constructor but not used)
            # We use the same data to avoid validation split - MLForecastingEngine will use X_train for training
            dummy_val_features = X_train.iloc[:1].copy()
            dummy_val_targets = y_train.iloc[:1].copy()
            
            # Create training dataframe with target
            train_data = X_train.copy()
            train_data[self.target_column] = y_train
            
            # Temporarily suppress logging from MLForecastingEngine's _prepare_data
            ml_logger = logging.getLogger('utils.ml')
            original_level = ml_logger.level
            ml_logger.setLevel(logging.WARNING)
            
            try:
                # Initialize MLForecastingEngine using models and parameters from utils/ml.py
                engine = MLForecastingEngine(
                    train_data=train_data,
                    validation_features=dummy_val_features,
                    validation_targets=dummy_val_targets,
                    feature_columns=list(X_train.columns),
                    target_column=self.target_column,
                    level_columns=self.level_columns,
                    date_column=self.date_column,
                    random_state=ML_MODELS_CONFIG.get("random_state", 42)
                )
            finally:
                ml_logger.setLevel(original_level)
            
            # Get model parameters from config
            model_params = self.model_params.get(algorithm, {})
            
            # Train the specific algorithm using MLForecastingEngine methods from utils/ml.py
            if algorithm == 'lightgbm':
                result = engine.train_lightgbm(params=model_params)
            elif algorithm == 'xgboost':
                result = engine.train_xgboost(params=model_params)
            elif algorithm == 'random_forest':
                result = engine.train_random_forest(params=model_params)
            else:
                logger.error(f"Unknown algorithm: {algorithm}")
                return None
            
            if result is None:
                return None
            
            model = result['model']
            logger.info(f"Successfully trained {algorithm} model on {len(X_train)} samples using MLForecastingEngine")
            return model
            
        except Exception as e:
            logger.error(f"Error training {algorithm} model: {str(e)}")
            return None
    
    def generate_predictions_for_intersection(self, intersection_values: dict, algorithm: str) -> Tuple[List[float], pd.DataFrame]:
        # Prepare training data using ALL available historical data
        X_train, y_train = self.prepare_training_data(intersection_values)
        
        if X_train is None or y_train is None:
            logger.warning(f"No training data available for intersection: {intersection_values}")
            return [], pd.DataFrame()
        
        # Train model
        model = self.train_model(X_train, y_train, algorithm)
        
        if model is None:
            logger.warning(f"Failed to train model for intersection: {intersection_values}")
            return [], pd.DataFrame()
        
        # Get future prediction data from future_features_df
        filter_conditions = []
        for col_name, col_value in intersection_values.items():
            filter_conditions.append(self.future_features_df[col_name] == col_value)
        
        # Apply all filter conditions
        if len(filter_conditions) == 1:
            future_data = self.future_features_df[filter_conditions[0]]
        else:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            future_data = self.future_features_df[combined_filter]
        
        if future_data.empty:
            logger.warning(f"No future features available for intersection: {intersection_values}")
            return [], pd.DataFrame()
        
        # Sort by date if date column exists
        if self.date_column in future_data.columns:
            future_data = future_data.sort_values(self.date_column)
        
        # Prepare features for prediction
        # Ensure all feature columns are present
        available_features = [col for col in self.feature_columns if col in future_data.columns]
        missing_features = [col for col in self.feature_columns if col not in future_data.columns]
        
        if missing_features:
            logger.warning(f"Missing features {missing_features} in future data for intersection {intersection_values}")
        
        if not available_features:
            logger.error(f"No feature columns available for prediction for intersection: {intersection_values}")
            return [], pd.DataFrame()
        
        X_pred = future_data[available_features].copy()
        
        # Fill missing values with mean from training data
        for col in X_pred.columns:
            if X_pred[col].isna().any():
                X_pred[col] = X_pred[col].fillna(X_train[col].mean() if col in X_train.columns else 0)
        
        # Generate predictions using model from MLForecastingEngine
        try:
            predictions = model.predict(X_pred)
            logger.info(f"Generated {len(predictions)} predictions for intersection: {intersection_values}")
            return predictions.tolist(), future_data.copy()
            
        except Exception as e:
            logger.error(f"Error generating predictions for intersection {intersection_values}: {str(e)}")
            return [], pd.DataFrame()
    
    def generate_all_predictions(self) -> pd.DataFrame:
        logger.info("Starting prediction generation for all intersections...")
        results = []
        
        for _, row in self.best_fit_df.iterrows():
            # Create intersection values dictionary dynamically
            intersection_values = {col: row[col] for col in self.level_columns}
            best_algorithm = row['best_fit_algorithm']
            
            logger.info(f"Processing intersection: {intersection_values} with algorithm: {best_algorithm}")
            
            # Generate predictions for this intersection using future features
            predictions, future_data = self.generate_predictions_for_intersection(intersection_values, best_algorithm)
            
            if not predictions or future_data.empty:
                logger.warning(f"No predictions generated for intersection: {intersection_values}")
                continue
            
            # Create result rows for each prediction
            for i, pred in enumerate(predictions):
                result_row = intersection_values.copy()
                result_row.update({
                    'date': future_data.iloc[i][self.date_column] if self.date_column in future_data.columns else None,
                    'prediction': pred,
                    'algorithm': best_algorithm
                })
                results.append(result_row)
        
        logger.info("Prediction generation completed")
        return pd.DataFrame(results)

