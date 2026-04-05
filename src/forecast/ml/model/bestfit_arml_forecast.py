import pandas as pd
import numpy as np
import sys
import os
import logging
from typing import Dict, List, Tuple
import joblib
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import ML_MODELS_CONFIG

logger = logging.getLogger(__name__)


class BestFitARMLForecaster:
    """
    Class to generate predictions using the best fit ML models for each intersection.
    """
    
    def __init__(self, best_fit_df: pd.DataFrame, feature_df: pd.DataFrame):
        """
        Initialize the forecaster with best fit models and feature data.
        
        Args:
            best_fit_df (pd.DataFrame): DataFrame with best fit algorithms for each intersection
            feature_df (pd.DataFrame): Feature data for training and prediction
        """
        self.best_fit_df = best_fit_df.copy()
        self.feature_df = feature_df.copy()
        self.level_columns = ML_MODELS_CONFIG["level_columns"]
        self.target_column = ML_MODELS_CONFIG["target_column"]
        self.date_column = ML_MODELS_CONFIG["date_column"]
        self.available_algorithms = ML_MODELS_CONFIG["available_algorithms"]
        self.model_params = ML_MODELS_CONFIG["model_params"]
        
        # Convert date column to datetime if it exists
        if self.date_column in self.feature_df.columns:
            self.feature_df[self.date_column] = pd.to_datetime(self.feature_df[self.date_column])
        
        # Get feature columns (exclude level columns, date, and target)
        self.feature_columns = [col for col in self.feature_df.columns 
                               if col not in self.level_columns + [self.date_column, self.target_column]]
        
        logger.info(f"Initialized BestFitMLForecaster with {len(self.feature_columns)} features")
        logger.info(f"Feature columns: {self.feature_columns}")
        
        # Model storage
        self.trained_models = {}
        
    def prepare_training_data(self, intersection_values: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data for a specific intersection, excluding the last cycle.
        
        Args:
            intersection_values (dict): Dictionary with level column names as keys and values
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training features and targets
        """
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
        
        # Remove the last cycle (365 days for daily data) for training
        validation_cycles = ML_MODELS_CONFIG["validation_cycles"]["daily"]  # Assuming daily frequency
        if len(intersection_data) > validation_cycles:
            training_data = intersection_data.iloc[:-validation_cycles].copy()
        else:
            training_data = intersection_data.copy()
        
        if training_data.empty:
            return None, None
        
        # Prepare features and targets
        X = training_data[self.feature_columns].copy()
        y = training_data[self.target_column].copy()

        df = X.copy()
        df[self.target_column] = y

        # Generate lag features from 1 to 6
        for lag in range(1, 7):
            df[f'{self.target_column}_lag_{lag}'] = df[self.target_column].shift(lag)

        df = df.iloc[6:].copy()

        X_lagged = df.drop(columns=[self.target_column])
        y_lagged = df[self.target_column]

        # Handle missing values
        X_lagged = X_lagged.fillna(X_lagged.mean())
        y_lagged = y_lagged.fillna(y_lagged.mean())

        return X_lagged, y_lagged
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, algorithm: str) -> object:
        """
        Train a model for the given algorithm.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Training targets
            algorithm (str): Algorithm name
        
        Returns:
            object: Trained model
        """
        try:
            if algorithm == 'lightgbm':
                import lightgbm as lgb
                model_params = self.model_params["lightgbm"].copy()
                model = lgb.LGBMRegressor(**model_params)
                model.fit(X, y)
                
            elif algorithm == 'xgboost':
                import xgboost as xgb
                model_params = self.model_params["xgboost"].copy()
                model = xgb.XGBRegressor(**model_params)
                model.fit(X, y)
                
            elif algorithm == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model_params = self.model_params["random_forest"].copy()
                model = RandomForestRegressor(**model_params)
                model.fit(X, y)
                
            else:
                logger.error(f"Unknown algorithm: {algorithm}")
                return None
            
            logger.info(f"Successfully trained {algorithm} model")
            return model
            
        except Exception as e:
            logger.error(f"Error training {algorithm} model: {str(e)}")
            return None
    
    def generate_predictions_for_intersection(self, intersection_values: dict, algorithm: str) -> List[float]:
        """
        Generate predictions for a specific intersection using its best fit algorithm on last cycle data.
        
        Args:
            intersection_values (dict): Dictionary with level column names as keys and values
            algorithm (str): Best fit algorithm for this intersection
            
        Returns:
            List[float]: List of predictions for the last cycle
        """
        # Prepare training data (excluding last cycle)
        X_train, y_train = self.prepare_training_data(intersection_values)
        
        if X_train is None or y_train is None:
            logger.warning(f"No training data available for intersection: {intersection_values}")
            return []
        
        # Train model
        model = self.train_model(X_train, y_train, algorithm)
        
        if model is None:
            logger.warning(f"Failed to train model for intersection: {intersection_values}")
            return []
        
        # Get prediction data (last cycle data)
        validation_cycles = ML_MODELS_CONFIG["validation_cycles"]["daily"]
        
        # Build filter conditions for prediction data
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
            return []
        
        # Sort by date
        if self.date_column in intersection_data.columns:
            intersection_data = intersection_data.sort_values(self.date_column)
        
        # Get the last cycle data for prediction
        if len(intersection_data) > validation_cycles:
            prediction_data = intersection_data.iloc[-validation_cycles:].copy()
        else:
            prediction_data = intersection_data.copy()
        
        # Prepare features for prediction
        X_pred = prediction_data[self.feature_columns].copy()
        X_pred = X_pred.fillna(X_pred.mean())

        try:
            # Predict directly on engineered feature columns.
            # Lag features are already included in self.feature_columns when created upstream.
            predictions = model.predict(X_pred)
            predictions = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
            logger.info(f"Generated {len(predictions)} predictions for intersection: {intersection_values}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions for intersection {intersection_values}: {str(e)}")
            return []
    
    def generate_all_predictions(self) -> pd.DataFrame:
        """
        Generate predictions for all intersections using their best fit algorithms on last cycle data.
        
        Returns:
            pd.DataFrame: DataFrame with predictions for all intersections
        """
        logger.info("Starting prediction generation for all intersections...")
        results = []
        
        for _, row in self.best_fit_df.iterrows():
            # Create intersection values dictionary dynamically
            intersection_values = {col: row[col] for col in self.level_columns}
            best_algorithm = row['best_fit_algorithm']
            
            logger.info(f"Processing intersection: {intersection_values} with algorithm: {best_algorithm}")
            
            # Generate predictions for this intersection
            predictions = self.generate_predictions_for_intersection(intersection_values, best_algorithm)
            
            if not predictions:
                logger.warning(f"No predictions generated for intersection: {intersection_values}")
                continue
            
            # Get prediction data for dates
            validation_cycles = ML_MODELS_CONFIG["validation_cycles"]["daily"]
            
            # Build filter conditions for prediction data
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
            
            # Sort by date
            if self.date_column in intersection_data.columns:
                intersection_data = intersection_data.sort_values(self.date_column)
            
            # Get the last cycle data for dates
            if len(intersection_data) > validation_cycles:
                prediction_data = intersection_data.iloc[-validation_cycles:].copy()
            else:
                prediction_data = intersection_data.copy()
            
            # Create result rows for each prediction
            for i, pred in enumerate(predictions):
                result_row = intersection_values.copy()
                result_row.update({
                    'date': prediction_data.iloc[i][self.date_column] if self.date_column in prediction_data.columns else None,
                    'prediction': pred,
                    'algorithm': best_algorithm
                })
                results.append(result_row)
        
        logger.info("Prediction generation completed")
        return pd.DataFrame(results)

