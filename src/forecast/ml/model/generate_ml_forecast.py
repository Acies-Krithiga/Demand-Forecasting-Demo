import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from utils.ml import MLForecastingEngine

# Add the project root to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import ML_MODELS_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLForecastGenerator:
    
    def __init__(self, feature_df: pd.DataFrame, 
                 level_columns=None, date_column=None, target_column=None, frequency=None):
        if level_columns is None:
            self.level_columns = ML_MODELS_CONFIG["level_columns"]
        else:
            self.level_columns = level_columns
            
        if date_column is None:
            self.date_column = ML_MODELS_CONFIG["date_column"]
        else:
            self.date_column = date_column
            
        if target_column is None:
            self.target_column = ML_MODELS_CONFIG["target_column"]
        else:
            self.target_column = target_column
            
        if frequency is None:
            self.frequency = ML_MODELS_CONFIG["frequency"]
        else:
            self.frequency = frequency
        
        # Get validation cycles and model parameters from config
        self.validation_cycles = ML_MODELS_CONFIG["validation_cycles"]
        self.validation_horizon = self.validation_cycles[self.frequency]
        self.model_params = ML_MODELS_CONFIG["model_params"]
        self.available_algorithms = ML_MODELS_CONFIG["available_algorithms"]
        self.minimum_data_points = ML_MODELS_CONFIG["minimum_data_points"]
        self.random_state = ML_MODELS_CONFIG["random_state"]
        
        # Load data
        self._load_data(feature_df)
    
    def _load_data(self, feature_df: pd.DataFrame):
        """Load the feature data."""
        
        # Use provided DataFrame
        self.feature_df = feature_df.copy()
        
        # Convert date column to datetime
        if self.date_column in self.feature_df.columns:
            self.feature_df[self.date_column] = pd.to_datetime(self.feature_df[self.date_column])
        
        # Get unique intersections using configurable level columns
        self.intersections = self.feature_df[self.level_columns].drop_duplicates()
        
        # Get feature columns (exclude target, date, and level columns)
        exclude_cols = [self.target_column, self.date_column] + self.level_columns
        self.feature_columns = [col for col in self.feature_df.columns if col not in exclude_cols]
        
        logger.info(f"Loaded data: {len(self.intersections)} intersections, {len(self.feature_columns)} features")
    
    def _prepare_intersection_data(self, intersection_values):
        filter_conditions = []
        for col_name, col_value in intersection_values.items():
            filter_conditions.append(self.feature_df[col_name] == col_value)
        
        # Apply all filter conditions
        if len(filter_conditions) == 1:
            data = self.feature_df[filter_conditions[0]].copy()
        else:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            data = self.feature_df[combined_filter].copy()
        
        if len(data) < self.validation_horizon + self.minimum_data_points:
            return None, None, None
        
        # Sort by date
        data = data.sort_values(self.date_column)
        data = data.set_index(self.date_column)
        
        # Split into train and validation - LAST validation_horizon days as validation
        split_point = len(data) - self.validation_horizon
        train_data = data.iloc[:split_point]  # Everything before the last validation_horizon days
        validation_data = data.iloc[split_point:]  # Last validation_horizon days
        
        # Separate validation features and targets
        validation_features = validation_data[self.feature_columns]
        validation_targets = validation_data[self.target_column]
        
        return train_data, validation_features, validation_targets
    
    def _generate_forecast_for_intersection(self, intersection_values: dict):

        # Prepare data
        train_data, validation_features, validation_targets = self._prepare_intersection_data(intersection_values)
        
        if train_data is None:
            return None
        
        try:
            # Create ML forecasting engine
            engine = MLForecastingEngine(
                train_data=train_data,
                validation_features=validation_features,
                validation_targets=validation_targets,
                feature_columns=self.feature_columns,
                target_column=self.target_column,
                level_columns=self.level_columns,
                date_column=self.date_column,
                random_state=self.random_state
            )
            
            # Train all models
            model_results = engine.train_all_models(self.model_params)
            
            # Get actual values from validation period ONLY
            actual_values = validation_targets.values
            
            # Create result structure with ONLY validation period dates
            base_result = {
                'date': validation_features.index.tolist(),  # Only validation dates
                'actual': actual_values.tolist()  # Only validation actuals
            }
            
            # Add level column values dynamically
            for col_name, col_value in intersection_values.items():
                base_result[col_name] = [col_value] * len(actual_values)
            
            # Add predictions from each model (only validation period)
            for algo_name in self.available_algorithms:
                if algo_name in model_results and model_results[algo_name] is not None:
                    predictions = model_results[algo_name]['val_predictions']
                    base_result[algo_name] = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)
                else:
                    base_result[algo_name] = [np.nan] * len(actual_values)
            
            return base_result
                
        except Exception as e:
            logger.error(f"Forecast failed for {intersection_values}: {str(e)}")
            return None
    
    def generate_all_forecasts(self):
        logger.info("Starting ML forecast generation...")
        start_time = datetime.now()
        
        all_forecasts = []
        successful_forecasts = 0
        failed_forecasts = 0
        
        for idx, row in self.intersections.iterrows():
            intersection_values = {col: row[col] for col in self.level_columns}
            intersection_str = "-".join([f"{k}:{v}" for k, v in intersection_values.items()])
            
            # Generate forecast
            forecast_result = self._generate_forecast_for_intersection(intersection_values)
            
            if forecast_result:
                all_forecasts.append(pd.DataFrame(forecast_result))
                successful_forecasts += 1
            else:
                failed_forecasts += 1
            
            # Progress update
            if (idx + 1) % 50 == 0 or idx == len(self.intersections) - 1:
                elapsed_time = datetime.now() - start_time
                progress_percent = ((idx + 1) / len(self.intersections)) * 100
                logger.info(f"Progress: {progress_percent:.1f}% ({idx + 1}/{len(self.intersections)})")
        
        # Combine all forecasts
        if all_forecasts:
            final_forecasts = pd.concat(all_forecasts, ignore_index=True)
            end_time = datetime.now()
            total_time = end_time - start_time
            
            logger.info(f"ML Forecast completed: {len(final_forecasts)} records, {successful_forecasts} successful, {failed_forecasts} failed in {total_time}")
            
            return final_forecasts
        else:
            logger.error("No forecasts generated")
            return None

