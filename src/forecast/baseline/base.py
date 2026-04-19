import warnings
from typing import List, Dict, Callable, Tuple
import sys
import os
import time
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import base forecasting models from utils.stat
from utils.stat import ForecastingEngine, AlgoParamExtractor

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import STATISTICAL_MODELS_CONFIG
logger = logging.getLogger(__name__)


class BaselineForecastingSystem:

    def __init__(self, sales_fact, level_columns=None, date_column=None, target_column=None, frequency=None):
        self.sales_fact = sales_fact        

        if level_columns is None:
            self.level_columns = STATISTICAL_MODELS_CONFIG["level_columns"]
        else:
            self.level_columns = level_columns
            
        if date_column is None:
            self.date_column = STATISTICAL_MODELS_CONFIG["date_column"]
        else:
            self.date_column = date_column
            
        if target_column is None:
            self.target_column = STATISTICAL_MODELS_CONFIG["target_column"]
        else:
            self.target_column = target_column
            
        if frequency is None:
            self.frequency = STATISTICAL_MODELS_CONFIG["frequency"]
        else:
            self.frequency = frequency
        
        # Get validation cycles from config
        self.validation_cycles = STATISTICAL_MODELS_CONFIG["validation_cycles"]
        self.confidence_interval_alpha = STATISTICAL_MODELS_CONFIG["confidence_interval_alpha"]
        self.minimum_data_points = STATISTICAL_MODELS_CONFIG["minimum_data_points"]
        self.baseline_models = STATISTICAL_MODELS_CONFIG.get("baseline_models", ["Moving Average", "Weighted Snaive"])
        self.algorithm_mapping = STATISTICAL_MODELS_CONFIG["algorithm_mapping"]
        
        self._prepare_data()
        
    def _prepare_data(self):
        if self.date_column in self.sales_fact.columns:
            self.sales_fact[self.date_column] = pd.to_datetime(self.sales_fact[self.date_column])
        
        # Generate unique level combinations from sales_fact
        self._generate_level_combinations()
    
    def _generate_level_combinations(self):
        """Generate unique combinations of level columns from sales_fact"""
        combinations_df = self.sales_fact[self.level_columns].drop_duplicates().reset_index(drop=True)
        
        self.valid_combinations = []
        
        for idx, row in combinations_df.iterrows():
            intersection_values = {col: row[col] for col in self.level_columns}
            
            # Check if this combination has enough data
            filter_conditions = []
            for col_name, col_value in intersection_values.items():
                filter_conditions.append(self.sales_fact[col_name] == col_value)
            
            if len(filter_conditions) == 1:
                data = self.sales_fact[filter_conditions[0]].copy()
            else:
                combined_filter = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    combined_filter = combined_filter & condition
                data = self.sales_fact[combined_filter].copy()
            
            validation_cycle = self._get_validation_cycle(len(data))
            if validation_cycle > 0:
                self.valid_combinations.append(intersection_values)

    def _get_validation_cycle(self, data_length: int) -> int:
        """Pick the largest validation window that still leaves enough training history."""
        default_cycle = self.validation_cycles[self.frequency]
        if data_length <= self.minimum_data_points:
            return 0
        return max(1, min(default_cycle, data_length - self.minimum_data_points))
        
    def _prepare_combination_data(self, intersection_values):

        filter_conditions = []
        for col_name, col_value in intersection_values.items():
            filter_conditions.append(self.sales_fact[col_name] == col_value)
        
        # Apply all filter conditions
        if len(filter_conditions) == 1:
            data = self.sales_fact[filter_conditions[0]].copy()
        else:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            data = self.sales_fact[combined_filter].copy()
        
        validation_cycle = self._get_validation_cycle(len(data))
        
        if validation_cycle <= 0:
            return None, None, None
            
        data = data.sort_values(self.date_column)
        
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.set_index(self.date_column)
        
        split_point = len(data) - validation_cycle
        train_data = data.iloc[:split_point]
        validation_data = data.iloc[split_point:]
        
        train_series = train_data[self.target_column]
        
        if self.frequency == 'daily':
            seasonal_periods = 365
        elif self.frequency == 'weekly':
            seasonal_periods = 52
        elif self.frequency == 'monthly':
            seasonal_periods = 12
        elif self.frequency == 'quarterly':
            seasonal_periods = 4
        else:
            seasonal_periods = 12
        
        return train_series, validation_data, seasonal_periods
    
    def _run_forecast(self, intersection_values):
        train_series, validation_data, seasonal_periods = self._prepare_combination_data(intersection_values)
        
        if train_series is None:
            return None
        
        # Get actual values from validation period
        actual_values = validation_data[self.target_column].values
        
        # Create base result structure with validation dates
        base_result = {
            'date': validation_data.index.tolist(),  # Use validation dates, not future dates
            'actual': actual_values.tolist()
        }
        
        # Add level column values dynamically
        for col_name, col_value in intersection_values.items():
            base_result[col_name] = [col_value] * len(actual_values)
        
        # Run forecasts for both baseline models
        for algo_name in self.baseline_models:
            try:
                # Create forecasting engine for this algorithm
                engine = ForecastingEngine(
                    train=train_series,
                    seasonal_periods=seasonal_periods,
                    forecast_horizon=len(validation_data),
                    confidence_interval_alpha=self.confidence_interval_alpha
                )
                
                method_name = self.algorithm_mapping[algo_name]
                method = getattr(engine, method_name)
                
                forecast_result = method()
                
                if forecast_result is not None and 'forecast' in forecast_result:
                    forecast_values = forecast_result['forecast']
                    
                    if len(forecast_values) >= len(actual_values):
                        forecast_values = forecast_values[:len(actual_values)]
                    else:
                        padded_forecast = np.full(len(actual_values), np.nan)
                        padded_forecast[:len(forecast_values)] = forecast_values
                        forecast_values = padded_forecast
                    
                    if hasattr(forecast_values, 'tolist'):
                        base_result[algo_name] = forecast_values.tolist()
                    else:
                        base_result[algo_name] = list(forecast_values)
                else:
                    base_result[algo_name] = [np.nan] * len(actual_values)
                    
            except Exception:
                base_result[algo_name] = [np.nan] * len(actual_values)
        
        result_df = pd.DataFrame(base_result)
        
        return result_df
    
    def run_forecasting(self):
        start_time = datetime.now()
        total_combinations = len(self.valid_combinations)
        logger.info(f"Starting baseline forecasting for {total_combinations} unique combinations")
        
        all_results = []
        
        for idx, intersection_values in enumerate(self.valid_combinations):
            try:
                result_df = self._run_forecast(intersection_values)
                if result_df is not None:
                    all_results.append(result_df)
            except Exception:
                continue
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            logger.info(f"Baseline forecasting completed: {len(final_results)} records in {total_time}")
            logger.info(f"Models used: {', '.join(self.baseline_models)}")
            return final_results
        else:
            return None
    

