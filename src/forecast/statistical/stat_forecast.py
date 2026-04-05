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


class DemandForecastingSystem:

    def __init__(self, rules_df, sales_fact, level_columns=None, date_column=None, target_column=None, frequency=None):
        self.rules_df = rules_df
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
        
        # Get validation cycles and algorithm mapping from config
        self.validation_cycles = STATISTICAL_MODELS_CONFIG["validation_cycles"]
        self.algorithm_mapping = STATISTICAL_MODELS_CONFIG["algorithm_mapping"]
        self.confidence_interval_alpha = STATISTICAL_MODELS_CONFIG["confidence_interval_alpha"]
        self.minimum_data_points = STATISTICAL_MODELS_CONFIG["minimum_data_points"]
        
        self._prepare_data()
        
    def _prepare_data(self):
        logger.info("Statistical forecasting system initializing...")
        
        if self.date_column in self.sales_fact.columns:
            self.sales_fact[self.date_column] = pd.to_datetime(self.sales_fact[self.date_column])
        
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
        
        validation_cycle = self.validation_cycles[self.frequency]
        
        if len(data) < validation_cycle + self.minimum_data_points:
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
    
    def _run_forecast(self, intersection_values, algorithms):
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
        
        for algo_idx, algo in enumerate(algorithms):
            algo_name = algo.strip()
            
            if algo_name in self.algorithm_mapping:
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
                        
                except Exception as e:
                    base_result[algo_name] = [np.nan] * len(actual_values)
            else:
                base_result[algo_name] = [np.nan] * len(actual_values)
        
        result_df = pd.DataFrame(base_result)
        
        return result_df
    
    def run_forecasting(self):
        logger.info("Statistical forecasting running...")
        start_time = datetime.now()
        total_combinations = len(self.rules_df)
        
        all_results = []
        successful_combinations = 0
        failed_combinations = 0
        
        for idx, row in self.rules_df.iterrows():
            try:
                # Create intersection values dictionary dynamically
                intersection_values = {col: row[col] for col in self.level_columns}
                algorithms_str = row['Algorithms_Used']
                
                algorithms = [algo.strip() for algo in algorithms_str.split(',')]
                
                result_df = self._run_forecast(intersection_values, algorithms)
                
                if result_df is not None:
                    all_results.append(result_df)
                    successful_combinations += 1
                else:
                    failed_combinations += 1
                
                # Progress update every 10 combinations
                if (idx + 1) % 10 == 0 or idx == total_combinations - 1:
                    progress_percent = ((idx + 1) / total_combinations) * 100
                    logger.info(f"Progress: {progress_percent:.1f}% ({idx + 1}/{total_combinations})")
                    
            except Exception as e:
                failed_combinations += 1
                continue
        
        end_time = datetime.now()
        total_time = end_time - start_time
        
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            logger.info(f"Statistical forecasting finished - {len(final_results)} records generated in {total_time}")
            return final_results
        else:
            logger.warning("No results generated")
            return None
    