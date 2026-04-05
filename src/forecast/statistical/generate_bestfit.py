import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from utils.stat import ForecastingEngine

# Add the project root to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import STATISTICAL_MODELS_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ForecastGenerator:
    """
    Generate forecasts using the best fit algorithms for each store-item intersection.
    """
    
    def __init__(self, best_fit_df: pd.DataFrame, sales_fact_df: pd.DataFrame, 
                 level_columns=None, date_column=None, target_column=None, frequency=None):
        """
        Initialize the ForecastGenerator.
        
        Args:
            best_fit_df (pd.DataFrame): DataFrame containing best fit results
            sales_fact_df (pd.DataFrame): DataFrame containing sales fact data
            level_columns (list): Columns that define the intersection level
            date_column (str): Date column name
            target_column (str): Target variable column name
            frequency (str): Data frequency ('daily', 'weekly', 'monthly')
        """
        # Set configuration from config if not provided
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
        
        # Get forecast cycles and algorithm mapping from config
        self.forecast_cycles = STATISTICAL_MODELS_CONFIG["forecast_cycles"]
        self.forecast_horizon = self.forecast_cycles[self.frequency]
        self.algorithm_mapping = STATISTICAL_MODELS_CONFIG["algorithm_mapping"]
        self.confidence_interval_alpha = STATISTICAL_MODELS_CONFIG["confidence_interval_alpha"]
        self.minimum_data_points = STATISTICAL_MODELS_CONFIG["minimum_data_points"]
        
        # Load data
        self._load_data(best_fit_df, sales_fact_df)
    
    def _load_data(self, best_fit_df: pd.DataFrame, sales_fact_df: pd.DataFrame):
        """Load the best fit results and sales fact data."""
        
        # Use provided DataFrames
        self.best_fit_df = best_fit_df.copy()
        
        # Process sales fact data
        self.sales_fact_df = sales_fact_df.copy()
        if self.date_column in self.sales_fact_df.columns:
            self.sales_fact_df[self.date_column] = pd.to_datetime(self.sales_fact_df[self.date_column])
        
        # Get unique intersections using configurable level columns
        self.intersections = self.best_fit_df[self.level_columns].drop_duplicates()
    
    def _prepare_intersection_data(self, intersection_values):
        """
        Prepare data for a specific intersection.
        
        Args:
            intersection_values (dict): Dictionary with level column names as keys and values
            
        Returns:
            tuple: (train_series, seasonal_periods, last_date) or (None, None, None) if insufficient data
        """
        # Build dynamic filter conditions
        filter_conditions = []
        for col_name, col_value in intersection_values.items():
            filter_conditions.append(self.sales_fact_df[col_name] == col_value)
        
        # Apply all filter conditions
        if len(filter_conditions) == 1:
            data = self.sales_fact_df[filter_conditions[0]].copy()
        else:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            data = self.sales_fact_df[combined_filter].copy()
        
        if len(data) < self.minimum_data_points:  # Minimum data requirement
            return None, None, None
        
        # Sort by date
        data = data.sort_values(self.date_column)
        data = data.set_index(self.date_column)
        
        # Get training series (use all available data)
        train_series = data[self.target_column]
        
        # Get the last date from the training data for this intersection
        last_date = train_series.index[-1]
        
        # Set seasonal periods based on frequency
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
        
        return train_series, seasonal_periods, last_date
    
    def _generate_forecast_for_intersection(self, intersection_values: dict, best_algorithm: str):
        """
        Generate forecast for a specific intersection using the best algorithm.
        
        Args:
            intersection_values (dict): Dictionary with level column names as keys and values
            best_algorithm (str): Best fit algorithm name
            
        Returns:
            dict: Forecast results or None if failed
        """
        # Prepare data
        train_series, seasonal_periods, last_date = self._prepare_intersection_data(intersection_values)
        
        if train_series is None:
            return None
        
        # Check if algorithm is supported
        if best_algorithm not in self.algorithm_mapping:
            return None
        
        try:
            # Create forecasting engine
            engine = ForecastingEngine(
                train=train_series,
                seasonal_periods=seasonal_periods,
                forecast_horizon=self.forecast_horizon,
                confidence_interval_alpha=self.confidence_interval_alpha
            )
            
            # Get the appropriate method
            method_name = self.algorithm_mapping[best_algorithm]
            method = getattr(engine, method_name)
            
            # Generate forecast
            forecast_result = method()
            
            if forecast_result and 'forecast' in forecast_result:
                forecast_values = forecast_result['forecast']
                
                # Create future dates starting from the day after the last date
                # last_date is already the last date from the training data for this intersection
                
                # Determine frequency string for pandas
                if self.frequency == 'daily':
                    freq_str = 'D'
                    start_date = last_date + timedelta(days=1)
                elif self.frequency == 'weekly':
                    freq_str = 'W'
                    start_date = last_date + timedelta(weeks=1)
                elif self.frequency == 'monthly':
                    freq_str = 'M'
                    start_date = last_date + timedelta(days=30)  # Approximate month
                else:
                    freq_str = 'D'
                    start_date = last_date + timedelta(days=1)
                
                future_dates = pd.date_range(
                    start=start_date,
                    periods=self.forecast_horizon,
                    freq=freq_str
                )
                
                # Log the date range for debugging
                intersection_str = "-".join([f"{k}:{v}" for k, v in intersection_values.items()])
                
                # Prepare results dynamically
                result = {}
                for col_name, col_value in intersection_values.items():
                    result[col_name] = [col_value] * self.forecast_horizon
                
                result.update({
                    'date': future_dates.tolist(),
                    'algorithm': [best_algorithm] * self.forecast_horizon,
                    'forecast': forecast_values.tolist() if hasattr(forecast_values, 'tolist') else list(forecast_values)
                })
                
                return result
            else:
                return None
                
        except Exception as e:
            return None
    
    def generate_all_forecasts(self):
        """
        Generate forecasts for all store-item intersections.
        
        Returns:
            pd.DataFrame: DataFrame containing all forecasts
        """
        logger.info("Forecast generation running...")
        start_time = datetime.now()
        
        all_forecasts = []
        successful_forecasts = 0
        failed_forecasts = 0
        
        for idx, row in self.intersections.iterrows():
            # Create intersection values dictionary dynamically
            intersection_values = {col: row[col] for col in self.level_columns}
            intersection_str = "-".join([f"{k}:{v}" for k, v in intersection_values.items()])
            
            # Build dynamic filter conditions for best fit lookup
            filter_conditions = []
            for col_name, col_value in intersection_values.items():
                filter_conditions.append(self.best_fit_df[col_name] == col_value)
            
            # Apply all filter conditions
            if len(filter_conditions) == 1:
                best_fit_row = self.best_fit_df[filter_conditions[0]]
            else:
                combined_filter = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    combined_filter = combined_filter & condition
                best_fit_row = self.best_fit_df[combined_filter]
            
            if best_fit_row.empty:
                failed_forecasts += 1
                continue
            
            best_algorithm = best_fit_row['best_fit_algorithm'].iloc[0]
            
            # Generate forecast
            forecast_result = self._generate_forecast_for_intersection(intersection_values, best_algorithm)
            
            if forecast_result:
                all_forecasts.append(pd.DataFrame(forecast_result))
                successful_forecasts += 1
            else:
                failed_forecasts += 1
            
            # Progress update
            if (idx + 1) % 10 == 0 or idx == len(self.intersections) - 1:
                elapsed_time = datetime.now() - start_time
                progress_percent = ((idx + 1) / len(self.intersections)) * 100
                logger.info(f"Progress: {progress_percent:.1f}% ({idx + 1}/{len(self.intersections)})")
        
        # Combine all forecasts
        if all_forecasts:
            final_forecasts = pd.concat(all_forecasts, ignore_index=True)
            end_time = datetime.now()
            total_time = end_time - start_time
            
            logger.info(f"Forecast generation finished - {len(final_forecasts)} records generated in {total_time}")
            
            return final_forecasts
        else:
            logger.error("No forecasts were generated")
            return None
