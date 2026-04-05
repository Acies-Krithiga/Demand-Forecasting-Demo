import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
import os
import logging

# Add the project root to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import STATISTICAL_MODELS_CONFIG

logger = logging.getLogger(__name__)


class BestFitAnalyzer:
    """
    Simple class to analyze forecast results and determine the best fit algorithm
    for each store-item intersection based on MAPE calculation.
    """
    
    def __init__(self, data_df: pd.DataFrame, level_columns=None, actual_column=None):
        """
        Initialize the BestFitAnalyzer with data dataframe.
        
        Args:
            data_df (pd.DataFrame): DataFrame containing actual values and forecast values
            level_columns (list): Columns that define the intersection level
            actual_column (str): Name of the column containing actual values
        """
        self.data_df = data_df.copy()
        
        # Set configuration from config if not provided
        if level_columns is None:
            self.level_columns = STATISTICAL_MODELS_CONFIG["level_columns"]
        else:
            self.level_columns = level_columns
            
        if actual_column is None:
            self.actual_column = 'actual'
        else:
            self.actual_column = actual_column
        
        # Identify algorithm columns (exclude level columns, actual, and common non-numeric columns)
        exclude_columns = set(self.level_columns + [self.actual_column, 'date', 'timestamp', 'time'])
        self.algorithm_columns = [col for col in self.data_df.columns 
                                 if col not in exclude_columns and self.data_df[col].notna().any()]
        
        # Get unique intersections
        self.intersections = self.data_df[self.level_columns].drop_duplicates()
    
    def calculate_mape(self, actual: pd.Series, forecast: pd.Series) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE)."""
        # Convert to numeric and remove NaN values
        actual_numeric = pd.to_numeric(actual, errors='coerce')
        forecast_numeric = pd.to_numeric(forecast, errors='coerce')
        
        mask = ~(actual_numeric.isna() | forecast_numeric.isna())
        actual_clean = actual_numeric[mask]
        forecast_clean = forecast_numeric[mask]
        
        if len(actual_clean) == 0:
            return np.nan
            
        # Avoid division by zero
        epsilon = 1e-8
        mape = np.mean(np.abs((actual_clean - forecast_clean) / (actual_clean + epsilon))) * 100
        return mape
    
    def find_best_fit_for_intersection(self, intersection_values: dict) -> Tuple[str, float]:
        """
        Find the best fit algorithm for a specific intersection.
        
        Args:
            intersection_values (dict): Dictionary with level column names as keys and values
        
        Returns:
            Tuple[str, float]: Best fit algorithm name and its MAPE value
        """
        # Build dynamic filter conditions
        filter_conditions = []
        for col_name, col_value in intersection_values.items():
            filter_conditions.append(self.data_df[col_name] == col_value)
        
        # Apply all filter conditions
        if len(filter_conditions) == 1:
            intersection_data = self.data_df[filter_conditions[0]]
        else:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            intersection_data = self.data_df[combined_filter]
        
        if intersection_data.empty:
            return None, np.nan
        
        actual_values = intersection_data[self.actual_column]
        mape_results = {}
        
        # Calculate MAPE for each algorithm
        for algorithm in self.algorithm_columns:
            forecast_values = intersection_data[algorithm]
            mape = self.calculate_mape(actual_values, forecast_values)
            mape_results[algorithm] = mape
        
        # Find best algorithm (lowest MAPE)
        valid_mape = {k: v for k, v in mape_results.items() if not np.isnan(v)}
        
        if not valid_mape:
            return None, np.nan
            
        best_algorithm = min(valid_mape, key=valid_mape.get)
        best_mape = valid_mape[best_algorithm]
        
        return best_algorithm, best_mape
    
    def analyze_all_intersections(self) -> pd.DataFrame:
        """
        Analyze all intersections and return results with only the 4 required columns.
        
        Returns:
            pd.DataFrame: DataFrame with store_id, item_id, best_fit_algorithm, best_mape
        """
        logger.info("BestFit analysis running...")
        results = []
        
        for _, row in self.intersections.iterrows():
            # Create intersection values dictionary dynamically
            intersection_values = {col: row[col] for col in self.level_columns}
            
            best_algorithm, best_mape = self.find_best_fit_for_intersection(intersection_values)
            
            # Create result row dynamically
            result_row = intersection_values.copy()
            result_row.update({
                'best_fit_algorithm': best_algorithm,
                'best_mape': best_mape
            })
            results.append(result_row)
        
        logger.info("BestFit analysis finished")
        return pd.DataFrame(results)

