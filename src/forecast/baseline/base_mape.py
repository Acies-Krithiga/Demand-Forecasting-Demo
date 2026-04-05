import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import STATISTICAL_MODELS_CONFIG

logger = logging.getLogger(__name__)


class BaselineMAPE:
    
    def __init__(self, df, level_columns=None, actual_column='actual', ma_column='Moving Average', sn_column='Weighted Snaive'):
        self.df = df
        self.actual_column = actual_column
        self.ma_column = ma_column
        self.sn_column = sn_column
        
        if level_columns is None:
            self.level_columns = STATISTICAL_MODELS_CONFIG["level_columns"]
        else:
            self.level_columns = level_columns
        
        # Calculate MAPE immediately
        self._calculate_mape()
    
    def calculate_mape(self, actual, forecast):
        """Calculate MAPE, handling zero and very small values"""
        actual = np.array(actual)
        forecast = np.array(forecast)
        
        # Filter out cases where actual is zero or very small (< 0.01)
        mask = np.abs(actual) >= 0.01
        actual_filtered = actual[mask]
        forecast_filtered = forecast[mask]
        
        if len(actual_filtered) == 0:
            return np.nan
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual_filtered - forecast_filtered) / actual_filtered)) * 100
        return mape
    
    def _calculate_mape(self):
        """Calculate MAPE for each level for both models"""
        logger.info("Calculating MAPE for each level...")
        
        results = []
        
        # Group by level columns
        grouped = self.df.groupby(self.level_columns)
        
        for name, group in grouped:
            # Convert tuple to dictionary if multiple level columns
            if isinstance(name, tuple):
                level_dict = {col: val for col, val in zip(self.level_columns, name)}
            else:
                level_dict = {self.level_columns[0]: name}
            
            # Get actual and forecast values
            actual = group[self.actual_column].values
            ma_forecast = group[self.ma_column].values
            sn_forecast = group[self.sn_column].values
            
            # Calculate MAPE for both models
            ma_mape = self.calculate_mape(actual, ma_forecast)
            sn_mape = self.calculate_mape(actual, sn_forecast)
            
            # Create result row
            result_row = level_dict.copy()
            result_row['ma_mape'] = ma_mape
            result_row['sn_mape'] = sn_mape
            
            results.append(result_row)
        
        self.mape_df = pd.DataFrame(results)
        
        logger.info(f"MAPE calculation completed for {len(self.mape_df)} levels")
    
    def run(self):
        """Return MAPE results"""
        return self.mape_df
