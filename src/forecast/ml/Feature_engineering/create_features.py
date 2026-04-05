import pandas as pd
import numpy as np
import logging
from config.config import FEATURE_CREATION_CONFIG

logger = logging.getLogger(__name__)


class CreateFeatures:
    def __init__(self, df: pd.DataFrame, level_columns=None, date_column=None, target_column=None):
        self.df = df.copy()

        if level_columns is None:
            self.level_columns = FEATURE_CREATION_CONFIG["level_columns"]
        else:
            self.level_columns = level_columns

        if date_column is None:
            self.date_column = FEATURE_CREATION_CONFIG["date_column"]
        else:
            self.date_column = date_column

        if target_column is None:
            self.target_column = FEATURE_CREATION_CONFIG["target_column"]
        else:
            self.target_column = target_column

        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
        
        # Get config parameters with defaults
        self.num_lags = FEATURE_CREATION_CONFIG.get("num_lags", 0)
        self.rolling_mean_windows = FEATURE_CREATION_CONFIG.get("rolling_mean_windows", [])
        self.rolling_std_windows = FEATURE_CREATION_CONFIG.get("rolling_std_windows", [])
        self.rolling_min_windows = FEATURE_CREATION_CONFIG.get("rolling_min_windows", [])
        self.rolling_max_windows = FEATURE_CREATION_CONFIG.get("rolling_max_windows", [])
        self.date_features = FEATURE_CREATION_CONFIG.get("date_features", {})
        self.cyclical_encoding = FEATURE_CREATION_CONFIG.get("cyclical_encoding", False)

    def create_lag_features(self):
        """Create lag features for the target variable"""
        if self.num_lags <= 0:
            logger.info("Skipping lag features: num_lags = 0")
            return self.df
        
        logger.info(f"Creating {self.num_lags} lag features...")
        
        # Sort dataframe by level columns and date
        df = self.df.sort_values(by=self.level_columns + [self.date_column])
        
        # Create lag features for each group
        for lag in range(1, self.num_lags + 1):
            self.df[f'{self.target_column}_lag_{lag}'] = df.groupby(self.level_columns)[self.target_column].shift(lag)
        
        logger.info(f"Created {self.num_lags} lag features")
        return self.df

    def create_rolling_mean_features(self):
        """Create rolling mean features for the target variable"""
        if not self.rolling_mean_windows:
            logger.info("Skipping rolling mean features: no windows specified")
            return self.df
        
        logger.info(f"Creating rolling mean features for windows: {self.rolling_mean_windows}")
        
        df = self.df.sort_values(by=self.level_columns + [self.date_column])
        
        for window in self.rolling_mean_windows:
            self.df[f'{self.target_column}_rolling_mean_{window}'] = (
                df.groupby(self.level_columns)[self.target_column].shift(1).rolling(window, min_periods=1).mean().values
            )
        
        logger.info(f"Created {len(self.rolling_mean_windows)} rolling mean features")
        return self.df

    def create_rolling_std_features(self):
        """Create rolling standard deviation features for the target variable"""
        if not self.rolling_std_windows:
            logger.info("Skipping rolling std features: no windows specified")
            return self.df
        
        logger.info(f"Creating rolling std features for windows: {self.rolling_std_windows}")
        
        df = self.df.sort_values(by=self.level_columns + [self.date_column])
        
        for window in self.rolling_std_windows:
            self.df[f'{self.target_column}_rolling_std_{window}'] = (
                df.groupby(self.level_columns)[self.target_column].shift(1).rolling(window, min_periods=1).std().values
            )
        
        logger.info(f"Created {len(self.rolling_std_windows)} rolling std features")
        return self.df

    def create_rolling_min_features(self):
        """Create rolling minimum features for the target variable"""
        if not self.rolling_min_windows:
            logger.info("Skipping rolling min features: no windows specified")
            return self.df
        
        logger.info(f"Creating rolling min features for windows: {self.rolling_min_windows}")
        
        df = self.df.sort_values(by=self.level_columns + [self.date_column])
        
        for window in self.rolling_min_windows:
            self.df[f'{self.target_column}_rolling_min_{window}'] = (
                df.groupby(self.level_columns)[self.target_column].shift(1).rolling(window, min_periods=1).min().values
            )
        
        logger.info(f"Created {len(self.rolling_min_windows)} rolling min features")
        return self.df

    def create_rolling_max_features(self):
        """Create rolling maximum features for the target variable"""
        if not self.rolling_max_windows:
            logger.info("Skipping rolling max features: no windows specified")
            return self.df
        
        logger.info(f"Creating rolling max features for windows: {self.rolling_max_windows}")
        
        df = self.df.sort_values(by=self.level_columns + [self.date_column])
        
        for window in self.rolling_max_windows:
            self.df[f'{self.target_column}_rolling_max_{window}'] = (
                df.groupby(self.level_columns)[self.target_column].shift(1).rolling(window, min_periods=1).max().values
            )
        
        logger.info(f"Created {len(self.rolling_max_windows)} rolling max features")
        return self.df

    def create_date_features(self):
        """Create date-based features"""
        logger.info("Creating date features...")
        
        df = self.df.copy()
        date_col = self.df[self.date_column]
        
        # Extract date components
        if self.date_features.get("day_of_week", False):
            if self.cyclical_encoding:
                self.df['day_of_week_sin'] = np.sin(2 * np.pi * date_col.dt.dayofweek / 7)
                self.df['day_of_week_cos'] = np.cos(2 * np.pi * date_col.dt.dayofweek / 7)
            else:
                self.df['day_of_week'] = date_col.dt.dayofweek
        
        if self.date_features.get("month", False):
            if self.cyclical_encoding:
                self.df['month_sin'] = np.sin(2 * np.pi * date_col.dt.month / 12)
                self.df['month_cos'] = np.cos(2 * np.pi * date_col.dt.month / 12)
            else:
                self.df['month'] = date_col.dt.month
        
        if self.date_features.get("quarter", False):
            if self.cyclical_encoding:
                self.df['quarter_sin'] = np.sin(2 * np.pi * date_col.dt.quarter / 4)
                self.df['quarter_cos'] = np.cos(2 * np.pi * date_col.dt.quarter / 4)
            else:
                self.df['quarter'] = date_col.dt.quarter
        
        if self.date_features.get("day_of_month", False):
            if self.cyclical_encoding:
                self.df['day_of_month_sin'] = np.sin(2 * np.pi * date_col.dt.day / 31)
                self.df['day_of_month_cos'] = np.cos(2 * np.pi * date_col.dt.day / 31)
            else:
                self.df['day_of_month'] = date_col.dt.day
        
        if self.date_features.get("week_of_year", False):
            week = date_col.dt.isocalendar().week
            if self.cyclical_encoding:
                self.df['week_of_year_sin'] = np.sin(2 * np.pi * week / 52)
                self.df['week_of_year_cos'] = np.cos(2 * np.pi * week / 52)
            else:
                self.df['week_of_year'] = week
        
        if self.date_features.get("is_weekend", False):
            self.df['is_weekend'] = (date_col.dt.dayofweek >= 5).astype(int)
        
        if self.date_features.get("is_month_start", False):
            self.df['is_month_start'] = date_col.dt.is_month_start.astype(int)
        
        if self.date_features.get("is_month_end", False):
            self.df['is_month_end'] = date_col.dt.is_month_end.astype(int)
        
        if self.date_features.get("is_quarter_start", False):
            self.df['is_quarter_start'] = date_col.dt.is_quarter_start.astype(int)
        
        if self.date_features.get("is_quarter_end", False):
            self.df['is_quarter_end'] = date_col.dt.is_quarter_end.astype(int)
        
        logger.info("Date features created successfully")
        return self.df

    def create_all_features(self):
        """Create all features based on configuration"""
        logger.info("Starting feature creation pipeline...")
        
        # Create all features in sequence
        self.create_lag_features()
        self.create_rolling_mean_features()
        self.create_rolling_std_features()
        self.create_rolling_min_features()
        self.create_rolling_max_features()
        self.create_date_features()
        
        logger.info("Feature creation pipeline completed successfully")
        return self.df

    def get_features_df(self):
        """Return the dataframe with all features"""
        return self.df
