import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.config import DATA_PREP_CONFIG
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    def __init__(self, config: Dict = None):
        # Use imported config as default if no config is passed
        if config is None:
            config = DATA_PREP_CONFIG
            
        self.config = config
        self.level_columns = config.get('level_columns', ['store_id', 'item_id'])
        self.date_column = config.get('date_column', 'date')
        self.target_column = config.get('target_column', 'units_sold')
        
        # Join configurations
        self.join_config = config.get('join_config', {
            'price': {'left_on': self.level_columns, 'right_on': self.level_columns},
            'calendar': {'left_on': self.date_column, 'right_on': self.date_column},
            'location': {'left_on': 'store_id', 'right_on': 'store_id'},
            'external': {'left_on': [self.date_column, 'region'], 'right_on': [self.date_column, 'region']},
            'promotions': {'left_on': [self.date_column] + self.level_columns, 
                         'right_on': [self.date_column] + self.level_columns}
        })
        
        pass
    

    
    def aggregate_sales(self, sales_df: pd.DataFrame) -> pd.DataFrame:
        agg_sales = sales_df.groupby(self.level_columns + [self.date_column])[self.target_column].sum().reset_index()
        return agg_sales
    
    def join_price_data(self, main_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:              
        join_keys = self.join_config['price']
        
        # Check for duplicate keys in price_df that could cause explosion
        price_duplicates = price_df.duplicated(subset=join_keys['right_on']).sum()
        if price_duplicates > 0:
            price_df = price_df.drop_duplicates(subset=join_keys['right_on'])
        
        result_df = main_df.merge(
            price_df, 
            left_on=join_keys['left_on'], 
            right_on=join_keys['right_on'], 
            how='left',
            suffixes=('', '_price')  # Add suffix to right table columns
        )
        
        return result_df
    
    def join_calendar_data(self, main_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        join_keys = self.join_config['calendar']
        result_df = main_df.merge(
            calendar_df,
            left_on=join_keys['left_on'],
            right_on=join_keys['right_on'],
            how='left',
            suffixes=('', '_calendar')  # Add suffix to right table columns
        )
        
        return result_df
    
    def join_location_data(self, main_df: pd.DataFrame, location_df: pd.DataFrame) -> pd.DataFrame:        
        join_keys = self.join_config['location']
        result_df = main_df.merge(
            location_df,
            left_on=join_keys['left_on'],
            right_on=join_keys['right_on'],
            how='left',
            suffixes=('', '_location')  # Add suffix to right table columns
        )
        
        return result_df
    
    def join_external_data(self, main_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:        
        join_keys = self.join_config['external']
        result_df = main_df.merge(
            external_df,
            left_on=join_keys['left_on'],
            right_on=join_keys['right_on'],
            how='left',
            suffixes=('', '_external')  # Add suffix to right table columns
        )
        
        return result_df
    
    def join_promotion_data(self, main_df: pd.DataFrame, promotion_df: pd.DataFrame) -> pd.DataFrame:
        join_keys = self.join_config['promotions']
        result_df = main_df.merge(
            promotion_df,
            left_on=join_keys['left_on'],
            right_on=join_keys['right_on'],
            how='left',
            suffixes=('', '_promotion')  # Add suffix to right table columns
        )
        
        return result_df
    
    def _standardize_date_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Standardize date column to datetime format for consistent joining."""
        if date_col in df.columns:
            # Convert to datetime, handling multiple formats
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Check for any NaT values (failed conversions)
            nat_count = df[date_col].isna().sum()
            if nat_count > 0:
                logger.warning(f"Found {nat_count} rows with invalid dates in {date_col} column")
        return df
    
    def prepare_master_dataset(self, 
                              sales_df: pd.DataFrame,
                              price_df: Optional[pd.DataFrame] = None,
                              calendar_df: Optional[pd.DataFrame] = None,
                              location_df: Optional[pd.DataFrame] = None,
                              external_df: Optional[pd.DataFrame] = None,
                              promotion_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        logger.info("Starting master dataset preparation...")
        
        # Step 1: Start with sales data (left table) and track initial row count
        main_df = sales_df.copy()
        initial_rows = len(main_df)
        logger.info(f"Initial sales data rows: {initial_rows}")
        
        # Standardize date column in sales data
        main_df = self._standardize_date_column(main_df, self.date_column)
        
        # Step 2: Join promotion data (sales ← promotion on date, store_id, item_id)
        if promotion_df is not None:
            promotion_df = promotion_df.copy()
            promotion_df = self._standardize_date_column(promotion_df, self.date_column)
            rows_before = len(main_df)
            main_df = self.join_promotion_data(main_df, promotion_df)
            logger.info(f"After promotion join: {len(main_df)} rows (expected: {rows_before})")
            del promotion_df
        
        # Step 3: Join calendar data (sales ← calendar on date)
        if calendar_df is not None:
            calendar_df = calendar_df.copy()
            calendar_df = self._standardize_date_column(calendar_df, self.date_column)
            rows_before = len(main_df)
            main_df = self.join_calendar_data(main_df, calendar_df)
            logger.info(f"After calendar join: {len(main_df)} rows (expected: {rows_before})")
            del calendar_df
        
        # Step 4: Join price data (sales ← price on store_id, item_id, wm_yr_wk)
        if price_df is not None:
            price_df = price_df.copy()
            rows_before = len(main_df)
            main_df = self.join_price_data(main_df, price_df)
            logger.info(f"After price join: {len(main_df)} rows (expected: {rows_before})")
            del price_df
        
        # Step 5: Join location data (sales ← location on store_id)
        if location_df is not None:
            location_df = location_df.copy()
            rows_before = len(main_df)
            main_df = self.join_location_data(main_df, location_df)
            logger.info(f"After location join: {len(main_df)} rows (expected: {rows_before})")
            del location_df
        
        # Step 6: Join external data (sales ← external on date, region)
        if external_df is not None:
            external_df = external_df.copy()
            external_df = self._standardize_date_column(external_df, self.date_column)
            rows_before = len(main_df)
            main_df = self.join_external_data(main_df, external_df)
            logger.info(f"After external join: {len(main_df)} rows (expected: {rows_before})")
            del external_df
        
        # Verify no data loss (left joins should preserve all rows)
        if len(main_df) != initial_rows:
            logger.warning(f"Data loss detected! Initial rows: {initial_rows}, Final rows: {len(main_df)}, Lost: {initial_rows - len(main_df)}")
        else:
            logger.info(f"All {initial_rows} rows preserved successfully")
        
        logger.info("Master dataset preparation completed.")

        # Fill missing promotion columns if they exist
        if 'promotion_flag' in main_df.columns:
            main_df['promotion_flag'] = main_df['promotion_flag'].fillna(0)
        if 'promotion_type' in main_df.columns:
            main_df['promotion_type'] = main_df['promotion_type'].fillna('No Promotion')
        if 'discount_percent' in main_df.columns:
            main_df['discount_percent'] = main_df['discount_percent'].fillna(0)
        if 'ad_spend' in main_df.columns:
            main_df['ad_spend'] = main_df['ad_spend'].fillna(0)
        if 'campaign_id' in main_df.columns:
            main_df['campaign_id'] = main_df['campaign_id'].fillna('No Event')
        
        # Convert date back to string format for consistency (YYYY-MM-DD)
        if self.date_column in main_df.columns:
            # Only convert valid dates, keep NaT as empty string or original value
            main_df[self.date_column] = main_df[self.date_column].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else ''
            )
            # Get date range for logging (filter out empty strings)
            valid_dates = main_df[self.date_column][main_df[self.date_column] != '']
            if len(valid_dates) > 0:
                logger.info(f"Final dataset: {len(main_df)} rows, date range: {valid_dates.min()} to {valid_dates.max()}")
            else:
                logger.warning("No valid dates found in final dataset!")
        
        return main_df
