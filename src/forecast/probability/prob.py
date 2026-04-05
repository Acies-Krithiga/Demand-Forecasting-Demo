import pandas as pd 
import numpy as np
from scipy import stats
from multiprocessing import Pool, cpu_count
import time
import sys
import logging
import platform

logger = logging.getLogger(__name__)


def _calculate_residuals_fast_helper(y, dates):

    window = min(365, len(y) // 2)
    min_periods = min(30, len(y) // 4)
    trend = pd.Series(y).rolling(window=window, min_periods=min_periods, center=True).mean()
    trend = trend.bfill().ffill().values
    
    if len(dates) > 365:
        dates_pd = pd.Series(pd.to_datetime(dates))
        doy = dates_pd.dt.dayofyear.values - 1
        
        seasonal = np.zeros(len(y))
        for d in range(365):
            mask = doy == d
            if mask.sum() > 0:
                seasonal[mask] = np.mean((y - trend)[mask])
        
        residuals = y - trend - seasonal
    else:
        residuals = y - trend
    
    return residuals


def _process_single_combination_helper(args):
    store_id, item_id, hist_subset, future_subset, quantiles = args
    
    if len(hist_subset) < 30:
        return None
    
    # Get historical data
    y = hist_subset['actual'].astype(float).values
    hist_dates = hist_subset['date'].values
    
    # Calculate residuals using fast method
    residuals = _calculate_residuals_fast_helper(y, hist_dates)
    residuals = residuals[~np.isnan(residuals)]
    
    if len(residuals) < 10:
        return None
    
    # Calculate residual statistics
    resid_mean = np.mean(residuals)  # Bias correction
    resid_std = np.std(residuals)    # Overall uncertainty
    
    # Day-of-year heteroscedasticity (different variance by day of year)
    hist_dates_pd = pd.Series(pd.to_datetime(hist_dates))
    doy = hist_dates_pd.dt.dayofyear.values - 1
    
    doy_std = np.zeros(365)
    for d in range(365):
        mask = doy == d
        if mask.sum() > 1:
            doy_std[d] = np.std(residuals[mask])
        else:
            doy_std[d] = resid_std
    
    # Fill any missing days with overall std
    doy_std_mean = np.mean(doy_std[doy_std > 0])
    doy_std = np.where(doy_std == 0, doy_std_mean, doy_std)
    
    # Get future forecasts
    valid_mask = ~pd.isna(future_subset['point_forecast'])
    if not valid_mask.any():
        return None
    
    valid_future = future_subset[valid_mask].copy()
    point_forecasts = valid_future['point_forecast'].values
    future_dates = pd.Series(pd.to_datetime(valid_future['date']))
    future_doy = future_dates.dt.dayofyear.values - 1
    
    # Get day-specific standard deviations
    std_adjs = doy_std[future_doy]
    
    # Pre-calculate z-scores for all quantiles
    z_scores = np.array([stats.norm.ppf(q) for q in quantiles])
    
    # Vectorized quantile calculation
    # Formula: quantile = point_forecast + bias + (z_score * std)
    quantile_values = (point_forecasts[:, np.newaxis] + resid_mean + 
                      (z_scores[np.newaxis, :] * std_adjs[:, np.newaxis]))
    quantile_values = np.maximum(quantile_values, 0)  # Ensure non-negative
    
    return {
        'store_id': store_id,
        'item_id': item_id,
        'dates': valid_future['date'].values,
        'quantile_values': quantile_values,
        'quantile_cols': [f'q{int(q*100):02d}' for q in quantiles]
    }


class ProbabilityForecasting:
    def __init__(self, hist_df, future_df, quantiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], n_jobs=-1):
        self.hist_df = hist_df
        self.future_df = future_df
        self.quantiles = quantiles
        self.n_jobs = n_jobs
        
        # Prepare dataframes
        if 'units_sold' in self.hist_df.columns:
            self.hist_df = self.hist_df.rename(columns={'units_sold': 'actual'})
        if 'forecast' in self.future_df.columns:
            self.future_df = self.future_df.rename(columns={'forecast': 'point_forecast'})
        
        # Validate required columns
        required_cols = ['store_id', 'item_id', 'date']
        for col in required_cols:
            if col not in self.hist_df.columns:
                raise ValueError(f"hist_df must contain '{col}' column")
            if col not in self.future_df.columns:
                raise ValueError(f"future_df must contain '{col}' column")
        
        if 'actual' not in self.hist_df.columns:
            raise ValueError("hist_df must contain 'actual' or 'units_sold' column")
        if 'point_forecast' not in self.future_df.columns:
            raise ValueError("future_df must contain 'point_forecast' or 'forecast' column")
        
        self.hist_df['date'] = pd.to_datetime(self.hist_df['date'])
        self.future_df['date'] = pd.to_datetime(self.future_df['date'])
        self.hist_df = self.hist_df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
        self.future_df = self.future_df.sort_values(['store_id', 'item_id', 'date']).reset_index(drop=True)
        self.hist_df = self.hist_df.drop_duplicates(subset=['store_id', 'item_id', 'date'], keep='last')
        self.future_df = self.future_df.drop_duplicates(subset=['store_id', 'item_id', 'date'], keep='last')


    def calculate_residuals(self):
        if 'actual' not in self.hist_df.columns:
            if 'units_sold' in self.hist_df.columns:
                self.hist_df = self.hist_df.rename(columns={'units_sold': 'actual'})
            else:
                raise ValueError("hist_df must contain 'actual' or 'units_sold' column")

        self.hist_df['residuals'] = np.nan
        unique_combos = self.hist_df[['store_id', 'item_id']].drop_duplicates()
        
        for idx, (_, row) in enumerate(unique_combos.iterrows()):
            store_id = row['store_id']
            item_id = row['item_id']
            
            mask = (self.hist_df['store_id'] == store_id) & (self.hist_df['item_id'] == item_id)
            subset = self.hist_df[mask].copy()
            
            if len(subset) < 30:
                continue
            
            y = subset['actual'].astype(float).values
            dates = subset['date'].values
            residuals = _calculate_residuals_fast_helper(y, dates)
            self.hist_df.loc[mask, 'residuals'] = residuals
        return self.hist_df

    def generate_probability_forecasts(self):
        start_time = time.time()
        n_workers = cpu_count() if self.n_jobs == -1 else self.n_jobs
        logger.info(f"Starting probability forecasting: {len(self.quantiles)} quantiles, "
                   f"{'parallel' if self.n_jobs != 1 else 'sequential'} processing ({n_workers} cores)")
        
        # Find common store-item combinations
        hist_combos = set(zip(self.hist_df['store_id'], self.hist_df['item_id']))
        future_combos = set(zip(self.future_df['store_id'], self.future_df['item_id']))
        common_combos = sorted(list(hist_combos.intersection(future_combos)))
        
        if len(common_combos) == 0:
            raise ValueError("No common store_id/item_id combinations found")
        
        # Initialize result dataframe
        result_df = self.future_df.copy()
        quantile_cols = [f'q{int(q*100):02d}' for q in self.quantiles]
        for col in quantile_cols:
            result_df[col] = np.nan
        
        # Prepare processing arguments
        process_args = []
        for store_id, item_id in common_combos:
            hist_subset = self.hist_df[(self.hist_df['store_id'] == store_id) & 
                                     (self.hist_df['item_id'] == item_id)].copy()
            future_subset = self.future_df[(self.future_df['store_id'] == store_id) & 
                                         (self.future_df['item_id'] == item_id)].copy()
            process_args.append((store_id, item_id, hist_subset, future_subset, self.quantiles))
        
        # Process (parallel or sequential)
        process_start = time.time()
        if self.n_jobs == 1:
            results = []
            for args in process_args:
                result = _process_single_combination_helper(args)
                if result is not None:
                    results.append(result)
        else:
            # Limit workers on Windows to avoid issues
            if platform.system() == 'Windows' and n_workers > 8:
                n_workers = 8
                logger.info(f"Limited workers to {n_workers} on Windows")
            
            try:
                with Pool(n_workers) as pool:
                    # Use chunksize to prevent hanging on large datasets
                    chunksize = max(1, len(process_args) // (n_workers * 4))
                    results = pool.map(_process_single_combination_helper, process_args, chunksize=chunksize)
                results = [r for r in results if r is not None]
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}. Falling back to sequential.")
                results = []
                for args in process_args:
                    result = _process_single_combination_helper(args)
                    if result is not None:
                        results.append(result)
        
        process_time = time.time() - process_start
        
        # Update result dataframe
        update_start = time.time()
        updated_count = 0
        
        for result in results:
            store_id = result['store_id']
            item_id = result['item_id']
            dates = pd.to_datetime(result['dates'])
            quantile_values = result['quantile_values']
            quantile_cols = result['quantile_cols']
            
            mask = ((result_df['store_id'] == store_id) & 
                    (result_df['item_id'] == item_id) & 
                    (result_df['date'].isin(dates)))
            indices = result_df[mask].index
            
            for j, q_col in enumerate(quantile_cols):
                date_map = dict(zip(dates, quantile_values[:, j]))
                result_df.loc[indices, q_col] = result_df.loc[indices, 'date'].map(date_map)
            
            updated_count += len(dates)
        
        update_time = time.time() - update_start
        total_time = time.time() - start_time
        
        logger.info(f"Completed probability forecasting: {len(results):,}/{len(process_args):,} combinations, "
                   f"{updated_count:,} records updated in {total_time:.2f}s "
                   f"(processing: {process_time:.2f}s, update: {update_time:.2f}s)")
        
        return result_df
