import pandas as pd 
import numpy as np
from datetime import timedelta
import sys
import os

# Add the project root to the path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.config import SEGMENTATION_CONFIG


class Segmentation:
    def __init__(self, df: pd.DataFrame, level_columns=None, date_column=None, target_column=None):
        self.df = df.copy()
        
        # Set configuration from config if not provided
        if level_columns is None:
            self.level_columns = SEGMENTATION_CONFIG["level_columns"]
        else:
            self.level_columns = level_columns
            
        if date_column is None:
            self.date_column = SEGMENTATION_CONFIG["date_column"]
        else:
            self.date_column = date_column
            
        if target_column is None:
            self.target_column = SEGMENTATION_CONFIG["target_column"]
        else:
            self.target_column = target_column

        self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors='coerce')

    # ------------------ INTERMITTENCY FUNCTION ------------------ 

    def intermitency(self, threshold=None):
        if threshold is None:
            threshold = SEGMENTATION_CONFIG["intermittency"]["threshold"]
        zero_sales_counts = self.df.groupby(self.level_columns)[self.target_column].transform(lambda x: (x == 0).sum())
        total_counts = self.df.groupby(self.level_columns)[self.target_column].transform('size')
        zero_ratio = zero_sales_counts / total_counts
        self.df['Intermitent'] = np.where(zero_ratio > threshold, 'Yes', 'NO')
        return self.df
    
    # ------------------ PRODUCT LIFE CYCLE FUNCTION ------------------


    def product_life_cycle(self, disco_periods=None, new_launch_periods=None):
        if disco_periods is None:
            disco_periods = SEGMENTATION_CONFIG["product_life_cycle"]["disco_periods"]
        if new_launch_periods is None:
            new_launch_periods = SEGMENTATION_CONFIG["product_life_cycle"]["new_launch_periods"]
        df = self.df.copy()
        plc_info = df.groupby(self.level_columns).agg(
            Introduction_Date=(self.date_column, 'min'),
            Last_Active_Date=(self.date_column, 'max'),
            Recent_Sales_Sum=(self.target_column, lambda x: x.tail(disco_periods).sum())
        ).reset_index()


        current_date = df[self.date_column].max()
        plc_info['NPI_Cutoff_Date'] = current_date - timedelta(days=new_launch_periods)
        plc_info['PLC_Status'] = pd.Series(index=plc_info.index, dtype='object')

        plc_info.loc[plc_info['Recent_Sales_Sum'] == 0, 'PLC_Status'] = "DISC"

        plc_info.loc[
            plc_info['Introduction_Date'] > plc_info['NPI_Cutoff_Date'],
            'PLC_Status'
        ] = "NEW LAUNCH"

        plc_info['PLC_Status'] = plc_info['PLC_Status'].fillna("MATURE")

        self.df = self.df.merge(plc_info[self.level_columns + ['PLC_Status']], on=self.level_columns, how='left')

        return self.df


    # ------------------ SERIES LENGTH CALCULATION FUNCTION ------------------

    def series_length_calculation(
        self,
        frequency=None
    ):
        if frequency is None:
            frequency = SEGMENTATION_CONFIG["series_length_calculation"]["frequency"]
        df = self.df.copy()
        group_cols = self.level_columns
        
        # Define periods per cycle based on frequency
        periods_per_cycle = {
            'daily': 365,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }
        
        if frequency not in periods_per_cycle:
            raise ValueError(f"Unsupported frequency: {frequency}. Use 'daily', 'weekly', 'monthly', or 'quarterly'")
        
        periods_per_year = periods_per_cycle[frequency]
        
        rows = []
        for keys, g in df.sort_values(by=self.date_column).groupby(group_cols):
            # Calculate number of periods in the series
            series_length = len(g)
            
            # Calculate number of cycles
            cycles = series_length / periods_per_year
            
            # Classify based on 2 cycles threshold
            cycle_classification = 'Greater than 2 cycles' if cycles >= 2 else 'Less than 2 cycles'
            
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row['Series_Length_Periods'] = series_length
            row['Series_Length_Cycles'] = round(cycles, 2)
            row['Cycle_Classification'] = cycle_classification
            rows.append(row)
        
        length_df = pd.DataFrame(rows)
        self.df = self.df.merge(length_df, on=group_cols, how='left')
        self.df.drop(columns = [ 'Series_Length_Periods', 'Series_Length_Cycles' ], inplace = True)
        return self.df


    # ------------------ VARIABILITY SEGMENTATION FUNCTION ------------------


    def variability_segmentation(
        self,
        cov_thresholds=None,
        cov_levels=None
    ):
        if cov_thresholds is None:
            cov_thresholds = SEGMENTATION_CONFIG["variability_segmentation"]["cov_thresholds"]
        if cov_levels is None:
            cov_levels = SEGMENTATION_CONFIG["variability_segmentation"]["cov_levels"]
        df = self.df.copy()

        if len(cov_levels) != len(cov_thresholds) + 1:
            raise ValueError("cov_levels must be exactly one longer than cov_thresholds")

        group_cols = self.level_columns

        variability_df = df.groupby(group_cols).agg({
            self.target_column: [np.mean, lambda x: np.std(x, ddof=1)]
        }).reset_index()

        avg_col = 'Avg_' + self.target_column
        std_dev_col = 'Std_' + self.target_column
        cov_col = 'CV_' + self.target_column
        cov_segment_col = 'Variability_Segment'

        variability_df.columns = group_cols + [avg_col, std_dev_col]

        variability_df[std_dev_col].fillna(0, inplace=True)
        variability_df[cov_col] = np.where(
            variability_df[avg_col] > 0,
            variability_df[std_dev_col] / variability_df[avg_col],
            0,
        )

        bins = np.array(cov_thresholds, dtype=float)
        idx = np.digitize(variability_df[cov_col].to_numpy(dtype=float), bins, right=False)
        idx = np.clip(idx, 0, len(cov_levels) - 1)
        variability_df[cov_segment_col] = np.array(cov_levels, dtype=object)[idx]

        result_cols = group_cols + [cov_col, cov_segment_col]
        self.df = self.df.merge(variability_df[result_cols], on=group_cols, how='left')

        self.df[cov_segment_col] = self.df[cov_segment_col].fillna(cov_levels[-1])
        self.df.drop(columns = f'CV_{self.target_column}', inplace = True)

        return self.df


    # ------------------ Volume SEGMENTATION FUNCTION ------------------

    def volume_segmentation(
        self,
        vol_thresholds=None,
        vol_levels=None,
        group_columns=None
    ):
        if vol_thresholds is None:
            vol_thresholds = SEGMENTATION_CONFIG["volume_segmentation"]["vol_thresholds"]
        if vol_levels is None:
            vol_levels = SEGMENTATION_CONFIG["volume_segmentation"]["vol_levels"]
        if group_columns is None:
            group_columns = SEGMENTATION_CONFIG["volume_segmentation"]["group_columns"]
        df = self.df.copy()

        if len(vol_levels) != len(vol_thresholds) + 1:
            raise ValueError("vol_levels must be exactly one longer than vol_thresholds")

        item_keys = list(group_columns) + self.level_columns

        item_volume = df.groupby(item_keys)[self.target_column].sum().reset_index()
        item_volume = item_volume.rename(columns={self.target_column: 'Item_Volume'})

        if len(group_columns) > 0:
            group_totals = item_volume.groupby(list(group_columns))['Item_Volume'].sum().reset_index()
            group_totals = group_totals.rename(columns={'Item_Volume': 'Group_Volume'})
            item_volume = item_volume.merge(group_totals, on=list(group_columns), how='left')
        else:
            total = item_volume['Item_Volume'].sum()
            item_volume['Group_Volume'] = total

        denom = item_volume['Group_Volume'].replace(0, np.nan)
        item_volume['Volume_Share'] = (item_volume['Item_Volume'] / denom).fillna(0.0)

        sort_cols = list(group_columns) + ['Volume_Share']
        item_volume = item_volume.sort_values(by=sort_cols, ascending=[True] * len(group_columns) + [False])

        if len(group_columns) > 0:
            item_volume['Cum_Share'] = item_volume.groupby(list(group_columns))['Volume_Share'].cumsum()
            item_volume['Cum_Share_Exclusive'] = item_volume['Cum_Share'] - item_volume['Volume_Share']
        else:
            item_volume['Cum_Share'] = item_volume['Volume_Share'].cumsum()
            item_volume['Cum_Share_Exclusive'] = item_volume['Cum_Share'] - item_volume['Volume_Share']

        bins = np.array(vol_thresholds, dtype=float)
        idx = np.digitize(item_volume['Cum_Share_Exclusive'].to_numpy(dtype=float), bins, right=False)
        idx = np.clip(idx, 0, len(vol_levels) - 1)
        item_volume['Volume_Segment'] = np.array(vol_levels, dtype=object)[idx]

        result_cols = item_keys + ['Volume_Share', 'Cum_Share', 'Volume_Segment']
        self.df = self.df.merge(item_volume[result_cols], on=item_keys, how='left')

        self.df['Volume_Segment'] = self.df['Volume_Segment'].fillna(vol_levels[-1])
        self.df.drop(columns = ['Volume_Share', 'Cum_Share'], inplace = True)

        return self.df


    # ------------------ TREND SEGMENTATION FUNCTION ------------------
    
    def trend_segmentation(
        self,
        trend_threshold=None
    ):
        if trend_threshold is None:
            trend_threshold = SEGMENTATION_CONFIG["trend_segmentation"]["trend_threshold"]
        df = self.df.copy()

        group_cols = self.level_columns

        rows = []
        for keys, g in df.sort_values(by=self.date_column).groupby(group_cols):
            series = g[self.target_column].to_numpy(dtype=float)
            if series.size < 2:
                slope = 0.0
            else:
                mask = np.isfinite(series)
                y = series[mask]
                if y.size < 2:
                    slope = 0.0
                else:
                    x = np.arange(1, y.size + 1, dtype=float)
                    slope = float(np.polyfit(x, y, 1)[0])
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row['Trend_Factor'] = slope
            rows.append(row)

        trend_df = pd.DataFrame(rows)

        conditions = [
            trend_df['Trend_Factor'] > trend_threshold,
            (trend_df['Trend_Factor'] <= trend_threshold) & (trend_df['Trend_Factor'] >= -trend_threshold),
            trend_df['Trend_Factor'] < -trend_threshold
        ]
        choices = ["UPWARD", "NO TREND", "DOWNWARD"]
        trend_df['Trend'] = np.select(conditions, choices, default=None)

        self.df = self.df.merge(trend_df, on=group_cols, how='left')
        self.df.drop(columns = 'Trend_Factor', inplace = True)

        return self.df
    



    
    def seasonality_detection(
        self,
        useACFForSeasonality=None,
        ACFDiff=None,
        ACFLowerThreshold=None,
        ACFUpperThreshold=None,
        seasonality_threshold=None,
        ACFSkipLags=None,
        RequiredACFLags=None,
        frequency=None,
        max_detected_periods=None
    ):
        if useACFForSeasonality is None:
            useACFForSeasonality = SEGMENTATION_CONFIG["seasonality_detection"]["useACFForSeasonality"]
        if ACFDiff is None:
            ACFDiff = SEGMENTATION_CONFIG["seasonality_detection"]["ACFDiff"]
        if ACFLowerThreshold is None:
            ACFLowerThreshold = SEGMENTATION_CONFIG["seasonality_detection"]["ACFLowerThreshold"]
        if ACFUpperThreshold is None:
            ACFUpperThreshold = SEGMENTATION_CONFIG["seasonality_detection"]["ACFUpperThreshold"]
        if seasonality_threshold is None:
            seasonality_threshold = SEGMENTATION_CONFIG["seasonality_detection"]["seasonality_threshold"]
        if ACFSkipLags is None:
            ACFSkipLags = SEGMENTATION_CONFIG["seasonality_detection"]["ACFSkipLags"]
        if RequiredACFLags is None:
            RequiredACFLags = SEGMENTATION_CONFIG["seasonality_detection"]["RequiredACFLags"]
        if frequency is None:
            frequency = SEGMENTATION_CONFIG["seasonality_detection"]["frequency"]
        if max_detected_periods is None:
            max_detected_periods = SEGMENTATION_CONFIG["seasonality_detection"]["max_detected_periods"]
        df = self.df.copy()
        group_cols = self.level_columns

        rows = []
        max_lag = Segmentation.default_max_lag(frequency)
        required_lags = Segmentation.parse_required_lags(RequiredACFLags)

        for keys, g in df.sort_values(by=self.date_column).groupby(group_cols):
            series = g[self.target_column].to_numpy(dtype=float)

            if not useACFForSeasonality:
                seasonal_diff = 0
                try:
                    from pmdarima.arima.utils import ndiffs  # type: ignore
                    seasonal_diff = int(ndiffs(series, alpha=0.05, test="kpss"))
                except Exception:
                    seasonal_diff = 0
                detected_periods = []
                has_seasonality = False
            else:
                y = Segmentation.difference_series(series, ACFDiff)
                if y.size < 3:
                    detected_periods = []
                    has_seasonality = False
                    seasonal_diff = ACFDiff
                else:
                    acf_vals = Segmentation.compute_acf(y, max_lag)
                    candidate_lags = required_lags if (required_lags and len(required_lags) > 0) else list(range(1, max_lag + 1))
                    candidate_lags = [lag for lag in candidate_lags if lag > max(1, ACFSkipLags)]

                    scored = []
                    for lag in candidate_lags:
                        val = acf_vals[lag] if lag < acf_vals.size else 0.0
                        if val >= seasonality_threshold and (val < ACFUpperThreshold) and (val > ACFLowerThreshold):
                            scored.append((lag, float(val)))

                    scored.sort(key=lambda t: t[1], reverse=True)
                    detected_periods = [lag for lag, _ in scored[:max_detected_periods]]

                    has_seasonality = len(detected_periods) > 0
                    seasonal_diff = ACFDiff

                    if has_seasonality and series.size < 2 * min(detected_periods):
                        has_seasonality = False

            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row['Seasonal'] = 'YES' if has_seasonality else 'NO'
            row['Seasonal_Periods'] = ','.join(str(p) for p in detected_periods) if detected_periods else None
            row['Seasonal_Diff'] = int(seasonal_diff)
            rows.append(row)

        season_df = pd.DataFrame(rows)
        self.df = self.df.merge(season_df, on=group_cols, how='left')
        self.df.drop(columns = ['Seasonal_Periods','Seasonal_Diff'], inplace = True)

        return self.df

    @staticmethod
    def difference_series(arr: np.ndarray, d: int) -> np.ndarray:
        result = np.asarray(arr, dtype=float)
        for _ in range(max(d, 0)):
            if result.size < 2:
                return result
            result = np.diff(result)
        return result

    @staticmethod
    def compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - np.nanmean(x)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n = x.size
        if n == 0:
            return np.zeros(max_lag + 1, dtype=float)
        denom = np.dot(x, x)
        if denom == 0:
            return np.zeros(max_lag + 1, dtype=float)
        acf = np.empty(max_lag + 1, dtype=float)
        for k in range(0, max_lag + 1):
            acf[k] = np.dot(x[: n - k], x[k:]) / denom
        return acf

    @staticmethod
    def default_max_lag(freq: str) -> int:
        f = (freq or '').lower()
        if f == 'weekly':
            return 52
        if f == 'quarterly':
            return 4
        return 12

    @staticmethod
    def parse_required_lags(spec: str):
        s = (spec or '').strip()
        if not s:
            return None
        try:
            return [int(tok) for tok in s.split(',') if tok.strip().isdigit()]
        except Exception:
            return None



 

    # ------------------ RUN ALL STEPS ------------------

    def run(self):
        # Execute all steps using config parameters
        self.intermitency()
        self.product_life_cycle()
        self.variability_segmentation()
        self.volume_segmentation()
        self.trend_segmentation()
        self.seasonality_detection()
        self.series_length_calculation()
        
        # Generate output columns dynamically based on level columns and segmentation results
        output_columns = self.level_columns + [
            "Intermitent",
            "PLC_Status", 
            "Cycle_Classification",
            "Variability_Segment",
            "Volume_Segment",
            "Trend",
            "Seasonal"
        ]
        
        # Filter to only include columns that exist in the dataframe
        present_cols = [c for c in output_columns if c in self.df.columns]
        summary = self.df[present_cols].drop_duplicates(subset=self.level_columns).reset_index(drop=True)
        return summary



    









    
    



    




    


    


