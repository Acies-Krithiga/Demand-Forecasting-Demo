import pandas as pd 
import numpy as np
import logging
import os
import sys
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config.config import STATISTICAL_MODELS_CONFIG
from utils.stat import ForecastingEngine

logger = logging.getLogger(__name__)


class BaseFuture:
    def __init__(self, sales_fact, level_columns=None, date_column=None, target_column=None, frequency=None, forecast_cycles=None):
        self.sales_fact = self._ensure_dataframe(sales_fact)

        # Config-driven defaults with user overrides
        self.level_columns = level_columns or STATISTICAL_MODELS_CONFIG["level_columns"]
        self.date_column = date_column or STATISTICAL_MODELS_CONFIG["date_column"]
        self.target_column = target_column or STATISTICAL_MODELS_CONFIG["target_column"]
        self.frequency = frequency or STATISTICAL_MODELS_CONFIG["frequency"]

        self.minimum_data_points = STATISTICAL_MODELS_CONFIG["minimum_data_points"]
        self.confidence_interval_alpha = STATISTICAL_MODELS_CONFIG["confidence_interval_alpha"]

        # Horizon can be overridden by user; else from config map
        default_horizon_map = STATISTICAL_MODELS_CONFIG.get("forecast_cycles", {})
        self.forecast_horizon = int(forecast_cycles) if forecast_cycles is not None else int(default_horizon_map.get(self.frequency, 12))
        self.baseline_models = ["Moving Average", "Weighted Snaive"]
        self.algorithm_mapping = {
            "Moving Average": "get_moving_avg_forecast",
            "Weighted Snaive": "get_weighted_snaive_forecast",
        }

        self._prepare_data()

    def _ensure_dataframe(self, data_or_path):
        if isinstance(data_or_path, pd.DataFrame):
            return data_or_path.copy()
        if isinstance(data_or_path, str):
            if not os.path.exists(data_or_path):
                raise FileNotFoundError(f"Sales fact file not found: {data_or_path}")
            return pd.read_csv(data_or_path)
        raise ValueError("sales_fact must be a pandas DataFrame or CSV file path")

    def _prepare_data(self):
        missing = [c for c in ([self.date_column, self.target_column] + self.level_columns) if c not in self.sales_fact.columns]
        if missing:
            raise ValueError(f"Missing required columns in sales data: {missing}")

        self.sales_fact[self.date_column] = pd.to_datetime(self.sales_fact[self.date_column])

        self._generate_level_combinations()

    def _generate_level_combinations(self):
        combinations_df = self.sales_fact[self.level_columns].drop_duplicates().reset_index(drop=True)

        # Include ALL unique intersections regardless of history length so every
        # intersection receives a future horizon output
        self.valid_combinations = [
            {col: row[col] for col in self.level_columns}
            for _, row in combinations_df.iterrows()
        ]

    def _get_seasonal_periods(self) -> int:
        if self.frequency == 'daily':
            return 365
        if self.frequency == 'weekly':
            return 52
        if self.frequency == 'monthly':
            return 12
        if self.frequency == 'quarterly':
            return 4
        return 12

    def _get_pandas_freq(self) -> str:
        if self.frequency == 'daily':
            return 'D'
        if self.frequency == 'weekly':
            return 'W'
        if self.frequency == 'monthly':
            return 'MS'
        if self.frequency == 'quarterly':
            return 'Q'
        return 'MS'

    def _prepare_series_for_combination(self, intersection_values):
        filters = [(self.sales_fact[col_name] == col_val) for col_name, col_val in intersection_values.items()]
        combined_filter = filters[0]
        for cond in filters[1:]:
            combined_filter = combined_filter & cond
        data = self.sales_fact[combined_filter].copy()
        # Even with limited history, proceed to produce forecasts for completeness
        data = data.sort_values(self.date_column)
        data = data.set_index(self.date_column)
        series = data[self.target_column]
        return series

    def _generate_future_index(self, last_date: pd.Timestamp) -> pd.DatetimeIndex:
        freq = self._get_pandas_freq()
        start = last_date + pd.tseries.frequencies.to_offset(freq)
        return pd.date_range(start=start, periods=self.forecast_horizon, freq=freq)

    def _forecast_for_combination(self, intersection_values) -> pd.DataFrame:
        train_series = self._prepare_series_for_combination(intersection_values)
        if train_series is None or train_series.empty:
            return None

        seasonal_periods = self._get_seasonal_periods()
        last_date = train_series.index.max()
        future_index = self._generate_future_index(last_date)

        base_result = {
            'date': future_index,
        }
        for col_name, col_value in intersection_values.items():
            base_result[col_name] = [col_value] * len(future_index)

        try:
            engine = ForecastingEngine(
                train=train_series,
                seasonal_periods=seasonal_periods,
                forecast_horizon=self.forecast_horizon,
                confidence_interval_alpha=self.confidence_interval_alpha
            )
        except Exception:
            for algo_name in self.baseline_models:
                base_result[algo_name] = [np.nan] * len(future_index)
            return pd.DataFrame(base_result)

        for algo_name in self.baseline_models:
            method_name = self.algorithm_mapping[algo_name]
            try:
                method = getattr(engine, method_name)
            except AttributeError:
                base_result[algo_name] = [np.nan] * len(future_index)
                continue

            try:
                forecast_result = method()
                if forecast_result is not None and 'forecast' in forecast_result:
                    forecast_values = forecast_result['forecast']
                    # Align length exactly to horizon
                    if len(forecast_values) >= self.forecast_horizon:
                        forecast_values = forecast_values[:self.forecast_horizon]
                    else:
                        padded = np.full(self.forecast_horizon, np.nan)
                        padded[:len(forecast_values)] = np.array(forecast_values)
                        forecast_values = padded

                    base_result[algo_name] = forecast_values.tolist() if hasattr(forecast_values, 'tolist') else list(forecast_values)
                else:
                    base_result[algo_name] = [np.nan] * self.forecast_horizon
            except Exception:
                base_result[algo_name] = [np.nan] * self.forecast_horizon

        return pd.DataFrame(base_result)

    def generate_future_forecast(self) -> pd.DataFrame:
        if not self.valid_combinations:
            logger.info("No valid level combinations with sufficient data found for future forecasting.")
            return None

        results = []
        for intersection_values in self.valid_combinations:
            try:
                df_forecast = self._forecast_for_combination(intersection_values)
                if df_forecast is not None and not df_forecast.empty:
                    results.append(df_forecast)
            except Exception:
                continue

        if not results:
            return None

        return pd.concat(results, ignore_index=True)

    @staticmethod
    def run_from_csv(csv_path: str, forecast_cycles: int = None) -> pd.DataFrame:
        """Convenience helper to run future forecasts directly from a CSV path.
        Ensures config level columns exist in the CSV and returns the combined future forecasts.
        """
        instance = BaseFuture(csv_path, forecast_cycles=forecast_cycles)
        return instance.generate_future_forecast()