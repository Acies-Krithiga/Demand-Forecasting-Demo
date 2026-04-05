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
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.forecasting.stl import STLForecast
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
try:
    from pmdarima.arima import AutoARIMA, ARIMA
except Exception:  # pragma: no cover
    AutoARIMA = None
    ARIMA = None

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None

try:
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.croston import Croston
    from sktime.forecasting.ets import AutoETS
    from sktime.forecasting.tbats import TBATS
    from sktime.forecasting.compose import make_reduction
except Exception:  # pragma: no cover
    ThetaForecaster = None
    Croston = None
    AutoETS = None
    TBATS = None
    make_reduction = None

try:
    from statsforecast.core import StatsForecast
except Exception:  # pragma: no cover
    StatsForecast = None

# Set up logging
logger = logging.getLogger(__name__)

class AlgoParamExtractor:
    
    def __init__(self):
        self.default_params = {
            'SES': {'alpha_lower': 0.1, 'alpha_upper': 0.9},
            'DES': {'alpha_lower': 0.1, 'alpha_upper': 0.9, 'beta_lower': 0.1, 'beta_upper': 0.9},
            'TES': {'alpha_lower': 0.1, 'alpha_upper': 0.9, 'beta_lower': 0.1, 'beta_upper': 0.9, 
                   'gamma_lower': 0.1, 'gamma_upper': 0.9, 'phi_lower': 0.8, 'phi_upper': 0.98},
            'SARIMA': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
            'ARIMA': {'order': (1, 1, 1)},
            'Moving Average': {'period': 3},
            'Theta': {'theta': 2.0},
            'Croston': {'alpha': 0.1},
            'ETS': {'model': 'ZZZ'},
            'TBATS': {'use_box_cox': True, 'use_trend': True, 'use_damped_trend': True},
            'Prophet': {'yearly_seasonality': True, 'weekly_seasonality': True, 'daily_seasonality': False},
            'STLF': {'seasonal_periods': 12},
            'Linear Regression': {'fit_intercept': True},
            'KNN Regression': {'n_neighbors': 5},
            'AR-NNET': {'hidden_layer_sizes': (100,), 'max_iter': 200},
            'Simple Snaive': {'seasonal_periods': 12},
            'Weighted Snaive': {'seasonal_periods': 12, 'weights': 'linear'},
            'Growth Snaive': {'seasonal_periods': 12, 'growth_factor': 1.0},
            'Simple AOA': {'seasonal_periods': 12},
            'Weighted AOA': {'seasonal_periods': 12, 'weights': 'linear'},
            'Growth AOA': {'seasonal_periods': 12, 'growth_factor': 1.0}
        }
    
    def extract_param_value(self, algorithm, parameter):
        params = self.default_params.get(algorithm, {})
        return params.get(parameter, 0.5)
    
    def get_params(self, algorithm: str, **kwargs) -> Dict:
        params = self.default_params.get(algorithm, {}).copy()
        params.update(kwargs)
        return params
    
    def extract_seasonal_periods(self, data: pd.Series, frequency: str = 'daily') -> int:
        if frequency == 'daily':
            return 365
        elif frequency == 'weekly':
            return 52
        elif frequency == 'monthly':
            return 12
        else:
            return 12
    
    def extract_trend_params(self, data: pd.Series) -> Dict:
        if len(data) > 1:
            trend_slope = np.polyfit(range(len(data)), data.values, 1)[0]
            return {
                'trend': 'increasing' if trend_slope > 0.01 else 'decreasing' if trend_slope < -0.01 else 'no_trend',
                'trend_strength': abs(trend_slope)
            }
        return {'trend': 'no_trend', 'trend_strength': 0}
    
    def extract_seasonality_params(self, data: pd.Series, frequency: str = 'daily') -> Dict:
        seasonal_periods = self.extract_seasonal_periods(data, frequency)
        
        if len(data) >= seasonal_periods * 2:
            seasonal_variance = np.var([data.iloc[i::seasonal_periods].mean() for i in range(seasonal_periods)])
            total_variance = np.var(data)
            seasonality_strength = seasonal_variance / total_variance if total_variance > 0 else 0
            
            return {
                'seasonal_periods': seasonal_periods,
                'seasonality_strength': seasonality_strength,
                'has_seasonality': seasonality_strength > 0.1
            }
        
        return {
            'seasonal_periods': seasonal_periods,
            'seasonality_strength': 0,
            'has_seasonality': False
        }


class ForecastingEngine:
    
    def __init__(self, train, seasonal_periods, forecast_horizon, confidence_interval_alpha=0.05):
        self.train = train
        self.seasonal_periods = seasonal_periods
        self.forecast_horizon = forecast_horizon
        self.confidence_interval_alpha = confidence_interval_alpha
        self.param_extractor = AlgoParamExtractor()
        
    def _make_lower_bound_non_zero(self, bound):
        return max(bound, 0.01)
    
    def _initialize_results(self):
        forecast = pd.Series([np.nan] * self.forecast_horizon)
        forecast_intervals = pd.DataFrame({
            'lower': [np.nan] * self.forecast_horizon,
            'upper': [np.nan] * self.forecast_horizon
        })
        fitted_params = {}
        return forecast, forecast_intervals, fitted_params
    
    def _get_exp_smoothing_forecast(self, estimator, model_name):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            fit = estimator.fit()
            forecast_values = fit.forecast(self.forecast_horizon)
            forecast = pd.Series(forecast_values)
            fitted_params = str(fit.params)
        except Exception as e:
            logger.error(f"{model_name} failed: {str(e)}")
        return forecast, forecast_intervals, fitted_params
    
    def get_ses_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            alpha_lower = self._make_lower_bound_non_zero(self.param_extractor.extract_param_value("SES", "alpha_lower"))
            alpha_upper = self.param_extractor.extract_param_value("SES", "alpha_upper")
            
            estimator = ETSModel(
                endog=self.train,
                error="add",
                initialization_method="estimated",
                bounds={"smoothing_level": (alpha_lower, alpha_upper)}
            )
            forecast, forecast_intervals, fitted_params = self._get_exp_smoothing_forecast(estimator, "SES")
        except Exception as e:
            logger.error(f"SES failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_des_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            alpha_lower = self._make_lower_bound_non_zero(self.param_extractor.extract_param_value("DES", "alpha_lower"))
            alpha_upper = self.param_extractor.extract_param_value("DES", "alpha_upper")
            beta_lower = self._make_lower_bound_non_zero(self.param_extractor.extract_param_value("DES", "beta_lower"))
            beta_upper = self.param_extractor.extract_param_value("DES", "beta_upper")
            
            estimator = ETSModel(
                endog=self.train,
                trend="add",
                error="add",
                initialization_method="estimated",
                bounds={
                    "smoothing_level": (alpha_lower, alpha_upper),
                    "smoothing_trend": (beta_lower, beta_upper)
                }
            )
            forecast, forecast_intervals, fitted_params = self._get_exp_smoothing_forecast(estimator, "DES")
        except Exception as e:
            logger.error(f"DES failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_tes_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            # Use faster Holt-Winters implementation
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            # Limit seasonal periods for speed (365 is too much for daily data)
            seasonal_periods = min(self.seasonal_periods, 30)  # Max 30 days
            
            model = ExponentialSmoothing(
                self.train,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
            
            fit = model.fit(optimized=True, remove_bias=False)
            forecast_values = fit.forecast(self.forecast_horizon)
            forecast = pd.Series(forecast_values)
            fitted_params = f"TES optimized (sp={seasonal_periods})"
        except Exception as e:
            logger.error(f"TES failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_moving_avg_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            periods = int(self.param_extractor.extract_param_value("Moving Average", "period"))
            
            # Simple moving average forecast
            last_values = self.train.tail(periods).values
            forecast_values = np.full(self.forecast_horizon, np.mean(last_values))
            forecast = pd.Series(forecast_values)
            fitted_params = f"MA Periods: {periods}"
        except Exception as e:
            logger.error(f"Moving Average failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_sarima_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            order = self.param_extractor.extract_param_value("SARIMA", "order")
            seasonal_order = self.param_extractor.extract_param_value("SARIMA", "seasonal_order")
            
            model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order)
            fit = model.fit(disp=False)
            forecast_values = fit.forecast(self.forecast_horizon)
            forecast = pd.Series(forecast_values)
            fitted_params = str(fit.params)
        except Exception as e:
            logger.error(f"SARIMA failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_auto_arima_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            if AutoARIMA is None:
                raise ImportError("pmdarima is not installed")
            model = AutoARIMA(start_p=0, start_q=0, max_p=3, max_q=3, seasonal=True, m=self.seasonal_periods)
            fit = model.fit(self.train)
            forecast_values = fit.predict(self.forecast_horizon)
            forecast = pd.Series(forecast_values)
            fitted_params = str(fit.params())
        except Exception as e:
            logger.error(f"Auto ARIMA failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_prophet_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            if Prophet is None:
                raise ImportError("prophet is not installed")
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': self.train.index,
                'y': self.train.values
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            model.fit(df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=self.forecast_horizon)
            forecast_df = model.predict(future)
            
            # Extract forecast values
            forecast_values = forecast_df['yhat'].tail(self.forecast_horizon).values
            forecast = pd.Series(forecast_values)
            fitted_params = "Prophet model fitted"
        except Exception as e:
            logger.error(f"Prophet failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_theta_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            if ThetaForecaster is None:
                raise ImportError("sktime is not installed")
            from sktime.forecasting.base import ForecastingHorizon
            fh = ForecastingHorizon(np.arange(1, self.forecast_horizon + 1))
            
            train_series_with_freq = self.train.copy()
            train_series_with_freq.index.freq = 'D'
            
            forecaster = ThetaForecaster(sp=1)
            forecaster.fit(train_series_with_freq)
            forecast_values = forecaster.predict(fh)
            forecast = pd.Series(forecast_values)
            fitted_params = "Theta method"
        except Exception as e:
            logger.error(f"Theta failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_croston_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            if Croston is None:
                raise ImportError("sktime is not installed")
            alpha = self.param_extractor.extract_param_value("Croston", "alpha")
            
            from sktime.forecasting.base import ForecastingHorizon
            fh = ForecastingHorizon(np.arange(1, self.forecast_horizon + 1))
            
            train_series_with_freq = self.train.copy()
            train_series_with_freq.index.freq = 'D'
            
            forecaster = Croston(smoothing=alpha)
            forecaster.fit(train_series_with_freq)
            forecast_values = forecaster.predict(fh)
            forecast = pd.Series(forecast_values)
            fitted_params = f"Croston alpha: {alpha}"
        except Exception as e:
            logger.error(f"Croston failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_ets_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            if AutoETS is None:
                raise ImportError("sktime is not installed")
            from sktime.forecasting.base import ForecastingHorizon
            fh = ForecastingHorizon(np.arange(1, self.forecast_horizon + 1))
            
            train_series_with_freq = self.train.copy()
            train_series_with_freq.index.freq = 'D'
            
            forecaster = AutoETS(auto=True, sp=self.seasonal_periods)
            forecaster.fit(train_series_with_freq)
            forecast_values = forecaster.predict(fh)
            forecast = pd.Series(forecast_values)
            fitted_params = "ETS auto model"
        except Exception as e:
            logger.error(f"ETS failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_tbats_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            if TBATS is None:
                raise ImportError("sktime is not installed")
            from sktime.forecasting.base import ForecastingHorizon
            fh = ForecastingHorizon(np.arange(1, self.forecast_horizon + 1))
            
            train_series_with_freq = self.train.copy()
            train_series_with_freq.index.freq = 'D'
            
            forecaster = TBATS(use_box_cox=True, use_trend=True, use_damped_trend=True)
            forecaster.fit(train_series_with_freq)
            forecast_values = forecaster.predict(fh)
            forecast = pd.Series(forecast_values)
            fitted_params = "TBATS model"
        except Exception as e:
            logger.error(f"TBATS failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_stlf_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.seasonal import STL
            
            stl = STL(self.train, seasonal=self.seasonal_periods)
            stl_fit = stl.fit()
            
            arima_model = ARIMA(stl_fit.resid, order=(1, 1, 1))
            arima_fit = arima_model.fit()
            
            forecast_values = arima_fit.forecast(self.forecast_horizon)
            forecast = pd.Series(forecast_values)
            fitted_params = "STLF model"
        except Exception as e:
            logger.error(f"STLF failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_linear_regression_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            # Create time features
            X = np.arange(len(self.train)).reshape(-1, 1)
            y = self.train.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future periods
            X_future = np.arange(len(self.train), len(self.train) + self.forecast_horizon).reshape(-1, 1)
            forecast_values = model.predict(X_future)
            forecast = pd.Series(forecast_values)
            fitted_params = f"Linear Regression: slope={model.coef_[0]:.4f}, intercept={model.intercept_:.4f}"
        except Exception as e:
            logger.error(f"Linear Regression failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_knn_regression_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            n_neighbors = int(self.param_extractor.extract_param_value("KNN Regression", "n_neighbors"))
            
            # Create time features
            X = np.arange(len(self.train)).reshape(-1, 1)
            y = self.train.values
            
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            model.fit(X, y)
            
            # Forecast future periods
            X_future = np.arange(len(self.train), len(self.train) + self.forecast_horizon).reshape(-1, 1)
            forecast_values = model.predict(X_future)
            forecast = pd.Series(forecast_values)
            fitted_params = f"KNN Regression: n_neighbors={n_neighbors}"
        except Exception as e:
            logger.error(f"KNN Regression failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_arnnet_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            from sklearn.neural_network import MLPRegressor
            
            # Create lagged features
            lags = min(5, len(self.train) // 2)
            X, y = [], []
            for i in range(lags, len(self.train)):
                X.append(self.train.iloc[i-lags:i].values)
                y.append(self.train.iloc[i])
            
            X = np.array(X)
            y = np.array(y)
            
            model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
            model.fit(X, y)
            
            # Forecast future periods
            forecast_values = []
            last_values = self.train.tail(lags).values
            
            for _ in range(self.forecast_horizon):
                pred = model.predict([last_values])[0]
                forecast_values.append(pred)
                last_values = np.append(last_values[1:], pred)
            
            forecast = pd.Series(forecast_values)
            fitted_params = f"AR-NNET: lags={lags}"
        except Exception as e:
            logger.error(f"AR-NNET failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_simple_snaive_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            # Use last seasonal period values
            forecast_values = []
            for i in range(self.forecast_horizon):
                seasonal_idx = i % self.seasonal_periods
                if len(self.train) > self.seasonal_periods:
                    forecast_values.append(self.train.iloc[-(self.seasonal_periods - seasonal_idx)])
                else:
                    forecast_values.append(self.train.iloc[-1])
            
            forecast = pd.Series(forecast_values)
            fitted_params = f"Simple Seasonal Naive: period={self.seasonal_periods}"
        except Exception as e:
            logger.error(f"Simple Seasonal Naive failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_weighted_snaive_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            # Use weighted average of recent seasonal periods
            forecast_values = []
            weights = np.linspace(0.1, 1.0, self.seasonal_periods)  # Linear weights
            
            for i in range(self.forecast_horizon):
                seasonal_idx = i % self.seasonal_periods
                if len(self.train) >= self.seasonal_periods:
                    seasonal_values = []
                    for j in range(min(3, len(self.train) // self.seasonal_periods)):
                        idx = len(self.train) - (j + 1) * self.seasonal_periods + seasonal_idx
                        if idx >= 0:
                            seasonal_values.append(self.train.iloc[idx])
                    
                    if seasonal_values:
                        weighted_avg = np.average(seasonal_values, weights=weights[:len(seasonal_values)])
                        forecast_values.append(weighted_avg)
                    else:
                        forecast_values.append(self.train.iloc[-1])
                else:
                    forecast_values.append(self.train.iloc[-1])
            
            forecast = pd.Series(forecast_values)
            fitted_params = f"Weighted Seasonal Naive: period={self.seasonal_periods}"
        except Exception as e:
            logger.error(f"Weighted Seasonal Naive failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_growth_snaive_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            growth_factor = self.param_extractor.extract_param_value("Growth Snaive", "growth_factor")
            
            # Calculate growth rate from recent periods
            if len(self.train) >= self.seasonal_periods * 2:
                recent_avg = self.train.tail(self.seasonal_periods).mean()
                previous_avg = self.train.iloc[-(self.seasonal_periods*2):-self.seasonal_periods].mean()
                growth_rate = (recent_avg - previous_avg) / previous_avg if previous_avg != 0 else 0
            else:
                growth_rate = 0
            
            # Apply growth to seasonal naive
            forecast_values = []
            for i in range(self.forecast_horizon):
                seasonal_idx = i % self.seasonal_periods
                if len(self.train) > self.seasonal_periods:
                    base_value = self.train.iloc[-(self.seasonal_periods - seasonal_idx)]
                    growth_adjusted = base_value * (1 + growth_rate * growth_factor)
                    forecast_values.append(max(0, growth_adjusted))  # Ensure non-negative
                else:
                    forecast_values.append(self.train.iloc[-1])
            
            forecast = pd.Series(forecast_values)
            fitted_params = f"Growth Seasonal Naive: period={self.seasonal_periods}, growth_rate={growth_rate:.4f}"
        except Exception as e:
            logger.error(f"Growth Seasonal Naive failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_simple_aoa_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            # Calculate averages for different time periods
            if len(self.train) >= self.seasonal_periods:
                seasonal_avg = self.train.tail(self.seasonal_periods).mean()
                overall_avg = self.train.mean()
                recent_avg = self.train.tail(min(7, len(self.train))).mean()
                
                # Simple average of averages
                aoa_value = (seasonal_avg + overall_avg + recent_avg) / 3
                forecast_values = [aoa_value] * self.forecast_horizon
            else:
                forecast_values = [self.train.mean()] * self.forecast_horizon
            
            forecast = pd.Series(forecast_values)
            fitted_params = f"Simple AOA: period={self.seasonal_periods}"
        except Exception as e:
            logger.error(f"Simple AOA failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_weighted_aoa_forecast(self):
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            if len(self.train) >= self.seasonal_periods:
                # Calculate weighted averages
                seasonal_avg = self.train.tail(self.seasonal_periods).mean()
                overall_avg = self.train.mean()
                recent_avg = self.train.tail(min(7, len(self.train))).mean()
                
                # Weighted average (recent gets more weight)
                weights = [0.5, 0.3, 0.2]  # recent, seasonal, overall
                weighted_aoa = weights[0] * recent_avg + weights[1] * seasonal_avg + weights[2] * overall_avg
                forecast_values = [weighted_aoa] * self.forecast_horizon
            else:
                forecast_values = [self.train.mean()] * self.forecast_horizon
            
            forecast = pd.Series(forecast_values)
            fitted_params = f"Weighted AOA: period={self.seasonal_periods}"
        except Exception as e:
            logger.error(f"Weighted AOA failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_growth_aoa_forecast(self):
        """Growth AOA"""
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            growth_factor = self.param_extractor.extract_param_value("Growth AOA", "growth_factor")
            
            if len(self.train) >= self.seasonal_periods:
                # Calculate growth rate
                recent_avg = self.train.tail(self.seasonal_periods).mean()
                previous_avg = self.train.iloc[-(self.seasonal_periods*2):-self.seasonal_periods].mean()
                growth_rate = (recent_avg - previous_avg) / previous_avg if previous_avg != 0 else 0
                
                # Calculate AOA with growth
                seasonal_avg = self.train.tail(self.seasonal_periods).mean()
                overall_avg = self.train.mean()
                
                base_aoa = (seasonal_avg + overall_avg) / 2
                growth_adjusted_aoa = base_aoa * (1 + growth_rate * growth_factor)
                
                forecast_values = [max(0, growth_adjusted_aoa)] * self.forecast_horizon
            else:
                forecast_values = [self.train.mean()] * self.forecast_horizon
            
            forecast = pd.Series(forecast_values)
            fitted_params = f"Growth AOA: period={self.seasonal_periods}, growth_rate={growth_rate:.4f}"
        except Exception as e:
            logger.error(f"Growth AOA failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
    
    def get_naive_random_walk_forecast(self):
        """Naive Random Walk"""
        forecast, forecast_intervals, fitted_params = self._initialize_results()
        try:
            # Use last value
            last_value = self.train.iloc[-1]
            forecast_values = [last_value] * self.forecast_horizon
            forecast = pd.Series(forecast_values)
            fitted_params = f"Naive Random Walk: last_value={last_value}"
        except Exception as e:
            logger.error(f"Naive Random Walk failed: {str(e)}")
        return {'forecast': forecast, 'intervals': forecast_intervals, 'params': fitted_params}
