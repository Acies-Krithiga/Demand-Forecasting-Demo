# Forecasting Models Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [AlgoParamExtractor Class](#algoparamextractor-class)
4. [ForecastingEngine Class](#forecastingengine-class)
5. [Forecasting Models](#forecasting-models)
   - [Smoothing Methods](#smoothing-methods)
   - [ARIMA Methods](#arima-methods)
   - [Advanced Statistical Methods](#advanced-statistical-methods)
   - [Machine Learning Methods](#machine-learning-methods)
   - [Baseline Methods](#baseline-methods)

---

## Overview

This documentation describes the forecasting models implemented in `utils/stat.py`. The script provides a comprehensive suite of 21 different forecasting algorithms suitable for various time series prediction scenarios.

The system is designed to:
- Support multiple forecasting algorithms with consistent interfaces
- Automatically extract default parameters for each method
- Handle seasonal and non-seasonal time series
- Provide forecasts with error handling and logging

---

## Architecture

The forecasting system consists of two main classes:

### 1. **AlgoParamExtractor**
- **Purpose**: Manages default parameters for all forecasting algorithms
- **Key Features**:
  - Stores default parameter configurations
  - Extracts parameter values for specific algorithms
  - Derives seasonal periods based on data frequency
  - Analyzes trend and seasonality in time series data

### 2. **ForecastingEngine**
- **Purpose**: Implements all forecasting methods
- **Key Features**:
  - Initializes with training data, seasonal periods, and forecast horizon
  - Provides individual methods for each forecasting algorithm
  - Returns forecasts in a consistent format
  - Includes error handling and logging

---

## AlgoParamExtractor Class

### Purpose
This class manages default parameters for all forecasting algorithms and provides utilities for analyzing time series characteristics.

### Key Methods

#### `__init__()`
Initializes the class with default parameter dictionaries for all 21 forecasting methods.

#### `extract_param_value(algorithm, parameter)`
- **Purpose**: Retrieves a specific parameter value for a given algorithm
- **Returns**: The parameter value or 0.5 as default
- **Example**: `extract_param_value("SES", "alpha_upper")` → Returns 0.9

#### `get_params(algorithm: str, **kwargs) -> Dict`
- **Purpose**: Gets all parameters for an algorithm with optional overrides
- **Returns**: Dictionary of parameters for the specified algorithm
- **Example**: `get_params("SES", alpha_upper=0.95)` → Returns SES params with alpha_upper=0.95

#### `extract_seasonal_periods(data, frequency='daily') -> int`
- **Purpose**: Determines seasonal periods based on data frequency
- **Returns**:
  - Daily data: 365 periods
  - Weekly data: 52 periods
  - Monthly data: 12 periods
  - Default: 12 periods

#### `extract_trend_params(data) -> Dict`
- **Purpose**: Analyzes trend in the time series
- **Returns**: Dictionary with:
  - `trend`: 'increasing', 'decreasing', or 'no_trend'
  - `trend_strength`: Absolute slope value

#### `extract_seasonality_params(data, frequency='daily') -> Dict`
- **Purpose**: Measures seasonal patterns in the data
- **Returns**: Dictionary with:
  - `seasonal_periods`: Number of seasonal periods
  - `seasonality_strength`: Strength of seasonality (0 to 1)
  - `has_seasonality`: Boolean indicating presence of strong seasonality

---

## ForecastingEngine Class

### Purpose
The main engine that implements all forecasting algorithms. Each method follows a consistent pattern and returns forecasts in a standardized format.

### Initialization

```python
engine = ForecastingEngine(
    train=data_series,           # Training data (pandas Series)
    seasonal_periods=12,          # Seasonal period length
    forecast_horizon=10,          # Number of future periods to forecast
    confidence_interval_alpha=0.05  # Confidence level (default 95%)
)
```

### Return Format

All forecasting methods return a dictionary with three components:
```python
{
    'forecast': pd.Series,           # Forecasted values
    'intervals': pd.DataFrame,        # Confidence intervals (lower, upper)
    'params': str                     # Fitted parameters description
}
```

---

## Forecasting Models

### Smoothing Methods

#### 1. SES - Simple Exponential Smoothing

**Full Name**: Simple Exponential Smoothing

**How It Works**:
- Applies exponentially decreasing weights to historical observations
- Most recent observations receive the highest weight
- Smooths out noise while capturing level patterns
- Uses a single smoothing parameter (alpha) that controls the rate of decay

**Mathematical Formula**:
```
Forecast(t+1) = alpha × Observation(t) + (1-alpha) × Forecast(t)
```

**Parameters**:
- `alpha_lower`: 0.1 (lower bound for smoothing parameter)
- `alpha_upper`: 0.9 (upper bound for smoothing parameter)

**When to Use**:
- Data with no clear trend or seasonality
- When you need a simple, fast forecasting method
- Suitable for data with stationary patterns

**Implementation Details**:
- Uses ETSModel from statsmodels
- Automatically estimates optimal alpha parameter
- Returns fitted parameters for model inspection

---

#### 2. DES - Double Exponential Smoothing (Holt's Method)

**Full Name**: Double Exponential Smoothing / Holt's Linear Trend Method

**How It Works**:
- Extends Simple Exponential Smoothing by adding a trend component
- Uses two smoothing parameters:
  - Alpha (α): smooths the level
  - Beta (β): smooths the trend
- Captures both the level and trend in the data

**Mathematical Formula**:
```
Level: l(t) = alpha × y(t) + (1-alpha) × (l(t-1) + b(t-1))
Trend: b(t) = beta × (l(t) - l(t-1)) + (1-beta) × b(t-1)
Forecast: F(t+h) = l(t) + h × b(t)
```

**Parameters**:
- `alpha_lower`: 0.1, `alpha_upper`: 0.9
- `beta_lower`: 0.1, `beta_upper`: 0.9

**When to Use**:
- Data with a clear trend but no seasonality
- Linear growth or decline patterns
- Medium-term forecasts with trending data

---

#### 3. TES - Triple Exponential Smoothing (Holt-Winters)

**Full Name**: Triple Exponential Smoothing / Holt-Winters Method

**How It Works**:
- Extends DES by adding a seasonal component
- Uses three smoothing parameters:
  - Alpha (α): smooths the level
  - Beta (β): smooths the trend
  - Gamma (γ): smooths the seasonality
- Handles both trend and seasonality simultaneously

**Mathematical Formula**:
```
Level: l(t) = alpha × (y(t) / s(t-m)) + (1-alpha) × (l(t-1) + b(t-1))
Trend: b(t) = beta × (l(t) - l(t-1)) + (1-beta) × b(t-1)
Seasonal: s(t) = gamma × (y(t) / l(t)) + (1-gamma) × s(t-m)
Forecast: F(t+h) = (l(t) + h × b(t)) × s(t+h-m×k)
```

**Parameters**:
- `alpha_lower`: 0.1, `alpha_upper`: 0.9
- `beta_lower`: 0.1, `beta_upper`: 0.9
- `gamma_lower`: 0.1, `gamma_upper`: 0.9
- `phi_lower`: 0.8, `phi_upper`: 0.98 (damping parameter)

**Special Implementation**:
- Limits seasonal periods to 30 days maximum for performance
- Uses optimized fitting for faster computation
- Auto-selects best additive or multiplicative model

**When to Use**:
- Data with both trend and seasonality
- Seasonal patterns (daily, weekly, monthly, yearly)
- When you need to capture all three components: level, trend, and seasonality

---

#### 4. Moving Average

**Full Name**: Simple Moving Average

**How It Works**:
- Calculates the average of the last N observations
- Projects this average into the future
- No trend or seasonal adjustments
- Simplest forecasting method

**Mathematical Formula**:
```
Forecast = Mean of last N observations
```

**Parameters**:
- `period`: 3 (number of periods to average)

**When to Use**:
- Very simple data with no clear patterns
- Quick baseline forecasts
- When other methods are too complex

**Limitations**:
- Assumes no trend or seasonality
- Responds slowly to changes
- Constant forecast (flat line)

---

### ARIMA Methods

#### 5. SARIMA - Seasonal AutoRegressive Integrated Moving Average

**Full Name**: Seasonal AutoRegressive Integrated Moving Average

**How It Works**:
- Combines ARIMA with seasonal differencing
- Models both non-seasonal and seasonal patterns
- Uses autoregressive (AR), differencing (I), and moving average (MA) components
- The (p,d,q) part handles non-seasonal patterns
- The (P,D,Q,s) part handles seasonal patterns

**Parameters Explained**:
- `order`: (p, d, q) = (1, 1, 1)
  - p: AR order (how many lagged values to include)
  - d: Differencing order (how many times to difference)
  - q: MA order (how many lagged forecast errors to include)
- `seasonal_order`: (P, D, Q, s) = (1, 1, 1, 12)
  - s: Seasonal period (12 for monthly data)

**When to Use**:
- Complex time series with seasonal patterns
- When you understand the ARIMA methodology
- When seasonal patterns are important
- Requires sufficient historical data (at least 2-3 seasons)

**Considerations**:
- Requires knowledge of time series analysis
- Parameter selection can be complex
- May be slower than simpler methods

---

#### 6. Auto ARIMA

**Full Name**: Automatic ARIMA

**How It Works**:
- Automatically selects optimal ARIMA parameters
- Tests multiple combinations of (p,d,q) parameters
- Uses information criteria (AIC, BIC) to select best model
- Includes automatic seasonal detection
- Searches parameter space and picks best model

**Parameters**:
- `start_p`: 0, `start_q`: 0 (starting values)
- `max_p`: 3, `max_q`: 3 (maximum values to test)
- `seasonal`: True (enable seasonal ARIMA)
- `m`: seasonal_periods (seasonal frequency)

**When to Use**:
- When you want optimal ARIMA without manual tuning
- When you're not sure of the best ARIMA parameters
- For automated forecasting systems
- When you want a balance between flexibility and automation

**Advantages**:
- Automatic parameter selection
- Handles both seasonal and non-seasonal data
- Based on robust statistical framework
- Often produces accurate forecasts

**Computational Time**:
- Slower than fixed ARIMA (tests multiple models)
- Worth the extra time for better accuracy

---

### Advanced Statistical Methods

#### 7. Prophet

**Full Name**: Facebook Prophet

**How It Works**:
- Decomposes time series into components:
  - **Trend**: Long-term direction (linear or logistic)
  - **Seasonality**: Recurring patterns (weekly, monthly, yearly)
  - **Holiday effects**: Special events
  - **Error term**: Random fluctuations
- Uses Bayesian methodology for forecasting
- Automatically handles missing data and outliers

**Components**:
```
y(t) = g(t) + s(t) + h(t) + ε(t)

where:
- g(t): trend component
- s(t): seasonal component
- h(t): holiday component
- ε(t): error term
```

**Parameters**:
- `yearly_seasonality`: True (captures annual patterns)
- `weekly_seasonality`: True (captures weekly patterns)
- `daily_seasonality`: False (disabled by default)

**When to Use**:
- Business time series with strong seasonal patterns
- Datasets with holidays and special events
- Data with missing values or outliers
- When you need interpretable components
- Multi-seasonal data (weekly + yearly patterns)

**Advantages**:
- Robust to outliers
- Handles missing data gracefully
- Easy to interpret
- Fast fitting
- Good for business forecasting
- Provides uncertainty intervals

**Data Format**:
Requires DataFrame with:
- `ds`: datetime column
- `y`: value column

---

#### 8. Theta Method

**Full Name**: Theta Forecasting Method

**How It Works**:
- Decomposes series into short-term and long-term components
- Uses theta parameter to emphasize different components
- Classic theta=2 linearly extrapolates the trend
- Combines different linear decompositions
- Winner of the M3 forecasting competition

**How It Works**:
1. Decompose series using the theta line
2. Extrapolate the theta line using time series regression
3. Combine extrapolations with different thetas
4. Apply seasonal adjustment if needed

**Parameters**:
- `theta`: 2.0 (classic value, emphasizes trend)
- `sp`: 1 (seasonal period)

**When to Use**:
- Very accurate for many time series
- When trend and seasonality are both present
- Medium to long-term forecasts
- Competition-level forecasting

**Advantages**:
- Proven accuracy (won M3 competition)
- Simple parameter selection
- Handles both trend and seasonality
- Robust to outliers

---

#### 9. Croston's Method

**Full Name**: Croston's Method for Intermittent Demand

**How It Works**:
- Designed specifically for intermittent/sparse demand
- Separates demand into two components:
  - **Demand size**: When demand occurs
  - **Demand interval**: Time between demands
- Uses exponential smoothing on both components
- Forecasts demand only for periods when demand is expected

**Mathematical Approach**:
```
If demand occurs in period t:
  - Update demand size: D(t) = alpha × X(t) + (1-alpha) × D(t-1)
  - Update interval: I(t) = alpha × k + (1-alpha) × I(t-1)
  
Forecast = D(t) / I(t) (only for periods with expected demand)
```

**Parameters**:
- `alpha`: 0.1 (smoothing parameter)
- `smoothing`: alpha value for exponential smoothing

**When to Use**:
- **Sparse demand**: Many zero values in time series
- **Intermittent demand**: Irregular demand patterns
- **Inventory management**: For items with irregular sales
- **Spare parts forecasting**: Typical use case

**Example Cases**:
- Slow-moving inventory items
- Items with demand only on certain occasions
- Products with irregular ordering patterns

**Advantages**:
- Specifically designed for intermittent demand
- Avoids over-forecasting zeros
- More accurate for sparse data than traditional methods

---

#### 10. ETS - Error Trend Seasonality

**Full Name**: Automatic ETS Model Selection

**How It Works**:
- Automatic selection from 30 possible ETS models
- Tries different combinations of:
  - **Error**: Additive (A) or Multiplicative (M)
  - **Trend**: None (N), Additive (A), Damped (Ad)
  - **Seasonal**: None (N), Additive (A), Multiplicative (M)
- Uses information criteria to select best model
- Example: ANA (Additive error, No trend, Additive seasonality)

**Model Selection**:
AutoETS tries all combinations and selects based on:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- Accuracy on training data

**Parameters**:
- `auto`: True (automatic selection)
- `sp`: seasonal_periods (seasonal frequency)

**When to Use**:
- When you're unsure of the best ETS model
- For automated forecasting systems
- When you need both trend and seasonal components
- For general-purpose forecasting

**Advantages**:
- Automatic model selection
- Flexible (handles many patterns)
- Based on proven ETS methodology
- Good accuracy for various data types

---

#### 11. TBATS - Trigonometric Box-Cox ARMA Trend Seasonality

**Full Name**: Trigonometric Box-Cox ARMA Trend Seasonality

**How It Works**:
- Uses trigonometric functions to model complex seasonality
- Applies Box-Cox transformation for data stabilization
- Includes ARMA component for error modeling
- Can handle multiple seasonal patterns simultaneously
- Captures complex seasonal structures

**Components**:
```
y(t) = T(t) + S1(t) + S2(t) + ... + E(t)

where:
- T(t): Trend (damped)
- S1(t), S2(t), ...: Multiple seasonal components
- E(t): ARMA error term
```

**Parameters**:
- `use_box_cox`: True (apply Box-Cox transformation)
- `use_trend`: True (include trend component)
- `use_damped_trend`: True (use damped trend)

**When to Use**:
- Complex seasonal patterns
- Multiple seasonal cycles
- When simpler methods fail
- High-frequency data with rich patterns

**Advantages**:
- Handles multiple seasonalities
- Flexible seasonal modeling
- Robust to outliers
- Can capture complex patterns

---

#### 12. STLF - STL Decomposition + ARIMA

**Full Name**: Seasonal and Trend decomposition using Loess + ARIMA

**How It Works**:
1. **Decomposes** time series using STL (Seasonal and Trend decomposition using Loess)
2. **Separates** data into: Trend, Seasonal, and Residual components
3. **Forecasts** the residual component using ARIMA
4. **Recombines** all components for final forecast

**Decomposition**:
```
y(t) = Trend(t) + Seasonal(t) + Residual(t)

Forecast:
1. Extract seasonal pattern
2. Model residual with ARIMA
3. Combine: Forecast = Trend + Seasonal + ARIMA(Residual)
```

**Parameters**:
- `seasonal_periods`: 12 (seasonal length)
- `order`: (1, 1, 1) for ARIMA on residuals

**When to Use**:
- Strong seasonal patterns
- When you want to explicitly separate components
- Interpreting seasonal vs. trend effects
- When seasonal pattern is stable

**Advantages**:
- Clear decomposition (interpretable)
- Separates seasonal and trend effects
- Robust to outliers
- Good for seasonal data

---

### Machine Learning Methods

#### 13. Linear Regression

**Full Name**: Linear Regression Forecasting

**How It Works**:
- Treats time as a feature
- Fits a linear line through historical data
- Extrapolates the trend into the future
- Simple: forecast = intercept + slope × time

**Mathematical Formula**:
```
y(t) = β₀ + β₁ × t + ε

where:
- β₀: intercept
- β₁: slope
- t: time index
- ε: error term

Forecast: ŷ(t+h) = β₀ + β₁ × (t+h)
```

**Parameters**:
- `fit_intercept`: True (include intercept term)

**When to Use**:
- Clear linear trend
- Simple data with steady growth/decline
- Quick baseline forecast
- When trend is the main driver

**Advantages**:
- Very simple and fast
- Interpretable (slope and intercept)
- Good for linear trends
- Robust to outliers

**Limitations**:
- Cannot capture seasonality
- Assumes constant growth rate
- Poor for non-linear patterns

---

#### 14. KNN Regression

**Full Name**: K-Nearest Neighbors Regression

**How It Works**:
- Uses time as a feature
- Finds K most similar historical time periods
- Predicts future value based on neighbors
- For time series: predicts based on recent history

**Algorithm**:
```
1. For time t in future:
   - Find k nearest neighbors in training data
   - Average their values
   - Use as forecast
```

**Parameters**:
- `n_neighbors`: 5 (number of neighbors to use)

**When to Use**:
- Non-linear patterns
- When local patterns are important
- When distance-based similarity makes sense
- Small datasets

**Advantages**:
- Captures non-linear patterns
- Simple concept
- No complex parameters

**Limitations**:
- Can be slow for large datasets
- Sensitive to k parameter
- May not generalize well

---

#### 15. AR-NNET - Autoregressive Neural Network

**Full Name**: Autoregressive Neural Network

**How It Works**:
- Uses past values as features (autoregression)
- Trains a Multi-Layer Perceptron (MLP) neural network
- Learns patterns in historical data
- Forecasts by recursively applying the model

**Structure**:
```
Input: [y(t-l), y(t-l+1), ..., y(t-1)]
       ↓ (Neural Network)
Output: y(t)

For future steps:
   - Use predicted value as input for next prediction
   - Recursively forecast multiple steps ahead
```

**Parameters**:
- `hidden_layer_sizes`: (100,) (one hidden layer with 100 neurons)
- `max_iter`: 200 (maximum training iterations)
- Number of lags: auto-selected (min of 5 or half of data length)

**Network Architecture**:
- Input layer: lagged values
- Hidden layer(s): 100 neurons with activation
- Output layer: single prediction

**When to Use**:
- Non-linear patterns
- Complex relationships
- When traditional methods fail
- Sufficient data available

**Advantages**:
- Captures complex patterns
- Non-linear modeling
- Flexible structure

**Limitations**:
- Requires sufficient data
- Can overfit with small datasets
- Less interpretable
- May be slow to train

**Implementation Details**:
- Automatically determines number of lags
- Uses autoregressive approach (sequential predictions)
- Each future prediction uses previous predictions

---

### Baseline Methods

#### 16. Simple Seasonal Naive

**How It Works**:
- Uses the value from the same period in the previous season
- No trend or smoothing applied
- Pure seasonal replication

**Example**:
- For monthly data: Next January = Last January
- For daily data: Same day next week = Same day last week

**Mathematical Formula**:
```
F(t+h) = y(t+h - seasonal_periods × k)

where k is the smallest integer such that (t+h - seasonal_periods × k) ≤ t
```

**When to Use**:
- Strong seasonal patterns
- Stable seasonality
- Quick baseline
- Benchmark for other methods

**Advantages**:
- Extremely simple
- Fast
- Good for strong seasonality
- No parameters to tune

**Limitations**:
- Ignores trend
- Doesn't adapt to changes
- Poor when seasonality changes

---

#### 17. Weighted Seasonal Naive

**How It Works**:
- Combines multiple recent seasonal periods
- Uses weighted average instead of single period
- Gives more weight to recent seasons
- More adaptive than simple seasonal naive

**Approach**:
```
1. For each forecast period, find values from same seasonal position in:
   - Last season
   - 2 seasons ago
   - 3 seasons ago
   
2. Apply weights (more recent = higher weight)
   F(t+h) = Σ w(i) × y(t+h - i×seasonal_periods)
```

**Weights**:
- Linear weights: 0.1 (oldest) to 1.0 (most recent)
- Considers up to 3 previous seasons

**When to Use**:
- Seasonal data with gradual changes
- When seasonality evolves over time
- Better baseline than simple seasonal naive
- Benchmark improvement

**Advantages**:
- Accounts for changing seasonality
- Simple yet effective
- More flexible than simple version
- Better for evolving patterns

---

#### 18. Growth Seasonal Naive

**How It Works**:
- Extends seasonal naive by adding growth adjustment
- Calculates growth rate from historical data
- Applies growth factor to seasonal forecast
- Combines seasonality with trend

**Growth Calculation**:
```
Recent_avg = Mean of last seasonal period
Previous_avg = Mean of previous seasonal period
Growth_rate = (Recent_avg - Previous_avg) / Previous_avg

Forecast = Seasonal_value × (1 + Growth_rate × growth_factor)
```

**Parameters**:
- `growth_factor`: 1.0 (multiplier for growth rate)

**When to Use**:
- Seasonal data with trend
- Growing or declining seasonal patterns
- When you want to incorporate growth
- More realistic than pure seasonal naive

**Advantages**:
- Combines seasonality and growth
- Simple approach to trend
- Better for trending seasonal data
- Practical for business forecasting

---

#### 19. Simple AOA - Average of Averages

**How It Works**:
- Calculates multiple averages from different periods
- Combines them into a single forecast
- Uses simple average of all component averages

**Components Averaged**:
```
1. Seasonal average: Mean of last seasonal period
2. Overall average: Mean of entire dataset
3. Recent average: Mean of recent periods (last 7 periods)

Forecast = (Seasonal_avg + Overall_avg + Recent_avg) / 3
```

**When to Use**:
- When you want a balanced view
- Quick general forecasts
- Baseline method
- When no clear pattern exists

**Advantages**:
- Simple and stable
- Balances different perspectives
- Robust approach
- Good baseline

---

#### 20. Weighted AOA - Weighted Average of Averages

**How It Works**:
- Same concept as Simple AOA
- Uses different weights for each component
- Gives more importance to recent data
- More sophisticated weighting scheme

**Weighted Formula**:
```
Forecast = w₁×Recent_avg + w₂×Seasonal_avg + w₃×Overall_avg

Weights:
- Recent avg: 0.5 (50%)
- Seasonal avg: 0.3 (30%)
- Overall avg: 0.2 (20%)
```

**When to Use**:
- When you want to emphasize recent data
- More sophisticated than simple AOA
- Quick but thoughtful forecast
- Business applications

**Advantages**:
- Emphasizes recent patterns
- Balanced approach
- Practical weighting
- Good for most cases

---

#### 21. Growth AOA - Growth-Adjusted Average of Averages

**How It Works**:
- Calculates AOA with growth adjustment
- Detects growth trend from historical data
- Applies growth rate to AOA forecast
- Combines averages with trend projection

**Growth-Adjusted Forecast**:
```
1. Calculate base AOA (from seasonal and overall averages)
2. Calculate growth rate (from recent vs. previous seasonal periods)
3. Apply growth: Forecast = Base_AOA × (1 + Growth_rate × growth_factor)
```

**Parameters**:
- `growth_factor`: 1.0 (growth adjustment factor)

**When to Use**:
- When you want both stability and growth
- Seasonal data with trend
- Balanced long-term forecast
- Pragmatic business forecasting

**Advantages**:
- Incorporates growth trend
- Balanced approach (multiple averages)
- Realistic for trending data
- Combines best of both worlds

---

#### 22. Naive Random Walk

**How It Works**:
- Uses the last observed value
- Assumes no change from current value
- Simplest possible forecast method
- Pure baseline benchmark

**Formula**:
```
F(t+h) = y(t) for all h

Simply repeats the last value
```

**When to Use**:
- Absolute simplest baseline
- Benchmark for comparisons
- When no pattern exists
- Testing purposes

**Advantages**:
- Extremely simple
- Fastest method
- Useful benchmark
- No assumptions

**Limitations**:
- Ignores all patterns
- Usually performs worst
- Only useful as baseline
- Flat forecast

---

## Model Selection Guide

### By Data Characteristics

**No Trend, No Seasonality**:
- ✅ SES (Simple Exponential Smoothing)
- ✅ Moving Average
- ✅ Naive Random Walk

**Trend Only**:
- ✅ DES (Double Exponential Smoothing)
- ✅ Linear Regression
- ✅ Croston (for sparse demand)

**Seasonality Only**:
- ✅ Simple Seasonal Naive
- ✅ Weighted Seasonal Naive
- ✅ TES (Triple Exponential Smoothing)

**Trend + Seasonality**:
- ✅ TES (Triple Exponential Smoothing/Holt-Winters)
- ✅ SARIMA
- ✅ Prophet
- ✅ Auto ARIMA
- ✅ STLF
- ✅ TBATS

**Complex Patterns**:
- ✅ Prophet
- ✅ TBATS
- ✅ AR-NNET
- ✅ ETS
- ✅ Theta Method

**Intermittent/Sparse Data**:
- ✅ Croston's Method
- ✅ Naive Methods

**Business Applications**:
- ✅ Prophet (handles holidays/events)
- ✅ Growth Seasonal Naive
- ✅ AOA methods
- ✅ Weighted methods

### By Forecast Horizon

**Short-term (1-7 periods)**:
- SES, DES, Moving Average
- Seasonal Naive methods
- Linear Regression

**Medium-term (8-30 periods)**:
- TES, SARIMA, Auto ARIMA
- Prophet, ETS
- Theta, TBATS

**Long-term (30+ periods)**:
- Prophet
- SARIMA
- Growth methods
- AOA methods

### By Data Volume

**Small dataset (<50 points)**:
- Simple methods: SES, DES
- Naive methods
- Moving Average

**Medium dataset (50-500 points)**:
- Most methods work well
- TES, SARIMA, Prophet

**Large dataset (500+ points)**:
- All methods applicable
- Complex methods: AR-NNET, TBATS
- Can use more sophisticated approaches

---

## Common Parameters

### Seasonal Periods

- **Daily data**: 365 (yearly seasonality)
- **Weekly data**: 52 (yearly seasonality)
- **Monthly data**: 12 (yearly seasonality)
- **Quarterly data**: 4 (yearly seasonality)

### Forecast Horizon

- Short-term: 1-7 periods ahead
- Medium-term: 8-30 periods ahead
- Long-term: 30+ periods ahead

### Confidence Intervals

- Default: 95% confidence (alpha = 0.05)
- Contains true value 95% of the time
- Helps quantify forecast uncertainty

---

## Usage Example

```python
import pandas as pd
from utils.stat import ForecastingEngine

# Prepare your data
data = pd.Series([...])  # Your time series

# Initialize the forecasting engine
engine = ForecastingEngine(
    train=data,
    seasonal_periods=12,
    forecast_horizon=10
)

# Run a forecast
result = engine.get_prophet_forecast()

# Access results
forecast_values = result['forecast']
confidence_intervals = result['intervals']
fitted_params = result['params']
```

---

## Error Handling

All methods include comprehensive error handling:
- Returns NaN forecasts if method fails
- Logs errors for debugging
- Never crashes the entire system
- Allows fallback to other methods

---

## Dependencies

Required libraries:
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `statsmodels`: Statistical models (ETS, SARIMA, STL)
- `sktime`: Time series forecasting models
- `pmdarima`: Auto ARIMA
- `prophet`: Facebook Prophet
- `sklearn`: Machine learning models

---

## Summary

This forecasting suite provides 21 diverse methods covering:
- **Smoothing methods**: SES, DES, TES, Moving Average
- **ARIMA methods**: SARIMA, Auto ARIMA
- **Advanced statistical**: Prophet, Theta, Croston, ETS, TBATS, STLF
- **Machine learning**: Linear Regression, KNN, AR-NNET
- **Baseline methods**: Seasonal Naive variants, AOA variants, Naive

Each method has specific strengths and is suited for different data characteristics and use cases. The system is designed to be flexible, allowing easy experimentation with different forecasting approaches.

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: DemandForecasting Team

