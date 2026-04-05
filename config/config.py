# config.py

DATA_PATHS = {
    "sales_fact": "data/sales_fact.csv",
    "price_fact": "data/price_fact.csv",
    "calendar_dim": "data/calendar_dim.csv",
    "inventory_fact": "data/inventory_fact.csv",
    "product_dim": "data/product_dim.csv",
    "location_dim": "data/location_dim.csv",
    "external_data": "data/external_data.csv",
    "promotion_fact": "data/promotion_fact.csv"
}

# Segmentation Configuration Parameters
SEGMENTATION_CONFIG = {
    # Core data configuration - USER CONFIGURABLE
    "level_columns": ["store_id", "item_id"],  # Columns that define the segmentation level
    "date_column": "date",  # Date column name
    "target_column": "units_sold",  # Target variable for analysis
    
    # Intermittency parameters
    "intermittency": {
        "threshold": 0.5
    },
    
    # Product Life Cycle parameters
    "product_life_cycle": {
        "disco_periods": 4,
        "new_launch_periods": 90
    },
    
    # Series Length Calculation parameters
    "series_length_calculation": {
        "frequency": "daily"  # 'daily'|'weekly'|'monthly'|'quarterly'
    },
    
    # Variability Segmentation parameters
    "variability_segmentation": {
        "cov_thresholds": (0.2,),
        "cov_levels": ("X", "Y")
    },
    
    # Volume Segmentation parameters
    "volume_segmentation": {
        "vol_thresholds": (0.5,),
        "vol_levels": ("A", "B"),
        "group_columns": []  # Additional grouping columns for volume analysis
    },
    
    # Trend Segmentation parameters
    "trend_segmentation": {
        "trend_threshold": 0.01
    },
    
    # Seasonality Detection parameters
    "seasonality_detection": {
        "useACFForSeasonality": True,
        "ACFDiff": 0,
        "ACFLowerThreshold": -0.10,
        "ACFUpperThreshold": 0.90,
        "seasonality_threshold": 0.64,
        "ACFSkipLags": 11,
        "RequiredACFLags": "365",
        "frequency": "daily",  # 'daily'|'weekly'|'monthly'|'quarterly'
        "max_detected_periods": 3
    }
}





# Data Preparation Configuration
DATA_PREP_CONFIG = {
    # Core data configuration - USER CONFIGURABLE
    "level_columns": ["store_id", "item_id"],  # Columns that define the forecast level
    "date_column": "date",  # Date column name
    "target_column": "units_sold",  # Target variable for forecasting
    
    # Join configurations for different data sources
    "join_config": {
        "promotions": {
            "left_on": ["date", "store_id", "item_id"], 
            "right_on": ["date", "store_id", "item_id"]
        },
        "calendar": {
            "left_on": "date", 
            "right_on": "date"
        },
        "price": {
            "left_on": ["store_id", "item_id", "wm_yr_wk"], 
            "right_on": ["store_id", "item_id", "wm_yr_wk"]
        },
        "location": {
            "left_on": "store_id", 
            "right_on": "store_id"
        },
        "external": {
            "left_on": ["date", "region"], 
            "right_on": ["date", "region"]
        }
    }
}

FEATURE_CREATION_CONFIG = {
    "level_columns": ["store_id", "item_id"],  
    "date_column": "date",  
    "target_column": "units_sold",
    
    # Lag Features Configuration
    "num_lags": 7,  # Number of lag periods to create (0 to disable lag features)
    
    # Rolling Window Features Configuration
    "rolling_mean_windows": [7, 14, 30],  # Windows for rolling mean (empty list to disable)
    "rolling_std_windows": [7, 14],  # Windows for rolling standard deviation (empty list to disable)
    "rolling_min_windows": [7, 14],  # Windows for rolling minimum (empty list to disable)
    "rolling_max_windows": [7, 14],  # Windows for rolling maximum (empty list to disable)
    
    # Date Features Configuration
    "date_features": {
        "day_of_week": True,  # Extract day of week (0-6)
        "month": True,  # Extract month (1-12)
        "quarter": True,  # Extract quarter (1-4)
        "day_of_month": True,  # Extract day of month (1-31)
        "week_of_year": True,  # Extract week of year (1-52/53)
        "is_weekend": True,  # Binary indicator for weekend
        "is_month_start": True,  # Binary indicator for month start
        "is_month_end": True,  # Binary indicator for month end
        "is_quarter_start": True,  # Binary indicator for quarter start
        "is_quarter_end": True  # Binary indicator for quarter end
    },
    
    # Cyclical Encoding for Date Features (if True, uses sin/cos encoding)
    "cyclical_encoding": True
}
    




# Statistical Models Configuration
STATISTICAL_MODELS_CONFIG = {
    # Core data configuration - USER CONFIGURABLE (same as segmentation)
    "level_columns": ["store_id", "item_id"],  # Columns that define the level
    "date_column": "date",  # Date column name
    "target_column": "units_sold",  # Target variable for forecasting
    
    # Validation and forecasting cycles - USER CONFIGURABLE
    "validation_cycles": {
        "daily": 365,    # Validation period for daily data (1 year)
        "weekly": 52,    # Validation period for weekly data (1 year)
        "monthly": 12,   # Validation period for monthly data (1 year)
        "quarterly": 4   # Validation period for quarterly data (1 year)
    },
    "forecast_cycles": {
        "daily": 365,    # Forecast horizon for daily data (1 year)
        "weekly": 52,    # Forecast horizon for weekly data (1 year)
        "monthly": 12,   # Forecast horizon for monthly data (1 year)
        "quarterly": 4   # Forecast horizon for quarterly data (1 year)
    },
    
    # Model parameters
    "frequency": "daily",  # Default frequency
    "confidence_interval_alpha": 0.05,
    "minimum_data_points": 30,  # Minimum data points required for forecasting
    
    # Baseline models configuration - USER CONFIGURABLE
    # List of baseline models to use for experimentation
    # Available models: 'Moving Average', 'Weighted Snaive', 'SES', 'TES', 'Growth Snaive', 
    # 'Weighted AOA', 'sARIMA', 'STLF', 'Theta', 'Simple Snaive', 'Auto ARIMA', 'Prophet', 
    # 'Croston', 'ETS', 'TBATS', 'Linear Regression', 'KNN Regression', 'AR-NNET', 
    # 'Simple AOA', 'Growth AOA', 'DES', 'Naive Random Walk'
    "baseline_models": ["Moving Average", "Weighted Snaive"],  # User can modify this list
    
    # Algorithm mapping (can be extended by users)
    "algorithm_mapping": {
        'SES': 'get_ses_forecast',
        'Moving Average': 'get_moving_avg_forecast',
        'TES': 'get_tes_forecast',
        'Weighted Snaive': 'get_weighted_snaive_forecast',
        'Growth Snaive': 'get_growth_snaive_forecast',
        'Weighted AOA': 'get_weighted_aoa_forecast',
        'sARIMA': 'get_sarima_forecast',
        'STLF': 'get_stlf_forecast',
        'Theta': 'get_theta_forecast',
        'Simple Snaive': 'get_simple_snaive_forecast',
        'Auto ARIMA': 'get_auto_arima_forecast',
        'Prophet': 'get_prophet_forecast',
        'Croston': 'get_croston_forecast',
        'ETS': 'get_ets_forecast',
        'TBATS': 'get_tbats_forecast',
        'Linear Regression': 'get_linear_regression_forecast',
        'KNN Regression': 'get_knn_regression_forecast',
        'AR-NNET': 'get_arnnet_forecast',
        'Simple AOA': 'get_simple_aoa_forecast',
        'Growth AOA': 'get_growth_aoa_forecast',
        'DES': 'get_des_forecast',
        'Naive Random Walk': 'get_naive_random_walk_forecast'
    }
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    # Core data configuration - USER CONFIGURABLE
    "method": "shap",
    "level_columns": ["store_id", "item_id"],  # Columns that define the level
    "date_column": "date",  # Date column name
    "target_column": "units_sold",  # Target variable for forecasting
    
    # Feature selection parameters - USER CONFIGURABLE
    "validation_cycles": 1,  # Number of cycles to use for validation
    "data_frequency": "daily",  # Data frequency: 'daily', 'weekly', 'monthly', 'quarterly'
    "feature_importance_threshold": 0.01,  # Minimum importance score to keep a feature
    "random_state": 42,  # Random state for reproducibility
    
    # Random Forest parameters - USER CONFIGURABLE
    "random_forest": {
        "n_estimators": 100,  # Number of trees in the forest
        "max_depth": None,  # Maximum depth of trees (None for no limit)
        "min_samples_split": 2,  # Minimum samples required to split a node
        "min_samples_leaf": 1,  # Minimum samples required at a leaf node
        "max_features": "sqrt",  # Number of features to consider for best split
        "n_jobs": -1  # Number of jobs to run in parallel (-1 for all cores)
    },
    
    # Data processing parameters - USER CONFIGURABLE
    "data_processing": {
        "chunk_size": 100000,  # Size of chunks when reading large files
        "categorical_threshold": 20,  # Max unique values to consider as categorical
        "categorical_ratio": 0.1  # Max ratio of unique values to total rows for categorical
    },

   
     "lasso": {
        "alphas": None,  # None lets LassoCV choose automatically
        "cv": 5,
        "max_iter": 10000,
        "coefficient_threshold": 1e-6,
        "top_k": None
    }
    ,
    # Output configuration - USER CONFIGURABLE
    "output": {
        "save_feature_importance": True,  # Save feature importance to CSV (optional)
        "output_dir": "data/outputs"  # Directory to save outputs
    }
}


# Machine Learning Models Configuration
ML_MODELS_CONFIG = {
    # Core data configuration - USER CONFIGURABLE (same as statistical)
    "level_columns": ["store_id", "item_id"],  # Columns that define the level
    "date_column": "date",  # Date column name
    "target_column": "units_sold",  # Target variable for forecasting
    
    # Validation and forecasting cycles - USER CONFIGURABLE
    "validation_cycles": {
        "daily": 365,    # Validation period for daily data (1 year)
        "weekly": 52,    # Validation period for weekly data (1 year)
        "monthly": 12,   # Validation period for monthly data (1 year)
        "quarterly": 4   # Validation period for quarterly data (1 year)
    },
    "forecast_cycles": {
        "daily": 365,    # Forecast horizon for daily data (1 year)
        "weekly": 52,    # Forecast horizon for weekly data (1 year)
        "monthly": 12,   # Forecast horizon for monthly data (1 year)
        "quarterly": 4   # Forecast horizon for quarterly data (1 year)
    },
    
    # Model parameters
    "frequency": "daily",  # Default frequency
    "minimum_data_points": 30,  # Minimum data points required for forecasting
    "random_state": 42,  # Random state for reproducibility
    
    # ML Model parameters - USER CONFIGURABLE
    "model_params": {
        "lightgbm": {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "num_boost_round": 1000
        },
        "xgboost": {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": 0
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1
        }
    },
    
    # Available ML algorithms
    "available_algorithms": ["lightgbm", "xgboost", "random_forest"]
}

