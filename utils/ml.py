import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor
import logging
import warnings
from datetime import datetime, timedelta
import os
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MLForecastingEngine:
    """
    Machine Learning Forecasting Engine supporting LightGBM, XGBoost, and Random Forest.
    """
    
    def __init__(self, 
                 train_data: pd.DataFrame,
                 validation_features: pd.DataFrame,
                 validation_targets: pd.Series,
                 feature_columns: List[str],
                 target_column: str,
                 level_columns: List[str],
                 date_column: str,
                 random_state: int = 42):
        """
        Initialize the ML Forecasting Engine.
        
        Args:
            train_data: Training data DataFrame
            validation_features: Validation features DataFrame (for prediction only)
            validation_targets: Validation targets Series (actual values for comparison)
            feature_columns: List of feature column names
            target_column: Target variable column name
            level_columns: Level columns (e.g., store_id, item_id)
            date_column: Date column name
            random_state: Random state for reproducibility
        """
        self.train_data = train_data.copy()
        self.validation_features = validation_features.copy()
        self.validation_targets = validation_targets.copy()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.level_columns = level_columns
        self.date_column = date_column
        self.random_state = random_state
        
        # Initialize models
        self.models = {}
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare training and validation data for ML models."""
        logger.info("Preparing data for ML models...")
        
        # Data is already processed by feature_selection.py, just extract features and target
        self.X_train = self.train_data[self.feature_columns]
        self.y_train = self.train_data[self.target_column]
        
        self.X_val = self.validation_features[self.feature_columns]
        self.y_val = self.validation_targets.values
        
        logger.info(f"Training data shape: {self.X_train.shape}")
        logger.info(f"Validation data shape: {self.X_val.shape}")
    
    def train_lightgbm(self, params: Dict = None) -> Dict:
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not available. Please install with: pip install lightgbm")
            return None
        
        logger.info("Training LightGBM model...")
        
        # Default parameters
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        }
        
        if params:
            default_params.update(params)
        
        # Create LightGBM datasets
        train_dataset = lgb.Dataset(self.X_train, label=self.y_train)
        
        # Train model
        model = lgb.train(
            default_params,
            train_dataset,
            num_boost_round=1000,
            callbacks=[lgb.log_evaluation(0)]
        )
        # Make predictions on validation data
        val_pred = model.predict(self.X_val)
        
        # Store model
        self.models['lightgbm'] = model
        
        logger.info("LightGBM training completed.")
        
        return {
            'model': model,
            'val_predictions': val_pred
        }
    
    def train_xgboost(self, params: Dict = None) -> Dict:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available. Please install with: pip install xgboost")
            return None
        
        logger.info("Training XGBoost model...")
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbosity': 0
        }
        
        if params:
            default_params.update(params)
        
        # Train model
        model = xgb.XGBRegressor(**default_params)
        model.fit(self.X_train, self.y_train, verbose=False)
        
        # Make predictions on validation data
        val_pred = model.predict(self.X_val)
        
        # Store model
        self.models['xgboost'] = model
        
        logger.info("XGBoost training completed.")
        
        return {
            'model': model,
            'val_predictions': val_pred
        }
    
    def train_random_forest(self, params: Dict = None) -> Dict:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        # Train model
        model = RandomForestRegressor(**default_params)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions on validation data
        val_pred = model.predict(self.X_val)
        
        # Store model
        self.models['random_forest'] = model
        
        logger.info("Random Forest training completed.")
        
        return {
            'model': model,
            'val_predictions': val_pred
        }
    
    def train_all_models(self, model_params: Dict = None) -> Dict:
        """Train all available models."""
        logger.info("Training all ML models...")
        
        if model_params is None:
            model_params = {}
        
        results = {}
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_params = model_params.get('lightgbm', {})
            results['lightgbm'] = self.train_lightgbm(lgb_params)
        
        # Train XGBoost
        if XGBOOST_AVAILABLE:
            xgb_params = model_params.get('xgboost', {})
            results['xgboost'] = self.train_xgboost(xgb_params)
        
        # Train Random Forest
        rf_params = model_params.get('random_forest', {})
        results['random_forest'] = self.train_random_forest(rf_params)
        
        return results
    
    def get_feature_importance(self, model_name: str) -> pd.DataFrame:
        """Get feature importance for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'lightgbm':
            importance = model.feature_importance(importance_type='gain')
        elif model_name == 'xgboost':
            importance = model.feature_importances_
        elif model_name == 'random_forest':
            importance = model.feature_importances_
        else:
            raise ValueError(f"Feature importance not supported for {model_name}")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'lightgbm':
            return model.predict(X)
        elif model_name in ['xgboost', 'random_forest']:
            return model.predict(X)
        else:
            raise ValueError(f"Prediction not supported for {model_name}")
