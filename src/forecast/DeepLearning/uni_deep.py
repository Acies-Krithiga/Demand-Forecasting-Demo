import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils.deep import build_lstm
from config.config import SEGMENTATION_CONFIG

logger = logging.getLogger(__name__)


class UniDeepForecaster:
    
    def __init__(self, 
                 df: pd.DataFrame,
                 level_columns: Optional[List[str]] = None,
                 date_column: Optional[str] = None,
                 target_column: Optional[str] = None,
                 sequence_length: int = 365,
                 hidden_size: int = 100,
                 epochs: int = 50,
                 batch_size: int = 32):
        if level_columns is None:
            self.level_columns = SEGMENTATION_CONFIG.get("level_columns", ["store_id", "item_id"])
        else:
            self.level_columns = level_columns
            
        if date_column is None:
            self.date_column = SEGMENTATION_CONFIG.get("date_column", "date")
        else:
            self.date_column = date_column
            
        if target_column is None:
            self.target_column = SEGMENTATION_CONFIG.get("target_column", "units_sold")
        else:
            self.target_column = target_column
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = {}
        self.scalers = {}
        self.intersection_data = {}
        self._prepare_data(df)
        self._prepare_intersections()
    
    def _prepare_data(self, df: pd.DataFrame):
        """Prepare the DataFrame for processing."""
        self.df = df.copy()
        
        # Convert date column to datetime
        if self.date_column in self.df.columns:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column])
        else:
            raise ValueError(f"Date column '{self.date_column}' not found in data")
        
        # Check if target column exists
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        # Check if level columns exist
        missing_cols = [col for col in self.level_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Level columns not found in data: {missing_cols}")
        
        # Sort by date
        self.df = self.df.sort_values([*self.level_columns, self.date_column])
        
        logger.info(f"Prepared {len(self.df)} rows of data")
    
    def _prepare_intersections(self):
        """Get unique intersections from the data."""
        self.intersections = self.df[self.level_columns].drop_duplicates().reset_index(drop=True)
        logger.info(f"Found {len(self.intersections)} unique intersections")
    
    def _filter_intersection_data(self, intersection_values: Dict[str, Any]) -> pd.DataFrame:
        # Build filter conditions
        filter_conditions = []
        for col_name, col_value in intersection_values.items():
            filter_conditions.append(self.df[col_name] == col_value)
        
        # Apply all filter conditions
        if len(filter_conditions) == 1:
            data = self.df[filter_conditions[0]].copy()
        else:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            data = self.df[combined_filter].copy()
        
        # Keep only date and target column
        data = data[[self.date_column, self.target_column]].copy()
        
        # Sort by date
        data = data.sort_values(self.date_column)
        
        # Remove duplicates if any (keep last)
        data = data.drop_duplicates(subset=[self.date_column], keep='last')
        
        # Reset index
        data = data.reset_index(drop=True)
        
        return data
    
    def _create_sequences(self, data: pd.Series, scaler: MinMaxScaler) -> tuple:
        scaled_data = scaler.transform(data.values.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i])
            y.append(scaled_data[i])
        
        return np.array(X), np.array(y)
    
    def _train_model_for_intersection(self, intersection_values: Dict[str, Any]) -> bool:
        intersection_str = "-".join([f"{k}:{v}" for k, v in intersection_values.items()])
        
        try:
            # Filter data for this intersection
            data = self._filter_intersection_data(intersection_values)
            
            if len(data) < self.sequence_length + 10:  # Need minimum data points
                logger.warning(f"Insufficient data for intersection {intersection_str}: {len(data)} rows")
                return False
            
            # Extract target values
            target_values = data[self.target_column].values
            
            # Initialize scaler and fit on training data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(target_values.reshape(-1, 1))
            
            # Create sequences
            X, y = self._create_sequences(pd.Series(target_values), scaler)
            
            if len(X) == 0:
                logger.warning(f"No sequences created for intersection {intersection_str}")
                return False
            
            # Reshape X for LSTM: (samples, time_steps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build and train model
            model = build_lstm(
                input_shape=(self.sequence_length, 1),
                hidden_size=self.hidden_size,
                output_size=1
            )
            
            # Train the model
            logger.info(f"Training model for intersection {intersection_str} with {len(X)} samples")
            model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                verbose=0  # Set to 1 for progress
            )
            
            # Store model, scaler, and data
            self.models[intersection_str] = model
            self.scalers[intersection_str] = scaler
            self.intersection_data[intersection_str] = data
            
            logger.info(f"Successfully trained model for intersection {intersection_str}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for intersection {intersection_str}: {str(e)}")
            return False
    
    def train_all_models(self):
        logger.info(f"Training models for {len(self.intersections)} intersections...")
        
        successful = 0
        failed = 0
        
        for idx, row in self.intersections.iterrows():
            intersection_values = {col: row[col] for col in self.level_columns}
            
            if self._train_model_for_intersection(intersection_values):
                successful += 1
            else:
                failed += 1
            
            # Progress update
            if (idx + 1) % 10 == 0 or idx == len(self.intersections) - 1:
                logger.info(f"Progress: {idx + 1}/{len(self.intersections)} (Success: {successful}, Failed: {failed})")
        
        logger.info(f"Training complete: {successful} successful, {failed} failed")
        return successful, failed



    
    def predict_future(self, 
                      intersection_values: Dict[str, Any],
                      n_periods: int = 30,
                      start_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        intersection_str = "-".join([f"{k}:{v}" for k, v in intersection_values.items()])
        
        if intersection_str not in self.models:
            raise ValueError(f"No trained model found for intersection: {intersection_str}")
        
        model = self.models[intersection_str]
        scaler = self.scalers[intersection_str]
        data = self.intersection_data[intersection_str]
        
        # Get last sequence_length values
        last_values = data[self.target_column].values[-self.sequence_length:]
        
        # Scale the last values
        scaled_last = scaler.transform(last_values.reshape(-1, 1)).flatten()
        
        # Prepare input sequence
        input_sequence = scaled_last.reshape(1, self.sequence_length, 1)
        
        # Generate predictions
        predictions = []
        current_sequence = scaled_last.copy()
        
        for _ in range(n_periods):
            # Predict next value
            next_pred = model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence: remove first, add prediction
            current_sequence = np.append(current_sequence[1:], next_pred[0, 0])
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Create date range
        if start_date is None:
            last_date = data[self.date_column].iloc[-1]
            start_date = last_date + pd.Timedelta(days=1)
        
        # Generate future dates (daily)
        future_dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
        
        # Create result DataFrame
        result = pd.DataFrame({
            self.date_column: future_dates,
            f'{self.target_column}_predicted': predictions
        })
        
        # Add intersection columns
        for col, val in intersection_values.items():
            result[col] = val
        
        return result
    
    def predict_all_future(self, n_periods: int = 30) -> pd.DataFrame:
        all_predictions = []
        
        for intersection_str in self.models.keys():
            # Parse intersection values from string
            parts = intersection_str.split("-")
            intersection_values = {}
            for part in parts:
                key, value = part.split(":")
                intersection_values[key] = value
            
            try:
                predictions = self.predict_future(intersection_values, n_periods=n_periods)
                all_predictions.append(predictions)
            except Exception as e:
                logger.error(f"Error predicting for {intersection_str}: {str(e)}")
        
        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load your DataFrame
    project_root = Path(__file__).resolve().parents[3]
    df = pd.read_csv(project_root / "data" / "outputs" / "df_feat_selected.csv")
    
    # Initialize forecaster with DataFrame
    forecaster = UniDeepForecaster(
        df=df,
        sequence_length=365,
        hidden_size=100,
        epochs=50,
        batch_size=32
    )
    
    # Train all models
    forecaster.train_all_models()
    
    # Make predictions for all intersections
    predictions = forecaster.predict_all_future(n_periods=30)
    print(f"\nGenerated {len(predictions)} predictions")
    print(predictions.head())
