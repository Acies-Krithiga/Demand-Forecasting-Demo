import tensorflow as tf
from typing import List, Dict, Any, Optional


def build_lstm(input_shape, hidden_size=100, output_size=1):
    """
    Build a simple LSTM model.
    
    Args:
        input_shape: Tuple (n_input, n_features) - sequence length and number of features
        hidden_size: Number of LSTM units (default: 100)
        output_size: Number of output units (default: 1)
    
    Returns:
        Compiled Keras Sequential model
    """
    n_input, n_features = input_shape
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(hidden_size, activation='relu', input_shape=(n_input, n_features)))
    model.add(tf.keras.layers.Dense(output_size))
    model.compile(optimizer='adam', loss='mse')
    return model




