import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Optional

# Assuming BaseObservationStrategy is in base.py. Adjust if their naming is slightly different!
from .base import BaseObservationStrategy


class VolatilityObservationStrategy(BaseObservationStrategy):
    """
    Observation strategy that calculates rolling volatility (Standard Deviation)
    alongside the raw features to give the RL agent context on market turbulence.
    """

    def __init__(self, window_size: int = 20, features: Optional[List[str]] = None):
        self.window_size = window_size
        # Default features if none provided
        self.features = features or ["open", "high", "low", "close", "volume"]

        # We add 1 to the shape because we are appending the volatility feature
        self.num_features = len(self.features) + 1

    def get_observation_space(self) -> spaces.Box:
        """Defines the shape and bounds of the observation matrix."""
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.num_features),
            dtype=np.float32
        )

    def get_observation(self, current_step: int, data: pd.DataFrame) -> np.ndarray:
        """
        Extracts the window of data, calculates rolling volatility,
        and returns the combined matrix.
        """
        start_idx = current_step - self.window_size

        # Extract the base features
        window_data = data[self.features].iloc[start_idx:current_step].copy()

        # Calculate Rolling Volatility on the 'close' price
        # Fill NaNs with 0 for the beginning of the dataset
        volatility = window_data["close"].rolling(window=5).std().fillna(0)

        # Append the new volatility column to the observation
        window_data["volatility"] = volatility

        # Return as a float32 numpy array for the Neural Network
        return window_data.to_numpy(dtype=np.float32)
