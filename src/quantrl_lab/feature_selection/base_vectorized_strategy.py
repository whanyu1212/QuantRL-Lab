from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class SignalType(Enum):
    # Just 3 possible signals for simplicity
    BUY = 1
    SELL = -1
    HOLD = 0


class VectorizedTradingStrategy(ABC):
    """
    Base strategy class for vectorized trading strategies.

    We will be using the results from vectorized trading strategies to
    decide on the feature selection process.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals for the given data.

        Args:
            data (pd.DataFrame): Input market data

        Returns:
            pd.Series: Generated trading signals
        """
        raise NotImplementedError

    @abstractmethod
    def get_required_columns(self) -> list:
        """
        Return list of required columns for this strategy.

        Raises:
            NotImplementedError: If not implemented

        Returns:
            list: List of required column names
        """
        raise NotImplementedError

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that required columns exist in data.

        Args:
            data (pd.DataFrame): Input market data

        Raises:
            ValueError: If required columns are missing

        Returns:
            bool: True if validation passes, False otherwise
        """
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return True
