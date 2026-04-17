from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List

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
        Generate discrete trading signals (Buy/Sell/Hold) for the given
        data.

        This is useful for backtesting distinct entry/exit rules.

        Args:
            data (pd.DataFrame): Input market data

        Returns:
            pd.Series: Generated trading signals (1, -1, 0)
        """
        raise NotImplementedError

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores (e.g., -1.0 to 1.0) for the
        given data.

        This is useful for RL observation features and signal analysis (IC).
        By default, it raises NotImplementedError, but subclasses should implement this
        to support advanced signal discovery.

        Args:
            data (pd.DataFrame): Input market data

        Returns:
            pd.Series: Continuous alpha scores
        """
        raise NotImplementedError("This strategy does not support continuous score generation.")

    @staticmethod
    def _rolling_zscore(series: "pd.Series", window: int) -> "pd.Series":
        """Normalize a series to roughly [-1, 1] via rolling Z-score /
        3."""
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return ((series - mean) / (std + 1e-9) / 3.0).clip(-1.0, 1.0)

    @abstractmethod
    def get_required_columns(self) -> list:
        """
        Return the list of DataFrame columns this strategy needs.

        Called after instantiation to validate that all indicator-generated
        columns were successfully resolved before signals are generated.

        Returns:
            list: List of required column names (actual column strings, not
                parameter names).
        """
        raise NotImplementedError

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve indicator-generated column names into constructor
        kwargs.

        A-7: Each strategy owns its own column-wiring logic so that
        ``AlphaRunner._resolve_strategy_args`` is a simple one-line dispatch
        rather than a long per-strategy if-chain.  Override in subclasses
        that need multi-column wiring (MACD, Bollinger Bands, Stochastic,
        ADX).  The default implementation handles the common single-column
        case (``indicator_col``).

        Args:
            new_cols: Columns added to the DataFrame by the indicator step.
            current_params: Strategy params already supplied by the user
                (never overridden).

        Returns:
            Dict of additional kwargs to pass to the strategy constructor.
        """
        resolved: Dict[str, Any] = {}
        if "indicator_col" not in current_params and new_cols:
            resolved["indicator_col"] = new_cols[0]
        return resolved

    def validate_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Return a list of required columns that are missing from *data*.

        A-2: Enables ``AlphaRunner`` to warn early when column wiring
        produced None values, rather than silently generating all-HOLD
        signals.

        Args:
            data (pd.DataFrame): DataFrame after indicator calculation.

        Returns:
            List of missing column names (empty list when all present).
        """
        return [col for col in self.get_required_columns() if col is not None and col not in data.columns]
