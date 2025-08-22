from dataclasses import dataclass, field
from typing import Dict

from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


@dataclass(frozen=True)
class AlpacaMappings:
    """Provides standardized mappings for Alpaca API data."""

    timeframes: Dict[str, TimeFrame] = field(
        default_factory=lambda: {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day,
            "1w": TimeFrame.Week,
            "1M": TimeFrame.Month,
        }
    )

    ohlcv_columns: Dict[str, str] = field(
        default_factory=lambda: {
            "symbol": "Symbol",
            "timestamp": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "trade_count": "Trade_count",
            "vwap": "VWAP",
        }
    )

    def get_timeframe(self, timeframe: str) -> TimeFrame:
        """
        Converts a string timeframe to an Alpaca TimeFrame object.

        Args:
            timeframe (str): The string representation of the timeframe (e.g., "1d", "1h").

        Returns:
            TimeFrame: The corresponding Alpaca TimeFrame object. Defaults to TimeFrame.Day.
        """
        return self.timeframes.get(timeframe, TimeFrame.Day)


# Create a single instance for use across the application
ALPACA_MAPPINGS = AlpacaMappings()
