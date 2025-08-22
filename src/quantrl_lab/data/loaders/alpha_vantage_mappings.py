from dataclasses import dataclass, field
from typing import Dict


# frozen = True for immutability
@dataclass(frozen=True)
class AlphaVantageColumnMapper:
    """Provides standardized column mappings for Alpha Vantage API
    responses."""

    standard: Dict[str, str] = field(
        default_factory=lambda: {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }
    )

    adjusted_daily: Dict[str, str] = field(
        default_factory=lambda: {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. adjusted close": "adj_close",
            "6. volume": "volume",
            "7. dividend amount": "dividend",
            "8. split coefficient": "split_coeff",
        }
    )

    def get_mapping(self, interval: str, adjusted: bool) -> Dict[str, str]:
        """
        Selects the appropriate column mapping based on the data
        interval and whether the data is adjusted.

        Args:
            interval (str): The time interval for the data (e.g., "1d", "5min").
            adjusted (bool): Flag indicating if the data is adjusted for splits and dividends.

        Returns:
            Dict[str, str]: The corresponding column mapping dictionary.
        """
        if interval == "1d" and adjusted:
            return self.adjusted_daily
        return self.standard


# Create a single instance for use across the application
ALPHA_VANTAGE_COLUMN_MAPPER = AlphaVantageColumnMapper()
