from dataclasses import dataclass, field
from typing import Dict


# frozen = True for immutability
@dataclass(frozen=True)
class AlphaVantageColumnMapper:
    """Provides standardized column mappings for Alpha Vantage API
    responses."""

    standard: Dict[str, str] = field(
        default_factory=lambda: {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume",
        }
    )

    adjusted_daily: Dict[str, str] = field(
        default_factory=lambda: {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. adjusted close": "Adj_close",
            "6. volume": "Volume",
            "7. dividend amount": "Dividend",
            "8. split coefficient": "Split_coeff",
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
