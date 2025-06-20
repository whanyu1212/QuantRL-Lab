from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from quantrl_lab.data.indicators.indicator_registry import IndicatorRegistry
from quantrl_lab.data.indicators.technical_indicators import *  # noqa: F401, F403
from quantrl_lab.data.loaders.alpaca_loader import AlpacaDataLoader
from quantrl_lab.data.loaders.alpha_vantage_loader import AlphaVantageDataLoader
from quantrl_lab.data.loaders.yfinance_loader import YfinanceDataloader


class DataSourceRegistry:

    # default configuration for data sources
    # can be overridden by passing sources dict to constructor
    DEFAULT_SOURCES = {
        "primary_source": AlpacaDataLoader,
        "fundamental_source": YfinanceDataloader,
        "macro_source": AlphaVantageDataLoader,
        "news_source": AlpacaDataLoader,
        # "analyst_source": FMPDataLoader,
        # "calendar_events_source": FMPDataLoader,
        # "sector_performance_source": FMPDataLoader,
    }

    def __init__(self, sources=None, **kwargs):
        """
        Initialize with configured data sources.

        Args:
            sources: Dictionary mapping source types to data source classes
            **kwargs: Individual source overrides
        """
        # Start with defaults
        self.sources = self.DEFAULT_SOURCES.copy()

        # Update with sources dict if provided
        if sources:
            self.sources.update(sources)

        # Allow individual overrides via kwargs
        self.sources.update(kwargs)

        # Initialize source instances
        self.data_sources = {}
        for source_type, source_class in self.sources.items():
            if source_class:
                if source_class.__name__ not in self.data_sources:
                    self.data_sources[source_class.__name__] = source_class()
                setattr(self, source_type, self.data_sources[source_class.__name__])

    def get_historical_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:

        # Use primary source to fetch historical data
        return self.primary_source.get_historical_ohlcv_data(
            symbols=symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            **kwargs,
        )

    def get_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[Union[str, Dict]]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Add technical indicators to existing OHLCV DataFrame.

        Args:
            df (pd.DataFrame): raw OHLCV data
            indicators (Optional[List[Union[str, Dict]]], optional): Defaults to None.

        Raises:
            ValueError: if input DataFrame is empty
            ValueError: if required columns are missing

        Returns:
            pd.DataFrame: DataFrame with added technical indicators
        """

        # Validate input DataFrame
        if df.empty:
            raise ValueError("Input DataFrame is empty. Technical indicators cannot be added.")

        # Check for required columns (case-insensitive)
        column_check = {col.lower(): col for col in df.columns}
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = []

        for req_col in required_cols:
            if req_col not in column_check and req_col.upper() not in column_check:
                missing_cols.append(req_col)

        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")

        result = df.copy()

        # Return original if no indicators specified
        if not indicators:
            return result

        available_indicators = set(IndicatorRegistry.list_all())

        for indicator_name in indicators:
            if indicator_name not in available_indicators:
                logger.warning(f"Indicator '{indicator_name}' not found in registry. Skipping.")
                continue
            try:
                # Extract custom parameters for this indicator if provided
                # Look for {indicator_name}_params in the kwargs
                custom_params = kwargs.get(f"{indicator_name}_params", {})

                # Apply the indicator with custom parameters
                logger.debug(f"Applying {indicator_name} with params: {custom_params}")
                result = IndicatorRegistry.apply(indicator_name, result, **custom_params)

            except Exception as e:
                logger.error(f"Failed to apply indicator '{indicator_name}' - {e}")

        return result
