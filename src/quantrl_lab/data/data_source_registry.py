from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

from quantrl_lab.data.loaders.alpaca_loader import AlpacaDataLoader
from quantrl_lab.data.loaders.alpha_vantage_loader import AlphaVantageDataLoader  # noqa: F401
from quantrl_lab.data.loaders.yfinance_loader import YfinanceDataloader  # noqa: F401


class DataSourceRegistry:

    # default configuration for data sources
    # can be overridden by passing sources dict to constructor
    DEFAULT_SOURCES = {
        "primary_source": AlpacaDataLoader,
        # "fundamental_source": YfinanceDataloader,
        # "macro_source": AlphaVantageDataLoader,
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
        """
        Fetch historical OHLCV data from the primary data source.

        Args:
            symbols (Union[str, List[str]]): Stock symbol(s) to fetch data for.
            start (Union[str, datetime]): Start date for the data.
            end (Optional[Union[str, datetime]], optional): End date for the data. Defaults to None.
            timeframe (str, optional): Timeframe for the data. Defaults to "1d".

        Returns:
            pd.DataFrame: Historical OHLCV data.
        """

        # Use primary source to fetch historical data
        # Handle different parameter naming conventions for different data sources
        primary_source_class = type(self.primary_source).__name__

        if primary_source_class == "AlphaVantageDataLoader":
            # Alpha Vantage uses start_date, end_date, interval
            return self.primary_source.get_historical_ohlcv_data(
                symbols=symbols,
                start_date=start,
                end_date=end,
                interval=timeframe,
                **kwargs,
            )
        else:
            # Alpaca and other sources use start, end, timeframe
            return self.primary_source.get_historical_ohlcv_data(
                symbols=symbols,
                start=start,
                end=end,
                timeframe=timeframe,
                **kwargs,
            )

    def get_news_data(
        self,
        symbols: str,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get news data for a symbol or list of symbols.

        Args:
            symbols (str): stock symbol(s)
            start (Union[str, datetime]): start date or timestamp
            end (Optional[Union[str, datetime]], optional): end date or timestamp.
            Defaults to None.

        Returns:
            pd.DataFrame: raw news data
        """

        return self.news_source.get_news_data(symbols=symbols, start=start, end=end, **kwargs)
