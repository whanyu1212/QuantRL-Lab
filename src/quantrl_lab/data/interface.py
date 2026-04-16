from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, List, Optional, Protocol, Set, Union, runtime_checkable

import pandas as pd


class DataSource(ABC):
    """Base class for all data sources."""

    SUPPORTED_FEATURES: ClassVar[Set[str]] = set()

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this data source."""
        pass

    def connect(self) -> None:
        """
        Connect to the data source.

        Default implementation for sources that don't require
        connections. Override this method if your data source needs
        connection management.
        """
        pass

    def disconnect(self) -> None:
        """
        Disconnect from the data source.

        Default implementation for sources that don't require
        connections. Override this method if your data source needs
        connection management.
        """
        pass

    def is_connected(self) -> bool:
        """
        Check if the data source is currently connected.

        Returns:
            True for sources that don't require connections (always available).
            Override this method if your data source needs connection management.
        """
        return True

    def list_available_instruments(
        self,
        instrument_type: Optional[str] = None,
        market: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Return a list of available instrument symbols or identifiers
        that this source can provide data for.

        Args:
            instrument_type: Optional filter by type (e.g., 'stock', 'future', 'option', 'crypto', 'forex').
            market: Optional filter by market (e.g., 'NASDAQ', 'NYSE', 'crypto_spot', 'crypto_futures').
            **kwargs: Provider-specific additional filter parameters.

        Returns:
            A list of string identifiers for the available instruments.

        Raises:
            NotImplementedError: If the data source doesn't support instrument discovery.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support instrument discovery. "
            "This data source may be designed for specific instruments only."
        )

    @property
    def supported_features(self) -> List[str]:
        """Return a list of supported features."""
        return sorted(self.SUPPORTED_FEATURES)

    def supports_feature(self, feature_name: str) -> bool:
        """Check if the data source supports a specific feature."""
        return feature_name in self.supported_features

    def __repr__(self) -> str:
        """
        Return a string representation of the data source.

        Returns:
            str: A string representation of the data source.
        """
        return f"<{self.__class__.__name__}(name='{self.source_name}', connected={self.is_connected()})>"


@runtime_checkable
class ConnectionManaged(Protocol):
    """
    Protocol for data sources that require explicit connection
    management.

    Sources implementing this protocol need to manage persistent
    connections, authentication sessions, or other stateful connections.
    """

    def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    def disconnect(self) -> None:
        """Close connection to the data source."""
        ...

    def is_connected(self) -> bool:
        """Check if currently connected to the data source."""
        ...


@runtime_checkable
class HistoricalDataCapable(Protocol):
    """Protocol for data sources that provide historical OHLCV data."""

    def get_historical_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """Get historical OHLCV data."""
        ...


@runtime_checkable
class NewsDataCapable(Protocol):
    """Protocol for data sources that provide news data."""

    def get_news_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Get news for specified symbols and time range."""
        ...


@runtime_checkable
class LiveDataCapable(Protocol):
    """
    Protocol for data sources with real-time data capabilities.

    It checks if the class has the following methods:
    - get_latest_quote
    - get_latest_trade
    """

    def get_latest_quote(self, symbols: Union[str, List[str]], **kwargs) -> pd.DataFrame:
        """Get latest market data."""
        ...

    def get_latest_trade(self, symbols: Union[str, List[str]], **kwargs) -> pd.DataFrame:
        """Get latest trade data."""
        ...


@runtime_checkable
class StreamingCapable(Protocol):
    """
    Protocol for data sources with streaming capabilities.

    It checks if the class has the following methods:
    - subscribe_to_updates
    - start_streaming
    - stop
    """

    async def subscribe_to_updates(
        self,
        symbol: str,
        data_type: str = "trades",
        **kwargs,
    ):
        """Subscribe to real-time data updates."""
        ...

    async def start_streaming(self):
        """Start the data stream."""
        ...

    async def stop_streaming(self):
        """Stop the data stream."""
        ...


@runtime_checkable
class FundamentalDataCapable(Protocol):
    """
    Protocol for data sources that provide fundamental data.

    It checks if the class has the following methods:
    - get_fundamental_data
    """

    def get_fundamental_data(self, symbols: str, metrics: List[str], **kwargs) -> pd.DataFrame:
        """Get fundamental data for specified symbols and metrics."""
        ...


@runtime_checkable
class MacroDataCapable(Protocol):
    """
    Protocol for data sources that provide macroeconomic data.

    It checks if the class has the following methods:
    - get_macro_data
    """

    def get_macro_data(
        self,
        indicators: Union[str, List[str]],
        start: Union[str, datetime],
        end: Union[str, datetime],
    ) -> pd.DataFrame:
        """Get macroeconomic data for specified indicators and time
        range."""
        ...


@runtime_checkable
class AnalystDataCapable(Protocol):
    """
    Protocol for data sources that provide analyst ratings and grades
    data.

    This includes analyst recommendations, upgrades/downgrades, price targets,
    and other research-based insights from financial analysts.

    It checks if the class has the following methods:
    - get_historical_grades
    - get_historical_rating
    """

    def get_historical_grades(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """
        Get historical analyst grades/recommendations for a symbol.

        Args:
            symbol: Stock symbol to fetch grades for
            **kwargs: Additional provider-specific parameters

        Returns:
            pd.DataFrame: Historical analyst grades data
        """
        ...

    def get_historical_rating(self, symbol: str, limit: int = 100, **kwargs: Any) -> pd.DataFrame:
        """
        Get historical analyst ratings for a symbol.

        Args:
            symbol: Stock symbol to fetch ratings for
            limit: Number of records to return (default: 100)
            **kwargs: Additional provider-specific parameters

        Returns:
            pd.DataFrame: Historical analyst ratings data
        """
        ...


@runtime_checkable
class SectorDataCapable(Protocol):
    """
    Protocol for data sources that provide sector and industry
    performance data.

    This includes historical performance metrics for market sectors and
    industries, enabling sector rotation and market trend analysis.

    It checks if the class has the following methods:
    - get_historical_sector_performance
    - get_historical_industry_performance
    """

    def get_historical_sector_performance(self, sector: str, **kwargs: Any) -> pd.DataFrame:
        """
        Get historical performance data for a specific market sector.

        Args:
            sector: Market sector name (e.g., "Energy", "Technology", "Healthcare")
            **kwargs: Additional provider-specific parameters

        Returns:
            pd.DataFrame: Historical sector performance data
        """
        ...

    def get_historical_industry_performance(self, industry: str, **kwargs: Any) -> pd.DataFrame:
        """
        Get historical performance data for a specific industry.

        Args:
            industry: Industry name (e.g., "Biotechnology", "Software", "Banks")
            **kwargs: Additional provider-specific parameters

        Returns:
            pd.DataFrame: Historical industry performance data
        """
        ...


@runtime_checkable
class CompanyProfileCapable(Protocol):
    """
    Protocol for data sources that provide company profile and metadata.

    This includes company information such as sector/industry classification,
    executive information, key financial metrics, and company details.

    It checks if the class has the following method:
    - get_company_profile
    """

    def get_company_profile(self, symbol: Union[str, List[str]], **kwargs: Any) -> pd.DataFrame:
        """
        Get company profile information including sector, industry, and
        key metrics.

        Args:
            symbol: Stock ticker symbol or list of symbols
            **kwargs: Additional provider-specific parameters

        Returns:
            pd.DataFrame: Company profile data with metadata
        """
        ...
