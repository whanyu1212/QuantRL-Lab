from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Union, runtime_checkable

import pandas as pd


class DataSource(ABC):
    """Base class for all data sources."""

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
        **kwargs,
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
        features = []

        if isinstance(self, HistoricalDataCapable):
            features.append("historical_bars")
        if isinstance(self, NewsDataCapable):
            features.append("news")
        if isinstance(self, LiveDataCapable):
            features.append("live_data")
        if isinstance(self, StreamingCapable):
            features.append("streaming")
        if isinstance(self, ConnectionManaged):
            features.append("connection_managed")

        # Check if instrument discovery is implemented (method is overridden)
        if (
            hasattr(self.__class__, "list_available_instruments")
            and self.__class__.list_available_instruments is not DataSource.list_available_instruments
        ):
            features.append("instrument_discovery")

        return features

    def supports_feature(self, feature_name: str) -> bool:
        """Check if the data source supports a specific feature."""
        return feature_name in self.supported_features

    def __repr__(self) -> str:
        """
        Return a string representation of the data source.

        Returns:
            str: A string representation of the data source.
        """
        return f"<{self.__class__.__name__}(name='{self.source_name}', connected={self.is_connected})>"


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
        start: Union[str, datetime],
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
    ) -> Union[pd.DataFrame, Dict]:
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

    async def subscribe(
        self,
        symbols: Union[str, List[str]],
        callback,
        data_type: str = "quotes",
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

    def get_fundamental_data(self, symbols: str, metrics: List[str], **kwargs) -> Union[pd.DataFrame, Dict]:
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
    ) -> Union[pd.DataFrame, Dict]:
        """Get macroeconomic data for specified indicators and time
        range."""
        ...
