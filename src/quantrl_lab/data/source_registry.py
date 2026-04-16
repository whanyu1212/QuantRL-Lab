from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Union

import pandas as pd

from quantrl_lab.data.interface import HistoricalDataCapable, NewsDataCapable
from quantrl_lab.data.sources.alpaca_loader import AlpacaDataLoader
from quantrl_lab.data.sources.alpha_vantage_loader import AlphaVantageDataLoader  # noqa: F401
from quantrl_lab.data.sources.fmp_loader import FMPDataSource
from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader  # noqa: F401


class DataSourceRegistry:
    """
    Registry for managing multiple data sources with factory pattern.

    Supports lazy initialization, multiple sources per type, and dynamic
    source discovery by capability.

    Example:
        >>> registry = DataSourceRegistry()
        >>> registry.register_source("alpaca_backup", lambda: AlpacaDataLoader())
        >>>
        >>> # Use primary source
        >>> data = registry.get_historical_ohlcv_data(...)
        >>>
        >>> # Or get specific source
        >>> backup = registry.get_source("alpaca_backup")
        >>> data = backup.get_historical_ohlcv_data(...)
    """

    # Default source configurations
    DEFAULT_SOURCES = {
        "primary_source": AlpacaDataLoader,
        "news_source": AlpacaDataLoader,
        "fundamental_source": FMPDataSource,
    }

    def __init__(self, sources: Optional[Dict[str, type]] = None, **kwargs: Any) -> None:
        """
        Initialize with configured data sources.

        Args:
            sources: Dictionary mapping source names to data source classes
            **kwargs: Individual source overrides (e.g., primary_source=YFinanceDataLoader)

        Example:
            >>> # Use defaults
            >>> registry = DataSourceRegistry()
            >>>
            >>> # Override primary source
            >>> registry = DataSourceRegistry(primary_source=YFinanceDataLoader)
            >>>
            >>> # Custom sources dict
            >>> registry = DataSourceRegistry(sources={
            ...     "primary_source": AlpacaDataLoader,
            ...     "backup_source": YFinanceDataLoader
            ... })
        """
        # Internal storage
        self._factories: Dict[str, Callable] = {}  # name -> factory function
        self._capabilities: Dict[str, Set[str]] = {}  # name -> declared capability set
        self._sources: Dict[str, Any] = {}  # name -> instantiated source (lazy)

        # Register default sources
        self._register_defaults()

        # Override with provided sources dict (backward compatibility)
        if sources:
            for name, source_class in sources.items():
                if source_class is not None:
                    self.register_source(
                        name,
                        self._make_factory(source_class),
                        override=True,
                        capabilities=getattr(source_class, "SUPPORTED_FEATURES", set()),
                    )

        # Override with kwargs (backward compatibility)
        for name, source_class in kwargs.items():
            if source_class is not None:
                self.register_source(
                    name,
                    self._make_factory(source_class),
                    override=True,
                    capabilities=getattr(source_class, "SUPPORTED_FEATURES", set()),
                )

    def _register_defaults(self) -> None:
        """Register default data sources."""
        for name, source_class in self.DEFAULT_SOURCES.items():
            if source_class is not None:
                self.register_source(
                    name,
                    self._make_factory(source_class),
                    capabilities=getattr(source_class, "SUPPORTED_FEATURES", set()),
                )

    @staticmethod
    def _make_factory(source_class: type) -> Callable:
        """
        Create a factory function for a source class.

        Args:
            source_class: Data source class to instantiate

        Returns:
            Factory function that creates source instances
        """

        def factory(**init_kwargs):
            return source_class(**init_kwargs)

        factory.source_class = source_class
        factory.supported_features = set(getattr(source_class, "SUPPORTED_FEATURES", set()))
        return factory

    def register_source(
        self,
        name: str,
        factory: Callable,
        override: bool = False,
        capabilities: Optional[Set[str]] = None,
    ) -> None:
        """
        Register a data source factory.

        Args:
            name: Unique name for this source (e.g., "alpaca_primary", "yfinance_backup")
            factory: Callable that returns a data source instance
            override: If True, replace existing registration
            capabilities: Explicit capability set used for source discovery

        Raises:
            ValueError: If source already registered and override=False

        Example:
            >>> registry.register_source("custom", lambda: YFinanceDataLoader())
            >>> registry.register_source("primary_source", lambda: AlpacaDataLoader(), override=True)
        """
        if name in self._factories and not override:
            raise ValueError(f"Source '{name}' already registered. Use override=True to replace.")
        self._factories[name] = factory
        self._capabilities[name] = self._resolve_capabilities(factory, capabilities)

    @staticmethod
    def _resolve_capabilities(factory: Callable, capabilities: Optional[Set[str]]) -> Set[str]:
        """Resolve capability metadata without forcing callers to
        provide it explicitly."""
        if capabilities is not None:
            return set(capabilities)

        factory_features = getattr(factory, "supported_features", None)
        if factory_features is not None:
            return set(factory_features)

        source_class = getattr(factory, "source_class", None)
        if source_class is not None:
            return set(getattr(source_class, "SUPPORTED_FEATURES", set()))

        try:
            instance = factory()
        except Exception:
            return set()

        instance_features = getattr(instance, "supported_features", None)
        if instance_features is not None:
            return set(instance_features)

        return set(getattr(instance.__class__, "SUPPORTED_FEATURES", set()))

    def get_source(self, name: str, **init_kwargs: Any) -> Any:
        """
        Get or create a data source instance (lazy initialization).

        Args:
            name: Source name to retrieve
            **init_kwargs: Initialization arguments for the source (if not yet created)

        Returns:
            Data source instance

        Raises:
            KeyError: If no factory registered for this name

        Example:
            >>> source = registry.get_source("primary_source")
            >>> data = source.get_historical_ohlcv_data(...)
        """
        # Lazy initialization - create on first access
        if name not in self._sources:
            if name not in self._factories:
                raise KeyError(f"No factory registered for source '{name}'")
            self._sources[name] = self._factories[name](**init_kwargs)
        return self._sources[name]

    def list_sources_by_capability(self, capability: str) -> List[str]:
        """
        Find all registered sources supporting a capability.

        Args:
            capability: Feature name (e.g., "historical_bars", "news", "streaming")

        Returns:
            List of source names that support the capability

        Example:
            >>> sources = registry.list_sources_by_capability("historical_bars")
            >>> print(sources)  # ["primary_source", "backup_source"]
        """
        return [name for name, features in self._capabilities.items() if capability in features]

    def list_all_sources(self) -> List[str]:
        """
        List all registered source names.

        Returns:
            List of all registered source names
        """
        return list(self._factories.keys())

    # Backward compatibility: lazy properties for common sources
    @property
    def primary_source(self) -> Any:
        """
        Get primary data source (lazy initialization).

        Returns:
            Primary source instance
        """
        return self.get_source("primary_source")

    @property
    def news_source(self) -> Any:
        """
        Get news data source (lazy initialization).

        Returns:
            News source instance
        """
        return self.get_source("news_source")

    # Existing methods preserved for backward compatibility
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
            symbols: Stock symbol(s) to fetch data for.
            start: Start date for the data.
            end: End date for the data. Defaults to None.
            timeframe: Timeframe for the data. Defaults to "1d".

        Returns:
            pd.DataFrame: Historical OHLCV data.

        Raises:
            RuntimeError: If primary_source doesn't implement HistoricalDataCapable protocol.
        """
        if not isinstance(self.primary_source, HistoricalDataCapable):
            raise RuntimeError(
                f"{self.primary_source.__class__.__name__} doesn't support historical data. "
                f"Please configure a data source that implements HistoricalDataCapable."
            )

        # Use primary source to fetch historical data
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
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get news data for a symbol or list of symbols.

        Args:
            symbols: Stock symbol(s)
            start: Start date or timestamp
            end: End date or timestamp. Defaults to None.
            **kwargs: Additional parameters passed to the news source

        Returns:
            pd.DataFrame: raw news data

        Raises:
            RuntimeError: If news_source doesn't implement NewsDataCapable protocol.
        """
        if not isinstance(self.news_source, NewsDataCapable):
            raise RuntimeError(
                f"{self.news_source.__class__.__name__} doesn't support news data. "
                f"Please configure a data source that implements NewsDataCapable."
            )

        return self.news_source.get_news_data(symbols=symbols, start=start, end=end, **kwargs)
