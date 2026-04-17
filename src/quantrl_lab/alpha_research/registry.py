from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from .base import VectorizedTradingStrategy


@dataclass
class StrategyMetadata:
    """
    Metadata for a registered vectorized trading strategy.

    A-7: Mirrors ``IndicatorMetadata`` in the data layer so the strategy
    registry is self-describing and tooling (e.g., auto-wiring, docs) can
    introspect strategies without importing every strategy class.

    Attributes:
        name (str): Registry key (e.g., ``"mean_reversion"``).
        strategy_class (Type[VectorizedTradingStrategy]): The concrete class.
        description (str): Human-readable description of the strategy logic.
        default_params (Dict[str, Any]): Default constructor kwargs
            (excluding column args, which are resolved at runtime).
    """

    name: str
    strategy_class: Type[VectorizedTradingStrategy]
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)


class VectorizedStrategyRegistry:
    """Registry for vectorized trading strategies."""

    _strategies: Dict[str, StrategyMetadata] = {}

    @classmethod
    def register(cls, name: str, description: str = "", default_params: Optional[Dict[str, Any]] = None) -> Callable:
        """
        Decorator to register a strategy class with metadata.

        Args:
            name (str): Registry key for the strategy.
            description (str): Human-readable description.
            default_params (Dict[str, Any], optional): Default constructor
                kwargs (excluding column args).

        Returns:
            Callable: Decorator that registers the strategy class.
        """

        def decorator(strategy_class: Type[VectorizedTradingStrategy]):
            if name in cls._strategies:
                import warnings

                warnings.warn(
                    f"Strategy '{name}' is already registered and will be overwritten.",
                    UserWarning,
                    stacklevel=2,
                )
            cls._strategies[name] = StrategyMetadata(
                name=name,
                strategy_class=strategy_class,
                description=description,
                default_params=default_params or {},
            )
            return strategy_class

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs) -> VectorizedTradingStrategy:
        """Create a strategy instance by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name].strategy_class(**kwargs)

    @classmethod
    def get(cls, name: str) -> Type[VectorizedTradingStrategy]:
        """Get the strategy class by name."""
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name].strategy_class

    @classmethod
    def get_metadata(cls, name: str) -> StrategyMetadata:
        """
        Get the full metadata for a strategy.

        Args:
            name (str): Registry key for the strategy.

        Raises:
            ValueError: If the strategy is not registered.

        Returns:
            StrategyMetadata: Metadata object for the strategy.
        """
        if name not in cls._strategies:
            raise ValueError(f"Unknown strategy: {name}. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List all registered strategies."""
        return list(cls._strategies.keys())
