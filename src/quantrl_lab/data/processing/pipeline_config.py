"""Typed configuration objects for data-processing pipelines."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

IndicatorConfig = Union[str, Dict[str, Any]]


@dataclass
class CleanupConfig:
    """Configuration for the cleanup step."""

    columns_to_drop: Optional[List[str]] = None
    keep_date: Optional[bool] = None
    keep_symbol: Optional[bool] = None


@dataclass
class SplitConfig:
    """Configuration for ratio- or date-based data splitting."""

    splits: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return plain dictionary form for splitter compatibility."""
        return dict(self.splits)


@dataclass
class CrossSectionalConfig:
    """Configuration for cross-sectional feature generation."""

    columns: List[str]
    methods: List[str] = field(default_factory=lambda: ["zscore"])
    date_column: Optional[str] = None


@dataclass
class ProcessingPipelineConfig:
    """Top-level typed configuration for ``DataProcessor`` assembly."""

    indicators: Optional[List[IndicatorConfig]] = None
    fillna_strategy: str = "neutral"
    split: Optional[SplitConfig] = None
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    numeric_conversion_columns: Optional[List[str]] = None
    strict_indicators: bool = False
    verbose: bool = False
    cross_sectional: Optional[CrossSectionalConfig] = None
