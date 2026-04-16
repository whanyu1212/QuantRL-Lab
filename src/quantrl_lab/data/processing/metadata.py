"""
Processing metadata dataclass.

Extracted from ``processor.py`` into its own module (fix for D-5) to break
the circular import chain where all pipeline step files imported from
``processor.py``, which itself imports from the pipeline and steps.

The dependency graph is now:
    steps/* → metadata.py          (leaf — no further processing imports)
    pipeline.py → metadata.py      (clean)
    processor.py → metadata.py     (clean)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class ProcessingMetadata:
    """
    Metadata collected during data processing pipeline.

    This dataclass tracks all transformations and operations applied during
    the data processing pipeline, providing transparency and reproducibility.

    Attributes:
        symbol (Optional[Union[str, List[str]]]): Stock symbol(s) being processed.
            Single symbol as string, multiple as list.
        date_ranges (Dict[str, Dict[str, str]]): Date ranges for each data split.
            Format: {"split_name": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
        fillna_strategy (str): Strategy used for filling missing sentiment scores.
            Options: "neutral" (fill with 0.0) or "fill_forward" (forward fill)
        technical_indicators (List[Union[str, Dict]]): List of technical indicators applied.
            Can contain strings ("SMA") or dicts ({"SMA": {"window": 20}})
        news_sentiment_applied (bool): Whether news sentiment analysis was performed.
        columns_dropped (List[str]): List of columns dropped during processing.
        original_shape (Tuple[int, int]): Shape of input data before processing (rows, cols).
        final_shapes (Dict[str, Tuple[int, int]]): Shapes of output data after processing.
            Format: {"split_name": (rows, cols)} or {"full_data": (rows, cols)}

    Examples:
        >>> metadata = ProcessingMetadata(
        ...     symbol="AAPL",
        ...     fillna_strategy="neutral",
        ...     original_shape=(1000, 7)
        ... )
        >>> metadata.technical_indicators = ["SMA", "RSI"]
        >>> metadata.to_dict()
    """

    symbol: Optional[Union[str, List[str]]] = None
    date_ranges: Dict[str, Dict[str, str]] = field(default_factory=dict)
    fillna_strategy: str = "neutral"
    technical_indicators: List[Union[str, Dict]] = field(default_factory=list)
    cross_sectional_features: List[str] = field(default_factory=list)
    news_sentiment_applied: bool = False
    analyst_data_applied: bool = False
    market_context_applied: bool = False
    alpha_selection_config: Optional[Dict] = None
    columns_dropped: List[str] = field(default_factory=list)
    required_non_null_columns: List[str] = field(default_factory=list)
    optional_feature_columns: List[str] = field(default_factory=list)
    original_shape: Tuple[int, int] = (0, 0)
    final_shapes: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def add_required_columns(self, columns: List[str]) -> None:
        """Track columns that must be present after processing."""
        for column in columns:
            if column not in self.required_non_null_columns and column not in self.optional_feature_columns:
                self.required_non_null_columns.append(column)

    def add_optional_columns(self, columns: List[str]) -> None:
        """Track sparse feature columns that should not drive row
        drops."""
        for column in columns:
            if column not in self.optional_feature_columns:
                self.optional_feature_columns.append(column)
            if column in self.required_non_null_columns:
                self.required_non_null_columns.remove(column)

    def to_dict(self) -> Dict:
        """
        Convert metadata to dictionary format for backward
        compatibility.

        Returns:
            Dict: Dictionary representation of metadata with all fields.

        Examples:
            >>> metadata = ProcessingMetadata(symbol="AAPL", original_shape=(100, 5))
            >>> result = metadata.to_dict()
            >>> assert result["symbol"] == "AAPL"
            >>> assert result["original_shape"] == (100, 5)
        """
        return {
            "symbol": self.symbol,
            "date_ranges": self.date_ranges,
            "fillna_strategy": self.fillna_strategy,
            "technical_indicators": self.technical_indicators,
            "cross_sectional_features": self.cross_sectional_features,
            "news_sentiment_applied": self.news_sentiment_applied,
            "analyst_data_applied": self.analyst_data_applied,
            "market_context_applied": self.market_context_applied,
            "alpha_selection_config": self.alpha_selection_config,
            "columns_dropped": self.columns_dropped,
            "required_non_null_columns": self.required_non_null_columns,
            "optional_feature_columns": self.optional_feature_columns,
            "original_shape": self.original_shape,
            "final_shapes": self.final_shapes,
        }
