"""Adapters for feeding alpha-research results into data processing."""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from quantrl_lab.data.partitioning import DateRangeSplitter, RatioSplitter
from quantrl_lab.data.processing import ProcessingPipelineConfig, SplitConfig

from .indicator_research import FEATURE_SELECTION_MODE, validate_selection_mode
from .selector import AlphaSelector


@dataclass
class AlphaSelectionConfig:
    """Configuration for alpha-driven indicator selection."""

    candidates: Optional[List[Dict[str, Any]]] = None
    metric: str = "ic"
    threshold: float = 0.0
    top_k: int = 5
    selection_mode: str = FEATURE_SELECTION_MODE

    def __post_init__(self):
        """Validate the requested selection mode."""
        self.selection_mode = validate_selection_mode(self.selection_mode)

    def to_dict(self) -> Dict[str, Any]:
        """Return plain dictionary form for selector calls and
        metadata."""
        return {
            "candidates": self.candidates,
            "metric": self.metric,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "selection_mode": self.selection_mode,
        }


def _coerce_selection_config(
    alpha_selection_config: Union[AlphaSelectionConfig, Dict[str, Any]],
) -> AlphaSelectionConfig:
    """Normalize selection config to the typed dataclass."""
    if isinstance(alpha_selection_config, AlphaSelectionConfig):
        return alpha_selection_config
    if isinstance(alpha_selection_config, dict):
        return AlphaSelectionConfig(**alpha_selection_config)
    raise TypeError("alpha_selection_config must be a dict or AlphaSelectionConfig.")


def _coerce_split_config(split_config: Optional[Union[SplitConfig, Dict[str, Any]]]) -> Optional[SplitConfig]:
    """Normalize split config to the typed dataclass."""
    if split_config is None:
        return None
    if isinstance(split_config, SplitConfig):
        return split_config
    if isinstance(split_config, dict):
        return SplitConfig(splits=split_config)
    raise TypeError("split_config must be a dict or SplitConfig.")


def _has_multiple_symbols(df: pd.DataFrame) -> bool:
    """Return True when the dataframe contains more than one symbol."""
    return "Symbol" in df.columns and df["Symbol"].dropna().nunique() > 1


def _split_raw_data(df: pd.DataFrame, split_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Split raw data using the same date/index handling as
    DataProcessor."""
    original_index_name = df.index.name
    index_name = original_index_name or "Date"
    has_datetime_index = hasattr(df.index, "dtype") and pd.api.types.is_datetime64_any_dtype(df.index)
    if has_datetime_index:
        df = df.reset_index()
        if df.columns[0] != index_name:
            df = df.rename(columns={df.columns[0]: index_name})

    splitter = (
        DateRangeSplitter(split_config)
        if any(isinstance(v, (tuple, list)) for v in split_config.values())
        else RatioSplitter(split_config)
    )
    split_data = splitter.split(df)

    if has_datetime_index:
        for key, split_df in split_data.items():
            if index_name in split_df.columns:
                split_data[key] = split_df.set_index(index_name)
                split_data[key].index.name = original_index_name

    return split_data


def select_indicators_for_processing(
    raw_data: pd.DataFrame,
    alpha_selection_config: Union[AlphaSelectionConfig, Dict[str, Any]],
    *,
    split_config: Optional[Union[SplitConfig, Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Select processing indicators from raw OHLCV data outside
    ``DataProcessor``.

    When ``split_config`` is provided, selection runs on the raw training split
    to avoid leakage.
    """
    if _has_multiple_symbols(raw_data):
        raise ValueError(
            "select_indicators_for_processing only supports single-symbol data. "
            "Use AlphaSelector.suggest_for_universe() for multi-symbol workflows."
        )

    selection_config = _coerce_selection_config(alpha_selection_config)
    selection_metadata = selection_config.to_dict()
    coerced_split = _coerce_split_config(split_config)

    if coerced_split is None:
        warnings.warn(
            "Selecting indicators without split_config uses the full dataset and may leak future information. "
            "Prefer selecting on an explicit training split.",
            UserWarning,
            stacklevel=2,
        )
        selection_data = raw_data
        selection_metadata["selected_from_split"] = "full_data"
    else:
        split_dict = coerced_split.to_dict()
        if "train" not in split_dict:
            raise ValueError("alpha selection for processing requires split_config to include a 'train' split.")
        split_data = _split_raw_data(raw_data, split_dict)
        if "train" not in split_data or split_data["train"].empty:
            raise ValueError("Training split for alpha selection is empty.")
        selection_data = split_data["train"]
        selection_metadata["selected_from_split"] = "train"

    selection_metadata["selection_rows"] = len(selection_data)
    selection_metadata["selection_verbose"] = verbose

    selector = AlphaSelector(selection_data, verbose=verbose)
    indicators = selector.suggest_indicators(
        candidates=selection_config.candidates,
        metric=selection_config.metric,
        threshold=selection_config.threshold,
        top_k=selection_config.top_k,
        selection_mode=selection_config.selection_mode,
    )
    return indicators, selection_metadata


def build_processing_config_from_alpha_selection(
    raw_data: pd.DataFrame,
    alpha_selection_config: Union[AlphaSelectionConfig, Dict[str, Any]],
    *,
    split_config: Optional[Union[SplitConfig, Dict[str, Any]]] = None,
    processing_config: Optional[ProcessingPipelineConfig] = None,
    verbose: bool = False,
) -> Tuple[ProcessingPipelineConfig, Dict[str, Any]]:
    """Build a ``ProcessingPipelineConfig`` populated with alpha-
    selected indicators."""
    resolved_config = deepcopy(processing_config) if processing_config is not None else ProcessingPipelineConfig()
    if resolved_config.indicators is not None:
        raise ValueError("processing_config.indicators must be None when building it from alpha selection.")

    coerced_split = _coerce_split_config(split_config)
    if coerced_split is not None:
        if resolved_config.split is not None and resolved_config.split.to_dict() != coerced_split.to_dict():
            raise ValueError("split_config conflicts with processing_config.split.")
        resolved_config.split = coerced_split

    selection_verbose = verbose or resolved_config.verbose
    indicators, selection_metadata = select_indicators_for_processing(
        raw_data,
        alpha_selection_config,
        split_config=resolved_config.split,
        verbose=selection_verbose,
    )

    resolved_config.indicators = indicators
    resolved_config.verbose = selection_verbose
    return resolved_config, selection_metadata
