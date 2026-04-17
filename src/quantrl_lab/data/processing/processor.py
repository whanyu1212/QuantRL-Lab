import json
import os
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from loguru import logger

# Import centralized configuration
from quantrl_lab.data.config import config
from quantrl_lab.data.partitioning import DateRangeSplitter, RatioSplitter
from quantrl_lab.data.processing.features.sentiment import SentimentFeatureGenerator

# Import new feature generators
from quantrl_lab.data.processing.features.technical import TechnicalFeatureGenerator

# D-5: ProcessingMetadata moved to its own module to break the circular import
# chain where all pipeline step files imported from processor.py.
# Re-exported here for backward compatibility with existing user code.
from quantrl_lab.data.processing.metadata import ProcessingMetadata  # noqa: F401
from quantrl_lab.data.processing.pipeline_config import (
    CleanupConfig,
    CrossSectionalConfig,
    ProcessingPipelineConfig,
    SplitConfig,
)

# Import sentiment modules
from quantrl_lab.data.processing.sentiment import (
    HuggingFaceProvider,
    SentimentConfig,
)


class DataProcessor:
    @staticmethod
    def load_indicators(file_path: str) -> List[Union[str, Dict]]:
        """
        Load indicator configuration from a YAML or JSON file.

        Args:
            file_path: Path to the configuration file (.yaml, .yml, or .json)

        Returns:
            List[Union[str, Dict]]: List of indicator configurations

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is unsupported or invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            with open(file_path, "r") as f:
                if ext in [".yaml", ".yml"]:
                    config_data = yaml.safe_load(f)
                elif ext == ".json":
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration format: {ext}. Use .yaml or .json")

            # Validate structure - expect a list or a dict with an 'indicators' key
            if isinstance(config_data, list):
                return config_data
            elif isinstance(config_data, dict) and "indicators" in config_data:
                return config_data["indicators"]
            else:
                raise ValueError("Invalid config structure. Expected a list or a dict with 'indicators' key.")

        except Exception as e:
            raise ValueError(f"Failed to load indicator config from {file_path}: {e}")

    def __init__(
        self,
        ohlcv_data: pd.DataFrame,
        *,
        news_data: Optional[pd.DataFrame] = None,
        analyst_grades: Optional[pd.DataFrame] = None,
        analyst_ratings: Optional[pd.DataFrame] = None,
        sector_performance: Optional[pd.DataFrame] = None,
        industry_performance: Optional[pd.DataFrame] = None,
        fundamental_data: Optional[pd.DataFrame] = None,
        macro_data: Optional[pd.DataFrame] = None,
        calendar_event_data: Optional[pd.DataFrame] = None,
        sentiment_config: Optional[SentimentConfig] = None,
        sentiment_provider: Optional[object] = None,
    ):
        if ohlcv_data is None:
            raise ValueError("Required parameter 'ohlcv_data' is missing.")

        self.ohlcv_data = ohlcv_data

        # Optional enrichment data
        self.news_data = news_data
        self.fundamental_data = fundamental_data
        self.macro_data = macro_data
        self.calendar_event_data = calendar_event_data

        # Analyst & market-context data
        self.analyst_grades = analyst_grades
        self.analyst_ratings = analyst_ratings
        self.sector_performance = sector_performance
        self.industry_performance = industry_performance

        # Sentiment configuration and provider
        self.sentiment_config = sentiment_config if sentiment_config is not None else SentimentConfig()
        self.sentiment_provider = sentiment_provider

        if self.sentiment_provider is None and self.news_data is not None:
            # Default to HuggingFaceProvider if news data is present but no provider given
            self.sentiment_provider = HuggingFaceProvider()

    def _has_multiple_symbols(self, df: pd.DataFrame) -> bool:
        """Return True when the dataframe contains more than one
        symbol."""
        return "Symbol" in df.columns and df["Symbol"].dropna().nunique() > 1

    @staticmethod
    def _reject_alpha_selection_config(alpha_selection_config: Any) -> None:
        """Reject deprecated in-processor alpha selection."""
        if alpha_selection_config is None:
            return

        warnings.warn(
            "`alpha_selection_config` on DataProcessor is deprecated and no longer performs alpha selection in the "
            "data-processing layer. Use quantrl_lab.alpha_research.build_processing_config_from_alpha_selection(...) "
            "or AlphaSelector explicitly, then pass indicators=... or pipeline_config=... to DataProcessor.",
            FutureWarning,
            stacklevel=3,
        )
        raise ValueError(
            "DataProcessor no longer performs alpha selection. Use "
            "quantrl_lab.alpha_research.build_processing_config_from_alpha_selection(...) "
            "or AlphaSelector explicitly before calling DataProcessor."
        )

    @staticmethod
    def _coerce_cleanup_config(cleanup_config: Optional[Union[CleanupConfig, Dict[str, Any]]]) -> CleanupConfig:
        """Normalize cleanup config to the typed dataclass."""
        if cleanup_config is None:
            return CleanupConfig()
        if isinstance(cleanup_config, CleanupConfig):
            return cleanup_config
        if isinstance(cleanup_config, dict):
            return CleanupConfig(**cleanup_config)
        raise TypeError("cleanup_config must be a dict or CleanupConfig.")

    @staticmethod
    def _coerce_cross_sectional_config(
        cross_sectional_config: Optional[Union[CrossSectionalConfig, Dict[str, Any]]],
    ) -> Optional[CrossSectionalConfig]:
        """Normalize cross-sectional config to the typed dataclass."""
        if cross_sectional_config is None:
            return None
        if isinstance(cross_sectional_config, CrossSectionalConfig):
            return cross_sectional_config
        if isinstance(cross_sectional_config, dict):
            return CrossSectionalConfig(**cross_sectional_config)
        raise TypeError("cross_sectional_config must be a dict or CrossSectionalConfig.")

    @staticmethod
    def _coerce_split_config(split_config: Optional[Union[SplitConfig, Dict[str, Any]]]) -> Optional[SplitConfig]:
        """Normalize split config to the typed dataclass."""
        if split_config is None:
            return None
        if isinstance(split_config, SplitConfig):
            return split_config
        if isinstance(split_config, dict):
            return SplitConfig(splits=split_config)
        raise TypeError("split_config must be a dict or SplitConfig.")

    def _normalize_pipeline_config(
        self,
        *,
        indicators: Optional[List[Union[str, Dict]]] = None,
        alpha_selection_config: Optional[Any] = None,
        fillna_strategy: str = "neutral",
        split_config: Optional[Union[SplitConfig, Dict[str, Any]]] = None,
        cross_sectional_config: Optional[Union[CrossSectionalConfig, Dict[str, Any]]] = None,
        cleanup_config: Optional[Union[CleanupConfig, Dict[str, Any]]] = None,
        numeric_conversion_columns: Optional[List[str]] = None,
        strict_indicators: bool = False,
        verbose: bool = False,
        pipeline_config: Optional[ProcessingPipelineConfig] = None,
        legacy_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ProcessingPipelineConfig:
        """Resolve explicit args, typed configs, and legacy kwargs into
        one typed config."""
        legacy_kwargs = dict(legacy_kwargs or {})
        self._reject_alpha_selection_config(alpha_selection_config)

        if pipeline_config is not None:
            if not isinstance(pipeline_config, ProcessingPipelineConfig):
                raise TypeError("pipeline_config must be a ProcessingPipelineConfig instance.")

            if (
                indicators is not None
                or alpha_selection_config is not None
                or fillna_strategy != "neutral"
                or split_config is not None
                or cross_sectional_config is not None
                or cleanup_config is not None
                or numeric_conversion_columns is not None
                or strict_indicators
                or verbose
                or legacy_kwargs
            ):
                raise ValueError("Use either pipeline_config or individual pipeline arguments, not both.")
            return deepcopy(pipeline_config)

        cleanup = self._coerce_cleanup_config(cleanup_config)

        if "columns_to_drop" in legacy_kwargs:
            warnings.warn(
                "`columns_to_drop` is deprecated; pass cleanup_config=CleanupConfig(columns_to_drop=...) instead.",
                FutureWarning,
                stacklevel=3,
            )
            if cleanup.columns_to_drop is not None:
                raise ValueError("Specify columns_to_drop via either cleanup_config or legacy kwargs, not both.")
            cleanup.columns_to_drop = legacy_kwargs.pop("columns_to_drop")

        if "columns_to_convert" in legacy_kwargs:
            warnings.warn(
                "`columns_to_convert` is deprecated; pass numeric_conversion_columns=[...] instead.",
                FutureWarning,
                stacklevel=3,
            )
            if numeric_conversion_columns is not None:
                raise ValueError(
                    "Specify numeric conversion columns via either "
                    "numeric_conversion_columns or legacy kwargs, not both."
                )
            numeric_conversion_columns = legacy_kwargs.pop("columns_to_convert")

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        return ProcessingPipelineConfig(
            indicators=deepcopy(indicators),
            fillna_strategy=fillna_strategy,
            split=self._coerce_split_config(split_config),
            cleanup=cleanup,
            numeric_conversion_columns=deepcopy(numeric_conversion_columns),
            strict_indicators=strict_indicators,
            verbose=verbose,
            cross_sectional=self._coerce_cross_sectional_config(cross_sectional_config),
        )

    def _resolve_cleanup_behavior(self, pipeline_config: ProcessingPipelineConfig) -> Tuple[bool, bool]:
        """Determine cleanup defaults based on split/panel
        configuration."""
        has_split = pipeline_config.split is not None
        preserve_symbol_by_default = self._has_multiple_symbols(self.ohlcv_data) and (
            has_split or pipeline_config.cross_sectional is not None
        )

        keep_date = pipeline_config.cleanup.keep_date if pipeline_config.cleanup.keep_date is not None else has_split
        keep_symbol = (
            pipeline_config.cleanup.keep_symbol
            if pipeline_config.cleanup.keep_symbol is not None
            else preserve_symbol_by_default
        )
        return keep_date, keep_symbol

    def _assemble_pipeline(
        self,
        pipeline_config: ProcessingPipelineConfig,
        resolved_indicators: Optional[List[Union[str, Dict]]],
    ):
        """Create a ``DataPipeline`` from the normalized
        configuration."""
        from quantrl_lab.data.processing.pipeline import DataPipeline
        from quantrl_lab.data.processing.steps import (
            AnalystEstimatesStep,
            ColumnCleanupStep,
            CrossSectionalStep,
            MarketContextStep,
            NumericConversionStep,
            SentimentEnrichmentStep,
            TechnicalIndicatorStep,
        )

        keep_date, keep_symbol = self._resolve_cleanup_behavior(pipeline_config)

        pipeline = DataPipeline()
        pipeline.add_step(
            TechnicalIndicatorStep(indicators=resolved_indicators, strict=pipeline_config.strict_indicators)
        )

        if self.analyst_grades is not None or self.analyst_ratings is not None:
            pipeline.add_step(AnalystEstimatesStep(grades_df=self.analyst_grades, ratings_df=self.analyst_ratings))

        if self.sector_performance is not None or self.industry_performance is not None:
            pipeline.add_step(
                MarketContextStep(sector_perf_df=self.sector_performance, industry_perf_df=self.industry_performance)
            )

        if self.news_data is not None:
            pipeline.add_step(
                SentimentEnrichmentStep(
                    news_data=self.news_data,
                    provider=self.sentiment_provider,
                    config=self.sentiment_config,
                    fillna_strategy=pipeline_config.fillna_strategy,
                )
            )

        pipeline.add_step(NumericConversionStep(columns=pipeline_config.numeric_conversion_columns))

        if pipeline_config.cross_sectional is not None:
            pipeline.add_step(
                CrossSectionalStep(
                    columns=pipeline_config.cross_sectional.columns,
                    methods=pipeline_config.cross_sectional.methods,
                    date_column=pipeline_config.cross_sectional.date_column,
                )
            )

        pipeline.add_step(
            ColumnCleanupStep(
                columns_to_drop=pipeline_config.cleanup.columns_to_drop,
                keep_date=keep_date,
                keep_symbol=keep_symbol,
            )
        )

        return pipeline

    def append_technical_indicators(
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
        # Return original if no indicators specified
        if not indicators:
            return df.copy()

        generator = TechnicalFeatureGenerator(indicators, strict=kwargs.pop("strict_indicators", False))
        return generator.generate(df, **kwargs)

    def append_news_sentiment_data(self, df: pd.DataFrame, fillna_strategy="neutral") -> pd.DataFrame:
        """
        Append news sentiment data to the OHLCV DataFrame.

        Args:
            df (pd.DataFrame): Input OHLCV DataFrame.
            fillna_strategy (str, optional): Strategy for handling missing sentiment scores. Defaults to "neutral".

        Raises:
            ValueError: If the input DataFrame is empty or if the strategy is unsupported.

        Returns:
            pd.DataFrame: DataFrame with appended news sentiment data.
        """
        if self.news_data is None or self.news_data.empty:
            logger.debug("No news data provided. Skipping sentiment analysis.")
            return df

        generator = SentimentFeatureGenerator(
            self.sentiment_provider, self.sentiment_config, self.news_data, fillna_strategy
        )
        return generator.generate(df)

    def drop_unwanted_columns(
        self,
        df: pd.DataFrame,
        columns_to_drop: Optional[List[str]] = None,
        keep_date: bool = False,
        keep_symbol: bool = False,
    ) -> pd.DataFrame:
        """
        Drop unwanted columns from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns_to_drop (Optional[List[str]], optional): List of column names to drop.
                If None, will drop default columns ('Date', 'Timestamp', 'Symbol'). Defaults to None.
            keep_date (bool): If True, date-related columns will not be dropped.
            keep_symbol (bool): If True, preserve ``Symbol`` under the default cleanup policy.
        Returns:
            pd.DataFrame: DataFrame with specified columns dropped.
        """
        from quantrl_lab.data.processing.steps import ColumnCleanupStep

        return ColumnCleanupStep(columns_to_drop=columns_to_drop, keep_date=keep_date, keep_symbol=keep_symbol).process(
            df, ProcessingMetadata()
        )

    def convert_columns_to_numeric(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert specified columns to numeric, handling date columns
        carefully.

        Args:
            df (pd.DataFrame): Input DataFrame
            columns (Optional[List[str]]): Specific columns to convert. If None, converts all object columns.

        Returns:
            pd.DataFrame: DataFrame with numeric conversions applied
        """
        from quantrl_lab.data.processing.steps import NumericConversionStep

        return NumericConversionStep(columns=columns).process(df, ProcessingMetadata())

    def build_pipeline(
        self,
        indicators: Optional[List[Union[str, Dict]]] = None,
        alpha_selection_config: Optional[Any] = None,
        fillna_strategy: str = "neutral",
        split_config: Optional[Union[SplitConfig, Dict[str, Any]]] = None,
        *,
        cross_sectional_config: Optional[Union[CrossSectionalConfig, Dict[str, Any]]] = None,
        cleanup_config: Optional[Union[CleanupConfig, Dict[str, Any]]] = None,
        numeric_conversion_columns: Optional[List[str]] = None,
        strict_indicators: bool = False,
        verbose: bool = False,
        pipeline_config: Optional[ProcessingPipelineConfig] = None,
        **kwargs: Any,
    ):
        """
        Assemble and return a ``DataPipeline`` without executing it.

        This is useful for downstream inspection, logging, and custom
        execution flows.
        """
        normalized_config = self._normalize_pipeline_config(
            indicators=indicators,
            alpha_selection_config=alpha_selection_config,
            fillna_strategy=fillna_strategy,
            split_config=split_config,
            cross_sectional_config=cross_sectional_config,
            cleanup_config=cleanup_config,
            numeric_conversion_columns=numeric_conversion_columns,
            strict_indicators=strict_indicators,
            verbose=verbose,
            pipeline_config=pipeline_config,
            legacy_kwargs=kwargs,
        )
        return self._assemble_pipeline(normalized_config, deepcopy(normalized_config.indicators))

    def data_processing_pipeline(
        self,
        indicators: Optional[List[Union[str, Dict]]] = None,
        alpha_selection_config: Optional[Any] = None,
        fillna_strategy: str = "neutral",
        split_config: Optional[Union[SplitConfig, Dict[str, Any]]] = None,
        *,
        cross_sectional_config: Optional[Union[CrossSectionalConfig, Dict[str, Any]]] = None,
        cleanup_config: Optional[Union[CleanupConfig, Dict[str, Any]]] = None,
        numeric_conversion_columns: Optional[List[str]] = None,
        strict_indicators: bool = False,
        verbose: bool = False,
        pipeline_config: Optional[ProcessingPipelineConfig] = None,
        **kwargs: Any,
    ) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict]:
        """
        Main data processing pipeline.

        Applies technical indicators, sentiment analysis, and data transformations

        This method uses the DataPipeline infrastructure internally.

        Args:
            indicators (Optional[List[Union[str, Dict]]], optional):
                List of indicators to apply. Supports:
                - String format: ["SMA", "RSI"]
                - Dict format: [{"SMA": {"window": 20}}, {"RSI": {"window": 14}}]
                Defaults to None (no indicators).
            alpha_selection_config: Deprecated. Alpha selection is no longer
                performed inside ``DataProcessor``. Use
                ``quantrl_lab.alpha_research.build_processing_config_from_alpha_selection(...)``
                or ``AlphaSelector`` explicitly.
            fillna_strategy (str, optional): Strategy for handling missing sentiment scores.
                Options: "neutral" (fill with 0.0) or "fill_forward" (forward fill).
                Defaults to "neutral".
            split_config (Optional[Dict], optional): Configuration for data splitting.
                If None, returns a single DataFrame. Otherwise, returns dict of DataFrames.
                Ratio-based: {'train': 0.7, 'test': 0.3}
                Date-based: {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')}
            cross_sectional_config: Optional configuration for cross-sectional
                feature generation on panel data.
            cleanup_config: Optional typed cleanup config replacing ad hoc
                column-drop kwargs.
            numeric_conversion_columns: Optional explicit columns to coerce to
                numeric before cleanup.
            strict_indicators: If True, fail fast when indicator application
                encounters unknown names or runtime errors.
            verbose: Enable additional pipeline logging.
            pipeline_config: Optional typed config object. Cannot be mixed with
                the other pipeline arguments.
            **kwargs: Deprecated compatibility kwargs such as
                ``columns_to_drop`` and ``columns_to_convert``.

        Returns:
            Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict]: A tuple containing:
                - Processed DataFrame if split_config is None
                - Dictionary of DataFrames if split_config is provided (keys: split names)
                - Metadata dictionary with processing information
        """
        from quantrl_lab.data.processing.steps import ColumnCleanupStep

        normalized_config = self._normalize_pipeline_config(
            indicators=indicators,
            alpha_selection_config=alpha_selection_config,
            fillna_strategy=fillna_strategy,
            split_config=split_config,
            cross_sectional_config=cross_sectional_config,
            cleanup_config=cleanup_config,
            numeric_conversion_columns=numeric_conversion_columns,
            strict_indicators=strict_indicators,
            verbose=verbose,
            pipeline_config=pipeline_config,
            legacy_kwargs=kwargs,
        )
        resolved_split_config = normalized_config.split.to_dict() if normalized_config.split is not None else None
        pipeline = self._assemble_pipeline(normalized_config, deepcopy(normalized_config.indicators))
        _, keep_symbol = self._resolve_cleanup_behavior(normalized_config)

        # Execute Pipeline
        # We pass symbol for metadata tracking if available
        symbol = None
        if "Symbol" in self.ohlcv_data.columns:
            unique_symbols = self.ohlcv_data["Symbol"].unique()
            symbol = unique_symbols[0] if len(unique_symbols) == 1 else None

        processed_data, metadata_obj = pipeline.execute(self.ohlcv_data, symbol=symbol)

        # Update metadata flags
        if self.analyst_grades is not None or self.analyst_ratings is not None:
            metadata_obj.analyst_data_applied = True
        if self.sector_performance is not None or self.industry_performance is not None:
            metadata_obj.market_context_applied = True
        # Handle Data Splitting (Post-Processing)
        # Debug: Check for columns with all NaN values before dropna
        if normalized_config.verbose:
            null_counts = processed_data.isnull().sum()
            all_null_cols = null_counts[null_counts == len(processed_data)]
            if not all_null_cols.empty:
                logger.warning("Columns with all NaN values: {columns}", columns=list(all_null_cols.index))

            logger.info("Before dropna: {rows} rows", rows=len(processed_data))
            logger.info("Columns in DataFrame: {columns}", columns=list(processed_data.columns))

        required_columns = [col for col in metadata_obj.required_non_null_columns if col in processed_data.columns]
        initial_len = len(processed_data)
        processed_data = processed_data.dropna(subset=required_columns or None)
        dropped_count = initial_len - len(processed_data)

        if normalized_config.verbose:
            if dropped_count > 0:
                logger.info(
                    "Dropped {count} rows containing NaNs in required columns: {columns}",
                    count=dropped_count,
                    columns=required_columns,
                )
            else:
                logger.info("No rows dropped (data is clean)")

        if normalized_config.verbose:
            logger.info("After dropna: {rows} rows", rows=len(processed_data))

        if resolved_split_config:
            split_data, split_metadata = self._split_data(processed_data, resolved_split_config)

            # Merge split metadata into pipeline metadata
            metadata_obj.date_ranges = split_metadata["date_ranges"]
            metadata_obj.final_shapes = split_metadata["final_shapes"]

            # Re-run cleanup after splitting so date columns can be removed while
            # preserving caller intent and panel identity defaults.
            for key in split_data:
                split_data[key] = ColumnCleanupStep(
                    columns_to_drop=normalized_config.cleanup.columns_to_drop,
                    keep_date=False,
                    keep_symbol=keep_symbol,
                ).process(split_data[key], metadata_obj)

            return split_data, metadata_obj.to_dict()
        else:
            # Handle metadata for non-split data (legacy logic port)
            date_column = next((col for col in config.DATE_COLUMNS if col in processed_data.columns), None)
            if date_column:
                dates = pd.to_datetime(processed_data[date_column])
                metadata_obj.date_ranges["full_data"] = {
                    "start": dates.min().strftime("%Y-%m-%d"),
                    "end": dates.max().strftime("%Y-%m-%d"),
                }
            metadata_obj.final_shapes["full_data"] = processed_data.shape

            return processed_data, metadata_obj.to_dict()

    def _split_data(self, df: pd.DataFrame, split_config: Dict) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Split the data into respective sets according to the config.

        This method now delegates to the new splitter classes (RatioSplitter or DateRangeSplitter)
        while maintaining backward compatibility with the existing API.

        Args:
            df (pd.DataFrame): input dataframe
            split_config (Dict): split config in dictionary format
                Example by ratio: {'train': 0.7, 'test': 0.3}
                Example by dates: {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')}

        Raises:
            ValueError: If date column not found for splitting or invalid config

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict]: datasets in dict and metadata
        """
        # If the DataFrame has a DatetimeIndex, promote it to a column so
        # RatioSplitter / DateRangeSplitter can sort by it without ambiguity.
        original_index_name = df.index.name
        index_name = original_index_name or "Date"
        has_datetime_index = hasattr(df.index, "dtype") and pd.api.types.is_datetime64_any_dtype(df.index)
        if has_datetime_index:
            df = df.reset_index()
            if df.columns[0] != index_name:
                df = df.rename(columns={df.columns[0]: index_name})

        # Determine split type based on config values
        is_date_based = any(isinstance(v, (tuple, list)) for v in split_config.values())

        if is_date_based:
            splitter = DateRangeSplitter(split_config)
        else:
            splitter = RatioSplitter(split_config)

        split_data = splitter.split(df)
        metadata = splitter.get_metadata()

        # Restore the DatetimeIndex on each split and drop the temporary column
        if has_datetime_index:
            for key in split_data:
                if index_name in split_data[key].columns:
                    split_data[key] = split_data[key].set_index(index_name)
                    split_data[key].index.name = original_index_name

        return split_data, metadata
