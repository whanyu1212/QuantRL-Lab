import json
import os
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

        generator = TechnicalFeatureGenerator(indicators)
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
        self, df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None, keep_date: bool = False
    ) -> pd.DataFrame:
        """
        Drop unwanted columns from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
            columns_to_drop (Optional[List[str]], optional): List of column names to drop.
                If None, will drop default columns ('Date', 'Timestamp', 'Symbol'). Defaults to None.
            keep_date (bool): If True, date-related columns will not be dropped.
        Returns:
            pd.DataFrame: DataFrame with specified columns dropped.
        """
        from quantrl_lab.data.processing.steps import ColumnCleanupStep

        return ColumnCleanupStep(columns_to_drop=columns_to_drop, keep_date=keep_date).process(df, ProcessingMetadata())

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

    def data_processing_pipeline(
        self,
        indicators: Optional[List[Union[str, Dict]]] = None,
        alpha_selection_config: Optional[Dict[str, Any]] = None,
        fillna_strategy: str = "neutral",
        split_config: Optional[Dict] = None,
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
            alpha_selection_config (Optional[Dict], optional):
                Configuration for dynamic alpha selection. If provided, the pipeline
                will automatically select and apply the best indicators.
                Keys: "metric" (default "ic"), "threshold", "top_k", "candidates".
            fillna_strategy (str, optional): Strategy for handling missing sentiment scores.
                Options: "neutral" (fill with 0.0) or "fill_forward" (forward fill).
                Defaults to "neutral".
            split_config (Optional[Dict], optional): Configuration for data splitting.
                If None, returns a single DataFrame. Otherwise, returns dict of DataFrames.
                Ratio-based: {'train': 0.7, 'test': 0.3}
                Date-based: {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')}
            **kwargs: Additional arguments:
                - columns_to_drop: List of columns to drop (overrides default)
                - columns_to_convert: List of columns to convert to numeric

        Returns:
            Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict]: A tuple containing:
                - Processed DataFrame if split_config is None
                - Dictionary of DataFrames if split_config is provided (keys: split names)
                - Metadata dictionary with processing information
        """
        from quantrl_lab.data.processing.pipeline import DataPipeline
        from quantrl_lab.data.processing.steps import (
            AnalystEstimatesStep,
            ColumnCleanupStep,
            MarketContextStep,
            NumericConversionStep,
            SentimentEnrichmentStep,
            TechnicalIndicatorStep,
        )

        # Resolve indicators — auto-select via AlphaSelector when requested
        if alpha_selection_config is not None and indicators is None:
            from quantrl_lab.alpha_research.selector import AlphaSelector

            selector = AlphaSelector(self.ohlcv_data, verbose=kwargs.get("verbose", False))
            indicators = selector.suggest_indicators(
                candidates=alpha_selection_config.get("candidates"),
                metric=alpha_selection_config.get("metric", "ic"),
                threshold=alpha_selection_config.get("threshold", 0.0),
                top_k=alpha_selection_config.get("top_k", 5),
            )

        # Build pipeline
        pipeline = DataPipeline()

        # 1. Technical Indicators
        pipeline.add_step(TechnicalIndicatorStep(indicators=indicators))

        # 2. Analyst Estimates
        if self.analyst_grades is not None or self.analyst_ratings is not None:
            pipeline.add_step(AnalystEstimatesStep(grades_df=self.analyst_grades, ratings_df=self.analyst_ratings))

        # 3. Market Context
        if self.sector_performance is not None or self.industry_performance is not None:
            pipeline.add_step(
                MarketContextStep(sector_perf_df=self.sector_performance, industry_perf_df=self.industry_performance)
            )

        # 4. Sentiment Enrichment (only if news data available)
        if self.news_data is not None:
            pipeline.add_step(
                SentimentEnrichmentStep(
                    news_data=self.news_data,
                    provider=self.sentiment_provider,
                    config=self.sentiment_config,
                    fillna_strategy=fillna_strategy,
                )
            )

        # 5. Numeric Conversion
        # Convert specified columns to numeric
        columns_to_convert = kwargs.get("columns_to_convert", None)
        pipeline.add_step(NumericConversionStep(columns=columns_to_convert))

        # 6. Column Cleanup
        # If columns_to_drop is passed, use it; otherwise rely on defaults in step
        # Note: We keep date columns if splitting is required later
        columns_to_drop = kwargs.get("columns_to_drop", None)
        # If splitting, we MUST keep date columns for the split operation
        # If not splitting, the pipeline step handles default date dropping unless overridden
        keep_date = split_config is not None

        # Configure Cleanup Step
        cleanup_step = ColumnCleanupStep(columns_to_drop=columns_to_drop, keep_date=keep_date)
        pipeline.add_step(cleanup_step)

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
        if alpha_selection_config is not None:
            metadata_obj.alpha_selection_config = alpha_selection_config

        # Handle Data Splitting (Post-Processing)
        # Debug: Check for columns with all NaN values before dropna
        verbose = kwargs.get("verbose", False)
        if verbose:
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

        if verbose:
            if dropped_count > 0:
                logger.info(
                    "Dropped {count} rows containing NaNs in required columns: {columns}",
                    count=dropped_count,
                    columns=required_columns,
                )
            else:
                logger.info("No rows dropped (data is clean)")

        if verbose:
            logger.info("After dropna: {rows} rows", rows=len(processed_data))

        if split_config:
            split_data, split_metadata = self._split_data(processed_data, split_config)

            # Merge split metadata into pipeline metadata
            metadata_obj.date_ranges = split_metadata["date_ranges"]
            metadata_obj.final_shapes = split_metadata["final_shapes"]

            # Drop date column after splitting if it wasn't supposed to be kept
            for key in split_data:
                # Re-run cleanup to drop date columns now that splitting is done
                # unless user explicitly asked to keep them via columns_to_drop logic?
                # For safety, we replicate old behavior: drop defaults
                split_data[key] = self.drop_unwanted_columns(
                    split_data[key], [config.DEFAULT_DATE_COLUMN, "Timestamp", "Symbol"]
                )

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

            # If we didn't split, we might still need to drop the date column if it was kept
            if not keep_date:
                pass

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
