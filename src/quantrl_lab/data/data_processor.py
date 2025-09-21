from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from rich.console import Console
from transformers import pipeline

from quantrl_lab.data.indicators.indicator_registry import IndicatorRegistry
from quantrl_lab.data.indicators.technical_indicators import *  # noqa: F401, F403

console = Console()


@dataclass
class SentimentConfig:  # default
    """Configuration for sentiment analysis."""

    model_name: str = "ProsusAI/finbert"
    text_column: str = "headline"
    date_column: str = "created_at"
    sentiment_score_column: str = "sentiment_score"
    device: int = -1  # -1 for CPU, 0 for GPU
    max_length: Optional[int] = None
    truncation: bool = True
    top_k: int = 1  # `return_all_scores` is deprecated

    # Supported models for validation
    SUPPORTED_MODELS: List[str] = field(
        default_factory=lambda: [
            "ProsusAI/finbert",
            "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            "ElKulako/cryptobert",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
        ]
    )

    def __post_init__(self):
        """Validate configuration after initialization."""

        if self.device < -1:
            raise ValueError("device must be -1 (CPU) or >= 0 (GPU)")

        if self.model_name not in self.SUPPORTED_MODELS:
            console.print(f"[yellow]⚠️  Warning: Model '{self.model_name}' not in supported models list.[/yellow]")
            console.print(f"[cyan]Supported models: {self.SUPPORTED_MODELS}[/cyan]")

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "model_name": self.model_name,
            "text_column": self.text_column,
            "date_column": self.date_column,
            "sentiment_score_column": self.sentiment_score_column,
            "device": self.device,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "top_k": self.top_k,
        }


class DataProcessor:
    def __init__(self, olhcv_data: pd.DataFrame, **kwargs):
        self.olhcv_data = olhcv_data  # minimal required data

        # === Optional data sources ===
        self.news_data = kwargs.get("news_data", None)
        self.fundamental_data = kwargs.get("fundamental_data", None)
        self.macro_data = kwargs.get("macro_data", None)
        self.calendar_event_data = kwargs.get("calendar_event_data", None)

        sentiment_config_input = kwargs.get("sentiment_config", {})
        if isinstance(sentiment_config_input, SentimentConfig):
            self.sentiment_config = sentiment_config_input
        elif isinstance(sentiment_config_input, dict):
            self.sentiment_config = SentimentConfig(**sentiment_config_input)
        else:
            self.sentiment_config = SentimentConfig()

        self._sentiment_pipeline = None

    def _get_sentiment_pipeline(self):
        """Lazy initialization of sentiment analysis pipeline."""
        if self._sentiment_pipeline is None:
            try:

                # Use GPU if available and device is set to 0
                device = 0 if torch.cuda.is_available() and self.sentiment_config.device == 0 else -1

                pipeline_kwargs = {
                    "model": self.sentiment_config.model_name,
                    "tokenizer": self.sentiment_config.model_name,
                    "device": device,
                    "truncation": self.sentiment_config.truncation,
                    "top_k": self.sentiment_config.top_k,
                }

                if self.sentiment_config.max_length:
                    pipeline_kwargs["max_length"] = self.sentiment_config.max_length

                self._sentiment_pipeline = pipeline("sentiment-analysis", **pipeline_kwargs)
                console.print(
                    f"[green]✓ Sentiment analysis pipeline initialized with model: "
                    f"{self.sentiment_config.model_name}[/green]"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load sentiment model: {e}")

        return self._sentiment_pipeline

    def _get_news_sentiment_scores(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of news articles. If sentiment scores are
        already present, use them. Otherwise, use the HF model to
        calculate them.

        Args:
            news_data (pd.DataFrame): DataFrame containing news articles.

        Raises:
            ValueError: If news_data is empty.
            ValueError: If text_column is not found in news_data.
            ValueError: If date_column is not found in news_data.
            ValueError: If sentiment analysis fails.
            ValueError: If no valid text data is found.

        Returns:
            pd.DataFrame: DataFrame containing sentiment scores at the daily level.
        """

        # === Input validation ===
        if news_data.empty:
            raise ValueError("News data cannot be empty")

        required_columns = [self.sentiment_config.date_column]
        # If sentiment score is not present, text column is required for analysis
        if self.sentiment_config.sentiment_score_column not in news_data.columns:
            required_columns.append(self.sentiment_config.text_column)

        for col in required_columns:
            if col not in news_data.columns:
                raise ValueError(f"Required column '{col}' not found. Available columns: {list(news_data.columns)}")

        # === Process sentiment scores ===
        if self.sentiment_config.sentiment_score_column in news_data.columns:
            console.print("[green]✓ Using pre-existing sentiment scores.[/green]")
            # Ensure the sentiment score column is numeric
            news_data[self.sentiment_config.sentiment_score_column] = pd.to_numeric(
                news_data[self.sentiment_config.sentiment_score_column], errors="coerce"
            )
        else:
            console.print("[cyan]Calculating sentiment scores using HF model...[/cyan]")
            # === Transformers pipeline initialization ===
            sentiment_pipeline = self._get_sentiment_pipeline()

            # === Fillna just in case ===
            texts_to_analyze = news_data[self.sentiment_config.text_column].fillna("").astype(str).tolist()

            if not texts_to_analyze:
                raise ValueError("No valid text data found for sentiment analysis")

            sentiments = sentiment_pipeline(texts_to_analyze)

            # Handle cases where each result might itself be a list
            scores = []
            for result in sentiments:
                if isinstance(result, list):
                    # Ensure result is not empty
                    if result:
                        scores.append(result[0].get("score", 0.0))
                    else:
                        scores.append(0.0)
                else:
                    scores.append(result.get("score", 0.0))

            news_data[self.sentiment_config.sentiment_score_column] = scores

        # === Process date column ===
        news_data[self.sentiment_config.date_column] = pd.to_datetime(
            news_data[self.sentiment_config.date_column]
        ).dt.date

        # === Group by date and calculate mean sentiment score ===
        news_data = (
            news_data.groupby(self.sentiment_config.date_column)
            .agg({self.sentiment_config.sentiment_score_column: "mean"})
            .reset_index()
        )
        news_data.rename(
            columns={
                self.sentiment_config.date_column: "Date",
                self.sentiment_config.sentiment_score_column: "sentiment_score",
            },
            inplace=True,
        )

        if news_data.empty:
            raise ValueError("No valid news data found after processing.")
        return news_data

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

        # Validate input DataFrame
        if df.empty:
            raise ValueError("Input DataFrame is empty. Technical indicators cannot be added.")

        # Check for required columns (case-insensitive)
        column_check = {col.lower(): col for col in df.columns}
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = []

        for req_col in required_cols:
            if req_col not in column_check and req_col.upper() not in column_check:
                missing_cols.append(req_col)

        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")

        result = df.copy()

        # Return original if no indicators specified
        if not indicators:
            return result

        available_indicators = set(IndicatorRegistry.list_all())

        for indicator_config in indicators:
            # Handle both string and dictionary formats
            if isinstance(indicator_config, str):
                # Simple string format: just the indicator name
                indicator_name = indicator_config
                custom_params = kwargs.get(f"{indicator_name}_params", {})
            elif isinstance(indicator_config, dict):
                # Dictionary format: {"IndicatorName": {"param1": value1, "param2": value2}}
                if len(indicator_config) != 1:
                    console.print(
                        f"[yellow]⚠️  Invalid indicator config format: {indicator_config}. " "Skipping.[/yellow]"
                    )
                    continue
                indicator_name = list(indicator_config.keys())[0]
                custom_params = indicator_config[indicator_name]
            else:
                console.print(
                    f"[yellow]⚠️  Invalid indicator config type: {type(indicator_config)}. " "Skipping.[/yellow]"
                )
                continue

            # Check if indicator exists in registry
            if indicator_name not in available_indicators:
                console.print(f"[yellow]⚠️  Indicator '{indicator_name}' not found in registry. Skipping.[/yellow]")
                continue

            try:
                # Handle different parameter formats
                if isinstance(custom_params, list):
                    # List of parameter dictionaries (e.g., [{"fast": 12, "slow": 26, "signal": 9},
                    # {"fast": 5, "slow": 15, "signal": 5}])
                    for param_set in custom_params:
                        if isinstance(param_set, dict):
                            console.print(f"[cyan]Applying {indicator_name} with params: {param_set}[/cyan]")
                            result = IndicatorRegistry.apply(indicator_name, result, **param_set)
                        else:
                            console.print(
                                f"[yellow]⚠️  Invalid parameter set format in list: {param_set}. Skipping.[/yellow]"
                            )
                elif isinstance(custom_params, dict) and any(isinstance(v, list) for v in custom_params.values()):
                    # Multiple parameter combinations (e.g., {"window": [10, 20, 50]})
                    import itertools

                    # Get all parameter combinations
                    param_names = list(custom_params.keys())
                    param_values = list(custom_params.values())

                    # Convert single values to lists for consistency
                    param_values = [v if isinstance(v, list) else [v] for v in param_values]

                    # Generate all combinations
                    for combination in itertools.product(*param_values):
                        params_dict = dict(zip(param_names, combination))
                        console.print(f"[cyan]Applying {indicator_name} with params: {params_dict}[/cyan]")
                        result = IndicatorRegistry.apply(indicator_name, result, **params_dict)
                elif isinstance(custom_params, dict):
                    # Single parameter dictionary
                    console.print(f"[cyan]Applying {indicator_name} with params: {custom_params}[/cyan]")
                    result = IndicatorRegistry.apply(indicator_name, result, **custom_params)
                else:
                    # Empty parameters or invalid format
                    console.print(f"[cyan]Applying {indicator_name} with default params[/cyan]")
                    result = IndicatorRegistry.apply(indicator_name, result)

            except Exception as e:
                console.print(f"[red]❌ Failed to apply indicator '{indicator_name}' - {e}[/red]")

        return result

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

        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot append news sentiment data.")

        if fillna_strategy not in ["neutral", "fill_forward"]:
            raise ValueError(
                f"Unsupported strategy: {fillna_strategy}. Supported strategies are 'neutral' and 'fill_forward'."
            )

        if self.news_data is None or self.news_data.empty:
            console.print("[yellow]⚠️  No news data provided. Skipping sentiment analysis.[/yellow]")
            return df

        sentiment_scores = self._get_news_sentiment_scores(self.news_data.copy())

        # Merge sentiment scores with OHLCV data
        # Ensure the date column in df is in the correct format for merging
        if "Date" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Date' column.")

        merged_data = pd.merge(df, sentiment_scores, on="Date", how="left")

        if fillna_strategy == "neutral":
            # Fill NaN sentiment scores with 0.0 for neutral strategy
            merged_data["sentiment_score"] = merged_data["sentiment_score"].fillna(0.0)
        elif fillna_strategy == "fill_forward":
            # Fill NaN sentiment scores with forward fill for fill-forward strategy
            merged_data["sentiment_score"] = merged_data["sentiment_score"].ffill()
        else:
            raise ValueError(
                f"Unsupported strategy: {fillna_strategy}. Supported strategies are 'neutral' and 'fill_forward'."
            )

        return merged_data

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
        if columns_to_drop is None:
            columns_to_drop = ["Date", "Timestamp", "Symbol"]
        elif not isinstance(columns_to_drop, list):
            raise ValueError("columns_to_drop must be a list of column names.")

        if keep_date:
            date_cols = ["Date", "date", "timestamp", "Timestamp"]
            columns_to_drop = [col for col in columns_to_drop if col not in date_cols]

        return df.drop(columns=columns_to_drop, errors="ignore")

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
        if columns is None:
            # Only convert object columns that are not date-like
            columns = []
            for col in df.columns:
                if df[col].dtype == "object":
                    # Skip columns that look like dates
                    if col.lower() in ["date", "timestamp", "created_at", "time"]:
                        continue
                    # Check if it's actually a date column by looking at the data
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample_val is not None:
                        try:
                            pd.to_datetime(sample_val)
                            # If conversion succeeds, it's probably a date column - skip it
                            continue
                        except (ValueError, TypeError):
                            # Not a date, safe to convert to numeric
                            columns.append(col)
        elif not isinstance(columns, list):
            raise ValueError("columns must be a list of column names.")

        for col in columns:
            if col in df.columns and df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def data_processing_pipeline(
        self,
        indicators: Optional[List[Union[str, Dict]]] = None,
        fillna_strategy: str = "neutral",
        split_config: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict]:
        """
        Main data processing pipeline.
        Args:
            indicators (Optional[List[Union[str, Dict]]], optional):
                List of indicators to apply. Defaults to None.
            fillna_strategy (str, optional): Strategy for handling missing sentiment scores.
                Defaults to "neutral".
            split_config (Optional[Dict], optional): Configuration for data splitting.
                If None, returns a single DataFrame. Otherwise, returns a dictionary of split DataFrames.
                Example by ratio: {'train': 0.7, 'test': 0.3}
                Example by dates: {'train': ('2020-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')}
        Returns:
            tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict]: A tuple containing:
                - Processed DataFrame (or dictionary of DataFrames if split)
                - Metadata dictionary containing processing information
        """
        # Initialize metadata collection
        metadata = {
            "symbol": None,
            "date_ranges": {},  # Changed from date_range to date_ranges
            "fillna_strategy": fillna_strategy,
            "technical_indicators": [],
            "news_sentiment_applied": False,
            "columns_dropped": [],
            "original_shape": self.olhcv_data.shape,
            "final_shapes": {},  # Changed from final_shape to final_shapes
        }

        # Extract symbol if available in the data
        if "Symbol" in self.olhcv_data.columns:
            unique_symbols = self.olhcv_data["Symbol"].unique()
            metadata["symbol"] = unique_symbols[0] if len(unique_symbols) == 1 else unique_symbols.tolist()

        # Track indicators applied
        if indicators:
            metadata["technical_indicators"] = indicators.copy() if isinstance(indicators, list) else [indicators]

        processed_data = self.append_technical_indicators(self.olhcv_data, indicators, **kwargs)

        if self.news_data is None:
            console.print("[yellow]⚠️  No news data provided. Skipping sentiment analysis.[/yellow]")
            final_data = processed_data
        else:
            data_w_sentiment = self.append_news_sentiment_data(processed_data, fillna_strategy)
            metadata["news_sentiment_applied"] = True
            final_data = data_w_sentiment

        # Drop unwanted columns if specified
        columns_to_drop = kwargs.get("columns_to_drop", None)
        if columns_to_drop is not None:
            final_data = self.drop_unwanted_columns(final_data, columns_to_drop)
            metadata["columns_dropped"] = columns_to_drop
        else:
            default_columns_to_drop = ["Date", "Timestamp", "Symbol"]
            # Only track actually dropped columns
            actually_dropped = [col for col in default_columns_to_drop if col in final_data.columns]
            final_data = self.drop_unwanted_columns(final_data, keep_date=True)  # Keep date for splitting
            metadata["columns_dropped"] = actually_dropped

        # Convert specified columns to numeric
        columns_to_convert = kwargs.get("columns_to_convert", None)
        if columns_to_convert is not None:
            final_data = self.convert_columns_to_numeric(final_data, columns_to_convert)
        else:
            final_data = self.convert_columns_to_numeric(final_data)

        final_data = final_data.dropna().reset_index(drop=True)

        if split_config:
            split_data, split_metadata = self._split_data(final_data, split_config)
            metadata["date_ranges"] = split_metadata["date_ranges"]
            metadata["final_shapes"] = split_metadata["final_shapes"]

            # Drop date column after splitting
            for key in split_data:
                split_data[key] = self.drop_unwanted_columns(split_data[key], ["Date", "Timestamp", "Symbol"])

            return split_data, metadata
        else:
            # Handle metadata for non-split data
            date_column = next(
                (col for col in ["Date", "date", "timestamp", "Timestamp"] if col in final_data.columns), None
            )
            if date_column:
                dates = pd.to_datetime(final_data[date_column])
                metadata["date_ranges"]["full_data"] = {
                    "start": dates.min().strftime("%Y-%m-%d"),
                    "end": dates.max().strftime("%Y-%m-%d"),
                }
            metadata["final_shapes"]["full_data"] = final_data.shape
            final_data = self.drop_unwanted_columns(final_data, ["Date", "Timestamp", "Symbol"])
            return final_data, metadata

    def _split_data(self, df: pd.DataFrame, split_config: Dict) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Split the data into respective sets according to the config.

        Args:
            df (pd.DataFrame): input dataframe
            split_config (Dict): split config in dictionary format

        Raises:
            ValueError: No date or timestamps to work with for splitting

        Returns:
            Tuple[Dict[str, pd.DataFrame], Dict]: datasets in dict and metadata
        """
        date_column = next((col for col in ["Date", "date", "timestamp", "Timestamp"] if col in df.columns), None)
        if not date_column:
            raise ValueError("Date column not found for splitting.")

        df[date_column] = pd.to_datetime(df[date_column]).dt.tz_localize(None)
        df = df.sort_values(by=date_column).reset_index(drop=True)

        split_data = {}
        metadata = {"date_ranges": {}, "final_shapes": {}}

        # Check if splitting by date ranges or ratios
        is_date_based = any(isinstance(v, (tuple, list)) for v in split_config.values())

        if is_date_based:
            for key, (start_date, end_date) in split_config.items():
                mask = (df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))
                subset = df[mask].copy()
                split_data[key] = subset
                if not subset.empty:
                    metadata["date_ranges"][key] = {
                        "start": subset[date_column].min().strftime("%Y-%m-%d"),
                        "end": subset[date_column].max().strftime("%Y-%m-%d"),
                    }
                    metadata["final_shapes"][key] = subset.shape
        else:  # Ratio-based split
            total_len = len(df)
            start_idx = 0
            for key, ratio in split_config.items():
                end_idx = start_idx + int(total_len * ratio)
                subset = df.iloc[start_idx:end_idx].copy()
                split_data[key] = subset
                if not subset.empty:
                    metadata["date_ranges"][key] = {
                        "start": subset[date_column].min().strftime("%Y-%m-%d"),
                        "end": subset[date_column].max().strftime("%Y-%m-%d"),
                    }
                    metadata["final_shapes"][key] = subset.shape
                start_idx = end_idx

        return split_data, metadata
