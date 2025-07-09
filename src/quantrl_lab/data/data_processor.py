from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from loguru import logger
from transformers import pipeline

from quantrl_lab.data.indicators.indicator_registry import IndicatorRegistry
from quantrl_lab.data.indicators.technical_indicators import *  # noqa: F401, F403


@dataclass
class SentimentConfig:  # default
    """Configuration for sentiment analysis."""

    model_name: str = "ProsusAI/finbert"
    text_column: str = "headline"
    date_column: str = "created_at"
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
            print(
                f"⚠️  Warning: Model '{self.model_name}' not in supported models list. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'model_name': self.model_name,
            'text_column': self.text_column,
            'date_column': self.date_column,
            'device': self.device,
            'max_length': self.max_length,
            'truncation': self.truncation,
            'top_k': self.top_k,
        }


class DataProcessor:
    def __init__(self, olhcv_data: pd.DataFrame, **kwargs):
        self.olhcv_data = olhcv_data  # minimal required data
        self.news_data = kwargs.get('news_data', None)
        self.fundamental_data = kwargs.get('fundamental_data', None)
        self.macro_data = kwargs.get('macro_data', None)

        sentiment_config_input = kwargs.get('sentiment_config', {})
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
                    'model': self.sentiment_config.model_name,
                    'tokenizer': self.sentiment_config.model_name,
                    'device': device,
                    'truncation': self.sentiment_config.truncation,
                    'top_k': self.sentiment_config.top_k,
                }

                if self.sentiment_config.max_length:
                    pipeline_kwargs['max_length'] = self.sentiment_config.max_length

                self._sentiment_pipeline = pipeline("sentiment-analysis", **pipeline_kwargs)
                logger.info(f"Sentiment analysis pipeline initialized with model: {self.sentiment_config.model_name}")
            except Exception as e:
                raise RuntimeError(f"Failed to load sentiment model: {e}")

        return self._sentiment_pipeline

    def _get_news_sentiment_scores(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment of news articles.

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

        if self.sentiment_config.text_column not in news_data.columns:
            raise ValueError(
                f"Text column '{self.sentiment_config.text_column}' not found. "
                f"Available columns: {list(news_data.columns)}"
            )

        if self.sentiment_config.date_column not in news_data.columns:
            raise ValueError(
                f"Date column '{self.sentiment_config.date_column}' not found. "
                f"Available columns: {list(news_data.columns)}"
            )

        # === Transformers pipeline initialization ===
        sentiment_pipeline = self._get_sentiment_pipeline()

        # === Fillna just in case ===
        texts_to_analyze = news_data[self.sentiment_config.text_column].fillna("").astype(str).tolist()

        if not texts_to_analyze:
            raise ValueError("No valid text data found")

        sentiments = sentiment_pipeline(texts_to_analyze)

        # Handle cases where each result might itself be a list
        scores = []
        for result in sentiments:
            if isinstance(result, list):
                scores.append(result[0]['score'])
            else:
                scores.append(result['score'])

        news_data['sentiment_score'] = scores

        # === Process date column ===
        news_data[self.sentiment_config.date_column] = pd.to_datetime(
            news_data[self.sentiment_config.date_column]
        ).dt.date

        # === Group by date and calculate mean sentiment score ===
        news_data = news_data.groupby(self.sentiment_config.date_column).agg({'sentiment_score': 'mean'}).reset_index()
        news_data.rename(columns={self.sentiment_config.date_column: 'Date'}, inplace=True)

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

        for indicator_name in indicators:
            if indicator_name not in available_indicators:
                logger.warning(f"Indicator '{indicator_name}' not found in registry. Skipping.")
                continue
            try:
                # Extract custom parameters for this indicator if provided
                # Look for {indicator_name}_params in the kwargs
                custom_params = kwargs.get(f"{indicator_name}_params", {})

                # Apply the indicator with custom parameters
                logger.info(f"Applying {indicator_name} with params: {custom_params}")
                result = IndicatorRegistry.apply(indicator_name, result, **custom_params)

            except Exception as e:
                logger.error(f"Failed to apply indicator '{indicator_name}' - {e}")

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

        sentiment_scores = self._get_news_sentiment_scores(self.news_data)

        # Merge sentiment scores with OHLCV data
        merged_data = pd.merge(df, sentiment_scores, on='Date', how='left')

        if fillna_strategy == "neutral":
            # Fill NaN sentiment scores with 0.0 for neutral strategy
            merged_data['sentiment_score'] = merged_data['sentiment_score'].fillna(0.0)
        elif fillna_strategy == "fill_forward":
            # Fill NaN sentiment scores with forward fill for fill-forward strategy
            merged_data['sentiment_score'] = merged_data['sentiment_score'].fillna(method='ffill')
        else:
            raise ValueError(
                f"Unsupported strategy: {fillna_strategy}. Supported strategies are 'neutral' and 'fill_forward'."
            )

        return merged_data

    def data_processing_pipeline(
        self, indicators: Optional[List[Union[str, Dict]]] = None, fillna_strategy: str = "neutral", **kwargs
    ) -> pd.DataFrame:
        """
        Main data processing pipeline.

        Args:
            indicators (Optional[List[Union[str, Dict]]], optional):
                List of indicators to apply. Defaults to None.
            strategy (str, optional): Strategy for handling missing sentiment scores.
                Defaults to "neutral".

        Returns:
            pd.DataFrame: Processed DataFrame with technical indicators and sentiment scores.
        """
        processed_data = self.append_technical_indicators(self.olhcv_data, indicators, **kwargs)

        if self.news_data is None:
            logger.warning("No news data provided. Skipping sentiment analysis.")
            return processed_data

        return self.append_news_sentiment_data(processed_data, fillna_strategy)
