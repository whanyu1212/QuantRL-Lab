"""Sentiment enrichment processing step."""

import pandas as pd
from loguru import logger

from quantrl_lab.data.processing.features.sentiment import SentimentFeatureGenerator
from quantrl_lab.data.processing.metadata import ProcessingMetadata
from quantrl_lab.data.processing.sentiment import SentimentConfig, SentimentProvider


class SentimentEnrichmentStep:
    """
    Add news sentiment scores to DataFrame.

    This step enriches OHLCV data with sentiment scores computed from
    news data. Requires news_data to be provided.

    Example:
        >>> step = SentimentEnrichmentStep(
        ...     news_data=news_df,
        ...     provider=HuggingFaceProvider(),
        ...     fillna_strategy="neutral"
        ... )
        >>> result = step.process(df, metadata)
    """

    def __init__(
        self,
        news_data: pd.DataFrame,
        provider: SentimentProvider = None,
        config: SentimentConfig = None,
        fillna_strategy: str = "neutral",
    ):
        """
        Initialize sentiment enrichment step.

        Args:
            news_data: DataFrame with news articles
            provider: Sentiment analysis provider (default: HuggingFaceProvider)
            config: Sentiment configuration
            fillna_strategy: Strategy for filling missing scores ("neutral" or "fill_forward")
        """
        self.news_data = news_data
        self.provider = provider
        self.config = config or SentimentConfig()
        self.fillna_strategy = fillna_strategy

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Add sentiment scores to DataFrame.

        Args:
            data: Input OHLCV DataFrame
            metadata: Processing metadata (updated with sentiment flag)

        Returns:
            DataFrame with sentiment scores added

        Raises:
            ValueError: If news_data is empty or invalid
        """
        if self.news_data is None or self.news_data.empty:
            logger.debug("No news data provided. Skipping sentiment analysis.")
            return data

        restored_index = False
        df = data
        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()
            restored_index = True

        generator = SentimentFeatureGenerator(
            self.provider,
            self.config,
            self.news_data,
            self.fillna_strategy,
        )
        result = generator.generate(df)

        if restored_index and "Date" in result.columns:
            result = result.set_index("Date")

        metadata.news_sentiment_applied = True
        metadata.fillna_strategy = self.fillna_strategy
        required_columns = [col for col in result.columns if col not in data.columns]
        if "sentiment_score" in result.columns:
            required_columns.append("sentiment_score")
        metadata.add_required_columns(required_columns)

        return result

    def get_step_name(self) -> str:
        """Return step name."""
        return "Sentiment Enrichment"
