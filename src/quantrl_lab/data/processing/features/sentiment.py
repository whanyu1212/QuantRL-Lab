"""Sentiment feature generator for news data."""

from typing import Dict

import pandas as pd

from quantrl_lab.data.processing.sentiment.config import SentimentConfig
from quantrl_lab.data.processing.sentiment.provider import SentimentProvider


class SentimentFeatureGenerator:
    """
    Generator for adding sentiment scores from news data.

    This generator uses a SentimentProvider to analyze news and merge
    sentiment scores with OHLCV data.

    Example:
        >>> from quantrl_lab.data.processing.features.sentiment import SentimentFeatureGenerator
        >>> from quantrl_lab.data.processing.sentiment import HuggingFaceProvider, HuggingFaceConfig, SentimentConfig
        >>> provider = HuggingFaceProvider(HuggingFaceConfig(batch_size=32))
        >>> config = SentimentConfig(text_column="headline", date_column="created_at")
        >>> generator = SentimentFeatureGenerator(provider, config, news_df)
        >>> enriched_df = generator.generate(ohlcv_df)
    """

    def __init__(
        self,
        sentiment_provider: SentimentProvider,
        sentiment_config: SentimentConfig,
        news_data: pd.DataFrame,
        fillna_strategy: str = "neutral",
        decay_rate: float = 0.8,
    ):
        """
        Initialize SentimentFeatureGenerator.

        Args:
            sentiment_provider (SentimentProvider): Provider for sentiment analysis.
            sentiment_config (SentimentConfig): Configuration for sentiment analysis.
            news_data (pd.DataFrame): News data containing text to analyze.
            fillna_strategy (str): Strategy for missing values ("neutral", "fill_forward", "exponential_decay").
            decay_rate (float): Decay factor for exponential decay (default: 0.8).
                                Only used if fillna_strategy="exponential_decay".

        Raises:
            ValueError: If news_data is empty or fillna_strategy is invalid.
        """
        if news_data is None or news_data.empty:
            raise ValueError("News data cannot be None or empty")

        valid_strategies = ["neutral", "fill_forward", "exponential_decay"]
        if fillna_strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported strategy: {fillna_strategy}. " f"Supported strategies are {', '.join(valid_strategies)}."
            )

        self.sentiment_provider = sentiment_provider
        self.sentiment_config = sentiment_config
        self.news_data = news_data
        self.fillna_strategy = fillna_strategy
        self.decay_rate = decay_rate

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate DataFrame with sentiment scores.

        Args:
            data (pd.DataFrame): Input OHLCV DataFrame with Date column.
            **kwargs: Additional parameters (unused, for protocol compatibility).

        Returns:
            pd.DataFrame: DataFrame with sentiment_score column added.

        Raises:
            ValueError: If data is empty or Date column missing.
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty. Cannot append news sentiment data.")

        if "Date" not in data.columns:
            raise ValueError("Input DataFrame must contain a 'Date' column.")

        # Get sentiment scores from provider
        sentiment_scores = self.sentiment_provider.analyze(self.news_data.copy(), self.sentiment_config)

        # Merge with OHLCV data
        # Convert both Date columns to date for merging compatibility
        data_copy = data.copy()
        data_copy["Date"] = pd.to_datetime(data_copy["Date"]).dt.date
        sentiment_scores["Date"] = pd.to_datetime(sentiment_scores["Date"]).dt.date

        merged_data = pd.merge(data_copy, sentiment_scores, on="Date", how="left")

        # Restore original Date column format
        merged_data["Date"] = data["Date"].values

        # Apply fillna strategy
        if self.fillna_strategy == "neutral":
            merged_data["sentiment_score"] = merged_data["sentiment_score"].fillna(0.0)
        elif self.fillna_strategy == "fill_forward":
            merged_data["sentiment_score"] = merged_data["sentiment_score"].ffill()
        elif self.fillna_strategy == "exponential_decay":
            # 1. Identify valid values (where data was present)
            valid_mask = merged_data["sentiment_score"].notna()

            # 2. Create groups for decay calculation
            # Each valid value starts a new group. cumsum() increments when True is encountered.
            groups = valid_mask.cumsum()

            # 3. Calculate "staleness" (days since last valid value)
            # cumcount() gives 0 for the valid value, 1 for the next, etc.
            days_since = merged_data.groupby(groups).cumcount()

            # 4. Forward fill the base values to decay from
            ffilled_scores = merged_data["sentiment_score"].ffill()

            # 5. Apply decay formula: Value * (decay_rate ^ days_since)
            # Note: For the valid day (days_since=0), decay_rate^0 = 1, so value is unchanged
            decay_factors = self.decay_rate**days_since
            merged_data["sentiment_score"] = ffilled_scores * decay_factors

            # Handle leading NaNs (ffill leaves them as NaN, multiplication by decay keeps them NaN)
            # Optional: Fill remaining leading NaNs with 0.0 if desired, but standard behavior is usually neutral
            merged_data["sentiment_score"] = merged_data["sentiment_score"].fillna(0.0)

        return merged_data

    def get_metadata(self) -> Dict:
        """
        Return metadata about feature generation.

        Returns:
            Dict: Dictionary containing feature generation details.
        """
        return {
            "type": "sentiment",
            "provider": self.sentiment_provider.__class__.__name__,
            "fillna_strategy": self.fillna_strategy,
            "decay_rate": self.decay_rate if self.fillna_strategy == "exponential_decay" else None,
            "news_records": len(self.news_data),
        }
