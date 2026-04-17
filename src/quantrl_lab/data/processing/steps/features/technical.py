"""Technical indicator processing step."""

from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from quantrl_lab.data.processing.features.technical import TechnicalFeatureGenerator
from quantrl_lab.data.processing.metadata import ProcessingMetadata


class TechnicalIndicatorStep:
    """
    Apply technical indicators to DataFrame.

    This step wraps TechnicalFeatureGenerator to add technical indicators
    as new columns. Indicators can be specified as strings (use defaults)
    or dicts (with custom parameters).

    Example:
        >>> step = TechnicalIndicatorStep(indicators=["SMA", {"RSI": {"window": 14}}])
        >>> result = step.process(df, metadata)
    """

    def __init__(self, indicators: Optional[List[Union[str, Dict]]] = None, strict: bool = False):
        """
        Initialize technical indicator step.

        Args:
            indicators: List of indicators to apply. Can be strings ("SMA")
                or dicts ({"SMA": {"window": 20}}).
            strict: If True, fail fast on unknown or broken indicators.
        """
        self.indicators = indicators or []
        self.strict = strict

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Apply technical indicators to DataFrame.

        Args:
            data: Input DataFrame with OHLCV data
            metadata: Processing metadata (updated with applied indicators)

        Returns:
            DataFrame with technical indicator columns added

        Raises:
            ValueError: If required columns are missing
        """
        if not self.indicators:
            return data.copy()

        generator = TechnicalFeatureGenerator(self.indicators, strict=self.strict)
        result = generator.generate(data)
        generator_metadata = generator.get_metadata()["last_run"]

        metadata.technical_indicators = self.indicators
        metadata.requested_technical_indicators = generator_metadata["requested"]
        metadata.applied_technical_indicators = generator_metadata["applied"]
        metadata.skipped_technical_indicators = generator_metadata["skipped"]
        metadata.failed_technical_indicators = generator_metadata["failed"]
        metadata.strict_indicators = self.strict
        metadata.add_required_columns([col for col in result.columns if col not in data.columns])
        logger.debug(
            "Applied technical indicators: requested={requested}, applied={applied}, "
            "skipped={skipped}, failed={failed}",
            requested=metadata.requested_technical_indicators,
            applied=metadata.applied_technical_indicators,
            skipped=metadata.skipped_technical_indicators,
            failed=metadata.failed_technical_indicators,
        )

        return result

    def get_step_name(self) -> str:
        """Return step name."""
        return "Technical Indicators"
