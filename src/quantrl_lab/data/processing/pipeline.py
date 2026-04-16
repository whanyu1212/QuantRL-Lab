"""
Data processing pipeline for composable transformations.

This module provides a builder-pattern pipeline for chaining data
transformations. Each step in the pipeline processes the DataFrame and
updates metadata.
"""

from typing import List, Tuple

import pandas as pd
from loguru import logger

from quantrl_lab.data.processing.metadata import ProcessingMetadata
from quantrl_lab.data.processing.steps.base import ProcessingStep


class DataPipeline:
    """
    Composable data processing pipeline.

    DataPipeline allows chaining multiple processing steps together using
    a builder pattern. Each step transforms the DataFrame and can update
    the processing metadata.

    Example:
        >>> pipeline = (DataPipeline()
        ...     .add_step(TechnicalIndicatorStep(indicators=["SMA", "RSI"]))
        ...     .add_step(SentimentEnrichmentStep(news_data=news_df))
        ...     .add_step(ColumnCleanupStep(columns_to_drop=["Date"])))
        >>>
        >>> result_df, metadata = pipeline.execute(raw_df)
        >>> print(metadata.technical_indicators)  # ["SMA", "RSI"]
    """

    def __init__(self):
        """Initialize empty pipeline."""
        self._steps: List[ProcessingStep] = []

    def add_step(self, step: ProcessingStep) -> "DataPipeline":
        """
        Add a processing step to the pipeline.

        Args:
            step: ProcessingStep instance to add

        Returns:
            Self for method chaining (builder pattern)

        Example:
            >>> pipeline = DataPipeline()
            >>> pipeline.add_step(TechnicalIndicatorStep(["SMA"]))
            >>> pipeline.add_step(ColumnCleanupStep())
        """
        self._steps.append(step)
        return self

    def execute(self, df: pd.DataFrame, symbol: str = None) -> Tuple[pd.DataFrame, ProcessingMetadata]:
        """
        Execute all steps in the pipeline.

        Processes the DataFrame through each step sequentially, maintaining
        metadata throughout the pipeline.

        Args:
            df: Input DataFrame to process
            symbol: Optional symbol name for metadata tracking

        Returns:
            Tuple of (processed DataFrame, processing metadata)

        Raises:
            ValueError: If any step raises a validation error

        Example:
            >>> pipeline = DataPipeline().add_step(TechnicalIndicatorStep(["SMA"]))
            >>> result_df, metadata = pipeline.execute(raw_df)
            >>> assert "SMA_20" in result_df.columns
        """
        # Initialize metadata
        metadata = ProcessingMetadata(
            symbol=symbol,
            original_shape=df.shape,
        )
        metadata.add_required_columns(
            [col for col in df.columns if col.lower() in {"open", "high", "low", "close", "volume"}]
        )

        # Execute steps sequentially
        result = df.copy()
        for i, step in enumerate(self._steps):
            step_name = step.get_step_name()
            logger.debug(f"Executing step {i+1}/{len(self._steps)}: {step_name}")

            try:
                result = step.process(result, metadata)
            except Exception as e:
                logger.error(f"Step '{step_name}' failed: {e}")
                raise

        # Finalize metadata
        metadata.final_shapes = {"processed": result.shape}

        return result, metadata

    def get_steps(self) -> List[ProcessingStep]:
        """
        Get list of all steps in the pipeline.

        Returns:
            List of ProcessingStep instances
        """
        return self._steps.copy()

    def __len__(self) -> int:
        """Return number of steps in pipeline."""
        return len(self._steps)

    def __repr__(self) -> str:
        """Return string representation of pipeline."""
        step_names = [step.get_step_name() for step in self._steps]
        return f"DataPipeline({len(self._steps)} steps: {step_names})"
