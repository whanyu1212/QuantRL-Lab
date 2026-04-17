"""Numeric conversion processing step."""

from typing import List, Optional

import pandas as pd

from quantrl_lab.data.config import config
from quantrl_lab.data.processing.metadata import ProcessingMetadata


class NumericConversionStep:
    """
    Convert DataFrame columns to numeric types.

    This step converts object columns to numeric, while preserving
    date columns. Useful for ensuring proper data types before
    feeding to ML models.

    Example:
        >>> step = NumericConversionStep(columns=["volume", "price"])
        >>> result = step.process(df, metadata)
    """

    def __init__(self, columns: Optional[List[str]] = None):
        """
        Initialize numeric conversion step.

        Args:
            columns (List[str], optional): Specific columns to convert. If None, converts
                all object columns excluding date columns. Defaults to None.
        """
        self.columns = columns

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Convert specified columns to numeric.

        Args:
            data (pd.DataFrame): Input DataFrame.
            metadata (ProcessingMetadata): Processing metadata (not modified).

        Returns:
            pd.DataFrame: DataFrame with numeric columns.

        Raises:
            ValueError: If columns is not a list.
        """
        df = data.copy()
        columns_to_convert = self.columns

        if columns_to_convert is None:
            columns_to_convert = []
            for col in df.columns:
                if df[col].dtype == "object":
                    if col == "Symbol":
                        continue
                    if col in config.DATE_COLUMNS or col.lower() in [c.lower() for c in config.DATE_COLUMNS]:
                        continue

                    # Sample a value to detect date columns not listed in DATE_COLUMNS
                    sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample_val is not None:
                        try:
                            pd.to_datetime(sample_val)
                            continue
                        except (ValueError, TypeError):
                            columns_to_convert.append(col)
        elif not isinstance(columns_to_convert, list):
            raise ValueError(f"Invalid type for 'columns': expected list, got {type(columns_to_convert).__name__}.")

        for col in columns_to_convert:
            if col in df.columns and df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_step_name(self) -> str:
        """Return step name."""
        return "Numeric Conversion"
