"""Column cleanup processing step."""

from typing import List, Optional

import pandas as pd

from quantrl_lab.data.config import config
from quantrl_lab.data.processing.metadata import ProcessingMetadata


class ColumnCleanupStep:
    """
    Drop unwanted columns from DataFrame.

    This step removes specified columns or default columns
    (Date, Timestamp, Symbol) from the DataFrame.

    Example:
        >>> step = ColumnCleanupStep(columns_to_drop=["Date", "Symbol"])
        >>> result = step.process(df, metadata)
    """

    def __init__(
        self,
        columns_to_drop: Optional[List[str]] = None,
        keep_date: bool = False,
        keep_symbol: bool = False,
    ):
        """
        Initialize column cleanup step.

        Args:
            columns_to_drop: List of columns to drop. If None, drops default columns.
            keep_date: If True, preserve date columns even if in columns_to_drop.
            keep_symbol: If True, preserve ``Symbol`` when using the default
                cleanup policy. Explicit caller-provided drop lists still win.
        """
        self._using_default_columns = columns_to_drop is None
        if columns_to_drop is None:
            self.columns_to_drop = [config.DEFAULT_DATE_COLUMN, "Timestamp", "Symbol"]
        elif not isinstance(columns_to_drop, list):
            raise ValueError(
                f"Invalid type for 'columns_to_drop': expected list, got {type(columns_to_drop).__name__}."
            )
        else:
            self.columns_to_drop = columns_to_drop

        self.keep_date = keep_date
        self.keep_symbol = keep_symbol

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Drop specified columns from DataFrame.

        Args:
            data: Input DataFrame
            metadata: Processing metadata (updated with dropped columns)

        Returns:
            DataFrame with columns removed
        """
        columns_to_drop = self.columns_to_drop.copy()

        if self.keep_date:
            columns_to_drop = [col for col in columns_to_drop if col not in config.DATE_COLUMNS]
        if self.keep_symbol and self._using_default_columns:
            columns_to_drop = [col for col in columns_to_drop if col != "Symbol"]

        # Track actually dropped columns
        actually_dropped = [col for col in columns_to_drop if col in data.columns]
        for column in actually_dropped:
            if column not in metadata.columns_dropped:
                metadata.columns_dropped.append(column)

        return data.drop(columns=columns_to_drop, errors="ignore")

    def get_step_name(self) -> str:
        """Return step name."""
        return "Column Cleanup"
