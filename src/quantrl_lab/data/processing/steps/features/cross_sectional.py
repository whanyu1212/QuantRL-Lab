"""Cross-sectional feature processing step."""

from typing import List, Optional

import pandas as pd
from loguru import logger

from quantrl_lab.data.config import config
from quantrl_lab.data.processing.metadata import ProcessingMetadata


class CrossSectionalStep:
    """
    Apply cross-sectional calculations across a basket of stocks.

    This step operates on panel data (multiple symbols). It groups by the Date
    (typically the index) and calculates relative metrics across all symbols
    present on that specific day.

    Supported methods:
    - "zscore": (value - daily_mean) / daily_std
    - "rank": Percentile rank (0.0 to 1.0)
    - "mean_centered": value - daily_mean

    Example:
        >>> step = CrossSectionalStep(columns=["RSI_14", "Volume"], methods=["zscore", "rank"])
        >>> result = step.process(df, metadata)
    """

    def __init__(self, columns: List[str], methods: Optional[List[str]] = None, date_column: Optional[str] = None):
        """
        Initialize cross-sectional step.

        Args:
            columns: List of feature column names to process (e.g., ["RSI_14", "Volume"]).
            methods: List of cross-sectional methods to apply.
                     Supported: "zscore", "rank", "mean_centered".
            date_column: Optional explicit date column to group by. If not
                provided, the step uses the first known date column or a
                ``DatetimeIndex``.
        """
        self.columns = columns
        self.methods = methods if methods is not None else ["zscore"]
        self.date_column = date_column
        self.supported_methods = {"zscore", "rank", "mean_centered"}

        # Validate methods
        for m in self.methods:
            if m not in self.supported_methods:
                raise ValueError(f"Unsupported cross-sectional method: {m}. Use one of {self.supported_methods}")

    def _resolve_grouping_key(self, data: pd.DataFrame) -> Optional[pd.Series]:
        """Resolve the date-like grouping key for cross-sectional
        transforms."""
        if self.date_column is not None:
            if self.date_column not in data.columns:
                logger.warning(
                    "CrossSectionalStep date_column '{column}' not found. Skipping.", column=self.date_column
                )
                return None
            return pd.Series(pd.to_datetime(data[self.date_column], errors="coerce"), index=data.index)

        for candidate in config.DATE_COLUMNS + ["Timestamp"]:
            if candidate in data.columns:
                return pd.Series(pd.to_datetime(data[candidate], errors="coerce"), index=data.index)

        if pd.api.types.is_datetime64_any_dtype(data.index):
            return pd.Series(pd.to_datetime(data.index), index=data.index)

        logger.warning("CrossSectionalStep requires a date column or DatetimeIndex. Skipping.")
        return None

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Apply cross-sectional calculations to DataFrame.

        Args:
            data: Input panel DataFrame (must have a 'Symbol' column and Date index)
            metadata: Processing metadata

        Returns:
            DataFrame with cross-sectional features added
        """
        if data.empty:
            return data

        if "Symbol" not in data.columns:
            logger.warning("CrossSectionalStep requires a 'Symbol' column. Skipping.")
            return data

        if data["Symbol"].nunique() < 2:
            logger.debug("CrossSectionalStep bypassed because only one symbol is present.")
            return data

        result = data.copy()

        grouping_key = self._resolve_grouping_key(result)
        if grouping_key is None:
            return data

        grouped = result.groupby(grouping_key)

        for col in self.columns:
            if col not in result.columns:
                logger.warning("Column '{column}' not found for cross-sectional processing.", column=col)
                continue

            for method in self.methods:
                new_col_name = f"{col}_cs_{method}"

                if method == "zscore":
                    # (x - mean) / std (add epsilon to avoid division by zero)
                    result[new_col_name] = grouped[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
                elif method == "rank":
                    # Percentile rank (0.0 to 1.0)
                    result[new_col_name] = grouped[col].transform(lambda x: x.rank(pct=True))
                elif method == "mean_centered":
                    # x - mean
                    result[new_col_name] = grouped[col].transform(lambda x: x - x.mean())

        # Update metadata to track these new features
        generated_columns = [
            f"{c}_cs_{m}" for c in self.columns for m in self.methods if f"{c}_cs_{m}" in result.columns
        ]
        metadata.cross_sectional_features.extend(
            [column for column in generated_columns if column not in metadata.cross_sectional_features]
        )
        metadata.add_required_columns(generated_columns)

        return result

    def get_step_name(self) -> str:
        """Return step name."""
        return "Cross-Sectional Features"
