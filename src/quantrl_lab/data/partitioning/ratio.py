"""Ratio-based data splitter for time series data."""

import math
from typing import Dict

import pandas as pd


class RatioSplitter:
    """
    Split DataFrame by ratio (e.g., 70% train, 30% test).

    This splitter divides data sequentially based on specified ratios,
    maintaining temporal order for time series data.

    Example:
        >>> splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        >>> splits = splitter.split(df)
        >>> train_df = splits["train"]
        >>> test_df = splits["test"]
    """

    def __init__(self, ratios: Dict[str, float]):
        """
        Initialize RatioSplitter.

        Args:
            ratios (Dict[str, float]): Dictionary mapping split names to ratios.
                Ratios must sum to 1.0. Example: {"train": 0.7, "test": 0.3}

        Raises:
            ValueError: If ratios do not sum to 1.0 or any ratio is invalid.
        """
        if not ratios:
            raise ValueError("Ratios dictionary cannot be empty")

        for name, ratio in ratios.items():
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"Invalid ratio for '{name}': {ratio}. Must be in range (0, 1]")

        total_ratio = sum(ratios.values())
        if not math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio:.6f}")

        self.ratios = ratios

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split DataFrame by ratio.

        Args:
            df (pd.DataFrame): Input DataFrame to split. Should be sorted by time.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of split DataFrames.

        Raises:
            ValueError: If DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")

        # Find date column for metadata
        date_column = next((col for col in ["Date", "date", "timestamp", "Timestamp"] if col in df.columns), None)

        if date_column:
            # Ensure date column is datetime and remove timezone
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column]).dt.tz_localize(None)
            df = df.sort_values(by=date_column).reset_index(drop=True)

        total_len = len(df)
        start_idx = 0
        split_data = {}
        metadata_ranges = {}
        metadata_shapes = {}

        ratio_items = list(self.ratios.items())
        for idx, (name, ratio) in enumerate(ratio_items):
            if idx == len(ratio_items) - 1:
                end_idx = total_len
            else:
                end_idx = start_idx + int(total_len * ratio)
            subset = df.iloc[start_idx:end_idx].copy()
            split_data[name] = subset

            # Track metadata
            if not subset.empty and date_column:
                metadata_ranges[name] = {
                    "start": subset[date_column].min().strftime("%Y-%m-%d"),
                    "end": subset[date_column].max().strftime("%Y-%m-%d"),
                }
            metadata_shapes[name] = subset.shape
            start_idx = end_idx

        # Store metadata for get_metadata() call
        self._last_metadata = {
            "date_ranges": metadata_ranges,
            "final_shapes": metadata_shapes,
        }

        return split_data

    def get_metadata(self) -> Dict:
        """
        Return metadata about the split.

        Returns:
            Dict: Dictionary containing:
                - type: "ratio"
                - ratios: Configuration used
                - date_ranges: Date ranges for each split (if date column exists)
                - final_shapes: Shape of each split DataFrame
        """
        metadata = {
            "type": "ratio",
            "ratios": self.ratios,
        }

        # Add metadata from last split operation if available
        if hasattr(self, "_last_metadata"):
            metadata.update(self._last_metadata)

        return metadata
