"""Analyst data processing steps."""

import pandas as pd
from loguru import logger

from quantrl_lab.data.processing.metadata import ProcessingMetadata
from quantrl_lab.data.utils import merge_asof_features


class AnalystEstimatesStep:
    """
    Merge analyst grades and ratings into the DataFrame.

    This step merges historical analyst data (grades, ratings) onto the main
    OHLCV DataFrame based on timestamps. Since analyst updates are sparse,
    values are forward-filled to represent the "current" analyst consensus
    at each time step.

    Attributes:
        grades_df (pd.DataFrame): Historical grades data.
        ratings_df (pd.DataFrame): Historical ratings data.
    """

    def __init__(self, grades_df: pd.DataFrame = None, ratings_df: pd.DataFrame = None):
        self.grades_df = grades_df
        self.ratings_df = ratings_df

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Merge and forward-fill analyst data.

        Args:
            data: Input OHLCV DataFrame (must have datetime index or 'Date'/'Timestamp' column)
            metadata: Processing metadata

        Returns:
            DataFrame with added analyst features
        """
        if (self.grades_df is None or self.grades_df.empty) and (self.ratings_df is None or self.ratings_df.empty):
            return data

        df = data.copy()

        # --- Process Grades ---
        if self.grades_df is not None and not self.grades_df.empty:
            grades = self.grades_df.copy()
            if "date" in grades.columns:
                dedupe_columns = ["date", "symbol"] if "symbol" in grades.columns else ["date"]
                grades = grades.sort_values(dedupe_columns).drop_duplicates(subset=dedupe_columns, keep="last")
                base_by_columns = ["Symbol"] if "Symbol" in df.columns and "symbol" in grades.columns else []
                feature_by_columns = ["symbol"] if base_by_columns else []
                df, added_columns = merge_asof_features(
                    df,
                    grades,
                    feature_date_column="date",
                    drop_columns=["symbol"],
                    base_by_columns=base_by_columns,
                    feature_by_columns=feature_by_columns,
                )
                metadata.add_optional_columns(added_columns)
            else:
                logger.warning("Analyst grades data is missing a 'date' column. Skipping grades enrichment.")

        # --- Process Ratings ---
        if self.ratings_df is not None and not self.ratings_df.empty:
            ratings = self.ratings_df.copy()
            if "date" in ratings.columns:
                dedupe_columns = ["date", "symbol"] if "symbol" in ratings.columns else ["date"]
                ratings = ratings.sort_values(dedupe_columns).drop_duplicates(subset=dedupe_columns, keep="last")
                base_by_columns = ["Symbol"] if "Symbol" in df.columns and "symbol" in ratings.columns else []
                feature_by_columns = ["symbol"] if base_by_columns else []
                df, added_columns = merge_asof_features(
                    df,
                    ratings,
                    feature_date_column="date",
                    drop_columns=["symbol", "rating"],
                    base_by_columns=base_by_columns,
                    feature_by_columns=feature_by_columns,
                )
                metadata.add_optional_columns(added_columns)
            else:
                logger.warning("Analyst ratings data is missing a 'date' column. Skipping ratings enrichment.")

        return df

    def get_step_name(self) -> str:
        return "Analyst Estimates Enrichment"
