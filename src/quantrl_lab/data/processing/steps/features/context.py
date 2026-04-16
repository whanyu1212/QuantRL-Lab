"""Market context processing steps."""

import pandas as pd

from quantrl_lab.data.processing.metadata import ProcessingMetadata
from quantrl_lab.data.utils import merge_asof_features


class MarketContextStep:
    """
    Merge broad market context (Sector/Industry performance) into the
    DataFrame.

    This step allows the agent to see how the specific stock's sector or
    industry is performing relative to the stock itself.

    Attributes:
        sector_perf_df (pd.DataFrame): Historical sector performance.
        industry_perf_df (pd.DataFrame): Historical industry performance.
    """

    def __init__(self, sector_perf_df: pd.DataFrame = None, industry_perf_df: pd.DataFrame = None):
        self.sector_perf_df = sector_perf_df
        self.industry_perf_df = industry_perf_df

    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        """
        Merge sector and industry data.

        Args:
            data: Input OHLCV DataFrame
            metadata: Processing metadata

        Returns:
            DataFrame with added context features (prefixed with sector_ or industry_)
        """
        if (self.sector_perf_df is None or self.sector_perf_df.empty) and (
            self.industry_perf_df is None or self.industry_perf_df.empty
        ):
            return data

        df = data.copy()

        # --- Merge Sector Performance ---
        if self.sector_perf_df is not None and not self.sector_perf_df.empty:
            sector_df = self.sector_perf_df.copy()
            if "date" in sector_df.columns:
                numeric_cols = sector_df.select_dtypes(include=["number"]).columns
                sector_payload = sector_df[["date", *numeric_cols]]
                df, added_columns = merge_asof_features(
                    df,
                    sector_payload,
                    feature_date_column="date",
                    prefix="sector_",
                )
                metadata.add_optional_columns(added_columns)

        # --- Merge Industry Performance ---
        if self.industry_perf_df is not None and not self.industry_perf_df.empty:
            ind_df = self.industry_perf_df.copy()
            if "date" in ind_df.columns:
                numeric_cols = ind_df.select_dtypes(include=["number"]).columns
                industry_payload = ind_df[["date", *numeric_cols]]
                df, added_columns = merge_asof_features(
                    df,
                    industry_payload,
                    feature_date_column="date",
                    prefix="industry_",
                )
                metadata.add_optional_columns(added_columns)

        return df

    def get_step_name(self) -> str:
        return "Market Context Enrichment"
