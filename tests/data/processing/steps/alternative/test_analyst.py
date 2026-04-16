"""Tests for AnalystEstimatesStep."""

import numpy as np
import pandas as pd

from quantrl_lab.data.processing.processor import ProcessingMetadata
from quantrl_lab.data.processing.steps.alternative.analyst import AnalystEstimatesStep


class TestAnalystEstimatesStep:
    """Test AnalystEstimatesStep functionality."""

    def test_asof_alignment_carries_latest_grade_forward(self):
        """Test that analyst grades are aligned using latest-known
        values."""
        # 1. Create Daily OHLCV Data (Starts Jan 3rd)
        dates = pd.date_range(start="2025-01-03", end="2025-01-10", freq="D")
        ohlcv_df = pd.DataFrame({"Date": dates, "Close": [100.0] * len(dates), "Volume": [1000] * len(dates)})

        # 2. Create Monthly Analyst Data (Jan 1st)
        grades_df = pd.DataFrame(
            {"date": [pd.Timestamp("2025-01-01")], "analystRatingsStrongBuy": [10.0], "symbol": ["TEST"]}
        )

        step = AnalystEstimatesStep(grades_df=grades_df)
        metadata = ProcessingMetadata()

        # 3. Process
        result = step.process(ohlcv_df, metadata)

        # 4. Verify
        # Jan 3rd should have the grade from Jan 1st (latest known analyst update)
        val_jan3 = result.loc[result["Date"] == "2025-01-03", "analystRatingsStrongBuy"].iloc[0]

        assert not np.isnan(val_jan3), "Analyst data should not be NaN for Jan 3rd"
        assert val_jan3 == 10.0
        assert "analystRatingsStrongBuy" in metadata.optional_feature_columns

    def test_asof_alignment_updates_correctly(self):
        """Test that grades update when a newer analyst release
        appears."""
        # Data spans Jan and Feb
        dates = pd.date_range(start="2025-01-28", end="2025-02-03", freq="D")
        ohlcv_df = pd.DataFrame({"Date": dates, "Close": [100.0] * len(dates)})

        # Grades for Jan 1 and Feb 1
        grades_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01")],
                "analystRatingsStrongBuy": [10.0, 20.0],
                "symbol": ["TEST", "TEST"],
            }
        )

        step = AnalystEstimatesStep(grades_df=grades_df)
        result = step.process(ohlcv_df, ProcessingMetadata())

        # Jan 31 should have Jan grade (10.0)
        val_jan31 = result.loc[result["Date"] == "2025-01-31", "analystRatingsStrongBuy"].iloc[0]
        assert val_jan31 == 10.0

        # Feb 1 should have Feb grade (20.0)
        val_feb1 = result.loc[result["Date"] == "2025-02-01", "analystRatingsStrongBuy"].iloc[0]
        assert val_feb1 == 20.0

    def test_rows_before_first_analyst_update_remain_nan(self):
        """Test dates before the first analyst update are kept with
        missing optional features."""
        dates = pd.date_range(start="2025-01-01", end="2025-01-05", freq="D")
        ohlcv_df = pd.DataFrame({"Date": dates, "Close": [100.0] * len(dates)})
        grades_df = pd.DataFrame(
            {"date": [pd.Timestamp("2025-01-03")], "analystRatingsStrongBuy": [10.0], "symbol": ["TEST"]}
        )

        result = AnalystEstimatesStep(grades_df=grades_df).process(ohlcv_df, ProcessingMetadata())

        assert np.isnan(result.loc[result["Date"] == "2025-01-01", "analystRatingsStrongBuy"].iloc[0])
        assert result.loc[result["Date"] == "2025-01-04", "analystRatingsStrongBuy"].iloc[0] == 10.0

    def test_panel_data_respects_symbol_boundaries(self):
        """Test analyst data is aligned per symbol in multi-symbol
        panels."""
        ohlcv_df = pd.DataFrame(
            {
                "Date": [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-02")],
                "Symbol": ["AAPL", "MSFT"],
                "Close": [100.0, 200.0],
            }
        )
        grades_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
                "symbol": ["AAPL", "MSFT"],
                "analystRatingsStrongBuy": [10.0, 99.0],
            }
        )

        result = AnalystEstimatesStep(grades_df=grades_df).process(ohlcv_df, ProcessingMetadata())

        aapl_value = result.loc[result["Symbol"] == "AAPL", "analystRatingsStrongBuy"].iloc[0]
        msft_value = result.loc[result["Symbol"] == "MSFT", "analystRatingsStrongBuy"].iloc[0]
        assert aapl_value == 10.0
        assert msft_value == 99.0
