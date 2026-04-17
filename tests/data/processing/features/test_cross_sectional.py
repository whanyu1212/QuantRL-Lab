"""Tests for CrossSectionalStep."""

import pandas as pd
import pytest

from quantrl_lab.data.processing.processor import ProcessingMetadata
from quantrl_lab.data.processing.steps.features.cross_sectional import CrossSectionalStep


@pytest.fixture
def panel_df():
    """Multi-symbol panel DataFrame with DatetimeIndex."""
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    symbols = ["AAPL", "MSFT", "GOOG"]
    rows = []
    for date in dates:
        for i, sym in enumerate(symbols):
            rows.append({"Symbol": sym, "RSI_14": 30.0 + i * 20, "Volume": 1000 + i * 500})
    idx = []
    for date in dates:
        for _ in symbols:
            idx.append(date)
    df = pd.DataFrame(rows, index=idx)
    df.index = pd.DatetimeIndex(df.index)
    return df


@pytest.fixture
def metadata():
    return ProcessingMetadata()


def test_zscore_method_creates_column(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["zscore"])
    result = step.process(panel_df, metadata)
    assert "RSI_14_cs_zscore" in result.columns


def test_rank_method_creates_column(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["rank"])
    result = step.process(panel_df, metadata)
    assert "RSI_14_cs_rank" in result.columns


def test_mean_centered_method_creates_column(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["mean_centered"])
    result = step.process(panel_df, metadata)
    assert "RSI_14_cs_mean_centered" in result.columns


def test_rank_values_between_zero_and_one(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["rank"])
    result = step.process(panel_df, metadata)
    ranks = result["RSI_14_cs_rank"].dropna()
    assert (ranks >= 0.0).all()
    assert (ranks <= 1.0).all()


def test_zscore_mean_approximately_zero(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["zscore"])
    result = step.process(panel_df, metadata)
    # For each date group, the mean of z-scores should be ~0
    for date, group in result.groupby(level=0):
        mean_z = group["RSI_14_cs_zscore"].mean()
        assert abs(mean_z) < 1e-6


def test_mean_centered_sum_approximately_zero(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["mean_centered"])
    result = step.process(panel_df, metadata)
    for date, group in result.groupby(level=0):
        assert abs(group["RSI_14_cs_mean_centered"].sum()) < 1e-9


def test_single_symbol_skipped(metadata):
    """Only 1 symbol — cross-sectional step should bypass and return df
    unchanged."""
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Symbol": ["AAPL"] * 3, "RSI_14": [30, 50, 70]}, index=dates)
    step = CrossSectionalStep(columns=["RSI_14"], methods=["zscore"])
    result = step.process(df, metadata)
    assert "RSI_14_cs_zscore" not in result.columns


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unsupported cross-sectional method"):
        CrossSectionalStep(columns=["RSI_14"], methods=["invalid_method"])


def test_missing_column_skipped(panel_df, metadata):
    """Gracefully skips columns not present in data."""
    step = CrossSectionalStep(columns=["NONEXISTENT_COL"], methods=["zscore"])
    result = step.process(panel_df, metadata)
    assert "NONEXISTENT_COL_cs_zscore" not in result.columns


def test_no_symbol_column_returns_unchanged(metadata):
    """DataFrame without Symbol column is returned as-is."""
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    df = pd.DataFrame({"RSI_14": [30, 50, 70]}, index=dates)
    step = CrossSectionalStep(columns=["RSI_14"], methods=["zscore"])
    result = step.process(df, metadata)
    assert "RSI_14_cs_zscore" not in result.columns


def test_empty_df_returns_empty(metadata):
    step = CrossSectionalStep(columns=["RSI_14"])
    result = step.process(pd.DataFrame(), metadata)
    assert result.empty


def test_step_name():
    step = CrossSectionalStep(columns=["RSI_14"])
    assert step.get_step_name() == "Cross-Sectional Features"


def test_multiple_methods(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["zscore", "rank"])
    result = step.process(panel_df, metadata)
    assert "RSI_14_cs_zscore" in result.columns
    assert "RSI_14_cs_rank" in result.columns


def test_metadata_updated(panel_df, metadata):
    step = CrossSectionalStep(columns=["RSI_14"], methods=["zscore"])
    step.process(panel_df, metadata)
    assert "RSI_14_cs_zscore" in metadata.cross_sectional_features


def test_date_column_supported_without_datetime_index(metadata):
    """Cross-sectional step should work with a date column instead of
    datetime index."""
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[1]],
            "Symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "RSI_14": [40.0, 60.0, 30.0, 70.0],
        }
    )

    step = CrossSectionalStep(columns=["RSI_14"], methods=["zscore"], date_column="Date")
    result = step.process(df, metadata)

    assert "RSI_14_cs_zscore" in result.columns
    for _, group in result.groupby("Date"):
        assert abs(group["RSI_14_cs_zscore"].mean()) < 1e-6
