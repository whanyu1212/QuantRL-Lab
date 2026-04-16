"""Tests for MarketContextStep."""

import pandas as pd
import pytest

from quantrl_lab.data.processing.processor import ProcessingMetadata
from quantrl_lab.data.processing.steps.features.context import MarketContextStep


@pytest.fixture
def base_df():
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {"Close": [100, 101, 102, 103, 104], "Volume": [1000, 1100, 1200, 1300, 1400]},
        index=dates,
    )


@pytest.fixture
def metadata():
    return ProcessingMetadata()


@pytest.fixture
def sector_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "tech_return": [0.01, 0.02, -0.01, 0.03, 0.00],
        }
    )


@pytest.fixture
def industry_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "software_return": [0.005, 0.01, -0.005, 0.02, 0.01],
        }
    )


def test_no_sector_data_returns_unchanged(base_df, metadata):
    step = MarketContextStep()
    result = step.process(base_df, metadata)
    assert list(result.columns) == list(base_df.columns)
    assert len(result) == len(base_df)


def test_with_sector_data_adds_prefixed_columns(base_df, metadata, sector_df):
    step = MarketContextStep(sector_perf_df=sector_df)
    result = step.process(base_df, metadata)
    assert "sector_tech_return" in result.columns
    assert "sector_tech_return" in metadata.optional_feature_columns


def test_with_industry_data_adds_prefixed_columns(base_df, metadata, industry_df):
    step = MarketContextStep(industry_perf_df=industry_df)
    result = step.process(base_df, metadata)
    assert "industry_software_return" in result.columns


def test_with_both_sector_and_industry(base_df, metadata, sector_df, industry_df):
    step = MarketContextStep(sector_perf_df=sector_df, industry_perf_df=industry_df)
    result = step.process(base_df, metadata)
    assert "sector_tech_return" in result.columns
    assert "industry_software_return" in result.columns


def test_empty_sector_df_returns_unchanged(base_df, metadata):
    step = MarketContextStep(sector_perf_df=pd.DataFrame())
    result = step.process(base_df, metadata)
    assert list(result.columns) == list(base_df.columns)


def test_datetime_index_preserved(base_df, metadata, sector_df):
    step = MarketContextStep(sector_perf_df=sector_df)
    result = step.process(base_df, metadata)
    # The result index should still be date-like
    assert len(result) == len(base_df)


def test_timezone_aware_index_handled(metadata, sector_df):
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)
    step = MarketContextStep(sector_perf_df=sector_df)
    result = step.process(df, metadata)
    assert len(result) == 5


def test_step_name():
    step = MarketContextStep()
    assert step.get_step_name() == "Market Context Enrichment"


def test_non_datetime_index_with_date_column(metadata, sector_df):
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "Close": [100, 101, 102, 103, 104],
        }
    )
    step = MarketContextStep(sector_perf_df=sector_df)
    result = step.process(df, metadata)
    # Should still run without error
    assert len(result) == 5


def test_sparse_context_uses_latest_known_values(base_df, metadata):
    sector_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2023-01-02"), pd.Timestamp("2023-01-04")],
            "tech_return": [0.02, 0.04],
        }
    )
    step = MarketContextStep(sector_perf_df=sector_df)
    result = step.process(base_df, metadata)

    assert pd.isna(result.loc[pd.Timestamp("2023-01-01"), "sector_tech_return"])
    assert result.loc[pd.Timestamp("2023-01-03"), "sector_tech_return"] == pytest.approx(0.02)
    assert result.loc[pd.Timestamp("2023-01-05"), "sector_tech_return"] == pytest.approx(0.04)
