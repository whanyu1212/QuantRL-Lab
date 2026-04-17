"""Tests for deprecated alpha-selection usage on DataProcessor."""

import pandas as pd
import pytest

from quantrl_lab.data.processing import DataProcessor


@pytest.fixture
def sample_single_symbol_data() -> pd.DataFrame:
    """Create simple single-symbol OHLCV data."""
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": range(12),
            "High": range(100, 112),
            "Low": range(10, 22),
            "Close": range(50, 62),
            "Volume": range(1000, 1012),
            "Symbol": ["AAPL"] * 12,
        }
    )


def test_data_processor_rejects_alpha_selection_shortcut(sample_single_symbol_data):
    """DataProcessor should no longer run alpha selection internally."""
    processor = DataProcessor(sample_single_symbol_data)

    with pytest.warns(FutureWarning, match="build_processing_config_from_alpha_selection"):
        with pytest.raises(ValueError, match="no longer performs alpha selection"):
            processor.data_processing_pipeline(
                alpha_selection_config={"metric": "ic"},
                split_config={"train": 0.7, "test": 0.3},
            )
