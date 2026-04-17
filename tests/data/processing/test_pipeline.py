"""Tests for high-level pipeline assembly and config objects."""

import pandas as pd
import pytest

from quantrl_lab.data.processing import (
    CleanupConfig,
    CrossSectionalConfig,
    DataPipeline,
    DataProcessor,
    ProcessingPipelineConfig,
    SplitConfig,
)
from quantrl_lab.data.processing.steps import ColumnCleanupStep, NumericConversionStep


@pytest.fixture
def panel_ohlcv_data() -> pd.DataFrame:
    """Create a small multi-symbol panel dataset with a Date column."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    for date in dates:
        rows.append(
            {
                "Date": date,
                "Symbol": "AAPL",
                "Open": 100.0,
                "High": 105.0,
                "Low": 99.0,
                "Close": 102.0,
                "Volume": 1_000_000,
            }
        )
        rows.append(
            {
                "Date": date,
                "Symbol": "MSFT",
                "Open": 200.0,
                "High": 210.0,
                "Low": 198.0,
                "Close": 205.0,
                "Volume": 1_500_000,
            }
        )
    return pd.DataFrame(rows)


def test_data_pipeline_exposes_step_names_and_description():
    """DataPipeline should provide lightweight introspection for
    assembled steps."""
    pipeline = DataPipeline().add_step(NumericConversionStep()).add_step(ColumnCleanupStep())

    assert pipeline.get_step_names() == ["Numeric Conversion", "Column Cleanup"]
    assert pipeline.describe() == {
        "step_count": 2,
        "steps": [
            {"index": 1, "name": "Numeric Conversion", "type": "NumericConversionStep"},
            {"index": 2, "name": "Column Cleanup", "type": "ColumnCleanupStep"},
        ],
    }


def test_build_pipeline_accepts_typed_pipeline_config(panel_ohlcv_data):
    """DataProcessor.build_pipeline should assemble a debuggable
    pipeline from config objects."""
    processor = DataProcessor(panel_ohlcv_data)
    pipeline = processor.build_pipeline(
        pipeline_config=ProcessingPipelineConfig(
            indicators=[{"SMA": {"window": 2}}],
            cross_sectional=CrossSectionalConfig(columns=["Close"], methods=["rank"]),
            split=SplitConfig(splits={"train": 0.6, "test": 0.4}),
        )
    )

    assert pipeline.get_step_names() == [
        "Technical Indicators",
        "Numeric Conversion",
        "Cross-Sectional Features",
        "Column Cleanup",
    ]
    assert pipeline.describe()["step_count"] == 4


def test_data_processing_pipeline_accepts_typed_pipeline_config(panel_ohlcv_data):
    """Typed config should flow through execution and preserve panel
    identity."""
    processor = DataProcessor(panel_ohlcv_data)
    result, metadata = processor.data_processing_pipeline(
        pipeline_config=ProcessingPipelineConfig(
            cross_sectional=CrossSectionalConfig(columns=["Close"], methods=["zscore", "rank"]),
            cleanup=CleanupConfig(),
        )
    )

    assert "Close_cs_zscore" in result.columns
    assert "Close_cs_rank" in result.columns
    assert "Symbol" in result.columns
    assert "Date" not in result.columns
    assert metadata["cross_sectional_features"] == ["Close_cs_zscore", "Close_cs_rank"]


def test_data_processing_pipeline_exposes_cross_sectional_args(panel_ohlcv_data):
    """Cross-sectional config should be usable without constructing the
    full pipeline config."""
    processor = DataProcessor(panel_ohlcv_data)
    result, metadata = processor.data_processing_pipeline(
        cross_sectional_config={"columns": ["Close"], "methods": ["rank"]},
    )

    assert "Close_cs_rank" in result.columns
    assert "Symbol" in result.columns
    assert metadata["cross_sectional_features"] == ["Close_cs_rank"]


def test_build_pipeline_rejects_mixed_pipeline_styles(panel_ohlcv_data):
    """Users should pick either explicit args or a pipeline config
    object."""
    processor = DataProcessor(panel_ohlcv_data)

    with pytest.raises(ValueError, match="Use either pipeline_config or individual pipeline arguments"):
        processor.build_pipeline(
            indicators=["SMA"],
            pipeline_config=ProcessingPipelineConfig(indicators=["RSI"]),
        )


def test_legacy_cleanup_kwargs_still_work_with_warning(panel_ohlcv_data):
    """Legacy kwargs should remain backward compatible while steering
    users to the typed config API."""
    processor = DataProcessor(panel_ohlcv_data)

    with pytest.warns(FutureWarning, match="cleanup_config"):
        result, _ = processor.data_processing_pipeline(columns_to_drop=["Date", "Symbol"])

    assert "Date" not in result.columns
    assert "Symbol" not in result.columns
