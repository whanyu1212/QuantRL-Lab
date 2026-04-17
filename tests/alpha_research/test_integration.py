"""Tests for alpha_research → data.processing integration helpers."""

import pandas as pd
import pytest

from quantrl_lab.alpha_research import (
    AlphaSelectionConfig,
    build_processing_config_from_alpha_selection,
    select_indicators_for_processing,
)
from quantrl_lab.data.processing import ProcessingPipelineConfig, SplitConfig


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


def test_select_indicators_for_processing_uses_train_split_only(monkeypatch, sample_single_symbol_data):
    """Alpha-selection helper should operate on the raw training split
    only."""
    captured = {}

    def fake_suggest_indicators(self, candidates=None, metric="ic", threshold=0.0, top_k=5, selection_mode="feature"):
        captured["rows"] = len(self.data)
        captured["start"] = pd.to_datetime(self.data["Date"]).min()
        captured["end"] = pd.to_datetime(self.data["Date"]).max()
        captured["selection_mode"] = selection_mode
        return [{"SMA": {"window": 3}}]

    monkeypatch.setattr(
        "quantrl_lab.alpha_research.selector.AlphaSelector.suggest_indicators",
        fake_suggest_indicators,
    )

    indicators, metadata = select_indicators_for_processing(
        sample_single_symbol_data,
        AlphaSelectionConfig(metric="ic", threshold=0.0, top_k=1),
        split_config=SplitConfig(splits={"train": ("2020-01-01", "2020-01-06"), "test": ("2020-01-07", "2020-01-12")}),
    )

    assert indicators == [{"SMA": {"window": 3}}]
    assert captured["rows"] == 6
    assert captured["start"] == pd.Timestamp("2020-01-01")
    assert captured["end"] == pd.Timestamp("2020-01-06")
    assert captured["selection_mode"] == "feature"
    assert metadata["selected_from_split"] == "train"


def test_build_processing_config_from_alpha_selection_populates_pipeline(monkeypatch, sample_single_symbol_data):
    """Adapter should return a ready-to-run processing config with
    selected indicators."""

    def fake_suggest_indicators(self, candidates=None, metric="ic", threshold=0.0, top_k=5, selection_mode="feature"):
        return [{"RSI": {"window": 14}}]

    monkeypatch.setattr(
        "quantrl_lab.alpha_research.selector.AlphaSelector.suggest_indicators",
        fake_suggest_indicators,
    )

    processing_config, metadata = build_processing_config_from_alpha_selection(
        sample_single_symbol_data,
        {"metric": "ic", "threshold": 0.0, "top_k": 1},
        split_config={"train": 0.7, "test": 0.3},
        processing_config=ProcessingPipelineConfig(verbose=True),
    )

    assert processing_config.indicators == [{"RSI": {"window": 14}}]
    assert processing_config.split == SplitConfig(splits={"train": 0.7, "test": 0.3})
    assert processing_config.verbose is True
    assert metadata["selection_mode"] == "feature"
    assert metadata["selected_from_split"] == "train"


def test_build_processing_config_rejects_existing_indicators(sample_single_symbol_data):
    """Existing manual indicators should not be mixed implicitly with
    alpha-selected ones."""
    with pytest.raises(ValueError, match="indicators must be None"):
        build_processing_config_from_alpha_selection(
            sample_single_symbol_data,
            {"metric": "ic"},
            processing_config=ProcessingPipelineConfig(indicators=["SMA"]),
        )


def test_select_indicators_for_processing_rejects_multi_symbol_input():
    """Single-series helper should direct panel workflows to
    suggest_for_universe."""
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "Date": list(dates) * 2,
            "Open": list(range(4)) + list(range(10, 14)),
            "High": list(range(100, 104)) + list(range(110, 114)),
            "Low": list(range(10, 14)) + list(range(20, 24)),
            "Close": list(range(50, 54)) + list(range(60, 64)),
            "Volume": [1000] * 8,
            "Symbol": ["AAPL"] * 4 + ["MSFT"] * 4,
        }
    )

    with pytest.raises(ValueError, match="suggest_for_universe"):
        select_indicators_for_processing(df, {"metric": "ic"}, split_config={"train": 0.5, "test": 0.5})
