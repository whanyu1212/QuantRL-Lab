"""Tests for TechnicalFeatureGenerator."""

import pandas as pd
import pytest

from quantrl_lab.data.processing.features.technical import TechnicalFeatureGenerator


class TestTechnicalFeatureGeneratorInit:
    """Test TechnicalFeatureGenerator initialization."""

    def test_init_with_valid_string_indicators(self):
        """Test initialization with valid string indicators."""
        indicators = ["SMA", "RSI"]
        generator = TechnicalFeatureGenerator(indicators)
        assert generator.indicators == indicators

    def test_init_with_valid_dict_indicators(self):
        """Test initialization with valid dictionary indicators."""
        indicators = [{"SMA": {"window": 10}}, {"RSI": {"window": 14}}]
        generator = TechnicalFeatureGenerator(indicators)
        assert generator.indicators == indicators

    def test_init_with_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="indicators list cannot be empty"):
            TechnicalFeatureGenerator([])

    def test_init_with_none_raises_error(self):
        """Test that None raises ValueError (or TypeError/AttributeError
        depending on implementation)."""
        # The implementation does `if not indicators`, so None will raise ValueError
        with pytest.raises(ValueError, match="indicators list cannot be empty"):
            TechnicalFeatureGenerator(None)


class TestTechnicalFeatureGeneratorGenerate:
    """Test TechnicalFeatureGenerator.generate() method."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        return pd.DataFrame(
            {
                "Date": pd.date_range("2023-01-01", periods=20, freq="D"),
                "Open": range(20),
                "High": range(5, 25),
                "Low": range(0, 20),
                "Close": range(2, 22),
                "Volume": range(1000, 1020),
            }
        )

    def test_generate_simple_indicators(self, sample_ohlcv_data):
        """Test generating simple indicators (defaults)."""
        generator = TechnicalFeatureGenerator(["SMA", "RSI"])
        result = generator.generate(sample_ohlcv_data)

        # SMA default window is 20 (might be NaN if len < 20, but column should exist)
        # RSI default window is 14
        assert "SMA_20" in result.columns
        assert "RSI_14" in result.columns

    def test_generate_custom_params_dict(self, sample_ohlcv_data):
        """Test generating indicators with custom params (dict
        format)."""
        generator = TechnicalFeatureGenerator([{"SMA": {"window": 5}}, {"RSI": {"window": 5}}])
        result = generator.generate(sample_ohlcv_data)

        assert "SMA_5" in result.columns
        assert "RSI_5" in result.columns

    def test_generate_custom_params_kwargs(self, sample_ohlcv_data):
        """Test generating indicators with custom params (kwargs)."""
        # Note: The implementation supports kwargs like `SMA_params={...}`
        # But this depends on how `enrich` is called.
        # The current implementation checks kwargs for `indicator_name_params`
        # if the config is a string.
        generator = TechnicalFeatureGenerator(["SMA"])
        result = generator.generate(sample_ohlcv_data, SMA_params={"window": 10})

        assert "SMA_10" in result.columns

    def test_generate_preserves_original_columns(self, sample_ohlcv_data):
        """Test that generation preserves original columns."""
        generator = TechnicalFeatureGenerator(["SMA"])
        result = generator.generate(sample_ohlcv_data)

        for col in sample_ohlcv_data.columns:
            assert col in result.columns

    def test_generate_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        generator = TechnicalFeatureGenerator(["SMA"])
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            generator.generate(empty_df)

    def test_generate_missing_required_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        generator = TechnicalFeatureGenerator(["SMA"])
        df = pd.DataFrame({"Date": ["2023-01-01"], "Open": [100]})

        with pytest.raises(ValueError, match="Missing required columns"):
            generator.generate(df)

    def test_generate_skips_unknown_indicators(self, sample_ohlcv_data):
        """Test that unknown indicators are skipped without error."""
        generator = TechnicalFeatureGenerator(["UNKNOWN_INDICATOR", "SMA"])
        result = generator.generate(sample_ohlcv_data)

        assert "SMA_20" in result.columns
        # No error raised

    def test_generate_strict_mode_raises_on_unknown_indicator(self, sample_ohlcv_data):
        """Strict mode should fail fast on unknown indicators."""
        generator = TechnicalFeatureGenerator(["UNKNOWN_INDICATOR", "SMA"], strict=True)

        with pytest.raises(ValueError, match="UNKNOWN_INDICATOR"):
            generator.generate(sample_ohlcv_data)


class TestTechnicalFeatureGeneratorMetadata:
    """Test TechnicalFeatureGenerator.get_metadata() method."""

    def test_metadata_structure(self):
        """Test metadata has correct structure."""
        indicators = ["SMA", "RSI"]
        generator = TechnicalFeatureGenerator(indicators)
        metadata = generator.get_metadata()

        assert "type" in metadata
        assert "indicators" in metadata
        assert metadata["type"] == "technical_indicators"
        assert metadata["indicators"] == indicators


class TestTechnicalFeatureGeneratorProtocol:
    """Test that TechnicalFeatureGenerator implements FeatureGenerator
    protocol."""

    def test_implements_protocol(self):
        """Test that generator implements FeatureGenerator protocol."""
        from quantrl_lab.data.processing.features.base import FeatureGenerator

        generator = TechnicalFeatureGenerator(["SMA"])
        assert isinstance(generator, FeatureGenerator)

    def test_has_generate_method(self):
        """Test that generator has generate() method."""
        generator = TechnicalFeatureGenerator(["SMA"])
        assert hasattr(generator, "generate")
        assert callable(generator.generate)

    def test_has_get_metadata_method(self):
        """Test that generator has get_metadata() method."""
        generator = TechnicalFeatureGenerator(["SMA"])
        assert hasattr(generator, "get_metadata")
        assert callable(generator.get_metadata)
