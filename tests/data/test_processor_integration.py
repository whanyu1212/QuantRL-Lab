"""Integration tests for DataProcessor with new splitters."""

import pandas as pd
import pytest

from quantrl_lab.data.processing import DataProcessor


class TestDataProcessorSplitting:
    """Test DataProcessor integration with new splitter modules."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2020-01-01", periods=365, freq="D")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": range(365),
                "High": range(100, 465),
                "Low": range(0, 365),
                "Close": range(50, 415),
                "Volume": range(1000, 1365),
                "Symbol": ["AAPL"] * 365,
            }
        )

    def test_ratio_based_split_backward_compatibility(self, sample_ohlcv_data):
        """Test that ratio-based splitting still works with existing
        API."""
        processor = DataProcessor(sample_ohlcv_data)

        split_config = {"train": 0.7, "test": 0.3}
        result, metadata = processor.data_processing_pipeline(split_config=split_config)

        # Check splits exist
        assert "train" in result
        assert "test" in result

        # Check sizes
        assert len(result["train"]) == 255  # 70% of 365
        assert len(result["test"]) == 110  # Remainder is kept in the last split

        # Check metadata
        assert metadata["date_ranges"]["train"]["start"] == "2020-01-01"
        assert "final_shapes" in metadata
        assert metadata["final_shapes"]["train"][0] == 255

    def test_date_based_split_backward_compatibility(self, sample_ohlcv_data):
        """Test that date-based splitting still works with existing
        API."""
        processor = DataProcessor(sample_ohlcv_data)

        split_config = {
            "train": ("2020-01-01", "2020-09-30"),
            "test": ("2020-10-01", "2020-12-30"),  # Data ends at 2020-12-30 (365 days)
        }
        result, metadata = processor.data_processing_pipeline(split_config=split_config)

        # Check splits exist
        assert "train" in result
        assert "test" in result

        # Date column is dropped after splitting, so check metadata instead
        assert metadata["date_ranges"]["train"]["start"] == "2020-01-01"
        assert metadata["date_ranges"]["train"]["end"] == "2020-09-30"
        assert metadata["date_ranges"]["test"]["start"] == "2020-10-01"
        assert metadata["date_ranges"]["test"]["end"] == "2020-12-30"

        # Check that Date column was dropped
        assert "Date" not in result["train"].columns
        assert "Date" not in result["test"].columns

    def test_three_way_split(self, sample_ohlcv_data):
        """Test three-way split (train/val/test)."""
        processor = DataProcessor(sample_ohlcv_data)

        split_config = {"train": 0.6, "val": 0.2, "test": 0.2}
        result, metadata = processor.data_processing_pipeline(split_config=split_config)

        assert len(result) == 3
        assert "train" in result
        assert "val" in result
        assert "test" in result

        # Verify total equals original (after dropna)
        total_len = sum(len(df) for df in result.values())
        assert total_len == len(sample_ohlcv_data)

    def test_no_split_returns_single_dataframe(self, sample_ohlcv_data):
        """Test that not providing split_config returns single
        DataFrame."""
        processor = DataProcessor(sample_ohlcv_data)

        result, metadata = processor.data_processing_pipeline()

        # Should return DataFrame, not dict
        assert isinstance(result, pd.DataFrame)
        assert "final_shapes" in metadata
        assert "full_data" in metadata["final_shapes"]

    def test_split_with_indicators(self, sample_ohlcv_data):
        """Test that splitting works correctly with technical
        indicators."""
        processor = DataProcessor(sample_ohlcv_data)

        indicators = ["SMA", "RSI"]
        split_config = {"train": 0.7, "test": 0.3}

        result, metadata = processor.data_processing_pipeline(indicators=indicators, split_config=split_config)

        # Check that indicators were applied before splitting
        assert "SMA_20" in result["train"].columns
        assert "RSI_14" in result["train"].columns
        assert "SMA_20" in result["test"].columns
        assert "RSI_14" in result["test"].columns

        # Check metadata tracks indicators
        assert "technical_indicators" in metadata
        assert "SMA" in metadata["technical_indicators"]

    def test_metadata_structure_unchanged(self, sample_ohlcv_data):
        """Test that metadata structure remains backward compatible."""
        processor = DataProcessor(sample_ohlcv_data)

        split_config = {"train": 0.7, "test": 0.3}
        result, metadata = processor.data_processing_pipeline(split_config=split_config)

        # Check all expected metadata fields exist
        required_fields = [
            "symbol",
            "date_ranges",
            "fillna_strategy",
            "technical_indicators",
            "news_sentiment_applied",
            "columns_dropped",
            "original_shape",
            "final_shapes",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

    def test_split_preserves_temporal_order(self, sample_ohlcv_data):
        """Test that splits maintain temporal ordering."""
        processor = DataProcessor(sample_ohlcv_data)

        split_config = {"train": 0.6, "val": 0.2, "test": 0.2}
        result, metadata = processor.data_processing_pipeline(split_config=split_config)

        # Train should come before val, val before test
        train_last = pd.to_datetime(metadata["date_ranges"]["train"]["end"])
        val_first = pd.to_datetime(metadata["date_ranges"]["val"]["start"])
        val_last = pd.to_datetime(metadata["date_ranges"]["val"]["end"])
        test_first = pd.to_datetime(metadata["date_ranges"]["test"]["start"])

        assert train_last < val_first
        assert val_last < test_first


class TestDataProcessorDirectSplitterUsage:
    """Test using splitters directly (new API)."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": range(100),
                "High": range(100, 200),
                "Low": range(0, 100),
                "Close": range(50, 150),
                "Volume": range(1000, 1100),
            }
        )

    def test_can_import_splitters_from_data_module(self):
        """Test that splitters are importable from main data module."""
        from quantrl_lab.data import DateRangeSplitter, RatioSplitter

        assert DateRangeSplitter is not None
        assert RatioSplitter is not None

    def test_use_ratio_splitter_directly(self, sample_ohlcv_data):
        """Test using RatioSplitter directly before processing."""
        from quantrl_lab.data import RatioSplitter

        # Create splitter
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})

        # Split data
        splits = splitter.split(sample_ohlcv_data)

        # Verify
        assert len(splits["train"]) == 70
        assert len(splits["test"]) == 30

    def test_use_date_splitter_directly(self, sample_ohlcv_data):
        """Test using DateRangeSplitter directly before processing."""
        from quantrl_lab.data import DateRangeSplitter

        # Create splitter
        splitter = DateRangeSplitter({"train": ("2020-01-01", "2020-02-29"), "test": ("2020-03-01", "2020-04-09")})

        # Split data
        splits = splitter.split(sample_ohlcv_data)

        # Verify
        assert len(splits["train"]) == 60  # Jan + Feb (leap year)
        assert len(splits["test"]) == 40  # Mar + 9 days of Apr


class TestDataProcessorCleanup:
    """Test data cleanup and dropna logic."""

    def test_indicator_warmup_rows_are_dropped(self):
        """Test that generated technical indicators still trim warm-up
        rows."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": range(10),
                "High": [100] * 10,
                "Low": [100] * 10,
                "Close": [100] * 10,
                "Volume": [100] * 10,
            }
        )

        processor = DataProcessor(df)
        result, _ = processor.data_processing_pipeline(indicators=[{"SMA": {"window": 3}}])

        assert len(result) == 8
        assert result.iloc[0]["Open"] == 2

    def test_optional_sparse_enrichment_does_not_drop_rows(self):
        """Test sparse optional enrichment no longer deletes valid OHLCV
        rows."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": range(10),
                "High": [100] * 10,
                "Low": [100] * 10,
                "Close": [100] * 10,
                "Volume": [100] * 10,
            }
        )
        analyst_grades = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-01-05")],
                "analystRatingsStrongBuy": [7.0],
                "symbol": ["AAPL"],
            }
        )

        processor = DataProcessor(df, analyst_grades=analyst_grades)
        result, metadata = processor.data_processing_pipeline()

        assert len(result) == 10
        assert "analystRatingsStrongBuy" in result.columns
        assert result.loc[result["Open"] == 0, "analystRatingsStrongBuy"].isna().all()
        assert "analystRatingsStrongBuy" in metadata["optional_feature_columns"]
