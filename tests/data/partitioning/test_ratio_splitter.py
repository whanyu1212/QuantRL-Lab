"""Tests for RatioSplitter."""

import pandas as pd
import pytest

from quantrl_lab.data.partitioning import RatioSplitter


class TestRatioSplitterInit:
    """Test RatioSplitter initialization."""

    def test_valid_ratios(self):
        """Test initialization with valid ratios."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        assert splitter.ratios == {"train": 0.7, "test": 0.3}

    def test_ratios_sum_to_one(self):
        """Test that ratios summing to 1.0 are valid."""
        splitter = RatioSplitter({"train": 0.6, "val": 0.2, "test": 0.2})
        assert sum(splitter.ratios.values()) == 1.0

    def test_ratios_sum_less_than_one_raises_error(self):
        """Test that ratios summing to < 1.0 are rejected."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RatioSplitter({"train": 0.7, "test": 0.2})

    def test_empty_ratios_raises_error(self):
        """Test that empty ratios dict raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            RatioSplitter({})

    def test_ratios_exceed_one_raises_error(self):
        """Test that ratios > 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            RatioSplitter({"train": 0.8, "test": 0.5})

    def test_negative_ratio_raises_error(self):
        """Test that negative ratios raise ValueError."""
        with pytest.raises(ValueError, match="Invalid ratio"):
            RatioSplitter({"train": -0.5, "test": 0.5})

    def test_zero_ratio_raises_error(self):
        """Test that zero ratio raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ratio"):
            RatioSplitter({"train": 0.0, "test": 1.0})

    def test_ratio_exceeds_one_individually_raises_error(self):
        """Test that individual ratio > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ratio"):
            RatioSplitter({"train": 1.5})


class TestRatioSplitterSplit:
    """Test RatioSplitter.split() method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
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

    def test_split_two_way(self, sample_data):
        """Test splitting data into train/test."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splits = splitter.split(sample_data)

        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) == 70
        assert len(splits["test"]) == 30

    def test_split_three_way(self, sample_data):
        """Test splitting data into train/val/test."""
        splitter = RatioSplitter({"train": 0.6, "val": 0.2, "test": 0.2})
        splits = splitter.split(sample_data)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"]) == 60
        assert len(splits["val"]) == 20
        assert len(splits["test"]) == 20

    def test_split_maintains_temporal_order(self, sample_data):
        """Test that splits maintain temporal ordering."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splits = splitter.split(sample_data)

        # Train should come before test
        train_last_date = splits["train"]["Date"].max()
        test_first_date = splits["test"]["Date"].min()
        assert train_last_date < test_first_date

    def test_split_preserves_data_integrity(self, sample_data):
        """Test that all data is preserved across splits."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splits = splitter.split(sample_data)

        combined_len = sum(len(df) for df in splits.values())
        assert combined_len == len(sample_data)

    def test_split_assigns_rounding_remainder_to_last_split(self):
        """Test that rounding leftovers are preserved in the final
        split."""
        df = pd.DataFrame({"value": range(365)})
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splits = splitter.split(df)

        assert len(splits["train"]) == 255
        assert len(splits["test"]) == 110
        assert sum(len(part) for part in splits.values()) == 365

    def test_split_empty_dataframe_raises_error(self):
        """Test that splitting empty DataFrame raises ValueError."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty DataFrame"):
            splitter.split(empty_df)

    def test_split_without_date_column(self):
        """Test splitting DataFrame without date column."""
        df = pd.DataFrame({"value": range(100)})
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splits = splitter.split(df)

        assert len(splits["train"]) == 70
        assert len(splits["test"]) == 30

    def test_split_with_different_date_column_names(self):
        """Test splitting with various date column names."""
        for date_col in ["Date", "date", "timestamp", "Timestamp"]:
            dates = pd.date_range("2023-01-01", periods=100, freq="D")
            df = pd.DataFrame({date_col: dates, "value": range(100)})

            splitter = RatioSplitter({"train": 0.7, "test": 0.3})
            splits = splitter.split(df)

            assert len(splits["train"]) == 70
            assert len(splits["test"]) == 30

    def test_split_with_timezone_aware_dates(self):
        """Test splitting with timezone-aware dates."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
        df = pd.DataFrame({"Date": dates, "value": range(100)})

        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splits = splitter.split(df)

        # Should remove timezone
        assert splits["train"]["Date"].dt.tz is None
        assert splits["test"]["Date"].dt.tz is None

    def test_split_with_unsorted_dates(self, sample_data):
        """Test that splitter sorts data by date."""
        # Shuffle the data
        shuffled = sample_data.sample(frac=1, random_state=42).reset_index(drop=True)

        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splits = splitter.split(shuffled)

        # Verify splits are sorted
        assert splits["train"]["Date"].is_monotonic_increasing
        assert splits["test"]["Date"].is_monotonic_increasing


class TestRatioSplitterMetadata:
    """Test RatioSplitter.get_metadata() method."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "Date": dates,
                "Open": range(100),
                "Close": range(50, 150),
            }
        )

    def test_metadata_before_split(self):
        """Test metadata before calling split()."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        metadata = splitter.get_metadata()

        assert metadata["type"] == "ratio"
        assert metadata["ratios"] == {"train": 0.7, "test": 0.3}

    def test_metadata_after_split(self, sample_data):
        """Test metadata after calling split()."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splitter.split(sample_data)
        metadata = splitter.get_metadata()

        assert metadata["type"] == "ratio"
        assert metadata["ratios"] == {"train": 0.7, "test": 0.3}
        assert "date_ranges" in metadata
        assert "final_shapes" in metadata

    def test_metadata_date_ranges(self, sample_data):
        """Test that metadata contains correct date ranges."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splitter.split(sample_data)
        metadata = splitter.get_metadata()

        assert "train" in metadata["date_ranges"]
        assert "test" in metadata["date_ranges"]
        assert metadata["date_ranges"]["train"]["start"] == "2023-01-01"
        assert metadata["date_ranges"]["test"]["end"] == "2023-04-10"

    def test_metadata_shapes(self, sample_data):
        """Test that metadata contains correct shapes."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splitter.split(sample_data)
        metadata = splitter.get_metadata()

        assert metadata["final_shapes"]["train"] == (70, 3)
        assert metadata["final_shapes"]["test"] == (30, 3)

    def test_metadata_without_date_column(self):
        """Test metadata when no date column exists."""
        df = pd.DataFrame({"value": range(100)})
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        splitter.split(df)
        metadata = splitter.get_metadata()

        # Should still have shapes but no date_ranges
        assert "final_shapes" in metadata
        assert metadata["final_shapes"]["train"] == (70, 1)


class TestRatioSplitterProtocol:
    """Test that RatioSplitter adheres to DataSplitter protocol."""

    def test_implements_protocol(self):
        """Test that RatioSplitter implements DataSplitter protocol."""
        from quantrl_lab.data.partitioning import DataSplitter

        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        assert isinstance(splitter, DataSplitter)

    def test_has_split_method(self):
        """Test that splitter has split() method."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        assert hasattr(splitter, "split")
        assert callable(splitter.split)

    def test_has_get_metadata_method(self):
        """Test that splitter has get_metadata() method."""
        splitter = RatioSplitter({"train": 0.7, "test": 0.3})
        assert hasattr(splitter, "get_metadata")
        assert callable(splitter.get_metadata)
