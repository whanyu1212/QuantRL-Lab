"""
Unit tests for data sources (YFinance, Alpaca, AlphaVantage, FMP).

These tests use mocking to avoid actual API calls.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantrl_lab.data.exceptions import AuthenticationError, DataSourceError, InvalidParametersError, RateLimitError
from quantrl_lab.data.interface import (
    FundamentalDataCapable,
    HistoricalDataCapable,
    LiveDataCapable,
    NewsDataCapable,
)
from quantrl_lab.data.sources.alpaca_loader import AlpacaDataLoader
from quantrl_lab.data.sources.alpha_vantage_loader import AlphaVantageDataLoader
from quantrl_lab.data.sources.fmp_loader import FMPDataSource
from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader


class TestYFinanceDataLoader:
    """Tests for YFinanceDataLoader class."""

    def test_implements_required_protocols(self):
        """Test that YFinanceDataLoader implements required
        protocols."""
        loader = YFinanceDataLoader()

        assert isinstance(loader, HistoricalDataCapable)
        assert isinstance(loader, FundamentalDataCapable)

    def test_source_name(self):
        """Test that source_name returns expected value."""
        loader = YFinanceDataLoader()

        assert loader.source_name == "Yahoo Finance"

    def test_is_connected_returns_true(self):
        """Test that is_connected returns True (uses HTTP requests)."""
        loader = YFinanceDataLoader()

        assert loader.is_connected() is True

    def test_connect_does_not_raise(self):
        """Test that connect doesn't raise an exception."""
        loader = YFinanceDataLoader()
        loader.connect()  # Should not raise

    def test_disconnect_does_not_raise(self):
        """Test that disconnect doesn't raise an exception."""
        loader = YFinanceDataLoader()
        loader.disconnect()  # Should not raise

    def test_get_historical_ohlcv_data_validates_symbols_type(self):
        """Test that get_historical_ohlcv_data validates symbols
        parameter type."""
        loader = YFinanceDataLoader()

        with pytest.raises(TypeError, match="symbols must be a string or list of strings"):
            loader.get_historical_ohlcv_data(
                symbols=123,  # Invalid type
                start="2023-01-01",
                end="2023-01-31",
            )

    def test_get_historical_ohlcv_data_validates_list_elements(self):
        """Test that get_historical_ohlcv_data validates list elements
        are strings."""
        loader = YFinanceDataLoader()

        with pytest.raises(ValueError, match="All elements in symbols list must be strings"):
            loader.get_historical_ohlcv_data(
                symbols=["AAPL", 123],  # Mixed types
                start="2023-01-01",
                end="2023-01-31",
            )

    def test_get_historical_ohlcv_data_validates_interval(self):
        """Test that get_historical_ohlcv_data validates interval
        parameter."""
        loader = YFinanceDataLoader()

        with pytest.raises(InvalidParametersError, match="Invalid interval"):
            loader.get_historical_ohlcv_data(
                symbols="AAPL",
                start="2023-01-01",
                end="2023-01-31",
                timeframe="invalid",
            )

    def test_get_historical_ohlcv_data_validates_date_order(self):
        """Test that get_historical_ohlcv_data validates start < end."""
        loader = YFinanceDataLoader()

        with pytest.raises(ValueError, match="Start date .* must be before or equal to end date"):
            loader.get_historical_ohlcv_data(
                symbols="AAPL",
                start="2023-12-31",
                end="2023-01-01",
            )

    @patch("quantrl_lab.data.sources.yfinance_loader.yf.download")
    def test_get_historical_ohlcv_data_returns_dataframe(self, mock_download):
        """Test that get_historical_ohlcv_data returns a DataFrame."""
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [150.0, 151.0],
                "High": [152.0, 153.0],
                "Low": [149.0, 150.0],
                "Close": [151.0, 152.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2023-01-01", periods=2),
        )

        loader = YFinanceDataLoader()
        result = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start="2023-01-01",
            end="2023-01-31",
        )

        assert isinstance(result, pd.DataFrame)
        assert "Symbol" in result.columns

    @patch("quantrl_lab.data.sources.yfinance_loader.yf.download")
    def test_get_historical_ohlcv_data_normalizes_multi_symbol_download(self, mock_download):
        """Test multi-symbol download normalization preserves every
        requested symbol."""
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL", "MSFT"]],
        )
        mock_download.return_value = pd.DataFrame(
            [
                [150.0, 300.0, 152.0, 302.0, 149.0, 299.0, 151.0, 301.0, 1000000, 2000000],
                [151.0, 301.0, 153.0, 303.0, 150.0, 300.0, 152.0, 302.0, 1100000, 2100000],
            ],
            index=pd.date_range("2023-01-01", periods=2),
            columns=columns,
        )

        loader = YFinanceDataLoader()
        result = loader.get_historical_ohlcv_data(
            symbols=["AAPL", "MSFT"],
            start="2023-01-01",
            end="2023-01-31",
        )

        assert isinstance(result, pd.DataFrame)
        assert set(result["Symbol"]) == {"AAPL", "MSFT"}
        assert len(result) == 4

    @patch("quantrl_lab.data.sources.yfinance_loader.yf.Ticker")
    def test_get_fundamental_data_returns_dataframe(self, mock_ticker):
        """Test that get_fundamental_data returns a DataFrame."""
        # Setup mocks for income statement, cash flow, and balance sheet
        mock_income = pd.DataFrame({"Revenue": [1000]}, index=[datetime(2023, 1, 1)])
        mock_cashflow = pd.DataFrame({"FreeCashFlow": [500]}, index=[datetime(2023, 1, 1)])
        mock_balance = pd.DataFrame({"TotalAssets": [5000]}, index=[datetime(2023, 1, 1)])

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.get_income_stmt.return_value = mock_income.T
        mock_ticker_instance.get_cashflow.return_value = mock_cashflow.T
        mock_ticker_instance.get_balance_sheet.return_value = mock_balance.T
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()
        result = loader.get_fundamental_data("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert "Symbol" in result.columns
        assert result["Symbol"].iloc[0] == "AAPL"


class TestAlpacaDataLoader:
    """Tests for AlpacaDataLoader class."""

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_implements_required_protocols(self, mock_stream, mock_client):
        """Test that AlpacaDataLoader implements required protocols."""
        # Reset singleton for testing
        AlpacaDataLoader._stock_stream_client_instance = None

        loader = AlpacaDataLoader()

        assert isinstance(loader, HistoricalDataCapable)
        assert isinstance(loader, LiveDataCapable)
        assert isinstance(loader, NewsDataCapable)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_source_name(self, mock_stream, mock_client):
        """Test that source_name returns expected value."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()

        assert loader.source_name == "Alpaca"

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_is_connected_with_valid_credentials(self, mock_stream, mock_client):
        """Test is_connected returns True with valid credentials."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()

        assert loader.is_connected() is True

    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_is_connected_without_client(self, mock_stream, mock_client):
        """Test is_connected returns False when historical client is
        None."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader(api_key="test", secret_key="test")

        # Simulate uninitialized client
        loader.stock_historical_client = None

        assert loader.is_connected() is False

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_connect_raises_without_credentials(self, mock_stream, mock_client):
        """Test that connect raises ValueError without credentials."""
        AlpacaDataLoader._stock_stream_client_instance = None
        loader = AlpacaDataLoader()
        loader.api_key = None
        loader.secret_key = None

        with pytest.raises(AuthenticationError, match="Alpaca API credentials not provided"):
            loader.connect()

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_get_historical_ohlcv_data_converts_string_to_list(self, mock_stream, mock_client):
        """Test that single symbol string is converted to list."""
        AlpacaDataLoader._stock_stream_client_instance = None

        # Setup mock response
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame(
            {
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.0],
                "volume": [1000000],
                "trade_count": [100],
                "vwap": [150.5],
            },
            index=pd.MultiIndex.from_tuples([("AAPL", datetime(2023, 1, 1))], names=["symbol", "timestamp"]),
        )

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = mock_bars
        mock_client.return_value = mock_client_instance

        loader = AlpacaDataLoader()
        result = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start="2023-01-01",
            end="2023-01-31",
        )

        assert isinstance(result, pd.DataFrame)

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_get_news_data_returns_dataframe(self, mock_stream, mock_client):
        """Test that get_news_data returns a DataFrame."""
        AlpacaDataLoader._stock_stream_client_instance = None

        mock_payload = {
            "news": [
                {
                    "headline": "Test headline",
                    "created_at": "2023-01-15T10:00:00Z",
                    "summary": "Test summary",
                }
            ],
            "next_page_token": None,
        }

        loader = AlpacaDataLoader()
        with patch.object(loader._news_request_wrapper, "make_request", return_value=mock_payload):
            result = loader.get_news_data(
                symbols="AAPL",
                start="2023-01-01",
                end="2023-01-31",
            )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["headline"] == "Test headline"

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    def test_get_news_data_silent_errors(self):
        """Test that silent_errors=True suppresses exceptions."""
        loader = AlpacaDataLoader()

        with patch.object(
            loader._news_request_wrapper, "make_request", side_effect=DataSourceError("Connection failed")
        ):
            result = loader.get_news_data(symbols="AAPL", start="2023-01-01", silent_errors=True)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    def test_get_news_data_raises_without_silent_errors(self):
        """Test that silent_errors=False propagates provider
        failures."""
        loader = AlpacaDataLoader()

        with patch.object(
            loader._news_request_wrapper, "make_request", side_effect=DataSourceError("Connection failed")
        ):
            with pytest.raises(DataSourceError, match="Connection failed"):
                loader.get_news_data(symbols="AAPL", start="2023-01-01", silent_errors=False)


class TestAlphaVantageDataLoader:
    """Tests for AlphaVantageDataLoader class."""

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_implements_required_protocols(self):
        """Test that AlphaVantageDataLoader implements required
        protocols."""
        loader = AlphaVantageDataLoader()

        assert isinstance(loader, HistoricalDataCapable)
        assert isinstance(loader, FundamentalDataCapable)
        assert isinstance(loader, NewsDataCapable)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_source_name(self):
        """Test that source_name returns expected value."""
        loader = AlphaVantageDataLoader()

        assert loader.source_name == "Alpha Vantage"

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_is_connected_returns_true(self):
        """Test that is_connected returns True (uses HTTP requests)."""
        loader = AlphaVantageDataLoader()

        assert loader.is_connected() is True

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_list_available_instruments_returns_empty_list(self):
        """Test that list_available_instruments returns empty list."""
        loader = AlphaVantageDataLoader()

        result = loader.list_available_instruments()

        assert result == []

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_historical_ohlcv_data_validates_timeframe(self):
        """Test that get_historical_ohlcv_data validates timeframe."""
        loader = AlphaVantageDataLoader()

        with pytest.raises(InvalidParametersError, match="Unsupported timeframe"):
            loader.get_historical_ohlcv_data(
                symbols="AAPL",
                start="2023-01-01",
                end="2023-01-31",
                timeframe="invalid",
            )

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_historical_ohlcv_data_daily(self):
        """Test get_historical_ohlcv_data with daily timeframe."""
        mock_response = {
            "Time Series (Daily)": {
                "2023-01-15": {
                    "1. open": "150.0",
                    "2. high": "152.0",
                    "3. low": "149.0",
                    "4. close": "151.0",
                    "5. volume": "1000000",
                },
            },
        }

        loader = AlphaVantageDataLoader()
        with patch.object(loader, "_make_api_request", return_value=mock_response):
            result = loader.get_historical_ohlcv_data(
                symbols="AAPL",
                start="2023-01-01",
                end="2023-01-31",
                timeframe="1d",
            )

        assert isinstance(result, pd.DataFrame)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_historical_ohlcv_data_intraday(self):
        """Test get_historical_ohlcv_data with intraday timeframe."""
        mock_response = {
            "Time Series (5min)": {
                "2023-01-15 10:00:00": {
                    "1. open": "150.0",
                    "2. high": "152.0",
                    "3. low": "149.0",
                    "4. close": "151.0",
                    "5. volume": "1000000",
                },
            },
        }

        loader = AlphaVantageDataLoader()
        with patch.object(loader, "_make_api_request", return_value=mock_response):
            result = loader.get_historical_ohlcv_data(
                symbols="AAPL",
                start="2023-01-01",
                end="2023-01-31",
                timeframe="5min",
            )

        assert isinstance(result, pd.DataFrame)

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_real_gdp_data_validates_interval(self):
        """Test that _get_real_gdp_data validates interval parameter."""
        loader = AlphaVantageDataLoader()

        with pytest.raises(InvalidParametersError, match="Invalid interval"):
            loader._get_real_gdp_data(interval="invalid")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_treasury_yield_data_validates_interval(self):
        """Test that _get_treasury_yield_data validates interval
        parameter."""
        loader = AlphaVantageDataLoader()

        with pytest.raises(InvalidParametersError, match="Invalid interval"):
            loader._get_treasury_yield_data(interval="invalid")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_treasury_yield_data_validates_maturity(self):
        """Test that _get_treasury_yield_data validates maturity
        parameter."""
        loader = AlphaVantageDataLoader()

        with pytest.raises(InvalidParametersError, match="Invalid maturity"):
            loader._get_treasury_yield_data(maturity="invalid")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_cpi_data_validates_interval(self):
        """Test that _get_cpi_data validates interval parameter."""
        loader = AlphaVantageDataLoader()

        with pytest.raises(InvalidParametersError, match="Invalid interval"):
            loader._get_cpi_data(interval="invalid")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_federal_funds_rate_validates_interval(self):
        """Test that _get_federal_funds_rate_data validates interval
        parameter."""
        loader = AlphaVantageDataLoader()

        with pytest.raises(InvalidParametersError, match="Invalid interval"):
            loader._get_federal_funds_rate_data(interval="invalid")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_make_api_request_handles_rate_limit(self):
        """Test that _make_api_request raises typed rate limit
        errors."""
        loader = AlphaVantageDataLoader(delay=0)
        with patch.object(
            loader._request_wrapper,
            "make_request",
            return_value={"Note": "API call frequency is 5 calls per minute"},
        ):
            with pytest.raises(RateLimitError):
                loader._make_api_request("TEST_FUNCTION", "AAPL")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_make_api_request_handles_error_message(self):
        """Test that _make_api_request raises invalid parameter
        errors."""
        loader = AlphaVantageDataLoader()
        with patch.object(loader._request_wrapper, "make_request", return_value={"Error Message": "Invalid API call"}):
            with pytest.raises(InvalidParametersError, match="Invalid API call"):
                loader._make_api_request("TEST_FUNCTION", "INVALID")

    @patch.dict("os.environ", {"ALPHA_VANTAGE_API_KEY": "test_key"})
    def test_get_news_data_expands_multi_symbol_sentiment(self):
        """Test news normalization emits one row per matched article-
        symbol pair."""
        loader = AlphaVantageDataLoader()
        with patch.object(
            loader,
            "_make_api_request",
            return_value={
                "feed": [
                    {
                        "title": "Chip demand improves",
                        "time_published": "20230115T100000",
                        "ticker_sentiment": [
                            {"ticker": "AAPL", "ticker_sentiment_score": "0.30"},
                            {"ticker": "MSFT", "ticker_sentiment_score": "0.70"},
                        ],
                    }
                ]
            },
        ):
            result = loader.get_news_data(["AAPL", "MSFT"], start="2023-01-01", end="2023-01-31")

        assert len(result) == 2
        assert set(result["Symbol"]) == {"AAPL", "MSFT"}
        assert set(result["sentiment_score"]) == {0.3, 0.7}


class TestDataSourceRegistry:
    """Tests for DataSourceRegistry class."""

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_default_sources_initialized(self, mock_stream, mock_client):
        """Test that default sources are initialized."""
        from quantrl_lab.data.source_registry import DataSourceRegistry

        AlpacaDataLoader._stock_stream_client_instance = None
        registry = DataSourceRegistry()

        assert hasattr(registry, "primary_source")
        assert hasattr(registry, "news_source")
        assert "primary_source" in registry.list_sources_by_capability("historical_bars")
        assert "news_source" in registry.list_sources_by_capability("news")

    @patch.dict("os.environ", {"ALPACA_API_KEY": "test_key", "ALPACA_SECRET_KEY": "test_secret"})
    @patch("quantrl_lab.data.sources.alpaca_loader.StockHistoricalDataClient")
    @patch("quantrl_lab.data.sources.alpaca_loader.StockDataStream")
    def test_get_historical_ohlcv_data_delegates_to_primary_source(self, mock_stream, mock_client):
        """Test that get_historical_ohlcv_data delegates to primary
        source."""
        from quantrl_lab.data.source_registry import DataSourceRegistry

        AlpacaDataLoader._stock_stream_client_instance = None

        # Setup mock response
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame(
            {
                "open": [150.0],
                "high": [152.0],
                "low": [149.0],
                "close": [151.0],
                "volume": [1000000],
                "trade_count": [100],
                "vwap": [150.5],
            },
            index=pd.MultiIndex.from_tuples([("AAPL", datetime(2023, 1, 1))], names=["symbol", "timestamp"]),
        )

        mock_client_instance = MagicMock()
        mock_client_instance.get_stock_bars.return_value = mock_bars
        mock_client.return_value = mock_client_instance

        registry = DataSourceRegistry()
        result = registry.get_historical_ohlcv_data(
            symbols="AAPL",
            start="2023-01-01",
            end="2023-01-31",
        )

        assert isinstance(result, pd.DataFrame)

    def test_register_source_infers_capabilities_from_factory_instance(self):
        """Test custom zero-arg factories remain discoverable by
        capability."""
        from quantrl_lab.data.source_registry import DataSourceRegistry

        registry = DataSourceRegistry(sources={})
        registry.register_source("custom", lambda: YFinanceDataLoader())

        assert "custom" in registry.list_sources_by_capability("historical_bars")
        assert "custom" in registry.list_sources_by_capability("fundamental_data")


class TestFMPDataSource:
    """Tests for FMPDataSource class."""

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_implements_required_protocols(self):
        """Test that FMPDataSource implements required protocols."""
        loader = FMPDataSource()
        assert isinstance(loader, HistoricalDataCapable)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_source_name(self):
        """Test that source_name returns expected value."""
        loader = FMPDataSource()
        assert loader.source_name == "FinancialModelingPrep"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_init_without_key_raises_error(self):
        """Test that initialization raises ValueError if no API key is
        provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(AuthenticationError, match="FMP API key must be provided"):
                FMPDataSource()

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_init_with_env_var(self):
        """Test that initialization works with environment variable."""
        loader = FMPDataSource()
        assert loader.api_key == "test_key"

    def test_init_with_arg(self):
        """Test that initialization works with argument."""
        loader = FMPDataSource(api_key="arg_key")
        assert loader.api_key == "arg_key"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_make_request_success(self):
        """Test _make_request handles successful response."""
        loader = FMPDataSource()

        # Mock the request wrapper's make_request method
        with patch.object(loader._request_wrapper, "make_request", return_value={"data": "test"}):
            result = loader._make_request("test-endpoint", {})
            assert result == {"data": "test"}

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_grades_success(self):
        """Test get_historical_grades returns DataFrame."""
        mock_data = [
            {
                "symbol": "AAPL",
                "date": "2024-01-01",
                "analystRatingsStrongBuy": 10,
                "analystRatingsBuy": 20,
                "analystRatingsHold": 5,
                "analystRatingsSell": 1,
                "analystRatingsStrongSell": 0,
            }
        ]

        loader = FMPDataSource()

        # Mock the request wrapper's make_request method
        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_historical_grades("AAPL")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "date" in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert df.iloc[0]["symbol"] == "AAPL"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_grades_empty(self):
        """Test get_historical_grades handles empty response."""
        loader = FMPDataSource()

        # Mock the request wrapper to return empty list
        with patch.object(loader._request_wrapper, "make_request", return_value=[]):
            df = loader.get_historical_grades("AAPL")

            assert isinstance(df, pd.DataFrame)
            assert df.empty

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_rating_success(self):
        """Test get_historical_rating returns DataFrame."""
        mock_data = [
            {
                "symbol": "AAPL",
                "date": "2024-01-01",
                "rating": "S",
                "ratingScore": 5,
                "ratingRecommendation": "Strong Buy",
                "ratingDetailsDCFScore": 5,
                "ratingDetailsROEScore": 5,
                "ratingDetailsROAScore": 5,
                "ratingDetailsDEScore": 5,
                "ratingDetailsPEScore": 5,
                "ratingDetailsPBScore": 5,
            }
        ]

        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_historical_rating("AAPL")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "date" in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert df.iloc[0]["symbol"] == "AAPL"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_rating_with_limit(self):
        """Test get_historical_rating respects limit."""
        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=[]) as mock_request:
            loader.get_historical_rating("AAPL", limit=50)

            # Check that request was called (limit is passed in params to _make_request)
            assert mock_request.called

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_ohlcv_data_daily_response_list(self):
        """Test get_historical_ohlcv_data for daily timeframe expecting
        list."""
        mock_data = [
            {
                "date": "2024-01-01",
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 152.0,
                "volume": 1000000,
            }
        ]

        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_historical_ohlcv_data("AAPL", start="2024-01-01")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "Timestamp" in df.columns
            assert "Open" in df.columns
            assert df.iloc[0]["Symbol"] == "AAPL"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_ohlcv_data_intraday(self):
        """Test get_historical_ohlcv_data for intraday timeframe."""
        mock_data = [
            {
                "date": "2024-01-01 10:00:00",
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 152.0,
                "volume": 1000000,
            }
        ]

        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_historical_ohlcv_data("AAPL", start="2024-01-01", timeframe="5min")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "Timestamp" in df.columns
            assert "Open" in df.columns
            assert df.iloc[0]["Symbol"] == "AAPL"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_sector_performance_success(self):
        """Test get_historical_sector_performance returns DataFrame."""
        mock_data = [
            {
                "date": "2024-01-01",
                "sector": "Energy",
                "changesPercentage": "2.5",
            },
            {
                "date": "2024-01-02",
                "sector": "Energy",
                "changesPercentage": "1.8",
            },
        ]

        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_historical_sector_performance("Energy")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert len(df) == 2
            assert "date" in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert df.iloc[0]["sector"] == "Energy"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_sector_performance_empty(self):
        """Test get_historical_sector_performance handles empty
        response."""
        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=[]):
            df = loader.get_historical_sector_performance("Energy")

            assert isinstance(df, pd.DataFrame)
            assert df.empty

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_sector_performance_invalid_sector(self):
        """Test get_historical_sector_performance validates sector
        parameter."""
        loader = FMPDataSource()

        with pytest.raises(InvalidParametersError, match="Sector must be a non-empty string"):
            loader.get_historical_sector_performance("")

        with pytest.raises(InvalidParametersError, match="Sector must be a non-empty string"):
            loader.get_historical_sector_performance(None)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_industry_performance_success(self):
        """Test get_historical_industry_performance returns
        DataFrame."""
        mock_data = [
            {
                "date": "2024-01-01",
                "industry": "Biotechnology",
                "changesPercentage": "3.2",
            },
            {
                "date": "2024-01-02",
                "industry": "Biotechnology",
                "changesPercentage": "2.1",
            },
        ]

        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_historical_industry_performance("Biotechnology")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert len(df) == 2
            assert "date" in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert df.iloc[0]["industry"] == "Biotechnology"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_industry_performance_empty(self):
        """Test get_historical_industry_performance handles empty
        response."""
        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=[]):
            df = loader.get_historical_industry_performance("Biotechnology")

            assert isinstance(df, pd.DataFrame)
            assert df.empty

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_historical_industry_performance_invalid_industry(self):
        """Test get_historical_industry_performance validates industry
        parameter."""
        loader = FMPDataSource()

        with pytest.raises(InvalidParametersError, match="Industry must be a non-empty string"):
            loader.get_historical_industry_performance("")

        with pytest.raises(InvalidParametersError, match="Industry must be a non-empty string"):
            loader.get_historical_industry_performance(None)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_company_profile_success(self):
        """Test get_company_profile returns DataFrame with company
        info."""
        mock_data = [
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "description": "Apple Inc. designs, manufactures, and markets smartphones...",
                "ceo": "Timothy D. Cook",
                "website": "https://www.apple.com",
                "exchange": "NASDAQ",
                "exchangeShortName": "NASDAQ",
                "mktCap": 3000000000000,
                "price": 175.50,
                "beta": 1.25,
                "volAvg": 50000000,
                "currency": "USD",
                "ipoDate": "1980-12-12",
            }
        ]

        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_company_profile("AAPL")

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert df.iloc[0]["symbol"] == "AAPL"
            assert df.iloc[0]["companyName"] == "Apple Inc."
            assert df.iloc[0]["sector"] == "Technology"
            assert df.iloc[0]["industry"] == "Consumer Electronics"

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_company_profile_empty(self):
        """Test get_company_profile handles empty response."""
        loader = FMPDataSource()

        with patch.object(loader._request_wrapper, "make_request", return_value=[]):
            df = loader.get_company_profile("INVALID")

            assert isinstance(df, pd.DataFrame)
            assert df.empty

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_company_profile_invalid_symbol(self):
        """Test get_company_profile validates symbol parameter."""
        loader = FMPDataSource()

        with pytest.raises(ValueError, match="Symbol at index 0 is empty or whitespace-only"):
            loader.get_company_profile("")

        with pytest.raises(TypeError):
            loader.get_company_profile(None)

    @patch.dict("os.environ", {"FMP_API_KEY": "test_key"})
    def test_get_company_profile_normalizes_symbol(self):
        """Test get_company_profile normalizes and validates symbol
        input."""
        mock_data = [
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
            }
        ]

        loader = FMPDataSource()

        # Test with list input (should use first symbol)
        with patch.object(loader._request_wrapper, "make_request", return_value=mock_data):
            df = loader.get_company_profile(["AAPL", "MSFT"])

            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert df.iloc[0]["symbol"] == "AAPL"
