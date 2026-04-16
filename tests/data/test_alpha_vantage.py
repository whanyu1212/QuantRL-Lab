"""Tests for AlphaVantageDataLoader (mocked HTTP responses)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantrl_lab.data.exceptions import APIConnectionError, InvalidParametersError, RateLimitError
from quantrl_lab.data.sources.alpha_vantage_loader import AlphaVantageDataLoader

DAILY_RESPONSE = {
    "Meta Data": {
        "1. Information": "Daily Prices",
        "2. Symbol": "AAPL",
    },
    "Time Series (Daily)": {
        "2023-01-03": {
            "1. open": "130",
            "2. high": "131",
            "3. low": "129",
            "4. close": "130.5",
            "5. volume": "1000000",
        },
        "2023-01-04": {
            "1. open": "131",
            "2. high": "132",
            "3. low": "130",
            "4. close": "131.5",
            "5. volume": "1100000",
        },
    },
}


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestAlphaVantageDataLoaderInit:
    def test_api_key_from_constructor(self):
        loader = AlphaVantageDataLoader(api_key="test_key")
        assert loader.api_key == "test_key"

    def test_default_rate_limit_delay(self):
        loader = AlphaVantageDataLoader(api_key="key")
        assert loader.rate_limit_delay == AlphaVantageDataLoader.DEFAULT_RATE_LIMIT_DELAY

    def test_source_name(self):
        loader = AlphaVantageDataLoader(api_key="key")
        assert loader.source_name == "Alpha Vantage"

    def test_is_connected_always_true(self):
        loader = AlphaVantageDataLoader(api_key="key")
        assert loader.is_connected() is True

    def test_connect_disconnect_no_error(self):
        loader = AlphaVantageDataLoader(api_key="key")
        loader.connect()
        loader.disconnect()

    def test_list_available_instruments_returns_empty(self):
        loader = AlphaVantageDataLoader(api_key="key")
        result = loader.list_available_instruments()
        assert result == []


class TestGetHistoricalOHLCVData:
    def test_daily_data_returns_dataframe(self):
        loader = AlphaVantageDataLoader(api_key="test_key", rate_limit_delay=0)
        with patch.object(loader, "_make_api_request", return_value=DAILY_RESPONSE):
            df = loader.get_historical_ohlcv_data("AAPL", timeframe="1d")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Close" in df.columns

    def test_empty_response_returns_empty_df(self):
        loader = AlphaVantageDataLoader(api_key="test_key", rate_limit_delay=0)
        with patch.object(loader, "_make_api_request", return_value={}):
            df = loader.get_historical_ohlcv_data("AAPL", timeframe="1d")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_error_message_in_response_raises(self):
        loader = AlphaVantageDataLoader(api_key="bad_key", rate_limit_delay=0)
        with patch.object(loader._request_wrapper, "make_request", return_value={"Error Message": "Invalid API call"}):
            with pytest.raises(InvalidParametersError):
                loader.get_historical_ohlcv_data("AAPL", timeframe="1d")

    def test_invalid_timeframe_raises(self):
        loader = AlphaVantageDataLoader(api_key="key")
        with pytest.raises(Exception):
            loader.get_historical_ohlcv_data("AAPL", timeframe="2w")

    def test_date_filtering_applied(self):
        loader = AlphaVantageDataLoader(api_key="test_key", rate_limit_delay=0)
        with patch.object(loader, "_make_api_request", return_value=DAILY_RESPONSE):
            df = loader.get_historical_ohlcv_data("AAPL", start="2023-01-04", end="2023-12-31", timeframe="1d")
        assert isinstance(df, pd.DataFrame)
        # Only the 2023-01-04 row should remain
        assert len(df) == 1


class TestMakeApiRequest:
    def test_raises_connection_error_on_failure(self):
        loader = AlphaVantageDataLoader(api_key="key", max_retries=1, delay=0, rate_limit_delay=0)
        with patch.object(
            loader._request_wrapper,
            "make_request",
            side_effect=APIConnectionError("Network error"),
        ):
            with pytest.raises(APIConnectionError):
                loader._make_api_request("TIME_SERIES_DAILY", symbol="AAPL")

    def test_rate_limit_note_raises(self):
        """API Note about rate limit should raise a typed exception."""
        loader = AlphaVantageDataLoader(api_key="key", max_retries=2, delay=0, rate_limit_delay=0)
        with patch.object(
            loader._request_wrapper,
            "make_request",
            return_value={"Note": "Thank you for using Alpha Vantage! API call frequency"},
        ):
            with pytest.raises(RateLimitError):
                loader._make_api_request("TIME_SERIES_DAILY", symbol="AAPL")
