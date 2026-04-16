"""Tests for HTTP request utilities module."""

import time
from unittest.mock import MagicMock, patch

import pytest

from quantrl_lab.data.exceptions import APIConnectionError, AuthenticationError, RateLimitError
from quantrl_lab.data.utils.request_utils import (
    HTTPRequestWrapper,
    RetryStrategy,
    create_default_wrapper,
)


class TestHTTPRequestWrapper:
    """Tests for HTTPRequestWrapper class."""

    def test_initialization(self):
        """Test wrapper initialization."""
        wrapper = HTTPRequestWrapper(
            max_retries=5,
            retry_strategy=RetryStrategy.EXPONENTIAL,
            base_delay=2.0,
        )
        assert wrapper.max_retries == 5
        assert wrapper.retry_strategy == RetryStrategy.EXPONENTIAL
        assert wrapper.base_delay == 2.0

    def test_successful_request(self):
        """Test successful HTTP request."""
        wrapper = HTTPRequestWrapper(max_retries=3)

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response

            result = wrapper.make_request("https://api.example.com/data")

            assert result == {"data": "test"}
            mock_request.assert_called_once()

    def test_retry_on_failure(self):
        """Test that wrapper retries on failure."""
        wrapper = HTTPRequestWrapper(max_retries=2, base_delay=0.01)

        with patch("requests.request") as mock_request:
            # First call fails, second succeeds
            mock_fail = MagicMock()
            mock_fail.status_code = 500
            mock_fail.url = "https://api.example.com/data"
            mock_fail.headers = {}
            mock_fail.json.return_value = {"error": "server error"}

            mock_success = MagicMock()
            mock_success.status_code = 200
            mock_success.url = "https://api.example.com/data"
            mock_success.headers = {}
            mock_success.json.return_value = {"data": "success"}

            mock_request.side_effect = [mock_fail, mock_success]

            result = wrapper.make_request("https://api.example.com/data")

            assert result == {"data": "success"}
            assert mock_request.call_count == 2

    def test_max_retries_exceeded(self):
        """Test that wrapper stops after max retries."""
        wrapper = HTTPRequestWrapper(max_retries=2, base_delay=0.01)

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"error": "server error"}
            mock_request.return_value = mock_response

            with pytest.raises(APIConnectionError):
                wrapper.make_request("https://api.example.com/data", raise_on_error=True)

            # Should be called max_retries + 1 times (initial + retries)
            assert mock_request.call_count == 3

    def test_rate_limiting(self):
        """Test rate limiting between requests."""
        wrapper = HTTPRequestWrapper(rate_limit_delay=0.1)

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response

            start_time = time.time()
            wrapper.make_request("https://api.example.com/data")
            wrapper.make_request("https://api.example.com/data")
            elapsed_time = time.time() - start_time

            # Second request should be delayed by rate_limit_delay
            assert elapsed_time >= 0.1

    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        wrapper = HTTPRequestWrapper(
            max_retries=3,
            retry_strategy=RetryStrategy.EXPONENTIAL,
            base_delay=0.1,
        )

        # Test delay calculation
        delay_0 = wrapper._calculate_retry_delay(0)
        delay_1 = wrapper._calculate_retry_delay(1)
        delay_2 = wrapper._calculate_retry_delay(2)

        assert delay_0 == 0.1  # base_delay * 2^0
        assert delay_1 == 0.2  # base_delay * 2^1
        assert delay_2 == 0.4  # base_delay * 2^2

    def test_linear_backoff(self):
        """Test linear backoff strategy."""
        wrapper = HTTPRequestWrapper(
            max_retries=3,
            retry_strategy=RetryStrategy.LINEAR,
            base_delay=0.5,
        )

        # All delays should be the same for linear
        delay_0 = wrapper._calculate_retry_delay(0)
        delay_1 = wrapper._calculate_retry_delay(1)
        delay_2 = wrapper._calculate_retry_delay(2)

        assert delay_0 == 0.5
        assert delay_1 == 0.5
        assert delay_2 == 0.5

    def test_no_retry_strategy(self):
        """Test no retry strategy."""
        wrapper = HTTPRequestWrapper(
            retry_strategy=RetryStrategy.NONE,
        )

        delay = wrapper._calculate_retry_delay(0)
        assert delay == 0

    def test_rate_limit_detection_http_429(self):
        """Test detection of HTTP 429 rate limit error."""
        wrapper = HTTPRequestWrapper()

        mock_response = MagicMock()
        mock_response.status_code = 429

        assert wrapper._is_rate_limit_error(mock_response) is True

    def test_rate_limit_detection_message(self):
        """Test detection of rate limit from message."""
        wrapper = HTTPRequestWrapper()

        mock_response = MagicMock()
        mock_response.status_code = 200

        json_data = {"error": "rate limit exceeded"}
        assert wrapper._is_rate_limit_error(mock_response, json_data) is True

        json_data = {"error": "too many requests"}
        assert wrapper._is_rate_limit_error(mock_response, json_data) is True

    def test_post_request(self):
        """Test POST request."""
        wrapper = HTTPRequestWrapper()

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"success": True}
            mock_request.return_value = mock_response

            result = wrapper.make_request("https://api.example.com/data", method="POST", json={"key": "value"})

            assert result == {"success": True}
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["method"] == "POST"
            assert call_args[1]["json"] == {"key": "value"}

    def test_request_with_headers(self):
        """Test request with custom headers."""
        wrapper = HTTPRequestWrapper()

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response

            headers = {"Authorization": "Bearer token"}
            wrapper.make_request("https://api.example.com/data", headers=headers)

            call_args = mock_request.call_args
            assert call_args[1]["headers"] == headers

    def test_request_with_params(self):
        """Test request with query parameters."""
        wrapper = HTTPRequestWrapper()

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response

            params = {"symbol": "AAPL", "limit": 100}
            wrapper.make_request("https://api.example.com/data", params=params)

            call_args = mock_request.call_args
            assert call_args[1]["params"] == params

    def test_request_timeout(self):
        """Test request with timeout."""
        wrapper = HTTPRequestWrapper(timeout=10.0)

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"data": "test"}
            mock_request.return_value = mock_response

            wrapper.make_request("https://api.example.com/data")

            call_args = mock_request.call_args
            assert call_args[1]["timeout"] == 10.0

    def test_custom_error_check(self):
        """Test custom error checking function."""
        wrapper = HTTPRequestWrapper()

        def custom_check(response):
            return response.get("status") == "error"

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"status": "error", "message": "test"}
            mock_request.return_value = mock_response

            with pytest.raises(ValueError, match="Custom error check failed"):
                wrapper.make_request("https://api.example.com/data", custom_error_check=custom_check)

    def test_non_json_response(self):
        """Test handling non-JSON response."""
        wrapper = HTTPRequestWrapper()

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.side_effect = ValueError("Not JSON")
            mock_response.text = "Plain text response"
            mock_request.return_value = mock_response

            result = wrapper.make_request("https://api.example.com/data")

            assert result == "Plain text response"

    def test_raise_on_error_false(self):
        """Test that raise_on_error=False returns None instead of
        raising."""
        wrapper = HTTPRequestWrapper(max_retries=0)

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"error": "server error"}
            mock_request.return_value = mock_response

            result = wrapper.make_request("https://api.example.com/data", raise_on_error=False)

            assert result is None

    def test_rate_limit_retry_multiplier(self):
        """Test that rate limit errors use custom retry multiplier."""
        wrapper = HTTPRequestWrapper(max_retries=1, base_delay=0.05)

        with patch("requests.request") as mock_request:
            # First request gets rate limited
            mock_rate_limit = MagicMock()
            mock_rate_limit.status_code = 429
            mock_rate_limit.url = "https://api.example.com/data"
            mock_rate_limit.headers = {}
            mock_rate_limit.json.return_value = {"error": "rate limit exceeded"}

            # Second request succeeds
            mock_success = MagicMock()
            mock_success.status_code = 200
            mock_success.url = "https://api.example.com/data"
            mock_success.headers = {}
            mock_success.json.return_value = {"data": "test"}

            mock_request.side_effect = [mock_rate_limit, mock_success]

            # Should apply rate_limit_retry_multiplier (default 2.0)
            start = time.time()
            result = wrapper.make_request("https://api.example.com/data", rate_limit_retry_multiplier=2.0)
            elapsed = time.time() - start

            assert result == {"data": "test"}
            # Should have delayed by base_delay * multiplier = 0.05 * 2 = 0.1
            assert elapsed >= 0.1

    def test_authentication_error_does_not_retry(self):
        """Test that auth failures raise immediately."""
        wrapper = HTTPRequestWrapper(max_retries=3, base_delay=0.01)

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {}
            mock_response.json.return_value = {"error": "unauthorized"}
            mock_request.return_value = mock_response

            with pytest.raises(AuthenticationError):
                wrapper.make_request("https://api.example.com/data")

            mock_request.assert_called_once()

    def test_rate_limit_error_exposes_retry_after(self):
        """Test Retry-After header propagation on rate limit
        failures."""
        wrapper = HTTPRequestWrapper(max_retries=0)

        with patch("requests.request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.url = "https://api.example.com/data"
            mock_response.headers = {"Retry-After": "7"}
            mock_response.json.return_value = {"error": "rate limit exceeded"}
            mock_request.return_value = mock_response

            with pytest.raises(RateLimitError) as exc_info:
                wrapper.make_request("https://api.example.com/data")

            assert exc_info.value.retry_after == 7


class TestCreateDefaultWrapper:
    """Tests for create_default_wrapper function."""

    def test_create_with_defaults(self):
        """Test creating wrapper with default settings."""
        wrapper = create_default_wrapper()

        assert wrapper.max_retries == 3
        assert wrapper.retry_strategy == RetryStrategy.EXPONENTIAL
        assert wrapper.base_delay == 1.0
        assert wrapper.timeout == 30.0

    def test_create_with_overrides(self):
        """Test creating wrapper with custom settings."""
        wrapper = create_default_wrapper(max_retries=5, base_delay=2.0, rate_limit_delay=1.5)

        assert wrapper.max_retries == 5
        assert wrapper.base_delay == 2.0
        assert wrapper.rate_limit_delay == 1.5
        # Other values should still use defaults
        assert wrapper.retry_strategy == RetryStrategy.EXPONENTIAL
