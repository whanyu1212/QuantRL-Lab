"""Tests for AsyncHTTPRequestWrapper."""

import asyncio
import re

import pytest
from aioresponses import aioresponses

from quantrl_lab.data.exceptions import APIConnectionError, RateLimitError
from quantrl_lab.data.utils.async_request_utils import AsyncHTTPRequestWrapper


class TestAsyncHTTPRequestWrapperInit:
    """Tests for AsyncHTTPRequestWrapper initialization."""

    def test_default_initialization(self):
        wrapper = AsyncHTTPRequestWrapper()
        assert wrapper.max_retries == 3
        assert wrapper.base_delay == 1.0
        assert wrapper.timeout is not None

    def test_custom_initialization(self):
        wrapper = AsyncHTTPRequestWrapper(
            max_retries=5,
            base_delay=2.0,
            concurrency=3,
            timeout=60.0,
        )
        assert wrapper.max_retries == 5
        assert wrapper.base_delay == 2.0

    def test_semaphore_created_with_concurrency(self):
        wrapper = AsyncHTTPRequestWrapper(concurrency=4)
        # Semaphore._value reflects the configured concurrency
        assert wrapper._semaphore._value == 4


class TestAsyncHTTPRequestWrapperRateLimitDetection:
    """Tests for _is_rate_limit_error logic."""

    def test_http_429_is_rate_limit(self):
        wrapper = AsyncHTTPRequestWrapper()
        assert wrapper._is_rate_limit_error(429, None) is True

    def test_200_is_not_rate_limit(self):
        wrapper = AsyncHTTPRequestWrapper()
        assert wrapper._is_rate_limit_error(200, None) is False

    def test_rate_limit_keyword_in_body(self):
        wrapper = AsyncHTTPRequestWrapper()
        assert wrapper._is_rate_limit_error(200, {"message": "rate limit exceeded"}) is True
        assert wrapper._is_rate_limit_error(200, {"message": "too many requests"}) is True
        assert wrapper._is_rate_limit_error(200, {"message": "api call frequency"}) is True
        assert wrapper._is_rate_limit_error(200, {"message": "throttle"}) is True

    def test_no_rate_limit_in_normal_body(self):
        wrapper = AsyncHTTPRequestWrapper()
        assert wrapper._is_rate_limit_error(200, {"data": "some result"}) is False


class TestAsyncHTTPRequestWrapperBackoff:
    """Tests for exponential backoff delay calculation."""

    def test_backoff_doubles_each_attempt(self):
        wrapper = AsyncHTTPRequestWrapper(base_delay=1.0)
        assert wrapper._backoff_delay(0) == 1.0
        assert wrapper._backoff_delay(1) == 2.0
        assert wrapper._backoff_delay(2) == 4.0

    def test_rate_limit_doubles_backoff(self):
        wrapper = AsyncHTTPRequestWrapper(base_delay=1.0)
        assert wrapper._backoff_delay(0, rate_limited=True) == 2.0
        assert wrapper._backoff_delay(1, rate_limited=True) == 4.0


class TestAsyncHTTPRequestWrapperMakeRequest:
    """Tests for make_request using aioresponses mock."""

    @pytest.mark.asyncio
    async def test_successful_request_returns_json(self):
        wrapper = AsyncHTTPRequestWrapper(max_retries=1, base_delay=0.01)
        url = "https://api.example.com/data"

        with aioresponses() as m:
            m.get(url, payload={"symbol": "AAPL", "price": 150.0})
            import aiohttp

            async with aiohttp.ClientSession() as session:
                result = await wrapper.make_request(session, url)

        assert result == {"symbol": "AAPL", "price": 150.0}

    @pytest.mark.asyncio
    async def test_raises_connection_error_after_all_retries_exhausted(self):
        wrapper = AsyncHTTPRequestWrapper(max_retries=2, base_delay=0.01)
        url = "https://api.example.com/data"

        with aioresponses() as m:
            m.get(url, status=500)
            m.get(url, status=500)
            m.get(url, status=500)
            import aiohttp

            async with aiohttp.ClientSession() as session:
                with pytest.raises(APIConnectionError):
                    await wrapper.make_request(session, url)

    @pytest.mark.asyncio
    async def test_retries_on_server_error_then_succeeds(self):
        wrapper = AsyncHTTPRequestWrapper(max_retries=2, base_delay=0.01)
        url = "https://api.example.com/data"

        with aioresponses() as m:
            m.get(url, status=500)
            m.get(url, payload={"data": "ok"})
            import aiohttp

            async with aiohttp.ClientSession() as session:
                result = await wrapper.make_request(session, url)

        assert result == {"data": "ok"}

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self):
        wrapper = AsyncHTTPRequestWrapper(max_retries=2, base_delay=0.01)
        url = "https://api.example.com/data"

        with aioresponses() as m:
            m.get(url, status=429, payload={"message": "rate limit"})
            m.get(url, payload={"data": "ok"})
            import aiohttp

            async with aiohttp.ClientSession() as session:
                result = await wrapper.make_request(session, url)

        assert result == {"data": "ok"}

    @pytest.mark.asyncio
    async def test_raises_rate_limit_error_when_exhausted(self):
        wrapper = AsyncHTTPRequestWrapper(max_retries=1, base_delay=0.01)
        url = "https://api.example.com/data"

        with aioresponses() as m:
            m.get(url, status=429, payload={"message": "rate limit"})
            m.get(url, status=429, payload={"message": "rate limit"})
            import aiohttp

            async with aiohttp.ClientSession() as session:
                with pytest.raises(RateLimitError):
                    await wrapper.make_request(session, url)

    @pytest.mark.asyncio
    async def test_passes_params_in_request(self):
        """Test that params are forwarded in the query string."""
        wrapper = AsyncHTTPRequestWrapper(max_retries=1, base_delay=0.01)
        url = "https://api.example.com/data"
        params = {"symbol": "AAPL", "limit": 100}

        with aioresponses() as m:
            # Use regex to match URL regardless of query param ordering
            m.get(re.compile(r"https://api\.example\.com/data.*"), payload={"result": "ok"})
            import aiohttp

            async with aiohttp.ClientSession() as session:
                result = await wrapper.make_request(session, url, params=params)

        assert result == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_passes_headers_in_request(self):
        wrapper = AsyncHTTPRequestWrapper(max_retries=1, base_delay=0.01)
        url = "https://api.example.com/data"
        headers = {"Authorization": "Bearer test-token"}

        with aioresponses() as m:
            m.get(url, payload={"data": "secure"})
            import aiohttp

            async with aiohttp.ClientSession() as session:
                result = await wrapper.make_request(session, url, headers=headers)

        assert result == {"data": "secure"}

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that the semaphore correctly gates concurrent
        requests."""
        wrapper = AsyncHTTPRequestWrapper(max_retries=1, base_delay=0.01, concurrency=2)
        url = "https://api.example.com/data"

        with aioresponses() as m:
            for _ in range(4):
                m.get(url, payload={"data": "ok"})
            import aiohttp

            async with aiohttp.ClientSession() as session:
                tasks = [wrapper.make_request(session, url) for _ in range(4)]
                results = await asyncio.gather(*tasks)

        assert all(r == {"data": "ok"} for r in results)
