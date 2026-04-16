"""
HTTP request utilities with unified retry logic and rate limiting.

This module provides a consistent HTTP request wrapper to eliminate
duplication and inconsistencies in retry/error handling across data
sources.
"""

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from loguru import logger

from quantrl_lab.data.exceptions import APIConnectionError, AuthenticationError, RateLimitError


class RetryStrategy(Enum):
    """Retry strategy types."""

    LINEAR = "linear"  # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    NONE = "none"  # No retries


class HTTPRequestWrapper:
    """
    Unified HTTP request wrapper with configurable retry logic and rate
    limiting.

    Features:
    - Configurable retry strategies (linear, exponential, none)
    - Rate limiting with automatic throttling
    - Detailed logging of requests and errors
    - Support for custom error detection
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        base_delay: float = 1.0,
        rate_limit_delay: float = 0.0,
        timeout: Optional[float] = None,
    ):
        """
        Initialize HTTP request wrapper.

        Args:
            max_retries: Maximum number of retry attempts
            retry_strategy: Retry strategy (LINEAR, EXPONENTIAL, or NONE)
            base_delay: Base delay in seconds for retry strategy
            rate_limit_delay: Minimum delay between requests (rate limiting)
            timeout: Request timeout in seconds (None = no timeout)
        """
        self.max_retries = max_retries
        self.retry_strategy = retry_strategy
        self.base_delay = base_delay
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self._last_request_time: float = 0

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting by sleeping if necessary."""
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                sleep_time = self.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)

    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            float: Delay in seconds
        """
        if self.retry_strategy == RetryStrategy.LINEAR:
            return self.base_delay
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL:
            return self.base_delay * (2**attempt)
        else:  # RetryStrategy.NONE
            return 0

    def _is_rate_limit_error(self, response: requests.Response, json_data: Optional[Dict] = None) -> bool:
        """
        Detect if error is due to rate limiting.

        Args:
            response: HTTP response object
            json_data: Optional parsed JSON data

        Returns:
            bool: True if rate limit error detected
        """
        # Check HTTP 429 status
        if response.status_code == 429:
            return True

        # Check for common rate limit messages in response
        if json_data:
            error_text = str(json_data).lower()
            rate_limit_indicators = [
                "rate limit",
                "too many requests",
                "api call frequency",
                "throttle",
            ]
            return any(indicator in error_text for indicator in rate_limit_indicators)

        return False

    def _is_auth_error(self, response: requests.Response, json_data: Optional[Any] = None) -> bool:
        """Detect authentication and authorization failures."""
        if response.status_code in {401, 403}:
            return True

        if json_data is not None:
            error_text = str(json_data).lower()
            auth_indicators = [
                "unauthorized",
                "forbidden",
                "invalid api key",
                "invalid token",
                "authentication",
                "permission",
                "not authenticated",
            ]
            return any(indicator in error_text for indicator in auth_indicators)

        return False

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> Optional[int]:
        """Parse Retry-After header when present."""
        retry_after = response.headers.get("Retry-After")
        if retry_after is None:
            return None

        try:
            return int(retry_after)
        except (TypeError, ValueError):
            return None

    def _build_http_exception(
        self,
        response: requests.Response,
        json_data: Optional[Any] = None,
    ) -> APIConnectionError:
        """Map an HTTP response to a domain-specific exception."""
        status_code = response.status_code
        message = f"HTTP {status_code} error while requesting {response.url}"

        if self._is_auth_error(response, json_data):
            return AuthenticationError(message)

        if self._is_rate_limit_error(response, json_data):
            return RateLimitError(message, retry_after=self._retry_after_seconds(response))

        return APIConnectionError(message)

    def make_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        rate_limit_retry_multiplier: float = 2.0,
        raise_on_error: bool = True,
        custom_error_check: Optional[Callable[[Any], bool]] = None,
    ) -> Union[Dict[str, Any], List[Any], str, None]:
        """
        Make an HTTP request with retry logic and rate limiting.

        Args:
            url: Target URL
            method: HTTP method ('GET', 'POST', etc.)
            params: URL query parameters
            headers: HTTP headers
            data: Form data for POST requests
            json: JSON data for POST requests
            rate_limit_retry_multiplier: Extra delay multiplier for rate limit errors
            raise_on_error: If True, raise exception on final failure
            custom_error_check: Optional function to check JSON response for errors

        Returns:
            Any: Parsed JSON response

        Raises:
            APIConnectionError: If raise_on_error=True and request fails
            ValueError: If custom_error_check detects an error

        Examples:
            >>> wrapper = HTTPRequestWrapper(max_retries=3)
            >>> data = wrapper.make_request('https://api.example.com/data', params={'symbol': 'AAPL'})
        """
        attempt = 0

        while attempt <= self.max_retries:
            try:
                # Apply rate limiting
                self._apply_rate_limit()

                # Make request
                logger.debug(f"HTTP {method} request to {url} (attempt {attempt + 1}/{self.max_retries + 1})")

                response = requests.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    data=data,
                    json=json,
                    timeout=self.timeout,
                )

                # Update last request time
                self._last_request_time = time.time()

                # Parse JSON
                try:
                    json_data = response.json()
                except ValueError:
                    json_data = None

                if response.status_code >= 400:
                    raise self._build_http_exception(response, json_data)

                # Custom error checking
                if custom_error_check and custom_error_check(json_data):
                    raise ValueError(f"Custom error check failed for response: {json_data}")

                # Success
                if json_data is None:
                    logger.debug(f"Request successful: {len(response.text)} bytes (text)")
                    return response.text

                logger.debug(f"Request successful: {len(str(json_data))} bytes")
                return json_data

            except (AuthenticationError, RateLimitError, APIConnectionError) as e:
                if isinstance(e, AuthenticationError):
                    logger.error(f"Authentication failed: {e}")
                    if raise_on_error:
                        raise
                    return None

                is_rate_limit = isinstance(e, RateLimitError)

                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(attempt)

                    # Extra delay for rate limit errors
                    if is_rate_limit:
                        delay *= rate_limit_retry_multiplier
                        logger.warning(f"Rate limit hit. Retrying in {delay:.2f}s...")
                    else:
                        logger.warning(f"Request failed: {e}. Retrying in {delay:.2f}s...")

                    time.sleep(delay)
                    attempt += 1
                else:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
                    if raise_on_error:
                        raise
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Request error: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    attempt += 1
                else:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
                    if raise_on_error:
                        raise APIConnectionError(f"Request failed after {self.max_retries + 1} attempts: {e}") from e
                    return None

            except Exception as e:
                logger.error(f"Unexpected error during request: {e}")
                if raise_on_error:
                    raise
                return None

        # Should never reach here
        return None


def create_default_wrapper(**kwargs) -> HTTPRequestWrapper:
    """
    Create a default HTTP request wrapper with sensible defaults.

    Args:
        **kwargs: Override default parameters

    Returns:
        HTTPRequestWrapper: Configured wrapper instance

    Examples:
        >>> wrapper = create_default_wrapper(max_retries=5, rate_limit_delay=1.0)
    """
    defaults = {
        "max_retries": 3,
        "retry_strategy": RetryStrategy.EXPONENTIAL,
        "base_delay": 1.0,
        "rate_limit_delay": 0.0,
        "timeout": 30.0,
    }
    defaults.update(kwargs)
    return HTTPRequestWrapper(**defaults)
