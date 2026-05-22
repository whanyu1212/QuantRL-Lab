"""
Async HTTP request utilities with rate limiting and retry logic.

Mirrors the interface of request_utils.py but built on aiohttp for use
in concurrent multi-symbol data fetching workflows.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from loguru import logger

try:
    import aiohttp
except ImportError as e:
    raise ImportError("aiohttp is required for async data fetching. Install with: pip install aiohttp") from e


RATE_LIMIT_INDICATORS = [
    "rate limit",
    "too many requests",
    "api call frequency",
    "throttle",
]


class AsyncHTTPRequestWrapper:
    """
    Async HTTP request wrapper with semaphore-based concurrency control,
    rate limit detection, and exponential backoff retries.

    The caller is responsible for creating and managing the aiohttp.ClientSession
    (use as an async context manager at the call site). The session is passed
    into make_request() to avoid per-request session creation overhead.

    Args:
        max_retries (int): Maximum retry attempts on failure.
        base_delay (float): Base delay in seconds for exponential backoff.
        concurrency (int): Max simultaneous in-flight requests (via asyncio.Semaphore).
        timeout (float): Per-request timeout in seconds.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        concurrency: int = 10,
        timeout: float = 30.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._semaphore = asyncio.Semaphore(concurrency)

    def _is_rate_limit_error(self, status: int, json_data: Optional[Any]) -> bool:
        """Detect rate limit errors from HTTP status or response
        body."""
        if status == 429:
            return True
        if json_data is not None:
            body_text = str(json_data).lower()
            return any(indicator in body_text for indicator in RATE_LIMIT_INDICATORS)
        return False

    def _backoff_delay(self, attempt: int, rate_limited: bool = False) -> float:
        """Exponential backoff; doubles for rate limit errors."""
        delay = self.base_delay * (2**attempt)
        return delay * 2 if rate_limited else delay

    async def make_request(
        self,
        session: "aiohttp.ClientSession",
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Make an async GET request with retry and rate limit handling.

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session managed by caller.
            url (str): Target URL.
            params (Dict[str, Any], optional): Query parameters. Defaults to None.
            headers (Dict[str, str], optional): HTTP headers. Defaults to None.

        Returns:
            Optional[Union[Dict[str, Any], List[Any]]]: Parsed JSON response,
                or None if all retries are exhausted.
        """

        async with self._semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    logger.debug(
                        "Async GET {url} (attempt {n}/{total})",
                        url=url,
                        n=attempt + 1,
                        total=self.max_retries + 1,
                    )
                    async with session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout,
                    ) as response:
                        try:
                            json_data = await response.json(content_type=None)
                        except Exception:
                            json_data = None

                        is_rate_limit = self._is_rate_limit_error(response.status, json_data)

                        if response.status == 200 and not is_rate_limit:
                            logger.debug("Async request successful: {url}", url=url)
                            return json_data

                        if attempt < self.max_retries:
                            delay = self._backoff_delay(attempt, rate_limited=is_rate_limit)
                            if is_rate_limit:
                                logger.warning(
                                    "Rate limit hit for {url}. Retrying in {delay:.1f}s...",
                                    url=url,
                                    delay=delay,
                                )
                            else:
                                logger.warning(
                                    "HTTP {status} for {url}. Retrying in {delay:.1f}s...",
                                    status=response.status,
                                    url=url,
                                    delay=delay,
                                )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                "Async request failed after {n} attempts: {url} (status={status})",
                                n=self.max_retries + 1,
                                url=url,
                                status=response.status,
                            )
                            return None

                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning("Timeout for {url}. Retrying in {delay:.1f}s...", url=url, delay=delay)
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "Async request timed out after {n} attempts: {url}", n=self.max_retries + 1, url=url
                        )
                        return None

                except Exception as e:
                    if attempt < self.max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning(
                            "Request error for {url}: {e}. Retrying in {delay:.1f}s...", url=url, e=e, delay=delay
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "Async request failed after {n} attempts: {url} — {e}", n=self.max_retries + 1, url=url, e=e
                        )
                        return None

        return None
