import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
from loguru import logger

from quantrl_lab.data.interface import (
    DataSource,
    FundamentalDataCapable,
    HistoricalDataCapable,
)
from quantrl_lab.utils.config import ALPHA_VANTAGE_API_BASE, FundamentalMetric


class AlphaVantageDataLoader(DataSource, FundamentalDataCapable, HistoricalDataCapable):
    """Alpha Vantage implementation that provides various datasets."""

    def __init__(
        self,
        api_key: str = None,
        max_retries: int = 3,
        delay: int = 5,
    ):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.max_retries = max_retries
        self.delay = delay

    @property
    def source_name(self) -> str:
        return "Alpha Vantage"

    def connect(self):
        """Alpha Vantage doesn't require explicit connection - it uses HTTP requests."""
        pass

    def disconnect(self):
        """Alpha Vantage doesn't require explicit connection - it uses HTTP requests."""
        pass

    def is_connected(self) -> bool:
        """
        Alpha Vantage uses HTTP requests - assume connected if no network issues.
        """
        return True

    def list_available_instruments(
        self,
        instrument_type: Optional[str] = None,
        market: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Alpha Vantage does not provide a direct API to list all
        available instruments.

        This method is a placeholder.
        """
        logger.warning("Alpha Vantage does not support listing available instruments.")
        return []

    def get_historical_ohlcv_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get historical OHLCV data from Alpha Vantage.

        Args:
            symbol: Stock symbol to fetch data for
            start_date: Start date for filtering (note: Alpha Vantage returns all data)
            end_date: End date for filtering (note: Alpha Vantage returns all data)
            interval: Time interval - "1d" for daily, or intraday intervals like "5min", "15min"
            **kwargs: Additional parameters including 'adjusted' (bool, default=True), 'outputsize', 'month', etc.

        Returns:
            Dict[str, Any]: Raw Alpha Vantage API response
        """
        # Extract adjusted parameter from kwargs, default to False because
        # daily adjusted data is behind the paywall

        adjusted = kwargs.pop("adjusted", False)

        logger.info(f"Fetching {interval} data for {symbol}")

        # Determine which API endpoint to use based on interval
        if interval == "1d":
            if adjusted:
                # Use daily adjusted data (includes dividend/split adjustments)
                raw_data = self._get_daily_adjusted_data(symbol, **kwargs)
                logger.debug(f"Using adjusted daily data for {symbol}")
            else:
                # Use regular daily data (raw prices)
                raw_data = self._get_daily_data(symbol, **kwargs)
                logger.debug(f"Using raw daily data for {symbol}")

        elif interval in ["1min", "5min", "15min", "30min", "60min"]:
            # Use intraday data (adjusted parameter doesn't apply to intraday)
            if adjusted:
                logger.warning("Adjusted prices not available for intraday data, using raw prices")
            raw_data = self._get_intraday_data(symbol, interval=interval, **kwargs)
            logger.debug(f"Using {interval} intraday data for {symbol}")
        else:
            raise ValueError(f"Unsupported interval: {interval}. Use '1d' or intraday intervals like '5min'")

        if not raw_data:
            logger.error(f"Failed to fetch data for {symbol}")
            return {}

        logger.info(f"Successfully retrieved {interval} data for {symbol}")
        return raw_data

    def _make_api_request(self, function: str, symbol: str, **params) -> Optional[Dict[str, Any]]:
        """Centralized private method for making Alpha Vantage API
        requests."""

        # Build URL parameters
        url_params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
            **params,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.get(ALPHA_VANTAGE_API_BASE, params=url_params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    # Check for Alpha Vantage specific errors
                    if "Error Message" in data:
                        logger.error(f"API Error for {symbol}: {data['Error Message']}")
                        return None

                    if "Note" in data and "API call frequency" in data.get("Note", ""):
                        logger.warning(f"Rate limit hit for {symbol}, retrying...")
                        if attempt < self.max_retries - 1:
                            wait_time = self.delay * (2**attempt)
                            time.sleep(wait_time)
                            continue
                        return None

                    logger.info(f"Successfully fetched {function} data for {symbol}")
                    return data

                elif response.status_code == 429:  # Rate limit
                    if attempt < self.max_retries - 1:
                        wait_time = self.delay * (2**attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue

                response.raise_for_status()

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {symbol} (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for {symbol} (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error for {symbol}: {e} (attempt {attempt + 1})")

            if attempt < self.max_retries - 1:
                time.sleep(self.delay * (attempt + 1))

        logger.error(f"Failed to fetch {function} data for {symbol} after {self.max_retries} attempts")
        return None

    def _get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company overview data from Alpha Vantage. This includes
        key statistics and company information.

        Args:
            symbol (str): Stock symbol to fetch data for.

        Returns:
            Optional[Dict[str, Any]]: data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("OVERVIEW", symbol)

    def _get_etf_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch ETF profile data from Alpha Vantage.
        This includes ETF holdings and other profile information.
        Note: ETF profile is not typically used in our context,
        but included for completeness.

        Args:
            symbol (str): ETF symbol to fetch data for.

        Returns:
            Optional[Dict[str, Any]]: data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("ETF_PROFILE", symbol)

    def _get_dividend_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch dividend payment history from Alpha Vantage. This includes
        historical dividend payments for the given symbol.

        Args:
            symbol (str): Stock symbol to fetch dividend data for.

        Returns:
            Optional[Dict[str, Any]]: data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("DIVIDENDS", symbol)

    def _get_splits_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch stock split history from Alpha Vantage.
        This includes historical stock splits for the given symbol.
        Note: not useful in our context, but included for completeness.

        Args:
            symbol (str): Stock symbol to fetch split data for.

        Returns:
            Optional[Dict[str, Any]]: data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("SPLITS", symbol)

    def _get_income_statement_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch income statement data from Alpha Vantage."""
        return self._make_api_request("INCOME_STATEMENT", symbol)

    def _get_balance_sheet_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch balance sheet data from Alpha Vantage."""
        return self._make_api_request("BALANCE_SHEET", symbol)

    def _get_cash_flow_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch cash flow data from Alpha Vantage."""
        return self._make_api_request("CASH_FLOW", symbol)

    def _get_earnings_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch earnings data from Alpha Vantage."""
        return self._make_api_request("EARNINGS", symbol)

    def _get_intraday_data(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: str = "full",
        month: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch intraday data from Alpha Vantage.

        Args:
            symbol: Stock symbol to fetch data for
            interval: Time interval (1min, 5min, 15min, 30min, 60min)
            outputsize: 'compact' or 'full'
            month: Optional month in YYYY-MM format for historical intraday data
            **kwargs: Additional Alpha Vantage API parameters

        Returns:
            Optional[Dict[str, Any]]: Intraday data or None if request fails
        """
        # Build parameters, only include non-None values
        params = {
            "interval": interval,
            "outputsize": outputsize,
        }

        if month:
            params["month"] = month

        # Add any additional kwargs
        params.update(kwargs)

        return self._make_api_request("TIME_SERIES_INTRADAY", symbol, **params)

    def _get_daily_data(self, symbol: str, outputsize: str = "full", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch daily time series data from Alpha Vantage.

        Args:
            symbol: Stock symbol to fetch data for
            outputsize: 'compact' (last 100 days) or 'full' (20+ years of data)
            **kwargs: Additional Alpha Vantage API parameters

        Returns:
            Optional[Dict[str, Any]]: Daily OHLCV data or None if request fails
        """
        params = {"outputsize": outputsize}
        params.update(kwargs)

        return self._make_api_request("TIME_SERIES_DAILY", symbol, **params)

    def _get_daily_adjusted_data(self, symbol: str, outputsize: str = "full", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch daily adjusted time series data from Alpha Vantage. This
        includes dividend and split adjustments.

        Args:
            symbol: Stock symbol to fetch data for
            outputsize: 'compact' (last 100 days) or 'full' (20+ years of data)
            **kwargs: Additional Alpha Vantage API parameters

        Returns:
            Optional[Dict[str, Any]]: Daily adjusted OHLCV data or None if request fails
        """
        params = {"outputsize": outputsize}
        params.update(kwargs)

        return self._make_api_request("TIME_SERIES_DAILY_ADJUSTED", symbol, **params)

    def get_fundamental_data(
        self, symbol: str, metrics: List[Union[FundamentalMetric, str]], **kwargs
    ) -> Union[pd.DataFrame, Dict]:
        """
        Get fundamental data for a single symbol by combining multiple
        Alpha Vantage API calls.

        Args:
            symbol: Stock symbol to fetch data for
            metrics: List of FundamentalMetric enums or strings
            return_format: 'dict' or 'dataframe' (default: 'dict')
            **kwargs: Additional parameters

        Returns:
            Dict with combined fundamental data or DataFrame
        """
        # return_format = kwargs.get("return_format", "dict")
        results = {}

        # Map metrics to private methods - use enum objects as keys consistently
        metric_methods = {
            FundamentalMetric.COMPANY_OVERVIEW: self._get_company_overview,
            # Remark: etf profile not useful in our context
            # FundamentalMetric.ETF_PROFILE: self._get_etf_profile,
            FundamentalMetric.DIVIDENDS: self._get_dividend_data,
            FundamentalMetric.SPLITS: self._get_splits_data,
            FundamentalMetric.INCOME_STATEMENT: self._get_income_statement_data,
            FundamentalMetric.BALANCE_SHEET: self._get_balance_sheet_data,
            FundamentalMetric.CASH_FLOW: self._get_cash_flow_data,
            FundamentalMetric.EARNINGS: self._get_earnings_data,
        }

        logger.info(f"Fetching fundamental data for {symbol}")

        for metric in metrics:
            # Handle both enum and string inputs
            if isinstance(metric, str):
                try:
                    metric_enum = FundamentalMetric(metric.lower())
                except ValueError:
                    logger.warning(f"Unknown metric '{metric}' for symbol {symbol}")
                    results[metric] = None
                    continue
            else:
                metric_enum = metric

            if metric_enum in metric_methods:
                method = metric_methods[metric_enum]
                data = method(symbol)

                if data:
                    results[metric_enum.value] = data
                    logger.debug(f"Successfully fetched {metric_enum.value} for {symbol}")
                else:
                    logger.warning(f"Failed to fetch {metric_enum.value} for {symbol}")
                    results[metric_enum.value] = None
            else:
                logger.warning(f"Unsupported metric '{metric_enum}' for symbol {symbol}")
                results[metric_enum.value] = None

        # TODO: Implement DataFrame conversion for cleaner output

        # Return format based on user preference
        # if return_format.lower() == "dataframe":
        #     return self._convert_to_dataframe(results, symbol)
        # else:
        return results
