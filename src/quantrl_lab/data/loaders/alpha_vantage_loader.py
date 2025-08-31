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
    MacroDataCapable,
    NewsDataCapable,
)
from quantrl_lab.data.loaders.alpha_vantage_mappings import (
    ALPHA_VANTAGE_COLUMN_MAPPER,
)
from quantrl_lab.utils.common import convert_datetime_to_alpha_vantage_format
from quantrl_lab.utils.config import (
    ALPHA_VANTAGE_API_BASE,
    FundamentalMetric,
    MacroIndicator,
)


class AlphaVantageDataLoader(
    DataSource,
    FundamentalDataCapable,
    HistoricalDataCapable,
    MacroDataCapable,
    NewsDataCapable,
):
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

    # Historical Data Method #
    def get_historical_ohlcv_data(
        self,
        symbols: str,
        interval: str = "1d",
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from Alpha Vantage.

        Args:
            symbol: Stock symbol to fetch data for
            interval: Time interval - "1d" for daily, or intraday intervals like "1min", "5min",
            "15min", "30min", "60min"
            start_date: Optional start date for filtering. If None, no start filtering is applied.
            end_date: Optional end date for filtering. If None, no end filtering is applied.
            **kwargs: Additional parameters including:
                     - 'adjusted' (bool, default=False): For daily data only
                     - 'outputsize' (str): "compact" or "full"
                     - 'month' (str): For intraday data, specify "YYYY-MM" for historical month

        For daily data: If start_date/end_date are None, defaults to outputsize='full'
        For intraday data: If 'month' is not specified, returns recent data (typically last 15-30 days)

        Returns:
            pd.DataFrame: OHLCV data, optionally filtered by date range
        """
        # Extract adjusted parameter from kwargs, default to False
        adjusted = kwargs.pop("adjusted", False)

        # Convert dates to datetime objects for filtering if provided
        parsed_start_date = None
        parsed_end_date = None

        if start_date is not None:
            parsed_start_date = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        if end_date is not None:
            parsed_end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        # Log what we're fetching
        log_msg = f"Fetching {interval} data for {symbols}"
        if parsed_start_date or parsed_end_date:
            if parsed_start_date and parsed_end_date:
                log_msg += f" from {parsed_start_date.date()} to {parsed_end_date.date()}"
            elif parsed_start_date:
                log_msg += f" from {parsed_start_date.date()} onwards"
            elif parsed_end_date:
                log_msg += f" up to {parsed_end_date.date()}"
        else:
            log_msg += " (all available data for given parameters)"
        logger.info(log_msg)

        # Determine which API endpoint to use based on interval
        if interval == "1d":
            # For daily data, default to full if no date filtering
            if not parsed_start_date and not parsed_end_date and "outputsize" not in kwargs:
                kwargs["outputsize"] = "full"
                logger.info("Defaulting to outputsize='full' for daily data with no date range specified")

            if adjusted:
                raw_data = self._get_daily_adjusted_data(symbols, **kwargs)
                logger.info(f"Using adjusted daily data for {symbols}")
            else:
                raw_data = self._get_daily_data(symbols, **kwargs)
                logger.info(f"Using raw daily data for {symbols}")

            time_series_key = "Time Series (Daily)"

        elif interval in ["1min", "5min", "15min", "30min", "60min"]:
            if adjusted:
                logger.warning("Adjusted prices not available for intraday data, using raw prices")

            # Log info about intraday data fetching
            if "month" in kwargs:
                logger.info(f"Fetching {interval} intraday data for {symbols} for month: {kwargs['month']}")
            else:
                logger.info(
                    f"Fetching {interval} intraday data for {symbols} (recent data - typically last 15-30 days)"
                )
                logger.info("For historical intraday data, specify 'month=\"YYYY-MM\"' in kwargs")

            raw_data = self._get_intraday_data(symbols, interval=interval, **kwargs)
            time_series_key = f"Time Series ({interval})"
        else:
            raise ValueError(
                f"Unsupported interval: {interval}. Use '1d' or intraday intervals like",
                "'1min', '5min', '15min', '30min', '60min'",
            )

        if not raw_data:
            logger.error(f"Failed to fetch data for {symbols}")
            return pd.DataFrame()

        # Extract time series data
        if time_series_key not in raw_data:
            logger.error(
                f"Expected key '{time_series_key}' not found in API response for {symbols}. "
                "This may be due to rate limits, invalid symbol, or no data available."
            )
            available_keys = list(raw_data.keys())
            logger.info(f"Available keys in response: {available_keys}")
            return pd.DataFrame()

        time_series = raw_data[time_series_key]

        if not time_series:
            logger.warning(f"Empty time series data for {symbols}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")

        # Get the appropriate column mapping and rename columns
        column_mapping = ALPHA_VANTAGE_COLUMN_MAPPER.get_mapping(interval, adjusted)
        df = df.rename(columns=column_mapping)

        # Drop any columns that weren't in our mapping
        expected_columns = list(set(column_mapping.values()))  # Remove duplicates
        df = df[df.columns.intersection(expected_columns)]

        # Convert string values to numeric
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        if "Adj_close" in df.columns:
            numeric_columns.append("Adj_close")
        if "Dividend" in df.columns:
            numeric_columns.append("Dividend")
        if "Split_coeff" in df.columns:
            numeric_columns.append("Split_coeff")

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Sort by date (Alpha Vantage returns newest first)
        df = df.sort_index()

        # Apply date filtering if dates are provided
        if parsed_start_date is not None:
            df = df[df.index >= parsed_start_date]
        if parsed_end_date is not None:
            df = df[df.index <= parsed_end_date]

        if df.empty:
            warning_msg = f"No data found for {symbols}"
            if parsed_start_date or parsed_end_date:
                warning_msg += " matching the specified date criteria"
            logger.warning(warning_msg)
        else:
            data_type = (
                "adjusted daily"
                if (interval == "1d" and adjusted)
                else (f"{interval} intraday" if interval != "1d" else "daily")
            )
            logger.info(f"Retrieved {len(df)} {data_type} records for {symbols}")

        df.reset_index(inplace=True)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        return df

    # Fundamental Data Method #
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
            **kwargs: Additional parameters, e.g., 'return_format'

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
                    logger.info(f"Successfully fetched {metric_enum.value} for {symbol}")
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

    # News Data Method #
    def get_news_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        limit: int = 50,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch news data for given symbols from Alpha Vantage. This
        method retrieves news articles related to the specified symbols
        within the given date range. It supports additional parameters
        like 'sort' and 'topics' to customize the news data.

        Args:
            symbols (Union[str, List[str]]): Symbols to fetch news for.
            start (Union[str, datetime]): Start datetime for news data.
            end (Optional[Union[str, datetime]], optional): End datetime for news.
            Defaults to None, which means current time.
            limit (int, optional): Maximum number of news items to fetch.
            Defaults to 50.
            **kwargs: Additional parameters for the API request,
            such as 'sort' or 'topics'.

        Returns:
            pd.DataFrame: DataFrame containing news data for the
            specified symbols.
        """
        # Convert dates to Alpha Vantage format
        time_from = convert_datetime_to_alpha_vantage_format(start)

        if end is None:
            end = datetime.now()
        time_to = convert_datetime_to_alpha_vantage_format(end)

        # Handle symbols - can be string or list
        if isinstance(symbols, str):
            tickers = symbols
        else:
            tickers = ",".join(symbols)

        logger.info(f"Fetching news for {tickers} from {time_from} to {time_to}")

        params = {
            "tickers": tickers,
            "time_from": time_from,
            "time_to": time_to,
            "limit": str(limit),
        }

        # Check for additional sort parameter in kwargs
        if "sort" in kwargs:
            params["sort"] = kwargs.pop("sort")
            logger.info(f"Using sort order: {params['sort']}")

        # Check for additional topics parameter in kwargs
        if "topics" in kwargs:
            params["topics"] = kwargs.pop("topics")
            logger.info(f"Using topics from kwargs: {params['topics']}")

        params.update(kwargs)

        # Make the API request (note: NEWS_SENTIMENT doesn't use symbol parameter,
        # it uses tickers instead, so we pass an empty string for symbol)
        news_data = self._make_api_request("NEWS_SENTIMENT", symbol="", **params)

        news_df = pd.DataFrame(news_data["feed"])

        # Rename time_published to created_at to standardize the column name
        if "time_published" in news_df.columns:
            news_df.rename(columns={"time_published": "created_at"}, inplace=True)

        # Convert created_at to datetime
        news_df["created_at"] = pd.to_datetime(news_df["created_at"], format="%Y%m%dT%H%M%S")
        news_df["Date"] = news_df["created_at"].dt.date

        news_df["sentiment_score"] = (
            news_df["ticker_sentiment"].apply(lambda x: self._find_ticker_sentiment(x, tickers)).astype(float)
        )

        return news_df

    def _find_ticker_sentiment(self, sentiment_list: List[Dict], ticker_symbol: str) -> Optional[float]:
        """
        Find the sentiment score for a specific ticker in the sentiment
        list.

        Args:
            sentiment_list (List[Dict]): A list of dictionaries
            containing sentiments for different tickers
            ticker_symbol (str): The ticker symbol to search
            for (e.g., 'AAPL').

        Returns:
            Optional[float]: The sentiment score for the specified ticker,
            or None if not found.
        """
        if not isinstance(sentiment_list, list):
            return None

        for item in sentiment_list:
            if item.get("ticker") == ticker_symbol:
                return item["ticker_sentiment_score"]
        return None

    # TODO: fix overengineering
    def get_macro_data(
        self,
        indicators: Union[str, List[str], Dict[str, Dict]],
        start: Union[str, datetime],
        end: Union[str, datetime],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get macroeconomic data for specified indicators. This method
        supports both standard indicator names and advanced dictionary
        format where each indicator can have its own parameters. e.g.: {
        "real_gdp": {"interval": "quarterly"}, "treasury_yield":
        {"interval": "monthly", "maturity": "10year"} }

        Args:
            indicators (Union[str, List[str], Dict[str, Dict]]): indicator(s) to fetch data for.
            start (Union[str, datetime]): time from
            end (Union[str, datetime]): time to
            **kwargs: Additional parameters for the API request

        Returns:
            Dict[str, Any]: Dictionary containing macroeconomic data for the
            specified indicators. Each key is the indicator name, and the value
            is the data fetched from Alpha Vantage.
        """

        # Handle different input formats
        if isinstance(indicators, dict):
            # Advanced dict format with per-indicator params
            return self._get_macro_data_with_params(indicators, **kwargs)
        else:
            # Standard format - convert to dict format internally
            if isinstance(indicators, (str, MacroIndicator)):
                indicators = [indicators]

            # Create dict with empty params for each indicator
            indicator_params = {ind: {} for ind in indicators}
            return self._get_macro_data_with_params(indicator_params, **kwargs)

    def _get_macro_data_with_params(
        self, indicator_params: Dict[Union[str, MacroIndicator], Dict], **global_kwargs
    ) -> Dict[str, Any]:
        """
        _summary_

        Args:
            indicator_params (Dict[Union[str, MacroIndicator], Dict]): _description_

        Returns:
            Dict[str, Any]: _description_
        """

        results = {}

        # Map indicators to private methods
        indicator_methods = {
            MacroIndicator.REAL_GDP: self._get_real_gdp_data,
            MacroIndicator.REAL_GDP_PER_CAPITA: self._get_real_gdp_per_capita_data,
            MacroIndicator.TREASURY_YIELD: self._get_treasury_yield_data,
            MacroIndicator.FEDERAL_FUNDS_RATE: self._get_federal_funds_rate_data,
            MacroIndicator.CPI: self._get_cpi_data,
            MacroIndicator.INFLATION: self._get_inflation_data,
            MacroIndicator.RETAIL_SALES: self._get_retail_sales_data,
            MacroIndicator.DURABLE_GOODS: self._get_durable_goods_data,
            MacroIndicator.UNEMPLOYMENT_RATE: self._get_unemployment_rate_data,
            MacroIndicator.NON_FARM_PAYROLL: self._get_non_farm_payroll_data,
        }

        logger.info(f"Fetching macro data for indicators: {list(indicator_params.keys())}")

        for indicator, ind_kwargs in indicator_params.items():
            # Convert string to enum if needed
            if isinstance(indicator, str):
                try:
                    indicator_enum = MacroIndicator(indicator.lower())
                except ValueError:
                    logger.warning(f"Unknown macro indicator '{indicator}'")
                    results[indicator] = None
                    continue
            else:
                indicator_enum = indicator

            if indicator_enum in indicator_methods:
                method = indicator_methods[indicator_enum]

                merged_kwargs = {**global_kwargs, **ind_kwargs}

                try:
                    method_kwargs = self._get_method_specific_kwargs(indicator_enum, merged_kwargs)

                    data = method(**method_kwargs)
                    if data:
                        results[indicator_enum.value] = data
                        logger.info(f"Successfully fetched {indicator_enum.value} data")
                    else:
                        logger.warning(f"Failed to fetch {indicator_enum.value} data")
                        results[indicator_enum.value] = None

                except ValueError as e:
                    logger.error(f"Parameter validation error for {indicator_enum.value}: {e}")
                    results[indicator_enum.value] = None
                except Exception as e:
                    logger.error(f"Error fetching {indicator_enum.value} data: {e}")
                    results[indicator_enum.value] = None
            else:
                logger.warning(f"Unsupported macro indicator '{indicator_enum}'")
                results[indicator_enum.value] = None

        return results

    def _get_method_specific_kwargs(self, indicator: MacroIndicator, kwargs: Dict) -> Dict:
        """
        Get method-specific parameters for macroeconomic indicators.

        Args:
            indicator (MacroIndicator): enum to filter kwargs for.
            kwargs (Dict): additional parameters for the API request.

        Raises:
            ValueError: If the interval or maturity parameters are invalid
            ValueError: If the indicator is not supported

        Returns:
            Dict: Filtered kwargs for the specific indicator method.
        """

        indicator_config = {
            MacroIndicator.REAL_GDP: {
                "params": ["interval"],
                "valid_intervals": ["quarterly", "annual"],
                "default_interval": "annual",
            },
            MacroIndicator.REAL_GDP_PER_CAPITA: {
                "params": [],
            },
            MacroIndicator.TREASURY_YIELD: {
                "params": ["interval", "maturity"],
                "valid_intervals": ["daily", "weekly", "monthly"],
                "valid_maturities": [
                    "3month",
                    "2year",
                    "5year",
                    "7year",
                    "10year",
                    "30year",
                ],
                "default_interval": "monthly",
                "default_maturity": "10year",
            },
            MacroIndicator.FEDERAL_FUNDS_RATE: {
                "params": ["interval"],
                "valid_intervals": ["daily", "weekly", "monthly"],
                "default_interval": "monthly",
            },
            MacroIndicator.CPI: {
                "params": ["interval"],
                "valid_intervals": ["semiannual", "monthly"],
                "default_interval": "monthly",
            },
            MacroIndicator.INFLATION: {"params": []},
            MacroIndicator.RETAIL_SALES: {"params": []},
            MacroIndicator.DURABLE_GOODS: {"params": []},
            MacroIndicator.UNEMPLOYMENT_RATE: {"params": []},
            MacroIndicator.NON_FARM_PAYROLL: {"params": []},
        }

        config = indicator_config.get(indicator, {"params": []})
        filtered_kwargs = {}

        # Handle interval parameter with validation
        if "interval" in config.get("params", []):
            interval = kwargs.get("interval", config.get("default_interval"))
            valid_intervals = config.get("valid_intervals", [])

            if interval and valid_intervals and interval not in valid_intervals:
                raise ValueError(
                    f"Invalid interval '{interval}' for {indicator.value}. " f"Valid options: {valid_intervals}"
                )

            if interval:
                filtered_kwargs["interval"] = interval

        # Handle maturity parameter with validation
        if "maturity" in config.get("params", []):
            maturity = kwargs.get("maturity", config.get("default_maturity"))
            valid_maturities = config.get("valid_maturities", [])

            if maturity and valid_maturities and maturity not in valid_maturities:
                raise ValueError(
                    f"Invalid maturity '{maturity}' for {indicator.value}. " f"Valid options: {valid_maturities}"
                )

            if maturity:
                filtered_kwargs["maturity"] = maturity

        for key, value in kwargs.items():
            if key not in ["interval", "maturity"] and key not in filtered_kwargs:
                filtered_kwargs[key] = value

        return filtered_kwargs

    # Common method for API requests #
    def _make_api_request(self, function: str, symbol: str = "", **params) -> Optional[Dict[str, Any]]:
        """Centralized private method for making Alpha Vantage API
        requests."""

        # Build URL parameters
        url_params = {
            "function": function,
            "apikey": self.api_key,
            **params,
        }

        # Only add symbol if it's provided (NEWS_SENTIMENT doesn't use symbol)
        if symbol:
            url_params["symbol"] = symbol

        for attempt in range(self.max_retries):
            try:
                response = requests.get(ALPHA_VANTAGE_API_BASE, params=url_params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    # Check for Alpha Vantage specific errors
                    if "Error Message" in data:
                        error_msg = f"API Error: {data['Error Message']}"
                        if symbol:
                            error_msg += f" for {symbol}"
                        logger.error(error_msg)
                        return None

                    if "Note" in data and "API call frequency" in data.get("Note", ""):
                        warning_msg = "Rate limit hit"
                        if symbol:
                            warning_msg += f" for {symbol}"
                        logger.warning(f"{warning_msg}, retrying...")

                        if attempt < self.max_retries - 1:
                            wait_time = self.delay * (2**attempt)
                            time.sleep(wait_time)
                            continue
                        return None

                    success_msg = f"Successfully fetched {function} data"
                    if symbol:
                        success_msg += f" for {symbol}"
                    logger.info(success_msg)
                    return data

                elif response.status_code == 429:  # Rate limit
                    if attempt < self.max_retries - 1:
                        wait_time = self.delay * (2**attempt)
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue

                response.raise_for_status()

            except requests.exceptions.Timeout:
                timeout_msg = f"Timeout (attempt {attempt + 1})"
                if symbol:
                    timeout_msg = f"Timeout for {symbol} (attempt {attempt + 1})"
                logger.warning(timeout_msg)
            except requests.exceptions.ConnectionError:
                conn_msg = f"Connection error (attempt {attempt + 1})"
                if symbol:
                    conn_msg = f"Connection error for {symbol} (attempt {attempt + 1})"
                logger.warning(conn_msg)
            except requests.exceptions.RequestException as e:
                req_msg = f"Request error: {e} (attempt {attempt + 1})"
                if symbol:
                    req_msg = f"Request error for {symbol}: {e} (attempt {attempt + 1})"
                logger.warning(req_msg)

            if attempt < self.max_retries - 1:
                time.sleep(self.delay * (attempt + 1))

        error_msg = f"Failed to fetch {function} data after {self.max_retries} attempts"
        if symbol:
            error_msg = f"Failed to fetch {function} data for {symbol} after {self.max_retries} attempts"
        logger.error(error_msg)
        return None

    # Private Fundamental Data Methods #
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
        """
        Fetch income statement data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch income statement for.

        Returns:
            Optional[Dict[str, Any]]: Income statement data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("INCOME_STATEMENT", symbol)

    def _get_balance_sheet_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch balance sheet data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch balance sheet for.

        Returns:
            Optional[Dict[str, Any]]: Balance sheet data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("BALANCE_SHEET", symbol)

    def _get_cash_flow_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch cash flow statement data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch cash flow statement for.

        Returns:
            Optional[Dict[str, Any]]: Cash flow statement data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("CASH_FLOW", symbol)

    def _get_earnings_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch earnings data from Alpha Vantage. This includes.

        Args:
            symbol (str): Stock symbol to fetch earnings data for.

        Returns:
            Optional[Dict[str, Any]]: Earnings data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("EARNINGS", symbol)

    # Private Historical Data Methods #

    def _get_intraday_data(
        self,
        symbol: str,
        interval: str = "5min",  # Valid Time intervals: 1min, 5min, 15min, 30min, 60min
        outputsize: str = "full",
        month: Optional[str] = None,
        **kwargs,  # not required but for future extensibility
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

    # Daily Adjusted Data Method #
    # Only available if you have a premium Alpha Vantage key #
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

    # Private Macro Data Methods #
    def _get_real_gdp_data(self, interval: str = "annual", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch real GDP data from Alpha Vantage.

        Args:
            interval (str, optional): Defaults to "annual". Available options are
            "quarterly" and "annual". Determines the frequency of the data.

            **kwargs: Additional parameters for the API request.

        Raises:
            ValueError: If the interval is not one of the valid options.

        Returns:
            Optional[Dict[str, Any]]: Real GDP data in dictionary format or
            None if request fails.
        """

        # Validate interval parameter
        if interval not in ["quarterly", "annual"]:
            raise ValueError(f"Invalid interval '{interval}'. Use 'quarterly' or 'annual'.")

        params = {"interval": interval}
        params.update(kwargs)

        return self._make_api_request("REAL_GDP", symbol="", **params)

    def _get_real_gdp_per_capita_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch real GDP per capita data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Real GDP per capita data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("REAL_GDP_PER_CAPITA", symbol="", **kwargs)

    def _get_treasury_yield_data(
        self, interval: str = "monthly", maturity: str = "10year", **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch treasury yield data from Alpha Vantage.

        Args:
            interval (str, optional): Defaults to "monthly".
            maturity (str, optional): Defaults to "10year".

        Raises:
            ValueError: If the interval or maturity parameters are invalid.
            ValueError: If the maturity is not one of the valid options.

        Returns:
            Optional[Dict[str, Any]]: Treasury yield data in dictionary format or
            None if request fails.
        """

        # Validate interval parameter
        valid_intervals = ["daily", "weekly", "monthly"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval '{interval}'. Use one of: {valid_intervals}")

        # Validate maturity parameter
        valid_maturities = ["3month", "2year", "5year", "7year", "10year", "30year"]
        if maturity not in valid_maturities:
            raise ValueError(f"Invalid maturity '{maturity}'. Use one of: {valid_maturities}")

        params = {"interval": interval, "maturity": maturity}
        params.update(kwargs)

        return self._make_api_request("TREASURY_YIELD", symbol="", **params)

    def _get_federal_funds_rate_data(self, interval: str = "monthly", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch federal funds rate data from Alpha Vantage.

        Args:
            interval (str, optional): Defaults to "monthly".

        Raises:
            ValueError: If the interval is not one of the valid options.

        Returns:
            Optional[Dict[str, Any]]: Federal funds rate data in dictionary format or
            None if request fails.
        """
        valid_intervals = ["daily", "weekly", "monthly"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval '{interval}'. Use one of: {valid_intervals}")

        params = {"interval": interval}
        params.update(kwargs)

        return self._make_api_request("FEDERAL_FUNDS_RATE", symbol="", **params)

    def _get_cpi_data(self, interval: str = "monthly", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch Consumer Price Index (CPI) data from Alpha Vantage.

        Args:
            interval (str, optional): Defaults to "monthly".

        Raises:
            ValueError: If the interval is not one of the valid options.

        Returns:
            Optional[Dict[str, Any]]: CPI data in dictionary format or
            None if request fails.
        """
        valid_intervals = ["semiannual", "monthly"]
        if interval not in valid_intervals:
            raise ValueError(f"Invalid interval '{interval}'. Use one of: {valid_intervals}")

        params = {"interval": interval}
        params.update(kwargs)

        return self._make_api_request("CPI", symbol="", **params)

    def _get_inflation_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch inflation data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Inflation data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("INFLATION", symbol="", **kwargs)

    def _get_retail_sales_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch retail sales data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Retail sales data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("RETAIL_SALES", symbol="", **kwargs)

    def _get_durable_goods_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch durable goods data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Durable goods data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("DURABLE_GOODS", symbol="", **kwargs)

    def _get_unemployment_rate_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch unemployment rate data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Unemployment rate data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("UNEMPLOYMENT", symbol="", **kwargs)

    def _get_non_farm_payroll_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch non-farm payroll data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Non-farm payroll data in dictionary format or
            None if request fails.
        """
        return self._make_api_request("NONFARM_PAYROLL", symbol="", **kwargs)
