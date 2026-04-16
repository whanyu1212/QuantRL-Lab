import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from quantrl_lab.data.config import (
    ALPHA_VANTAGE_API_BASE,
    FundamentalMetric,
    MacroIndicator,
)
from quantrl_lab.data.exceptions import APIConnectionError, AuthenticationError, InvalidParametersError, RateLimitError
from quantrl_lab.data.interface import (
    DataSource,
    FundamentalDataCapable,
    HistoricalDataCapable,
    MacroDataCapable,
    NewsDataCapable,
)
from quantrl_lab.data.processing.mappings import ALPHA_VANTAGE_COLUMN_MAPPER
from quantrl_lab.data.utils import (
    HTTPRequestWrapper,
    RetryStrategy,
    convert_columns_to_numeric,
    format_av_datetime,
    log_dataframe_info,
    normalize_date_range,
    normalize_symbols,
)


class AlphaVantageDataLoader(
    DataSource,
    FundamentalDataCapable,
    HistoricalDataCapable,
    MacroDataCapable,
    NewsDataCapable,
):
    """Alpha Vantage implementation that provides various datasets."""

    SUPPORTED_FEATURES = {"historical_bars", "fundamental_data", "macro_data", "news"}

    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 5
    DEFAULT_RATE_LIMIT_DELAY = 1.2
    NUMERIC_COLUMNS = ["Open", "High", "Low", "Close", "Volume", "Adj_close"]

    def __init__(
        self,
        api_key: str = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        delay: int = DEFAULT_RETRY_DELAY,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    ):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        self.max_retries = max_retries
        self.delay = delay
        self.rate_limit_delay = rate_limit_delay
        self._request_wrapper = HTTPRequestWrapper(
            max_retries=max_retries,
            retry_strategy=RetryStrategy.EXPONENTIAL,
            base_delay=float(delay),
            rate_limit_delay=rate_limit_delay,
            timeout=30.0,
        )

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
        """Alpha Vantage uses HTTP requests - assume connected if no network issues."""
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
        symbols: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from Alpha Vantage.

        Args:
            symbols (str): Stock symbol to fetch data for.
            start (Union[str, datetime], optional): Start date for filtering.
                If None, no start filtering is applied. Defaults to None.
            end (Union[str, datetime], optional): End date for filtering.
                If None, no end filtering is applied. Defaults to None.
            timeframe (str, optional): Time interval - "1d" (daily), or intraday
                ("1min", "5min", "15min", "30min", "60min"). Defaults to "1d".
            **kwargs: Additional parameters. 'adjusted' (bool) enables split/dividend
                adjustment for daily data (premium). 'outputsize' (str) is "compact" or "full"
                (premium). 'month' (str, "YYYY-MM") fetches historical intraday month (premium).

        Returns:
            pd.DataFrame: OHLCV data, optionally filtered by date range.

        Note:
            outputsize='full' and historical intraday 'month' parameter require a premium API key.
            Rate limit: 25 requests/day, 1 request/second burst limit on the free tier.
        """
        adjusted = kwargs.pop("adjusted", False)

        parsed_start_date = None
        parsed_end_date = None

        if start is not None or end is not None:
            if start is not None and end is not None:
                parsed_start_date, parsed_end_date = normalize_date_range(
                    start, end, default_end_to_now=False, validate_order=True
                )
            elif start is not None:
                from quantrl_lab.data.utils import normalize_date

                parsed_start_date = normalize_date(start)
            elif end is not None:
                from quantrl_lab.data.utils import normalize_date

                parsed_end_date = normalize_date(end)

        if parsed_start_date or parsed_end_date:
            if parsed_start_date and parsed_end_date:
                logger.info(
                    "Fetching {timeframe} data for {symbol} from {start} to {end}",
                    timeframe=timeframe,
                    symbol=symbols,
                    start=parsed_start_date.date(),
                    end=parsed_end_date.date(),
                )
            elif parsed_start_date:
                logger.info(
                    "Fetching {timeframe} data for {symbol} from {start} onwards",
                    timeframe=timeframe,
                    symbol=symbols,
                    start=parsed_start_date.date(),
                )
            else:
                logger.info(
                    "Fetching {timeframe} data for {symbol} up to {end}",
                    timeframe=timeframe,
                    symbol=symbols,
                    end=parsed_end_date.date(),
                )
        else:
            logger.info(
                "Fetching {timeframe} data for {symbol} (all available data)",
                timeframe=timeframe,
                symbol=symbols,
            )

        if timeframe == "1d":
            # Default to compact to avoid premium-only "full" output size
            if "outputsize" not in kwargs:
                kwargs["outputsize"] = "compact"
                logger.debug(
                    "Defaulting to outputsize='compact' (last 100 data points). "
                    "Use outputsize='full' with premium API key for 20+ years of data."
                )

            if adjusted:
                raw_data = self._get_daily_adjusted_data(symbols, **kwargs)
                logger.debug(f"Using adjusted daily data for {symbols}")
            else:
                raw_data = self._get_daily_data(symbols, **kwargs)
                logger.debug(f"Using raw daily data for {symbols}")

            time_series_key = "Time Series (Daily)"

        elif timeframe in ["1min", "5min", "15min", "30min", "60min"]:
            if adjusted:
                logger.warning("Adjusted prices not available for intraday data, using raw prices")

            if "month" in kwargs:
                logger.info(f"Fetching {timeframe} intraday data for {symbols} for month: {kwargs['month']}")
            else:
                logger.info(
                    f"Fetching {timeframe} intraday data for {symbols} (recent data - typically last 15-30 days)"
                )
                logger.debug("For historical intraday data, specify 'month=\"YYYY-MM\"' in kwargs")

            raw_data = self._get_intraday_data(symbols, interval=timeframe, **kwargs)
            time_series_key = f"Time Series ({timeframe})"
        else:
            raise InvalidParametersError(
                f"Unsupported timeframe: {timeframe}. Use '1d' or intraday intervals like "
                "'1min', '5min', '15min', '30min', '60min'"
            )

        if not raw_data:
            logger.error(f"Failed to fetch data for {symbols}")
            return pd.DataFrame()

        if time_series_key not in raw_data:
            logger.error(
                f"Expected key '{time_series_key}' not found in API response for {symbols}. "
                "This may be due to rate limits, invalid symbol, or no data available."
            )
            available_keys = list(raw_data.keys())
            logger.debug(f"Available keys in response: {available_keys}")
            return pd.DataFrame()

        time_series = raw_data[time_series_key]

        if not time_series:
            logger.warning(f"Empty time series data for {symbols}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(time_series, orient="index")

        column_mapping = ALPHA_VANTAGE_COLUMN_MAPPER.get_mapping(timeframe, adjusted)
        df = df.rename(columns=column_mapping)

        expected_columns = list(set(column_mapping.values()))
        df = df[df.columns.intersection(expected_columns)]

        numeric_columns = self.NUMERIC_COLUMNS.copy()
        if "Dividend" in df.columns:
            numeric_columns.append("Dividend")
        if "Split_coeff" in df.columns:
            numeric_columns.append("Split_coeff")

        df = convert_columns_to_numeric(df, columns=numeric_columns, errors="coerce")

        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        # Alpha Vantage returns newest first, so sort ascending
        df = df.sort_index()

        if parsed_start_date is not None:
            df = df[df.index >= parsed_start_date]
        if parsed_end_date is not None:
            df = df[df.index <= parsed_end_date]

        if df.empty:
            if parsed_start_date or parsed_end_date:
                logger.warning(
                    "No data found for {symbol} matching the specified date criteria",
                    symbol=symbols,
                )
            else:
                logger.warning("No data found for {symbol}", symbol=symbols)
        else:
            data_type = (
                "adjusted daily"
                if (timeframe == "1d" and adjusted)
                else (f"{timeframe} intraday" if timeframe != "1d" else "daily")
            )
            log_dataframe_info(df, f"Retrieved {data_type} records", symbol=symbols)

        df.reset_index(inplace=True)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        return df

    def get_fundamental_data(
        self, symbol: str, metrics: List[Union[FundamentalMetric, str]], **kwargs: Any
    ) -> Union[pd.DataFrame, Dict]:
        """
        Get fundamental data for a single symbol by combining multiple
        Alpha Vantage API calls.

        Args:
            symbol (str): Stock symbol to fetch data for.
            metrics (List[Union[FundamentalMetric, str]]): List of FundamentalMetric enums or strings.
            **kwargs: Additional parameters. 'return_format' (str) is 'dict' or 'dataframe'.
                Defaults to 'dict'.

        Returns:
            Union[pd.DataFrame, Dict]: Dict with combined fundamental data.
        """
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
                    logger.success(f"Successfully fetched {metric_enum.value} for {symbol}")
                else:
                    logger.warning(f"Failed to fetch {metric_enum.value} for {symbol}")
                    results[metric_enum.value] = None
            else:
                logger.warning(f"Unsupported metric '{metric_enum}' for symbol {symbol}")
                results[metric_enum.value] = None

        return results

    def get_news_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        limit: int = 50,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch news data for given symbols from Alpha Vantage.

        Retrieves news articles related to the specified symbols within the given date range.
        Supports additional parameters like 'sort' and 'topics' to customize the news data.

        Args:
            symbols (Union[str, List[str]]): Symbols to fetch news for.
            start (Union[str, datetime]): Start datetime for news data.
            end (Union[str, datetime], optional): End datetime for news. Defaults to None (current time).
            limit (int, optional): Maximum number of news items to fetch. Defaults to 50.
            **kwargs: Additional parameters for the API request, such as 'sort' or 'topics'.

        Returns:
            pd.DataFrame: DataFrame containing news data for the specified symbols.
        """
        time_from = format_av_datetime(start)

        if end is None:
            end = datetime.now()
        time_to = format_av_datetime(end)

        symbol_list = normalize_symbols(symbols)
        tickers = ",".join(symbol_list)

        logger.info(f"Fetching news for {tickers} from {time_from} to {time_to}")

        params = {
            "tickers": tickers,
            "time_from": time_from,
            "time_to": time_to,
            "limit": str(limit),
        }

        if "sort" in kwargs:
            params["sort"] = kwargs.pop("sort")
            logger.debug(f"Using sort order: {params['sort']}")

        if "topics" in kwargs:
            params["topics"] = kwargs.pop("topics")
            logger.debug(f"Using topics from kwargs: {params['topics']}")

        params.update(kwargs)

        news_data = self._make_api_request("NEWS_SENTIMENT", symbol="", **params)

        news_df = pd.DataFrame(news_data["feed"]) if news_data and "feed" in news_data else pd.DataFrame()

        if news_df.empty:
            logger.warning(f"No news data retrieved for {tickers}. This may be due to no data being available.")
            return news_df

        if "time_published" in news_df.columns:
            news_df.rename(columns={"time_published": "created_at"}, inplace=True)
        else:
            logger.error(f"Expected 'time_published' column not found in news data for {tickers}")
            logger.debug(f"Available columns: {list(news_df.columns)}")
            return pd.DataFrame()

        news_df["created_at"] = pd.to_datetime(news_df["created_at"], format="%Y%m%dT%H%M%S")
        news_df["Date"] = news_df["created_at"].dt.date
        news_df = self._expand_news_rows(news_df, symbol_list)

        logger.success(f"Retrieved {len(news_df)} news items for {tickers}")

        return news_df

    def _expand_news_rows(self, news_df: pd.DataFrame, requested_symbols: List[str]) -> pd.DataFrame:
        """Normalize Alpha Vantage news into one row per article-symbol
        pair."""
        if "ticker_sentiment" not in news_df.columns:
            result = news_df.copy()
            if len(requested_symbols) == 1:
                result["Symbol"] = requested_symbols[0]
            return result

        expanded_rows: List[pd.Series] = []
        for _, row in news_df.iterrows():
            sentiment_map = self._extract_ticker_sentiments(row.get("ticker_sentiment"), requested_symbols)
            if sentiment_map:
                for symbol, score in sentiment_map.items():
                    row_copy = row.copy()
                    row_copy["Symbol"] = symbol
                    row_copy["sentiment_score"] = score
                    expanded_rows.append(row_copy)
            elif len(requested_symbols) == 1:
                row_copy = row.copy()
                row_copy["Symbol"] = requested_symbols[0]
                row_copy["sentiment_score"] = None
                expanded_rows.append(row_copy)

        if not expanded_rows:
            return pd.DataFrame(columns=list(news_df.columns) + ["Symbol", "sentiment_score"])

        return pd.DataFrame(expanded_rows).reset_index(drop=True)

    @staticmethod
    def _extract_ticker_sentiments(
        sentiment_list: List[Dict[str, Any]],
        requested_symbols: List[str],
    ) -> Dict[str, Optional[float]]:
        """Extract per-symbol sentiment scores for the requested
        tickers."""
        if not isinstance(sentiment_list, list):
            return {}

        requested = set(requested_symbols)
        matches: Dict[str, Optional[float]] = {}
        for item in sentiment_list:
            ticker = item.get("ticker")
            if ticker not in requested:
                continue

            score = item.get("ticker_sentiment_score")
            try:
                matches[ticker] = float(score) if score is not None else None
            except (TypeError, ValueError):
                matches[ticker] = None

        return matches

    def get_macro_data(
        self,
        indicators: Union[str, List[str], Dict[str, Dict]],
        start: Union[str, datetime],
        end: Union[str, datetime],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Get macroeconomic data for specified indicators.

        Supports both standard indicator names and advanced dictionary format where each
        indicator can have its own parameters, e.g.:
        {"real_gdp": {"interval": "quarterly"}, "treasury_yield": {"interval": "monthly", "maturity": "10year"}}

        Args:
            indicators (Union[str, List[str], Dict[str, Dict]]): Indicator(s) to fetch data for.
            start (Union[str, datetime]): Start date.
            end (Union[str, datetime]): End date.
            **kwargs: Additional parameters for the API request.

        Returns:
            Dict[str, Any]: Dictionary containing macroeconomic data for the specified indicators.
                Each key is the indicator name, and the value is the fetched data.
        """
        if isinstance(indicators, dict):
            return self._get_macro_data_with_params(indicators, **kwargs)
        else:
            if isinstance(indicators, (str, MacroIndicator)):
                indicators = [indicators]

            indicator_params = {ind: {} for ind in indicators}
            return self._get_macro_data_with_params(indicator_params, **kwargs)

    def _get_macro_data_with_params(
        self, indicator_params: Dict[Union[str, MacroIndicator], Dict], **global_kwargs
    ) -> Dict[str, Any]:
        """
        Fetch macro data for multiple indicators with per-indicator
        parameter overrides.

        Args:
            indicator_params (Dict[Union[str, MacroIndicator], Dict]): Mapping from indicator
                to its specific keyword arguments.
            **global_kwargs: Global parameters applied to all indicators unless overridden.

        Returns:
            Dict[str, Any]: Mapping from indicator name to fetched data (or None on failure).
        """
        results = {}

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
                        logger.success(f"Successfully fetched {indicator_enum.value} data")
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
            indicator (MacroIndicator): Enum to filter kwargs for.
            kwargs (Dict): Additional parameters for the API request.

        Returns:
            Dict: Filtered kwargs for the specific indicator method.

        Raises:
            ValueError: If the interval or maturity parameters are invalid.
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

        if "interval" in config.get("params", []):
            interval = kwargs.get("interval", config.get("default_interval"))
            valid_intervals = config.get("valid_intervals", [])

            if interval and valid_intervals and interval not in valid_intervals:
                raise InvalidParametersError(
                    f"Invalid interval '{interval}' for {indicator.value}. Valid options: {valid_intervals}"
                )

            if interval:
                filtered_kwargs["interval"] = interval

        if "maturity" in config.get("params", []):
            maturity = kwargs.get("maturity", config.get("default_maturity"))
            valid_maturities = config.get("valid_maturities", [])

            if maturity and valid_maturities and maturity not in valid_maturities:
                raise InvalidParametersError(
                    f"Invalid maturity '{maturity}' for {indicator.value}. Valid options: {valid_maturities}"
                )

            if maturity:
                filtered_kwargs["maturity"] = maturity

        for key, value in kwargs.items():
            if key not in ["interval", "maturity"] and key not in filtered_kwargs:
                filtered_kwargs[key] = value

        return filtered_kwargs

    def _make_api_request(self, function: str, symbol: str = "", **params) -> Optional[Dict[str, Any]]:
        """
        Centralized private method for making Alpha Vantage API
        requests.

        Args:
            function (str): Alpha Vantage function name (e.g., 'TIME_SERIES_DAILY').
            symbol (str, optional): Stock symbol. Omitted for macro/news endpoints. Defaults to "".
            **params: Additional query parameters for the API request.

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON response, or None if all retries are exhausted.
        """
        if not self.api_key:
            raise AuthenticationError("Alpha Vantage API key not provided")

        url_params = {
            "function": function,
            "apikey": self.api_key,
            **params,
        }

        if symbol:
            url_params["symbol"] = symbol

        data = self._request_wrapper.make_request(
            ALPHA_VANTAGE_API_BASE,
            params=url_params,
            raise_on_error=True,
        )

        if not isinstance(data, dict):
            raise APIConnectionError(f"Unexpected response while fetching {function} data")

        if "Error Message" in data:
            error_msg = f"API Error: {data['Error Message']}"
            if symbol:
                error_msg += f" for {symbol}"
            raise InvalidParametersError(error_msg)

        if "Note" in data and "api call frequency" in data.get("Note", "").lower():
            error_msg = data["Note"]
            if symbol:
                error_msg = f"{error_msg} for {symbol}"
            raise RateLimitError(error_msg)

        if "Information" in data:
            info_message = data["Information"]
            lowered = info_message.lower()
            if any(
                indicator in lowered
                for indicator in [
                    "request per second",
                    "requests per day",
                    "rate limit",
                    "api call frequency",
                    "thank you for using alpha vantage",
                ]
            ):
                raise RateLimitError(info_message)
            if "api key" in lowered or "unauthorized" in lowered:
                raise AuthenticationError(info_message)
            raise APIConnectionError(info_message)

        success_msg = f"Successfully fetched {function} data"
        if symbol:
            success_msg += f" for {symbol}"
        logger.info(success_msg)
        return data

    def _get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company overview data from Alpha Vantage including key
        statistics and company info.

        Args:
            symbol (str): Stock symbol to fetch data for.

        Returns:
            Optional[Dict[str, Any]]: Data in dictionary format, or None if request fails.
        """
        return self._make_api_request("OVERVIEW", symbol)

    def _get_etf_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch ETF profile data from Alpha Vantage including ETF holdings
        and other profile info.

        Note: ETF profile is not typically used in our context, but included for completeness.

        Args:
            symbol (str): ETF symbol to fetch data for.

        Returns:
            Optional[Dict[str, Any]]: Data in dictionary format, or None if request fails.
        """
        return self._make_api_request("ETF_PROFILE", symbol)

    def _get_dividend_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch dividend payment history from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch dividend data for.

        Returns:
            Optional[Dict[str, Any]]: Data in dictionary format, or None if request fails.
        """
        return self._make_api_request("DIVIDENDS", symbol)

    def _get_splits_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch stock split history from Alpha Vantage.

        Note: not useful in our context, but included for completeness.

        Args:
            symbol (str): Stock symbol to fetch split data for.

        Returns:
            Optional[Dict[str, Any]]: Data in dictionary format, or None if request fails.
        """
        return self._make_api_request("SPLITS", symbol)

    def _get_income_statement_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch income statement data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch income statement for.

        Returns:
            Optional[Dict[str, Any]]: Income statement data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("INCOME_STATEMENT", symbol)

    def _get_balance_sheet_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch balance sheet data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch balance sheet for.

        Returns:
            Optional[Dict[str, Any]]: Balance sheet data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("BALANCE_SHEET", symbol)

    def _get_cash_flow_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch cash flow statement data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch cash flow statement for.

        Returns:
            Optional[Dict[str, Any]]: Cash flow statement data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("CASH_FLOW", symbol)

    def _get_earnings_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch earnings data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch earnings data for.

        Returns:
            Optional[Dict[str, Any]]: Earnings data in dictionary format, or None if request fails.
        """
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
            symbol (str): Stock symbol to fetch data for.
            interval (str, optional): Time interval (1min, 5min, 15min, 30min, 60min).
                Defaults to "5min".
            outputsize (str, optional): 'compact' or 'full'. Defaults to "full".
            month (str, optional): Month in YYYY-MM format for historical intraday data.
                Defaults to None.
            **kwargs: Additional Alpha Vantage API parameters.

        Returns:
            Optional[Dict[str, Any]]: Intraday data, or None if request fails.
        """
        params = {
            "interval": interval,
            "outputsize": outputsize,
        }

        if month:
            params["month"] = month

        params.update(kwargs)

        return self._make_api_request("TIME_SERIES_INTRADAY", symbol, **params)

    def _get_daily_data(self, symbol: str, outputsize: str = "full", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch daily time series data from Alpha Vantage.

        Args:
            symbol (str): Stock symbol to fetch data for.
            outputsize (str, optional): 'compact' (last 100 days) or 'full' (20+ years of data).
                Defaults to "full".
            **kwargs: Additional Alpha Vantage API parameters.

        Returns:
            Optional[Dict[str, Any]]: Daily OHLCV data, or None if request fails.
        """
        params = {"outputsize": outputsize}
        params.update(kwargs)

        return self._make_api_request("TIME_SERIES_DAILY", symbol, **params)

    def _get_daily_adjusted_data(self, symbol: str, outputsize: str = "full", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch daily adjusted time series data from Alpha Vantage.

        Includes dividend and split adjustments. Requires a premium API key.

        Args:
            symbol (str): Stock symbol to fetch data for.
            outputsize (str, optional): 'compact' (last 100 days) or 'full' (20+ years of data).
                Defaults to "full".
            **kwargs: Additional Alpha Vantage API parameters.

        Returns:
            Optional[Dict[str, Any]]: Daily adjusted OHLCV data, or None if request fails.
        """
        params = {"outputsize": outputsize}
        params.update(kwargs)

        return self._make_api_request("TIME_SERIES_DAILY_ADJUSTED", symbol, **params)

    def _get_real_gdp_data(self, interval: str = "annual", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch real GDP data from Alpha Vantage.

        Args:
            interval (str, optional): Data frequency. Available options are "quarterly" and
                "annual". Defaults to "annual".
            **kwargs: Additional parameters for the API request.

        Returns:
            Optional[Dict[str, Any]]: Real GDP data in dictionary format, or None if request fails.

        Raises:
            ValueError: If the interval is not one of the valid options.
        """
        if interval not in ["quarterly", "annual"]:
            raise InvalidParametersError(f"Invalid interval '{interval}'. Use 'quarterly' or 'annual'.")

        params = {"interval": interval}
        params.update(kwargs)

        return self._make_api_request("REAL_GDP", symbol="", **params)

    def _get_real_gdp_per_capita_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch real GDP per capita data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Real GDP per capita data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("REAL_GDP_PER_CAPITA", symbol="", **kwargs)

    def _get_treasury_yield_data(
        self, interval: str = "monthly", maturity: str = "10year", **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch treasury yield data from Alpha Vantage.

        Args:
            interval (str, optional): Data frequency. Defaults to "monthly".
            maturity (str, optional): Bond maturity (e.g., "10year"). Defaults to "10year".
            **kwargs: Additional parameters for the API request.

        Returns:
            Optional[Dict[str, Any]]: Treasury yield data in dictionary format,
                or None if request fails.

        Raises:
            ValueError: If the interval or maturity parameters are invalid.
        """
        valid_intervals = ["daily", "weekly", "monthly"]
        if interval not in valid_intervals:
            raise InvalidParametersError(f"Invalid interval '{interval}'. Use one of: {valid_intervals}")

        valid_maturities = ["3month", "2year", "5year", "7year", "10year", "30year"]
        if maturity not in valid_maturities:
            raise InvalidParametersError(f"Invalid maturity '{maturity}'. Use one of: {valid_maturities}")

        params = {"interval": interval, "maturity": maturity}
        params.update(kwargs)

        return self._make_api_request("TREASURY_YIELD", symbol="", **params)

    def _get_federal_funds_rate_data(self, interval: str = "monthly", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch federal funds rate data from Alpha Vantage.

        Args:
            interval (str, optional): Data frequency. Defaults to "monthly".
            **kwargs: Additional parameters for the API request.

        Returns:
            Optional[Dict[str, Any]]: Federal funds rate data in dictionary format,
                or None if request fails.

        Raises:
            ValueError: If the interval is not one of the valid options.
        """
        valid_intervals = ["daily", "weekly", "monthly"]
        if interval not in valid_intervals:
            raise InvalidParametersError(f"Invalid interval '{interval}'. Use one of: {valid_intervals}")

        params = {"interval": interval}
        params.update(kwargs)

        return self._make_api_request("FEDERAL_FUNDS_RATE", symbol="", **params)

    def _get_cpi_data(self, interval: str = "monthly", **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch Consumer Price Index (CPI) data from Alpha Vantage.

        Args:
            interval (str, optional): Data frequency. Defaults to "monthly".
            **kwargs: Additional parameters for the API request.

        Returns:
            Optional[Dict[str, Any]]: CPI data in dictionary format, or None if request fails.

        Raises:
            ValueError: If the interval is not one of the valid options.
        """
        valid_intervals = ["semiannual", "monthly"]
        if interval not in valid_intervals:
            raise InvalidParametersError(f"Invalid interval '{interval}'. Use one of: {valid_intervals}")

        params = {"interval": interval}
        params.update(kwargs)

        return self._make_api_request("CPI", symbol="", **params)

    def _get_inflation_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch inflation data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Inflation data in dictionary format, or None if request fails.
        """
        return self._make_api_request("INFLATION", symbol="", **kwargs)

    def _get_retail_sales_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch retail sales data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Retail sales data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("RETAIL_SALES", symbol="", **kwargs)

    def _get_durable_goods_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch durable goods data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Durable goods data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("DURABLE_GOODS", symbol="", **kwargs)

    def _get_unemployment_rate_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch unemployment rate data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Unemployment rate data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("UNEMPLOYMENT", symbol="", **kwargs)

    def _get_non_farm_payroll_data(self, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Fetch non-farm payroll data from Alpha Vantage.

        Returns:
            Optional[Dict[str, Any]]: Non-farm payroll data in dictionary format,
                or None if request fails.
        """
        return self._make_api_request("NONFARM_PAYROLL", symbol="", **kwargs)
