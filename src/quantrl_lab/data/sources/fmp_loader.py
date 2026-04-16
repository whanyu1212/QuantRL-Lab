import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import aiohttp

import pandas as pd
from loguru import logger

from quantrl_lab.data.exceptions import AuthenticationError, InvalidParametersError
from quantrl_lab.data.interface import (
    AnalystDataCapable,
    CompanyProfileCapable,
    DataSource,
    HistoricalDataCapable,
    SectorDataCapable,
)
from quantrl_lab.data.utils import (
    AsyncHTTPRequestWrapper,
    HTTPRequestWrapper,
    RetryStrategy,
    convert_to_dataframe_safe,
    format_date_to_string,
    get_single_symbol,
    log_dataframe_info,
    normalize_date_range,
    normalize_symbols,
    standardize_ohlcv_dataframe,
    validate_symbols,
)


class FMPDataSource(
    DataSource,
    HistoricalDataCapable,
    AnalystDataCapable,
    SectorDataCapable,
    CompanyProfileCapable,
):
    """
    Financial Modeling Prep data source for historical stock data and
    analyst insights.

    Supports both end-of-day (daily) and intraday data.
    Intraday timeframes: 5min, 15min, 30min, 1hour, 4hour
    Daily timeframe: 1d

    Implements the following protocols:
    - HistoricalDataCapable: OHLCV data (daily and intraday)
    - AnalystDataCapable: Analyst grades and ratings data
    - SectorDataCapable: Historical sector and industry performance data
    - CompanyProfileCapable: Company profile and metadata
    """

    SUPPORTED_FEATURES = {"historical_bars", "analyst_data", "sector_data", "company_profile"}

    BASE_URL = "https://financialmodelingprep.com/stable"
    RATE_LIMIT_SLEEP = 1  # seconds
    INTRADAY_TIMEFRAMES = {"5min", "15min", "30min", "1hour", "4hour"}

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP data source.

        Args:
            api_key (str, optional): FMP API key. If not provided, will try to read from
                FMP_API_KEY environment variable.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise AuthenticationError("FMP API key must be provided or set in FMP_API_KEY environment variable")

        self._request_wrapper = HTTPRequestWrapper(
            max_retries=3,
            retry_strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            rate_limit_delay=self.RATE_LIMIT_SLEEP,
            timeout=30.0,
        )

    @property
    def source_name(self) -> str:
        return "FinancialModelingPrep"

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Any:
        """
        Make an HTTP request to the FMP API with retry logic.

        Args:
            endpoint (str): API endpoint path.
            params (Dict[str, Any]): Query parameters.

        Returns:
            Any: JSON response data.

        Raises:
            requests.HTTPError: If the request fails after retries.
        """
        params["apikey"] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"

        return self._request_wrapper.make_request(
            url=url,
            method="GET",
            params=params,
            raise_on_error=True,
        )

    def _get_intraday_data(
        self,
        symbol: str,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]],
        timeframe: str,
        nonadjusted: bool = False,
    ) -> pd.DataFrame:
        """
        Get intraday OHLCV data from FMP historical-chart endpoint.

        Args:
            symbol (str): Stock symbol to fetch data for.
            start (Union[str, datetime]): Start date for historical data.
            end (Union[str, datetime], optional): End date for historical data.
            timeframe (str): Intraday timeframe (5min, 15min, 30min, 1hour, 4hour).
            nonadjusted (bool, optional): If True, returns unadjusted prices. Defaults to False.

        Returns:
            pd.DataFrame: Intraday OHLCV data with standardized column names.
        """
        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        start_str = format_date_to_string(start_dt)
        end_str = format_date_to_string(end_dt)

        logger.info(
            "Fetching {timeframe} intraday data for {symbol} from {start} to {end}",
            timeframe=timeframe,
            symbol=symbol,
            start=start_str,
            end=end_str,
        )

        endpoint = f"historical-chart/{timeframe}"
        params = {
            "symbol": symbol,
            "from": start_str,
            "to": end_str,
            "nonadjusted": str(nonadjusted).lower(),
        }

        data = self._make_request(endpoint, params)

        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=symbol)
        if df.empty:
            return df

        column_mapping = {
            "date": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        df = standardize_ohlcv_dataframe(
            df,
            column_mapping=column_mapping,
            symbol=symbol,
            timestamp_col="Timestamp",
            add_date=True,
            sort_data=True,
            convert_numeric=True,
        )

        log_dataframe_info(df, f"Fetched {timeframe} intraday data", symbol=symbol)
        return df

    def get_historical_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from FMP (daily or intraday).

        Args:
            symbols (Union[str, List[str]]): Stock symbol(s) to fetch data for.
            start (Union[str, datetime], optional): Start date for historical data.
            end (Union[str, datetime], optional): End date for historical data.
            timeframe (str, optional): Timeframe - "1d" for daily, or intraday:
                "5min", "15min", "30min", "1hour", "4hour". Defaults to "1d".
            **kwargs: Additional arguments including 'nonadjusted' (bool) for intraday data.

        Returns:
            pd.DataFrame: OHLCV data with standardized column names.

        Raises:
            ValueError: If timeframe is not supported.
        """
        if start is None:
            raise InvalidParametersError("FMP requires a 'start' date for historical data.")

        # FMP only supports single symbols
        symbol = get_single_symbol(symbols, warn_on_multiple=True)

        if timeframe in self.INTRADAY_TIMEFRAMES:
            nonadjusted = kwargs.get("nonadjusted", False)
            return self._get_intraday_data(symbol, start, end, timeframe, nonadjusted)

        if timeframe != "1d":
            logger.warning(f"Timeframe {timeframe} not supported by FMP. Using daily (1d) data.")

        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        start_str = format_date_to_string(start_dt)
        end_str = format_date_to_string(end_dt)

        logger.info(
            "Fetching EOD data for {symbol} from {start} to {end}",
            symbol=symbol,
            start=start_str,
            end=end_str,
        )

        endpoint = "historical-price-eod/full"
        params = {
            "symbol": symbol,
            "from": start_str,
            "to": end_str,
        }

        data = self._make_request(endpoint, params)

        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=symbol)
        if df.empty:
            return df

        column_mapping = {
            "date": "Timestamp",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

        df = standardize_ohlcv_dataframe(
            df,
            column_mapping=column_mapping,
            symbol=symbol,
            timestamp_col="Timestamp",
            add_date=True,
            sort_data=True,
            convert_numeric=True,
        )

        log_dataframe_info(df, "Fetched EOD data", symbol=symbol)
        return df

    def get_historical_grades(self, symbol: str) -> pd.DataFrame:
        """
        Get historical analyst grades for a symbol.

        Args:
            symbol (str): Stock symbol to fetch data for.

        Returns:
            pd.DataFrame: Historical grades data.
        """
        endpoint = "grades-historical"
        params = {"symbol": symbol}

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No historical grades found for symbol: {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(f"Empty grades dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)

        logger.success(
            "Fetched {n} historical grades for {symbol}",
            n=len(df),
            symbol=symbol,
        )

        return df

    def get_historical_rating(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Get historical ratings for a symbol.

        Args:
            symbol (str): Stock symbol to fetch data for.
            limit (int, optional): Number of records to return. Defaults to 100.

        Returns:
            pd.DataFrame: Historical ratings data.
        """
        endpoint = "ratings-historical"
        params = {"symbol": symbol, "limit": limit}

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No historical ratings found for symbol: {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        if df.empty:
            logger.warning(f"Empty ratings dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)

        logger.success(
            "Fetched {n} historical ratings for {symbol}",
            n=len(df),
            symbol=symbol,
        )

        return df

    def get_historical_sector_performance(self, sector: str, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Get historical performance data for a specific market sector.

        This endpoint provides historical performance metrics for market sectors,
        allowing analysis of sector trends and performance over time.

        Args:
            sector (str): Market sector name (e.g., "Energy", "Technology", "Healthcare",
                "Financials", "Consumer Cyclical", "Industrials", "Basic Materials",
                "Consumer Defensive", "Real Estate", "Utilities", "Communication Services").
            start (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to API default.
            end (str, optional): End date in 'YYYY-MM-DD' format. Defaults to API default.

        Returns:
            pd.DataFrame: Historical sector performance data with columns including date,
                sector, and performance metrics.

        Raises:
            ValueError: If sector is invalid or API request fails.

        Example:
            >>> source = FMPDataSource()
            >>> df = source.get_historical_sector_performance("Energy", start="2024-01-01", end="2024-12-31")
            >>> print(df.head())
        """
        if not sector or not isinstance(sector, str):
            raise InvalidParametersError("Sector must be a non-empty string")

        logger.info("Fetching historical performance for sector: {sector}", sector=sector)

        endpoint = "historical-sector-performance"
        params = {"sector": sector}

        if start:
            params["from"] = start
        if end:
            params["to"] = end

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No historical sector performance data found for sector: {sector}")
            return pd.DataFrame()

        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=sector)

        if df.empty:
            logger.warning(f"Empty sector performance dataset returned for sector: {sector}")
            return pd.DataFrame()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)

        log_dataframe_info(df, "Fetched sector performance", symbol=sector)

        logger.success(
            "Fetched {n} records of historical sector performance for {sector}",
            n=len(df),
            sector=sector,
        )

        return df

    def get_historical_industry_performance(self, industry: str, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Get historical performance data for a specific industry.

        This endpoint provides historical performance metrics for industries,
        enabling long-term trend analysis and industry evolution tracking.

        Args:
            industry (str): Industry name (e.g., "Biotechnology", "Software", "Banks",
                "Oil & Gas", "Semiconductors", "Insurance", "Auto Manufacturers",
                "Pharmaceuticals", "Consumer Electronics", "Aerospace & Defense").
            start (str, optional): Start date in 'YYYY-MM-DD' format. Defaults to API default.
            end (str, optional): End date in 'YYYY-MM-DD' format. Defaults to API default.

        Returns:
            pd.DataFrame: Historical industry performance data with columns including date,
                industry, and performance metrics.

        Raises:
            ValueError: If industry is invalid or API request fails.

        Example:
            >>> source = FMPDataSource()
            >>> df = source.get_historical_industry_performance("Biotechnology", start="2024-01-01", end="2024-12-31")
            >>> print(df.head())
        """
        if not industry or not isinstance(industry, str):
            raise InvalidParametersError("Industry must be a non-empty string")

        logger.info("Fetching historical performance for industry: {industry}", industry=industry)

        endpoint = "historical-industry-performance"
        params = {"industry": industry}

        if start:
            params["from"] = start
        if end:
            params["to"] = end

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No historical industry performance data found for industry: {industry}")
            return pd.DataFrame()

        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=industry)

        if df.empty:
            logger.warning(f"Empty industry performance dataset returned for industry: {industry}")
            return pd.DataFrame()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)

        log_dataframe_info(df, "Fetched industry performance", symbol=industry)

        logger.success(
            "Fetched {n} records of historical industry performance for {industry}",
            n=len(df),
            industry=industry,
        )

        return df

    def get_company_profile(self, symbol: Union[str, List[str]]) -> pd.DataFrame:
        """
        Get company profile information including sector, industry, and
        key metrics.

        This endpoint provides comprehensive company information including business
        description, sector/industry classification, executive information, and
        key financial metrics.

        Args:
            symbol (Union[str, List[str]]): Stock ticker symbol (e.g., "AAPL", "MSFT") or
                list of symbols (only first symbol will be used if list is provided).

        Returns:
            pd.DataFrame: Company profile data with columns including symbol, companyName,
                sector, industry, description, ceo, website, exchange, mktCap, price, beta,
                volAvg, currency, ipoDate, address, fullTimeEmployees, and asset type flags.

        Raises:
            ValueError: If symbol is invalid or API request fails.

        Example:
            >>> source = FMPDataSource()
            >>> profile = source.get_company_profile("AAPL")
            >>> print(f"Sector: {profile.iloc[0].get('sector')}")
            >>> print(f"Industry: {profile.iloc[0].get('industry')}")
            >>> print(f"CEO: {profile.iloc[0].get('ceo')}")

        Use Cases:
            - Get sector/industry classification for stocks
            - Screen stocks by sector or industry
            - Retrieve company metadata for analysis
            - Build company information datasets
        """
        symbols = normalize_symbols(symbol)
        validate_symbols(symbols)
        symbol = get_single_symbol(symbols)

        if not symbol or not isinstance(symbol, str):
            raise InvalidParametersError("Symbol must be a non-empty string")

        logger.info("Fetching company profile for: {symbol}", symbol=symbol)

        endpoint = "profile"
        params = {"symbol": symbol}

        data = self._make_request(endpoint, params)

        if not data or not isinstance(data, list):
            logger.warning(f"No company profile data found for symbol: {symbol}")
            return pd.DataFrame()

        df = convert_to_dataframe_safe(data, expected_min_rows=0, symbol=symbol)

        if df.empty:
            logger.warning(f"Empty company profile dataset returned for symbol: {symbol}")
            return pd.DataFrame()

        if not df.empty:
            company_name = df.iloc[0].get("companyName", "Unknown")
            sector = df.iloc[0].get("sector", "N/A")
            industry = df.iloc[0].get("industry", "N/A")

            logger.success(
                "Fetched company profile for {symbol}: {name} ({sector} - {industry})",
                symbol=symbol,
                name=company_name,
                sector=sector,
                industry=industry,
            )

        return df

    def _get_async_wrapper(self) -> AsyncHTTPRequestWrapper:
        """Return a shared async wrapper configured for FMP rate
        limits."""
        return AsyncHTTPRequestWrapper(
            max_retries=3,
            base_delay=1.0,
            concurrency=5,  # Conservative for FMP free tier
            timeout=30.0,
        )

    async def _async_request(
        self,
        session: "aiohttp.ClientSession",
        endpoint: str,
        params: Dict[str, Any],
        wrapper: AsyncHTTPRequestWrapper,
    ) -> Any:
        """
        Async equivalent of _make_request().

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session.
            endpoint (str): API endpoint path.
            params (Dict[str, Any]): Query parameters (apikey is added automatically).
            wrapper (AsyncHTTPRequestWrapper): Configured async request wrapper.

        Returns:
            Any: JSON response data.
        """
        params = {**params, "apikey": self.api_key}
        url = f"{self.BASE_URL}/{endpoint}"
        return await wrapper.make_request(session, url, params=params)

    async def async_fetch_ohlcv(
        self,
        session: "aiohttp.ClientSession",
        symbol: str,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
    ) -> Tuple[str, pd.DataFrame]:
        """
        Async fetch of EOD OHLCV data for a single symbol.

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session.
            symbol (str): Stock symbol to fetch.
            start (Union[str, datetime]): Start date for historical data.
            end (Union[str, datetime], optional): End date for historical data.
            timeframe (str, optional): Timeframe string. Defaults to "1d".

        Returns:
            Tuple[str, pd.DataFrame]: Tuple of (symbol, df).
        """
        wrapper = self._get_async_wrapper()
        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        params = {
            "symbol": symbol,
            "from": format_date_to_string(start_dt),
            "to": format_date_to_string(end_dt),
        }
        data = await self._async_request(session, "historical-price-eod/full", params, wrapper)
        df = convert_to_dataframe_safe(data or [], symbol=symbol)
        if not df.empty:
            column_mapping = {
                "date": "Timestamp",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            df = standardize_ohlcv_dataframe(
                df,
                column_mapping=column_mapping,
                symbol=symbol,
                timestamp_col="Timestamp",
                add_date=True,
                sort_data=True,
                convert_numeric=True,
            )
        return symbol, df

    async def async_fetch_ratings(
        self,
        session: "aiohttp.ClientSession",
        symbol: str,
        limit: int = 500,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Async fetch of historical analyst ratings.

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session.
            symbol (str): Stock symbol to fetch.
            limit (int, optional): Maximum number of records to return. Defaults to 500.

        Returns:
            Tuple[str, pd.DataFrame]: Tuple of (symbol, df).
        """
        wrapper = self._get_async_wrapper()
        data = await self._async_request(session, "ratings-historical", {"symbol": symbol, "limit": limit}, wrapper)
        if not data or not isinstance(data, list):
            return symbol, pd.DataFrame()
        df = pd.DataFrame(data)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
        return symbol, df

    async def async_fetch_company_profile(
        self,
        session: "aiohttp.ClientSession",
        symbol: str,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Async fetch of company profile (sector, industry).

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session.
            symbol (str): Stock symbol to fetch.

        Returns:
            Tuple[str, pd.DataFrame]: Tuple of (symbol, df).
        """
        wrapper = self._get_async_wrapper()
        data = await self._async_request(session, "profile", {"symbol": symbol}, wrapper)
        df = convert_to_dataframe_safe(data or [], symbol=symbol)
        return symbol, df

    async def async_fetch_sector_perf(
        self,
        session: "aiohttp.ClientSession",
        sector: str,
        start: str,
        end: str,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Async fetch of historical sector performance.

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session.
            sector (str): Market sector name.
            start (str): Start date in 'YYYY-MM-DD' format.
            end (str): End date in 'YYYY-MM-DD' format.

        Returns:
            Tuple[str, pd.DataFrame]: Tuple of (sector, df).
        """
        wrapper = self._get_async_wrapper()
        params = {"sector": sector, "from": start, "to": end}
        data = await self._async_request(session, "historical-sector-performance", params, wrapper)
        df = convert_to_dataframe_safe(data or [], symbol=sector)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
        return sector, df

    async def async_fetch_industry_perf(
        self,
        session: "aiohttp.ClientSession",
        industry: str,
        start: str,
        end: str,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Async fetch of historical industry performance.

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session.
            industry (str): Industry name.
            start (str): Start date in 'YYYY-MM-DD' format.
            end (str): End date in 'YYYY-MM-DD' format.

        Returns:
            Tuple[str, pd.DataFrame]: Tuple of (industry, df).
        """
        wrapper = self._get_async_wrapper()
        params = {"industry": industry, "from": start, "to": end}
        data = await self._async_request(session, "historical-industry-performance", params, wrapper)
        df = convert_to_dataframe_safe(data or [], symbol=industry)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
        return industry, df
