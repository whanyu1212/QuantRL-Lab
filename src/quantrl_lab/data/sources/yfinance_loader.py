import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple, Union

import pandas as pd
import yfinance as yf
from loguru import logger

from quantrl_lab.data.config import (
    YFinanceInterval,
    financial_columns,
)
from quantrl_lab.data.exceptions import APIConnectionError, InvalidParametersError
from quantrl_lab.data.interface import (
    DataSource,
    FundamentalDataCapable,
    HistoricalDataCapable,
)
from quantrl_lab.data.utils import log_dataframe_info, normalize_date_range, normalize_symbols


class YFinanceDataLoader(DataSource, FundamentalDataCapable, HistoricalDataCapable):
    """Yahoo Finance implementation that provides market data and
    fundamental data."""

    SUPPORTED_FEATURES = {"historical_bars", "fundamental_data"}

    def __init__(
        self,
        max_retries: int = 3,
        delay: int = 1,
    ):
        # Do not initialize ticker-related variables here to keep the object reusable
        self.max_retries = max_retries
        self.delay = delay

    @property
    def source_name(self) -> str:
        return "Yahoo Finance"

    def connect(self):
        """yfinance doesn't require explicit connection - it uses HTTP requests."""
        pass

    def disconnect(self):
        """yfinance doesn't require explicit connection - it uses HTTP requests."""
        pass

    def is_connected(self) -> bool:
        """yfinance uses HTTP requests - assume connected if no network issues."""
        return True

    def list_available_instruments(
        self,
        instrument_type: Optional[str] = None,
        market: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        logger.warning("Yahoo Finance does not support listing available instruments.")
        return []

    def get_fundamental_data(
        self,
        symbol: str,
        frequency: str = "quarterly",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get all fundamental data for a symbol including income
        statement, cash flow, and balance sheet.

        Args:
            symbol (str): Stock symbol; only a single symbol is supported.
            frequency (str, optional): Frequency of data. Defaults to "quarterly".
            **kwargs: Additional yfinance parameters.

        Returns:
            pd.DataFrame: DataFrame with raw fundamental data.
        """
        income_statement = self._get_income_statement(symbol, frequency=frequency)
        cash_flow = self._get_cash_flow(symbol, frequency=frequency)
        balance_sheet = self._get_balance_sheet(symbol, frequency=frequency)

        df = income_statement.merge(cash_flow, on="Date", how="outer")
        df = df.merge(balance_sheet, on="Date", how="outer")

        df["Symbol"] = symbol

        essential_columns = [
            "Date",
            "Symbol",
        ] + financial_columns.get_all_statement_columns()
        available_columns = [col for col in essential_columns if col in df.columns]

        return df[available_columns]

    def _get_income_statement(self, symbol: str, frequency: str = "quarterly") -> pd.DataFrame:
        """
        Get income statement for a symbol.

        Args:
            symbol (str): Stock symbol; only a single symbol is supported.
            frequency (str, optional): Frequency of data. Defaults to "quarterly".

        Returns:
            pd.DataFrame: DataFrame with raw income statement data.
        """
        logger.info("Fetching income statement for {symbol}", symbol=symbol)
        ticker = yf.Ticker(symbol)
        df = ticker.get_income_stmt(freq=frequency).T.reset_index(names="Date")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def _get_cash_flow(self, symbol: str, frequency: str = "quarterly") -> pd.DataFrame:
        """
        Get cash flow statement for a symbol.

        Args:
            symbol (str): Stock symbol; only a single symbol is supported.
            frequency (str, optional): Frequency of data. Defaults to "quarterly".

        Returns:
            pd.DataFrame: DataFrame with raw cash flow data.
        """
        logger.info("Fetching cash flow statement for {symbol}", symbol=symbol)
        ticker = yf.Ticker(symbol)
        df = ticker.get_cashflow(freq=frequency).T.reset_index(names="Date")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def _get_balance_sheet(self, symbol: str, frequency: str = "quarterly") -> pd.DataFrame:
        """
        Get balance sheet for a symbol.

        Args:
            symbol (str): Stock symbol; only a single symbol is supported.
            frequency (str, optional): Frequency of data. Defaults to "quarterly".

        Returns:
            pd.DataFrame: DataFrame with raw balance sheet data.
        """
        logger.info("Fetching balance sheet for {symbol}", symbol=symbol)
        ticker = yf.Ticker(symbol)
        df = ticker.get_balance_sheet(freq=frequency).T.reset_index(names="Date")
        df["Date"] = pd.to_datetime(df["Date"])
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
        Get historical OHLCV data for a list of symbols.

        Args:
            symbols (Union[str, List[str]]): A single symbol or a list of symbols.
            start (Union[str, datetime], optional): Start date or datetime.
            end (Union[str, datetime], optional): End date or datetime.
            timeframe (str, optional): Bar interval. Defaults to "1d".
            **kwargs: Additional yfinance parameters, including 'period' (e.g., '1y', 'max').

        Returns:
            pd.DataFrame: Output dataframe with OHLCV data (raw).

        Raises:
            ValueError: If all elements in 'symbols' are not strings.
            TypeError: If 'symbols' is not a string or list of strings.
            ValueError: If interval is invalid.
            ValueError: If start or end date is invalid.
            ValueError: If start date is not before end date.
            ValueError: If 1 min interval start date is not within 30 days from today.
        """
        symbol_list = normalize_symbols(symbols)

        if timeframe not in YFinanceInterval.values():
            raise InvalidParametersError(f"Invalid interval. Must be one of {YFinanceInterval.values()}.")

        period = kwargs.pop("period", None)
        start_dt, end_dt = None, None

        if start is not None:
            start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True, validate_order=True)

            # Yahoo Finance restricts 1m interval data to the last 30 days
            if timeframe == "1m" and start_dt < datetime.now() - timedelta(days=30):
                raise InvalidParametersError(
                    "For 1 min interval, the start date must be within 30 days from the current date."
                )
        elif period is None:
            logger.warning("Neither 'start' nor 'period' provided. Defaulting to period='1mo'")
            period = "1mo"

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                download_kwargs = {
                    "interval": timeframe,
                    "group_by": "column",
                    "auto_adjust": False,
                    "progress": False,
                    "threads": len(symbol_list) > 1,
                    **kwargs,
                }
                if start_dt is not None:
                    download_kwargs["start"] = start_dt
                    download_kwargs["end"] = end_dt
                else:
                    download_kwargs["period"] = period

                result = yf.download(symbol_list, **download_kwargs)
                df_result = self._normalize_download_result(result, symbol_list)
                log_dataframe_info(df_result, f"Fetched OHLCV data for {len(symbol_list)} symbol(s)")
                return df_result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Failed to fetch data for {symbols} (attempt {attempt}/{max_retries}): {error}",
                        symbols=symbol_list,
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        error=str(e),
                    )
                    time.sleep(self.delay)
                else:
                    logger.error(
                        "Failed to fetch data for {symbols} after {max_retries} retries: {error}",
                        symbols=symbol_list,
                        max_retries=self.max_retries,
                        error=str(e),
                    )
                    raise APIConnectionError(
                        f"Failed to fetch data for {symbol_list} after {self.max_retries} retries"
                    ) from e

        if last_error is not None:
            raise APIConnectionError(f"Failed to fetch data for {symbol_list}") from last_error

        return pd.DataFrame()

    def _normalize_download_result(self, data: pd.DataFrame, symbol_list: List[str]) -> pd.DataFrame:
        """Normalize `yf.download` output into the library's row-wise
        format."""
        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            symbol_level = self._detect_symbol_level(data.columns, symbol_list)
            frames = []
            for symbol in symbol_list:
                if symbol not in data.columns.get_level_values(symbol_level):
                    continue
                symbol_frame = data.xs(symbol, axis=1, level=symbol_level, drop_level=True).copy()
                normalized = self._normalize_single_result(symbol_frame.reset_index(), symbol)
                if not normalized.empty:
                    frames.append(normalized)

            if not frames:
                return pd.DataFrame()

            return pd.concat(frames, ignore_index=True)

        return self._normalize_single_result(data.reset_index(), symbol_list[0])

    @staticmethod
    def _detect_symbol_level(columns: pd.MultiIndex, symbol_list: List[str]) -> int:
        """Detect which MultiIndex level corresponds to ticker
        symbols."""
        symbol_candidates = set(symbol_list)
        for level in range(columns.nlevels):
            if symbol_candidates.issubset(set(columns.get_level_values(level))):
                return level
        raise APIConnectionError("Unexpected yfinance column structure for multi-symbol download")

    @staticmethod
    def _normalize_single_result(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize a single-symbol history DataFrame."""
        if df.empty:
            return df

        result = df.copy()
        if "Datetime" in result.columns and "Date" not in result.columns:
            result = result.rename(columns={"Datetime": "Date"})
        elif "index" in result.columns and "Date" not in result.columns:
            result = result.rename(columns={"index": "Date"})

        result["Symbol"] = symbol
        if "Date" in result.columns:
            result = result.sort_values("Date").reset_index(drop=True)

        return result

    def _fetch_single_symbol(
        self,
        symbol: str,
        start_dt: Optional[datetime],
        end_dt: Optional[datetime],
        timeframe: str,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol (blocking). Used by
        async_fetch_ohlcv.

        Args:
            symbol (str): Stock symbol to fetch.
            start_dt (datetime, optional): Start datetime.
            end_dt (datetime, optional): End datetime.
            timeframe (str): Bar interval.
            period (str, optional): yfinance period string (e.g. '1mo'). Defaults to None.

        Returns:
            pd.DataFrame: OHLCV data with reset index.
        """
        ticker = yf.Ticker(symbol)
        if start_dt is not None:
            data = ticker.history(start=start_dt, end=end_dt, interval=timeframe).assign(Symbol=symbol)
        else:
            data = ticker.history(period=period or "1mo", interval=timeframe).assign(Symbol=symbol)
        return data.reset_index()

    async def async_fetch_ohlcv(
        self,
        symbol: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
    ) -> Tuple[str, pd.DataFrame]:
        """
        Async wrapper around yfinance OHLCV fetch for a single symbol.

        Uses asyncio.to_thread() to run the blocking yfinance SDK call in a
        background thread, keeping the event loop free for concurrent fetches.

        Args:
            symbol (str): Stock symbol to fetch.
            start (Union[str, datetime], optional): Start date or datetime.
            end (Union[str, datetime], optional): End date or datetime.
            timeframe (str, optional): Bar interval. Defaults to "1d".

        Returns:
            Tuple[str, pd.DataFrame]: Tuple of (symbol, DataFrame). DataFrame is empty on failure.
        """
        start_dt, end_dt = None, None
        if start is not None:
            start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)

        try:
            df = await asyncio.to_thread(self._fetch_single_symbol, symbol, start_dt, end_dt, timeframe)
            if df.empty:
                logger.warning("async_fetch_ohlcv: empty result for {symbol}", symbol=symbol)
            return symbol, df
        except Exception as e:
            logger.error("async_fetch_ohlcv failed for {symbol}: {e}", symbol=symbol, e=e)
            return symbol, pd.DataFrame()
