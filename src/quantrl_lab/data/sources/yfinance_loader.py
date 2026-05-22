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
from quantrl_lab.data.exceptions import InvalidParametersError
from quantrl_lab.data.interface import (
    DataSource,
    FundamentalDataCapable,
    HistoricalDataCapable,
)
from quantrl_lab.data.utils import log_dataframe_info, normalize_date_range, normalize_symbols


class YFinanceDataLoader(DataSource, FundamentalDataCapable, HistoricalDataCapable):
    """Yahoo Finance implementation that provides market data and
    fundamental data."""

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
        # TODO
        pass

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

        for attempt in range(self.max_retries):
            try:
                result = pd.DataFrame()
                for symbol in symbol_list:
                    ticker = yf.Ticker(symbol)
                    if start_dt is not None:
                        data = ticker.history(start=start_dt, end=end_dt, interval=timeframe, **kwargs).assign(
                            Symbol=symbol
                        )
                    else:
                        data = ticker.history(period=period, interval=timeframe, **kwargs).assign(Symbol=symbol)
                    result = pd.concat([result, data])

                df_result = result.reset_index()
                log_dataframe_info(df_result, f"Fetched OHLCV data for {len(symbol_list)} symbol(s)")
                return df_result

            except Exception as e:
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
                    return pd.DataFrame()

    def _fetch_single_symbol(
        self,
        symbol: str,
        start_dt: Optional[datetime],
        end_dt: Optional[datetime],
        timeframe: str,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol (blocking).

        Used by async_fetch_ohlcv.

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
