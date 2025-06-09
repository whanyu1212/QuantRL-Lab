import time
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pandas as pd
import yfinance as yf
from loguru import logger

from quantrl_lab.data.interface import (
    DataSource,
    FundamentalDataCapable,
    HistoricalDataCapable,
)
from quantrl_lab.utils.config import (
    YFinanceInterval,
    financial_columns,
)


class YfinanceDataloader(DataSource, FundamentalDataCapable, HistoricalDataCapable):
    """YF implementation that provides market data and fundamental
    data."""

    def __init__(
        self,
        max_retries: int = 3,
        delay: int = 1,
    ):
        # Remark:
        # Do not initialize the ticker related variables here
        # or else the class object will not be reusable
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
        """
        yfinance uses HTTP requests - assume connected if no network issues.
        """
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
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get all the fundamental related data for a symbol, including
        income statement, cash flow, and balance sheet.

        Args:
            symbol (str): Stock symbol, only a single symbol is supported.
            Defaults to None.
            frequency (str, optional):Defaults to "quarterly".

        Returns:
            pd.DataFrame: DataFrame with raw fundamental data
        """

        # Get the financial statements
        income_statement = self._get_income_statement(symbol, frequency=frequency)
        cash_flow = self._get_cash_flow(symbol, frequency=frequency)
        balance_sheet = self._get_balance_sheet(symbol, frequency=frequency)

        # Merge all the dataframes
        df = income_statement.merge(cash_flow, on="Date", how="outer")
        df = df.merge(balance_sheet, on="Date", how="outer")

        # Add symbol column
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
            symbol (str): Stock symbol, only a single symbol is supported.
            frequency (str, optional): Defaults to "quarterly".

        Returns:
            pd.DataFrame: DataFrame with raw income statement data
        """
        logger.info(f"Fetching income statement for {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.get_income_stmt(freq=frequency).T.reset_index(names="Date")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def _get_cash_flow(self, symbol: str, frequency: str = "quarterly") -> pd.DataFrame:
        """
        Get cash flow statement for a symbol.

        Args:
            symbol (str): Stock symbol, only a single symbol is supported.
            frequency (str, optional): Defaults to "quarterly".

        Returns:
            pd.DataFrame: DataFrame with raw cash flow data
        """
        logger.info(f"Fetching cash flow statement for {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.get_cashflow(freq=frequency).T.reset_index(names="Date")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def _get_balance_sheet(self, symbol: str, frequency: str = "quarterly") -> pd.DataFrame:
        """
        Get balance sheet for a symbol.

        Args:
            symbol (str): Stock symbol, only a single symbol is supported.
            frequency (str, optional): Defaults to "quarterly".

        Returns:
            pd.DataFrame: DataFrame with raw balance sheet data
        """
        logger.info(f"Fetching balance sheet for {symbol}")
        ticker = yf.Ticker(symbol)
        df = ticker.get_balance_sheet(freq=frequency).T.reset_index(names="Date")
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def get_historical_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Union[str, datetime],
        timeframe: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a list of symbols.

        Args:
            symbols (Union[str, List[str]]): A single symbol or a list of symbols.
            start (Union[str, datetime]): start date or datetime
            end (Union[str, datetime]): end date or datetime
            timeframe (str, optional): period. Defaults to "1d".

        Raises:
             ValueError: all elements in 'symbols' must be strings
             TypeError: 'symbols' must be a string or a list of strings
             ValueError: Invalid interval
             ValueError: Invalid start or end date
             ValueError: Start date should be before end date
             ValueError: For 1 min interval, the start date must be within
             30 days from the current date

        Returns:
            pd.DataFrame: output dataframe with OHLCV data (raw)
        """

        # --------- Runtime Error Handling ------------
        if isinstance(symbols, str):
            # Convert single string to a list with one element
            symbols = [symbols]
        elif isinstance(symbols, list):
            # Check if all elements are strings
            if not all(isinstance(symbol, str) for symbol in symbols):
                raise ValueError("All elements in 'symbols' must be strings.")
            symbols = symbols
        else:
            # Handle any other type (neither string nor list)
            raise TypeError("'symbols' must be a string or a list of strings.")

        if timeframe not in YFinanceInterval.values():
            raise ValueError(f"Invalid interval. Must be one of {YFinanceInterval.values()}.")

        try:
            pd.to_datetime(start)
            pd.to_datetime(end)
        except ValueError:
            raise ValueError("Invalid start or end date.")

        if start > end:
            raise ValueError("Start date should be before end date.")

        if timeframe == "1m" and start < datetime.now() - timedelta(days=30):
            # This is the rule set by Yahoo Finance
            raise ValueError("For 1 min interval, the start date must be within " "30 days from the current date.")

        for _ in range(self.max_retries):
            try:
                result = pd.DataFrame()
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start, end=end, period=timeframe).assign(Symbol=symbol)
                    result = pd.concat([result, data])
                return result.reset_index()
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbols} - {e}")
                time.sleep(self.delay)

        else:
            logger.error(f"Failed to fetch data for {symbols} after {self.max_retries} retries.")
            return None


if __name__ == "__main__":
    # Example usage
    yf_loader = YfinanceDataloader()

    # Get fundamental data
    print("Fetching fundamental data for PLTR...")
    fund_data = yf_loader.get_fundamental_data("PLTR")
    print(f"Retrieved {len(fund_data)} rows with {len(fund_data.columns)} key_cols")
    print(fund_data)

    # Get historical data
    print("\nFetching historical data for PLTR...")
    hist_data = yf_loader.get_historical_ohlcv_data("PLTR", start="2023-01-01", end="2025-03-10")
    print(f"Retrieved {len(hist_data)} days of historical data")
    print(hist_data.head())
