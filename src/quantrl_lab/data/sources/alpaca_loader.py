import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    import aiohttp

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.models import Trade
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from loguru import logger

from quantrl_lab.data.exceptions import AuthenticationError, DataSourceError, InvalidParametersError
from quantrl_lab.data.interface import (
    ConnectionManaged,
    DataSource,
    HistoricalDataCapable,
    LiveDataCapable,
    NewsDataCapable,
    StreamingCapable,
)
from quantrl_lab.data.processing.mappings import ALPACA_MAPPINGS
from quantrl_lab.data.utils import (
    HTTPRequestWrapper,
    RetryStrategy,
    add_date_column_from_timestamp,
    format_date_to_string,
    log_dataframe_info,
    normalize_date_range,
    normalize_symbols,
    standardize_ohlcv_columns,
)


class AlpacaDataLoader(
    DataSource,
    HistoricalDataCapable,
    LiveDataCapable,
    StreamingCapable,
    NewsDataCapable,
    ConnectionManaged,
):
    """Alpaca implementation that provides market data from Alpaca
    APIs."""

    SUPPORTED_FEATURES = {
        "historical_bars",
        "live_data",
        "streaming",
        "news",
        "connection_managed",
    }

    NEWS_API_BASE_URL = "https://data.alpaca.markets/v1beta1/news"
    DEFAULT_NEWS_SORT = "desc"
    DEFAULT_NEWS_LIMIT = 50

    _stock_stream_client_instance = None

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        stock_historical_client: StockHistoricalDataClient = None,
        stock_stream_client: StockDataStream = None,
    ):
        # `or` returns the first truthy value, so env vars serve as fallback when args are None
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")

        if stock_historical_client is not None:
            self.stock_historical_client = stock_historical_client
        else:
            self.stock_historical_client = StockHistoricalDataClient(self.api_key, self.secret_key)

        if AlpacaDataLoader._stock_stream_client_instance is None:
            AlpacaDataLoader._stock_stream_client_instance = StockDataStream(self.api_key, self.secret_key)
        self.stock_stream_client = AlpacaDataLoader._stock_stream_client_instance
        self._news_request_wrapper = HTTPRequestWrapper(
            max_retries=3,
            retry_strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            timeout=30.0,
        )

        self.subscribers = {"quotes": [], "trades": [], "bars": []}
        self._subscribed_symbols = set()

    @property
    def source_name(self) -> str:
        return "Alpaca"

    def connect(self) -> None:
        """
        Connect to the historical data client of Alpaca.

        Reinitializes the stock historical client with current credentials.

        Raises:
            AuthenticationError: If API credentials are not provided.
        """
        if not self.api_key or not self.secret_key:
            raise AuthenticationError("Alpaca API credentials not provided")
        self.stock_historical_client = StockHistoricalDataClient(self.api_key, self.secret_key)

    def disconnect(self) -> None:
        """Disconnect from the historical data client."""
        if self.stock_historical_client:
            self.stock_historical_client.close()

    def is_connected(self) -> bool:
        """
        Check if the historical client is initialized and credentials
        are valid.

        Returns:
            bool: True if the client is initialized with valid credentials, False otherwise.
        """
        try:
            return self.stock_historical_client is not None and (
                self.api_key is not None and self.secret_key is not None
            )
        except Exception:
            return False

    def list_available_instruments(
        self,
        instrument_type: Optional[str] = None,
        market: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        logger.warning("AlpacaDataLoader does not currently implement instrument discovery.")
        return []

    def get_historical_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from Alpaca.

        `end` is not compulsory and defaults to today if not provided.

        Args:
            symbols (Union[str, List[str]]): Stock symbol(s) to fetch data for.
            start (Union[str, datetime], optional): Start date for historical data.
            end (Union[str, datetime], optional): End date for historical data. Defaults to today.
            timeframe (str, optional): The bar timeframe (1d, 1h, 1m, etc.). Defaults to "1d".
            **kwargs: Additional arguments to pass to Alpaca API.

        Returns:
            pd.DataFrame: Raw OHLCV data.
        """
        if start is None:
            raise InvalidParametersError("Alpaca requires a 'start' date for historical data.")

        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        symbol_list = normalize_symbols(symbols)

        logger.info(
            "Fetching historical data for {symbols} from {start} to {end} with timeframe {timeframe}",
            symbols=symbol_list,
            start=start_dt,
            end=end_dt,
            timeframe=timeframe,
        )

        alpaca_timeframe = ALPACA_MAPPINGS.get_timeframe(timeframe)

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol_list,
            timeframe=alpaca_timeframe,
            start=start_dt,
            end=end_dt,
            **kwargs,
        )

        bars = self.stock_historical_client.get_stock_bars(request_params)
        bars_df = bars.df.reset_index()
        bars_df = standardize_ohlcv_columns(bars_df, ALPACA_MAPPINGS.ohlcv_columns)
        bars_df = add_date_column_from_timestamp(bars_df, timestamp_col="Timestamp")

        num_symbols = len(set(bars_df["Symbol"])) if "Symbol" in bars_df.columns else 1
        log_dataframe_info(
            bars_df,
            f"Fetched OHLCV data for {num_symbols} symbol(s)",
            symbol=None,
        )

        return bars_df

    def get_latest_quote(self, symbol: str, **kwargs: Any) -> Dict:
        """
        Get the latest quote for a symbol from Alpaca.

        Args:
            symbol (str): Stock symbol to fetch quote for.
            **kwargs: Additional arguments such as feed type.

        Returns:
            Dict: Output dictionary.
        """
        request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        return self.stock_historical_client.get_stock_latest_quote(request_params)

    def get_latest_trade(self, symbol: str, **kwargs: Any) -> Dict:
        """
        Get the latest trade for a symbol from Alpaca.

        Args:
            symbol (str): Stock symbol to fetch trade for.
            **kwargs: Additional arguments such as feed type.

        Returns:
            Dict: Output dictionary.
        """
        request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
        return self.stock_historical_client.get_stock_latest_trade(request_params)

    async def _trade_handler(self, trade_data: Trade):
        """Process incoming trade data."""
        logger.debug("Trade: {data}", data=trade_data)

    async def subscribe_to_updates(self, symbol: str, data_type: str = "trades") -> None:
        """
        Subscribe to real-time market data updates.

        Args:
            symbol (str): The stock symbol to subscribe to.
            data_type (str, optional): The type of data to subscribe to
                ('trades', 'quotes', 'bars'). Defaults to "trades".
        """
        if data_type == "trades":
            self.stock_stream_client.subscribe_trades(self._trade_handler, symbol)
        elif data_type == "quotes":

            async def quote_handler(data):
                logger.debug("Quote: {data}", data=data)

            self.stock_stream_client.subscribe_quotes(quote_handler, symbol)
        elif data_type == "bars":

            async def bar_handler(data):
                logger.debug("Bar: {data}", data=data)

            self.stock_stream_client.subscribe_bars(bar_handler, symbol)
        else:
            logger.error("Unknown data type '{data_type}' for subscription", data_type=data_type)
            return

        self._subscribed_symbols.add(symbol)
        logger.success("Subscribed to {data_type} for {symbol}", data_type=data_type, symbol=symbol)

    async def start_streaming(self):
        """Initialize, subscribe, and run the data stream."""
        logger.info("Initializing stream...")
        try:
            if not self._subscribed_symbols:
                logger.warning("No symbols subscribed. Call subscribe_to_updates() first.")
                return
            await self.stock_stream_client._run_forever()
        except KeyboardInterrupt:
            logger.info("Stream stopped by user.")
        except Exception as e:
            logger.exception("An error occurred while streaming: {e}", e=e)

    async def stop_streaming(self):
        """Stop the WebSocket connection and clean up resources."""
        logger.info("Stopping WebSocket connection...")
        await self.stock_stream_client.stop_ws()
        logger.success("WebSocket connection stopped")

    def get_news_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        limit: int = 50,
        include_content: bool = False,
        verbose: bool = False,
        silent_errors: bool = False,
        **kwargs: Any,
    ) -> Union[pd.DataFrame, Dict]:
        """
        Get news for specified symbols from Alpaca News API.

        Args:
            symbols (Union[str, List[str]]): Stock symbol(s) to fetch news for.
            start (Union[str, datetime]): Start date for news.
            end (Union[str, datetime], optional): End date for news. Defaults to today.
            limit (int, optional): Number of news items per request. Defaults to 50.
            include_content (bool, optional): Whether to include full article content. Defaults to False.
            verbose (bool, optional): Whether to log progress. Defaults to False.
            silent_errors (bool, optional): Whether to suppress connection errors. Defaults to False.
            **kwargs: Additional parameters.

        Returns:
            pd.DataFrame: News data.
        """
        symbol_list = normalize_symbols(symbols)
        symbols_str = ",".join(symbol_list)

        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        start_str = format_date_to_string(start_dt)
        end_str = format_date_to_string(end_dt)

        # Alpaca's Python SDK does not include a News API client, so requests is used directly
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        params = {
            "symbols": symbols_str,
            "start": start_str,
            "end": end_str,
            "limit": limit,
            "include_content": str(include_content).lower(),
            "sort": self.DEFAULT_NEWS_SORT,
        }

        all_news = []
        page_token = None
        page_count = 0

        if verbose:
            logger.info(
                "Fetching news for {symbols} from {start} to {end} (limit={limit}, include_content={include})",
                symbols=symbols_str,
                start=start_str,
                end=end_str,
                limit=limit,
                include=include_content,
            )

        while True:
            if page_token:
                params["page_token"] = page_token

            try:
                data = self._news_request_wrapper.make_request(
                    self.NEWS_API_BASE_URL,
                    headers=headers,
                    params=params,
                    raise_on_error=True,
                )
                news_items = data.get("news", [])

                if not news_items:
                    break

                all_news.extend(news_items)
                page_count += 1

                if verbose:
                    logger.debug(
                        "Fetched page {page} (total_items={total})",
                        page=page_count,
                        total=len(all_news),
                    )

                page_token = data.get("next_page_token")
                if not page_token:
                    break

            except DataSourceError as e:
                if silent_errors:
                    logger.debug("Silent failure fetching news: {e}", e=e)
                    return pd.DataFrame()
                raise

        if verbose:
            logger.success("Total news items fetched: {n}", n=len(all_news))

        if all_news:
            return pd.DataFrame(all_news)
        else:
            return pd.DataFrame()

    async def async_fetch_news(
        self,
        session: "aiohttp.ClientSession",
        symbol: str,
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        limit: int = 50,
        include_content: bool = False,
    ) -> Tuple[str, pd.DataFrame]:
        """
        Async fetch of Alpaca news for a single symbol.

        Ports the pagination loop from get_news_data() using aiohttp instead
        of requests, keeping the event loop free during I/O waits.

        Args:
            session (aiohttp.ClientSession): Shared aiohttp session.
            symbol (str): Stock symbol to fetch news for.
            start (Union[str, datetime]): Start date for news.
            end (Union[str, datetime], optional): End date for news. Defaults to today.
            limit (int, optional): Number of news items per request. Defaults to 50.
            include_content (bool, optional): Whether to include full article content. Defaults to False.

        Returns:
            Tuple[str, pd.DataFrame]: Tuple of (symbol, DataFrame). DataFrame is empty on failure.
        """
        start_dt, end_dt = normalize_date_range(start, end, default_end_to_now=True)
        start_str = format_date_to_string(start_dt)
        end_str = format_date_to_string(end_dt)

        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        params = {
            "symbols": symbol,
            "start": start_str,
            "end": end_str,
            "limit": limit,
            "include_content": str(include_content).lower(),
            "sort": self.DEFAULT_NEWS_SORT,
        }

        all_news = []
        page_token = None

        try:
            while True:
                if page_token:
                    params["page_token"] = page_token

                async with session.get(self.NEWS_API_BASE_URL, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json(content_type=None)

                news_items = data.get("news", [])
                if not news_items:
                    break
                all_news.extend(news_items)

                page_token = data.get("next_page_token")
                if not page_token:
                    break

        except Exception as e:
            logger.error("async_fetch_news failed for {symbol}: {e}", symbol=symbol, e=e)

        df = pd.DataFrame(all_news) if all_news else pd.DataFrame()
        return symbol, df
