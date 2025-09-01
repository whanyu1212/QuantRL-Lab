import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import dateutil.parser
import pandas as pd
import requests
from alpaca.data import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.models import Trade
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
)
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

from quantrl_lab.data.interface import (
    ConnectionManaged,
    DataSource,
    HistoricalDataCapable,
    LiveDataCapable,
    NewsDataCapable,
    StreamingCapable,
)
from quantrl_lab.data.loaders.alpaca_mappings import ALPACA_MAPPINGS

console = Console()


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

    _stock_stream_client_instance = None

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        stock_historical_client: StockHistoricalDataClient = None,
        stock_stream_client: StockDataStream = None,
    ):
        # `or` operator works by returning the first truthy value or the last value if all are falsy # noqa E501
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")

        if stock_historical_client is not None:
            self.stock_historical_client = stock_historical_client
        else:
            self.stock_historical_client = StockHistoricalDataClient(self.api_key, self.secret_key)

        if AlpacaDataLoader._stock_stream_client_instance is None:
            AlpacaDataLoader._stock_stream_client_instance = StockDataStream(self.api_key, self.secret_key)
        self.stock_stream_client = AlpacaDataLoader._stock_stream_client_instance

        # event subscribers
        self.subscribers = {"quotes": [], "trades": [], "bars": []}
        self._subscribed_symbols = set()

    @property
    def source_name(self) -> str:
        return "Alpaca"

    def connect(self) -> StockHistoricalDataClient:
        """
        Connect to the historical data client of Alpaca.

        Returns:
            StockHistoricalDataClient: The historical data client.
        """
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided")
        return StockHistoricalDataClient(self.api_key, self.secret_key)

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
        # TODO
        pass

    def get_historical_ohlcv_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        timeframe: str = "1d",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from Alpaca. `end` is not compulsory
        and defaults to today if not provided.

        Args:
            symbols: Stock symbol(s) to fetch data for
            start: Start date for historical data
            end: End date for historical data (defaults to today)
            timeframe: The bar timeframe (1d, 1h, 1m, etc.)
            **kwargs: Additional arguments to pass to Alpaca API

        Returns:
            pd.DataFrame: raw OHLCV data
        """

        console.print(
            f"[green]Fetching historical data for {symbols} from {start} to {end} with timeframe {timeframe}[/green]"
        )

        # Convert string inputs to proper types
        if isinstance(start, str):
            start = dateutil.parser.parse(start)

        if end is None:
            end = datetime.now()
        elif isinstance(end, str):
            end = dateutil.parser.parse(end)

        # TODO: may need error handling for intraday data
        # Convert timeframe string to Alpaca TimeFrame object
        alpaca_timeframe = ALPACA_MAPPINGS.get_timeframe(timeframe)

        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]

        # Request parameters
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=alpaca_timeframe,
            start=start,
            end=end,
            **kwargs,
        )

        # Get the bars
        bars = self.stock_historical_client.get_stock_bars(request_params)

        # Return as DataFrame
        bars_df = bars.df.reset_index()
        # Upper case to standardize with other data sources
        bars_df.rename(
            columns=ALPACA_MAPPINGS.ohlcv_columns,
            inplace=True,
        )
        bars_df["Date"] = bars_df["Timestamp"].dt.date
        return bars_df

    def get_latest_quote(self, symbol: str, **kwargs) -> Dict:
        """
        Get the latest quote for a symbol from Alpaca.

        Args:
            symbols (str): symbol to fetch quote for
            **kwargs: additional arguments such as feed type

        Returns:
            Dict: output dictionary
        """

        request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        return self.stock_historical_client.get_stock_latest_quote(request_params)

    def get_latest_trade(self, symbol: str, **kwargs) -> Dict:
        """
        Get the latest trade for a symbol from Alpaca.

        Args:
            symbols (str): symbol to fetch trade for
            **kwargs: additional arguments such as feed type

        Returns:
            Dict: output dictionary
        """
        request_params = StockLatestTradeRequest(symbol_or_symbols=symbol)
        return self.stock_historical_client.get_stock_latest_trade(request_params)

    async def _trade_handler(self, trade_data: Trade):
        """Processes incoming trade data."""
        console.print("--- New Trade Received ---")
        console.print(f"Symbol: {trade_data.symbol}")
        console.print(f"Price: {trade_data.price}")
        console.print(f"Volume: {trade_data.size}")
        console.print(f"Timestamp: {trade_data.timestamp}")
        console.print("--------------------------\n")

    async def subscribe_to_updates(self, symbol: str, data_type: str = "trades") -> None:
        """
        Subscribe to real-time market data updates.

        Args:
            symbol (str): The stock symbol to subscribe to.
            data_type (str): The type of data to subscribe to ('trades', 'quotes', 'bars').
        """
        if data_type == "trades":
            self.stock_stream_client.subscribe_trades(self._trade_handler, symbol)
        elif data_type == "quotes":
            # Define or use a quote handler
            async def quote_handler(data):
                console.print(f"[green]Received quote: {data}[/green]")

            self.stock_stream_client.subscribe_quotes(quote_handler, symbol)
        elif data_type == "bars":
            # Define or use a bar handler
            async def bar_handler(data):
                console.print(f"[blue]Received bar: {data}[/blue]")

            self.stock_stream_client.subscribe_bars(bar_handler, symbol)
        else:
            console.print(f"[red]Error: Unknown data type '{data_type}'[/red]")
            return

        self._subscribed_symbols.add(symbol)
        console.print(f"Subscribed to {data_type} for {symbol}")

    async def start_streaming(self):
        """Initializes, subscribes, and runs the data stream."""
        console.print("Initializing stream...")
        try:
            if not self._subscribed_symbols:
                console.print("[yellow]No symbols subscribed. Call subscribe_to_updates() first.[/yellow]")
                return
            await self.stock_stream_client._run_forever()
        except KeyboardInterrupt:
            console.print("Stream stopped by user.")
        except Exception as e:
            console.print(f"An error occurred: {e}")

    async def stop_streaming(self):
        """Stop the WebSocket connection and clean up resources."""
        await self.stock_stream_client.stop_ws()

    def get_news_data(
        self,
        symbols: Union[str, List[str]],
        start: Union[str, datetime],
        end: Optional[Union[str, datetime]] = None,
        limit: int = 50,
        include_content: bool = False,
        **kwargs,
    ) -> Union[pd.DataFrame, Dict]:
        """
        Get news for specified symbols from Alpaca News API.

        Args:
            symbols: Stock symbol(s) to fetch news for
            start: Start date for news
            end: End date for news (defaults to today)
            limit: Number of news items per request
            include_content: Whether to include full article content
            **kwargs: Additional parameters

        Returns:
            pd.DataFrame: News data
        """

        # Convert symbols list to comma-separated string if needed
        if isinstance(symbols, list):
            symbols = ",".join(symbols)

        # Convert datetime objects to string if needed
        if isinstance(start, datetime):
            start = start.strftime("%Y-%m-%d")

        if end is not None and isinstance(end, datetime):
            end = end.strftime("%Y-%m-%d")
        elif end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        # For some reason Alpaca's Python SDK doesn't
        # have a client for the News API
        # so we'll use requests directly

        # Too lazy to keep a separate config just for this 1 url
        base_url = "https://data.alpaca.markets/v1beta1/news"
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

        params = {
            "symbols": symbols,
            "start": start,
            "end": end,
            "limit": limit,
            "include_content": str(include_content).lower(),
            "sort": "desc",
        }

        all_news = []
        page_token = None
        page_count = 0

        # Create progress bar for news fetching
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # We don't know the total number of pages, so we'll use an indeterminate progress
            task = progress.add_task(f"[cyan]Fetching news for {symbols}...", total=None)

            while True:
                # Add page token if we have one
                if page_token:
                    params["page_token"] = page_token

                try:
                    response = requests.get(base_url, headers=headers, params=params)
                    response.raise_for_status()  # Raise exception for HTTP errors

                    data = response.json()
                    news_items = data.get("news", [])

                    if not news_items:
                        break

                    all_news.extend(news_items)
                    page_count += 1

                    # Update progress description with current stats
                    progress.update(
                        task, description=f"[cyan]Fetched page {page_count} ({len(all_news)} news items total)..."
                    )

                    # Check if there's a next page
                    page_token = data.get("next_page_token")
                    if not page_token:
                        break

                except requests.exceptions.RequestException as e:
                    console.print(f"[red]Error fetching news: {e}[/red]")
                    break

        console.print(f"[green]✓ Total news items fetched: {len(all_news)}[/green]")

        # Convert to DataFrame
        if all_news:
            return pd.DataFrame(all_news)
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    async def main():
        alpaca_client = AlpacaDataLoader()
        # Test historical data
        console.print("\n[bold blue]Testing historical data:[/bold blue]")
        df = alpaca_client.get_historical_ohlcv_data("AAPL", start="2023-01-01", end="2023-01-10")
        console.print(df.head())

        # Test news data
        console.print("\n[bold blue]Testing news data:[/bold blue]")
        news_df = alpaca_client.get_news_data("AAPL", start="2023-01-01", end="2023-01-10", limit=10)
        if not news_df.empty:
            console.print(news_df.iloc[:5][["headline", "created_at", "summary"]])

        # Set up the subscription
        console.print("\n[bold blue]Testing websocket:[/bold blue]")

        # Set up the subscription
        await alpaca_client.subscribe_to_updates("AAPL")

        # Start streaming data
        try:
            console.print("[cyan]Starting WebSocket connection...[/cyan]")
            await alpaca_client.start_streaming()
        except KeyboardInterrupt:
            console.print("[yellow]Closing connection...[/yellow]")
        finally:
            await alpaca_client.stop_streaming()

    asyncio.run(main())
