"""
Example: Fetching data from Alpaca

Alpaca provides:
- Historical OHLCV data
- Latest quotes and trades
- News data

Requirements:
- Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file
- Or pass them directly to the AlpacaDataLoader constructor

Note: This example excludes live/streaming data.
"""

from dotenv import load_dotenv

from quantrl_lab.data.exceptions import DataSourceError
from quantrl_lab.data.sources import AlpacaDataLoader

# Load environment variables from .env file
load_dotenv()


def main():
    # Initialize the data loader
    # API keys are loaded from environment variables by default
    loader = AlpacaDataLoader()

    # Verify connection
    if not loader.is_connected():
        print("ERROR: Alpaca API credentials not configured.")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")
        return

    print("=" * 60)
    print("Alpaca Data Examples")
    print("=" * 60)

    def run_example_step(title, fn):
        print(f"\n{title}")
        print("-" * 40)
        try:
            fn()
        except DataSourceError as exc:
            print(f"Skipped: {exc}")
        except Exception as exc:
            print(f"Skipped: {exc}")

    # ------------------------------------------------------------------
    # Example 1: Fetch historical OHLCV data
    # ------------------------------------------------------------------
    def example_daily():
        df = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start="2024-01-01",
            end="2024-03-01",
            timeframe="1d",
        )
        print(f"Retrieved {len(df)} daily bars for AAPL")
        print(df.head())

    run_example_step("[1] Historical OHLCV Data", example_daily)

    # ------------------------------------------------------------------
    # Example 2: Fetch data for multiple symbols
    # ------------------------------------------------------------------
    def example_multi():
        df_multi = loader.get_historical_ohlcv_data(
            symbols=["AAPL", "GOOGL", "TSLA"],
            start="2024-01-01",
            end="2024-02-01",
            timeframe="1d",
        )
        print(f"Retrieved {len(df_multi)} total bars")
        print(f"Symbols: {df_multi['Symbol'].unique().tolist()}")

    run_example_step("[2] Multiple Symbols", example_multi)

    # ------------------------------------------------------------------
    # Example 3: Fetch hourly data
    # ------------------------------------------------------------------
    def example_hourly():
        df_hourly = loader.get_historical_ohlcv_data(
            symbols="NVDA",
            start="2024-01-15",
            end="2024-01-20",
            timeframe="1h",
        )
        print(f"Retrieved {len(df_hourly)} hourly bars for NVDA")
        print(df_hourly.head())

    run_example_step("[3] Hourly Data", example_hourly)

    # ------------------------------------------------------------------
    # Example 4: Get latest quote
    # ------------------------------------------------------------------
    run_example_step("[4] Latest Quote", lambda: print(f"Latest quote for AAPL: {loader.get_latest_quote('AAPL')}"))

    # ------------------------------------------------------------------
    # Example 5: Get latest trade
    # ------------------------------------------------------------------
    run_example_step("[5] Latest Trade", lambda: print(f"Latest trade for AAPL: {loader.get_latest_trade('AAPL')}"))

    # ------------------------------------------------------------------
    # Example 6: Fetch news data
    # ------------------------------------------------------------------
    def example_news():
        news_df = loader.get_news_data(
            symbols="AAPL",
            start="2024-01-01",
            end="2024-01-15",
            limit=10,
            include_content=False,
        )
        if not news_df.empty:
            print(f"Retrieved {len(news_df)} news articles")
            print("\nSample headlines:")
            for _, row in news_df.head(3).iterrows():
                print(f"  - {row['headline'][:80]}...")
        else:
            print("No news articles found for the specified period.")

    run_example_step("[6] News Data", example_news)

    # ------------------------------------------------------------------
    # Example 7: News for multiple symbols
    # ------------------------------------------------------------------
    def example_multi_news():
        news_multi = loader.get_news_data(
            symbols=["AAPL", "TSLA", "NVDA"],
            start="2024-01-01",
            end="2024-01-10",
            limit=20,
        )
        if not news_multi.empty:
            print(f"Retrieved {len(news_multi)} news articles")
            print(f"Columns: {news_multi.columns.tolist()}")

    run_example_step("[7] News for Multiple Symbols", example_multi_news)

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
