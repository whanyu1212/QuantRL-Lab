"""
Example: Fetching data from Alpaca.

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

    # ------------------------------------------------------------------
    # Example 1: Fetch historical OHLCV data
    # ------------------------------------------------------------------
    print("\n[1] Historical OHLCV Data")
    print("-" * 40)

    df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        start="2024-01-01",
        end="2024-03-01",
        timeframe="1d",  # Daily bars
    )
    print(f"Retrieved {len(df)} daily bars for AAPL")
    print(df.head())

    # ------------------------------------------------------------------
    # Example 2: Fetch data for multiple symbols
    # ------------------------------------------------------------------
    print("\n[2] Multiple Symbols")
    print("-" * 40)

    df_multi = loader.get_historical_ohlcv_data(
        symbols=["AAPL", "GOOGL", "TSLA"],
        start="2024-01-01",
        end="2024-02-01",
        timeframe="1d",
    )
    print(f"Retrieved {len(df_multi)} total bars")
    print(f"Symbols: {df_multi['Symbol'].unique().tolist()}")

    # ------------------------------------------------------------------
    # Example 3: Fetch hourly data
    # ------------------------------------------------------------------
    print("\n[3] Hourly Data")
    print("-" * 40)

    df_hourly = loader.get_historical_ohlcv_data(
        symbols="NVDA",
        start="2024-01-15",
        end="2024-01-20",
        timeframe="1h",  # Hourly bars
    )
    print(f"Retrieved {len(df_hourly)} hourly bars for NVDA")
    print(df_hourly.head())

    # ------------------------------------------------------------------
    # Example 4: Get latest quote
    # ------------------------------------------------------------------
    print("\n[4] Latest Quote")
    print("-" * 40)

    quote = loader.get_latest_quote("AAPL")
    print(f"Latest quote for AAPL: {quote}")

    # ------------------------------------------------------------------
    # Example 5: Get latest trade
    # ------------------------------------------------------------------
    print("\n[5] Latest Trade")
    print("-" * 40)

    trade = loader.get_latest_trade("AAPL")
    print(f"Latest trade for AAPL: {trade}")

    # ------------------------------------------------------------------
    # Example 6: Fetch news data
    # ------------------------------------------------------------------
    print("\n[6] News Data")
    print("-" * 40)

    news_df = loader.get_news_data(
        symbols="AAPL",
        start="2024-01-01",
        end="2024-01-15",
        limit=10,
        include_content=False,  # Set True to include full article content
    )

    if not news_df.empty:
        print(f"Retrieved {len(news_df)} news articles")
        print("\nSample headlines:")
        for _, row in news_df.head(3).iterrows():
            print(f"  - {row['headline'][:80]}...")
    else:
        print("No news articles found for the specified period.")

    # ------------------------------------------------------------------
    # Example 7: News for multiple symbols
    # ------------------------------------------------------------------
    print("\n[7] News for Multiple Symbols")
    print("-" * 40)

    news_multi = loader.get_news_data(
        symbols=["AAPL", "TSLA", "NVDA"],
        start="2024-01-01",
        end="2024-01-10",
        limit=20,
    )

    if not news_multi.empty:
        print(f"Retrieved {len(news_multi)} news articles")
        print(f"Columns: {news_multi.columns.tolist()}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
