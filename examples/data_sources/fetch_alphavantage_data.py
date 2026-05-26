"""
Example: Fetching data from Alpha Vantage.

Alpha Vantage provides:
- Historical OHLCV data (daily and intraday)
- Fundamental data (company overview, financials, earnings)
- News sentiment data
- Macroeconomic indicators (GDP, CPI, unemployment, etc.)

Requirements:
- Set ALPHA_VANTAGE_API_KEY in your .env file
- Or pass it directly to the AlphaVantageDataLoader constructor

Note: Free tier limitations:
- 25 API requests per day, 1 request per second
- outputsize='full' requires premium (only last 100 days available on free tier)
- Historical intraday data with 'month' parameter requires premium
- Some examples below may require a premium API key
"""

from dotenv import load_dotenv

from quantrl_lab.data.config import FundamentalMetric, MacroIndicator
from quantrl_lab.data.sources import AlphaVantageDataLoader

# Load environment variables from .env file
load_dotenv()


def main():
    # Initialize the data loader
    loader = AlphaVantageDataLoader()

    if not loader.api_key:
        print("ERROR: Alpha Vantage API key not configured.")
        print("Please set ALPHA_VANTAGE_API_KEY in your .env file.")
        return

    print("=" * 60)
    print("Alpha Vantage Data Examples")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Example 1: Fetch daily OHLCV data (recent, free tier compatible)
    # ------------------------------------------------------------------
    print("\n[1] Daily OHLCV Data (last 100 trading days)")
    print("-" * 40)

    # Free tier: outputsize='compact' returns last 100 data points
    # Premium tier: use outputsize='full' for 20+ years of data
    df = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        timeframe="1d",
        # Note: Don't specify start/end dates with free tier, as outputsize='compact'
        # only returns the most recent 100 trading days
    )
    if not df.empty:
        print(f"Retrieved {len(df)} daily bars for AAPL")
        print(df.head())

    # ------------------------------------------------------------------
    # Example 2: Fetch intraday data (PREMIUM FEATURE)
    # ------------------------------------------------------------------
    print("\n[2] Intraday Data (5-minute bars) - REQUIRES PREMIUM")
    print("-" * 40)

    # Note: Historical intraday with 'month' parameter requires premium API key
    # Free tier only gets recent intraday data (last 1-2 trading days)
    df_intraday = loader.get_historical_ohlcv_data(
        symbols="AAPL",
        timeframe="5min",
        # month="2024-01",  # Premium feature: historical month data
    )
    if not df_intraday.empty:
        print(f"Retrieved {len(df_intraday)} intraday bars")
        print(df_intraday.head())
    else:
        print("No intraday data retrieved (may require premium API key)")

    # ------------------------------------------------------------------
    # Example 3: Company overview (fundamental data)
    # ------------------------------------------------------------------
    print("\n[3] Company Overview")
    print("-" * 40)

    fundamentals = loader.get_fundamental_data(
        symbol="AAPL",
        metrics=[FundamentalMetric.COMPANY_OVERVIEW],
    )

    if fundamentals.get("company_overview"):
        overview = fundamentals["company_overview"]
        print(f"Company: {overview.get('Name')}")
        print(f"Sector: {overview.get('Sector')}")
        print(f"Industry: {overview.get('Industry')}")
        print(f"Market Cap: {overview.get('MarketCapitalization')}")
        print(f"PE Ratio: {overview.get('PERatio')}")

    # ------------------------------------------------------------------
    # Example 4: Financial statements
    # ------------------------------------------------------------------
    print("\n[4] Financial Statements")
    print("-" * 40)

    financials = loader.get_fundamental_data(
        symbol="MSFT",
        metrics=[
            FundamentalMetric.INCOME_STATEMENT,
            FundamentalMetric.BALANCE_SHEET,
            FundamentalMetric.CASH_FLOW,
        ],
    )

    for metric, data in financials.items():
        if data:
            print(f"  {metric}: Retrieved successfully")

    # ------------------------------------------------------------------
    # Example 5: Earnings data
    # ------------------------------------------------------------------
    print("\n[5] Earnings Data")
    print("-" * 40)

    earnings = loader.get_fundamental_data(
        symbol="GOOGL",
        metrics=[FundamentalMetric.EARNINGS],
    )

    if earnings.get("earnings"):
        earnings_data = earnings["earnings"]
        if "quarterlyEarnings" in earnings_data:
            print(f"Retrieved {len(earnings_data['quarterlyEarnings'])} quarterly earnings reports")

    # ------------------------------------------------------------------
    # Example 6: News sentiment data
    # ------------------------------------------------------------------
    print("\n[6] News Sentiment Data")
    print("-" * 40)

    news_df = loader.get_news_data(
        symbols="AAPL",
        start="2024-01-01",
        end="2024-01-15",
        limit=10,
    )

    if not news_df.empty:
        print(f"Retrieved {len(news_df)} news articles")
        print("\nSample articles:")
        for _, row in news_df.head(3).iterrows():
            title = row.get("title", "N/A")[:60]
            sentiment = row.get("sentiment_score", "N/A")
            print(f"  - {title}... (sentiment: {sentiment})")

    # ------------------------------------------------------------------
    # Example 7: Macroeconomic data - GDP
    # ------------------------------------------------------------------
    print("\n[7] Macroeconomic Data - GDP")
    print("-" * 40)

    macro_data = loader.get_macro_data(
        indicators=[MacroIndicator.REAL_GDP],
        start="2020-01-01",
        end="2024-01-01",
    )

    if macro_data.get("real_gdp"):
        gdp_data = macro_data["real_gdp"]
        if "data" in gdp_data:
            print(f"Retrieved {len(gdp_data['data'])} GDP data points")
            print("Latest GDP values:")
            for item in gdp_data["data"][:3]:
                print(f"  {item['date']}: ${item['value']} billion")

    # ------------------------------------------------------------------
    # Example 8: Treasury yields
    # ------------------------------------------------------------------
    print("\n[8] Treasury Yields")
    print("-" * 40)

    treasury = loader.get_macro_data(
        indicators={
            MacroIndicator.TREASURY_YIELD: {
                "interval": "monthly",
                "maturity": "10year",
            }
        },
        start="2023-01-01",
        end="2024-01-01",
    )

    if treasury.get("treasury_yield"):
        yield_data = treasury["treasury_yield"]
        if "data" in yield_data:
            print(f"Retrieved {len(yield_data['data'])} yield data points")
            print("Latest 10-year Treasury yields:")
            for item in yield_data["data"][:3]:
                print(f"  {item['date']}: {item['value']}%")

    # ------------------------------------------------------------------
    # Example 9: Multiple macro indicators
    # ------------------------------------------------------------------
    print("\n[9] Multiple Macro Indicators")
    print("-" * 40)

    multi_macro = loader.get_macro_data(
        indicators=[
            MacroIndicator.CPI,
            MacroIndicator.UNEMPLOYMENT_RATE,
            MacroIndicator.INFLATION,
        ],
        start="2023-01-01",
        end="2024-01-01",
    )

    for indicator, data in multi_macro.items():
        if data and "data" in data:
            print(f"  {indicator}: {len(data['data'])} data points")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: Alpha Vantage has rate limits. If you encounter errors,")
    print("wait a moment and try again.")


if __name__ == "__main__":
    main()
