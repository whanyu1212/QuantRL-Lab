"""
Example: Fetching data from Yahoo Finance

Yahoo Finance provides free access to:
- Historical OHLCV (price) data
- Fundamental data (income statement, balance sheet, cash flow)

No API key required.
"""

from datetime import datetime

from quantrl_lab.data.sources import YFinanceDataLoader


def run_example_step(title, fn):
    """Run one example step and keep the script alive on provider
    failures."""
    print(f"\n{title}")
    print("-" * 40)
    try:
        fn()
    except Exception as exc:
        print(f"Skipped: {exc}")


def main():
    # Initialize the data loader
    loader = YFinanceDataLoader()

    print("=" * 60)
    print("Yahoo Finance Data Examples")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Example 1: Fetch historical OHLCV data for a single symbol
    # ------------------------------------------------------------------
    def example_single_symbol():
        df = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start="2024-01-01",
            end="2024-06-01",
            timeframe="1d",  # Daily data
        )
        print(f"Retrieved {len(df)} rows for AAPL")
        print(df.head())

    run_example_step("[1] Historical OHLCV Data (Single Symbol)", example_single_symbol)

    # ------------------------------------------------------------------
    # Example 2: Fetch historical data for multiple symbols
    # ------------------------------------------------------------------
    def example_multi_symbol():
        df_multi = loader.get_historical_ohlcv_data(
            symbols=["AAPL", "GOOGL", "MSFT"],
            start="2024-01-01",
            end="2024-03-01",
            timeframe="1d",
        )
        print(f"Retrieved {len(df_multi)} total rows")
        print(f"Symbols: {df_multi['Symbol'].unique().tolist()}")
        print(df_multi.head(10))

    run_example_step("[2] Historical OHLCV Data (Multiple Symbols)", example_multi_symbol)

    # ------------------------------------------------------------------
    # Example 3: Fetch intraday data (1-minute bars)
    # Note: Yahoo Finance limits 1-minute data to last 30 days
    # ------------------------------------------------------------------
    def example_intraday():
        from datetime import timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)

        df_intraday = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date,
            end=end_date,
            timeframe="1m",  # 1-minute bars
        )
        print(f"Retrieved {len(df_intraday)} intraday bars")
        print(df_intraday.head())

    run_example_step("[3] Intraday Data (1-minute bars)", example_intraday)

    # ------------------------------------------------------------------
    # Example 4: Fetch fundamental data
    # ------------------------------------------------------------------
    def example_fundamentals():
        fundamentals = loader.get_fundamental_data(
            symbol="AAPL",
            frequency="quarterly",
        )
        print(f"Retrieved {len(fundamentals)} quarters of fundamental data")
        print(f"Columns: {fundamentals.columns.tolist()[:10]}...")
        print(fundamentals.head())

    run_example_step("[4] Fundamental Data", example_fundamentals)

    # ------------------------------------------------------------------
    # Example 5: Individual financial statements
    # ------------------------------------------------------------------
    def example_statements():
        income = loader._get_income_statement("MSFT", frequency="quarterly")
        print(f"Income Statement: {len(income)} quarters")

        balance = loader._get_balance_sheet("MSFT", frequency="quarterly")
        print(f"Balance Sheet: {len(balance)} quarters")

        cashflow = loader._get_cash_flow("MSFT", frequency="quarterly")
        print(f"Cash Flow: {len(cashflow)} quarters")

    run_example_step("[5] Individual Financial Statements", example_statements)

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
