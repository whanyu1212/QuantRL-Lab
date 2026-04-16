"""
Example: Fetching data from Financial Modeling Prep (FMP)

FMP provides access to:
- Historical OHLCV (price) data (daily and intraday)
- Analyst grades and ratings
- Company profile information (sector, industry, key metrics)
- Historical sector performance
- Historical industry performance

Requires API key.
"""

from dotenv import load_dotenv

from quantrl_lab.data.exceptions import AuthenticationError, DataSourceError
from quantrl_lab.data.sources import FMPDataSource

# Load environment variables from .env file
load_dotenv()


def main():
    # Initialize the data loader
    # API key can be provided as argument or via FMP_API_KEY environment variable
    try:
        loader = FMPDataSource()
    except AuthenticationError:
        print("ERROR: FMP API key not configured.")
        print("Please set FMP_API_KEY in your .env file.")
        return

    print("=" * 60)
    print("Financial Modeling Prep (FMP) Data Examples")
    print("=" * 60)

    def run_example_step(title, fn):
        print(f"\n{title}")
        print("-" * 40)
        try:
            fn()
        except DataSourceError as exc:
            print(f"Skipped: {exc}")

    # ------------------------------------------------------------------
    # Example 1: Fetch historical OHLCV data (daily)
    # ------------------------------------------------------------------
    def example_daily():
        df = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start="2025-01-01",
            end="2026-01-01",
            timeframe="1d",
        )
        print(f"Retrieved {len(df)} rows for AAPL")
        print(df.head())

    run_example_step("[1] Historical OHLCV Data (Daily)", example_daily)

    # ------------------------------------------------------------------
    # Example 2: Fetch intraday data (5-minute bars)
    # ------------------------------------------------------------------
    start_date = "2025-01-01"
    end_date = "2026-01-01"

    def example_intraday():
        df_intraday = loader.get_historical_ohlcv_data(
            symbols="AAPL",
            start=start_date,
            end=end_date,
            timeframe="5min",
        )
        print(f"Retrieved {len(df_intraday)} 5-minute bars")
        print(df_intraday.head())

    run_example_step("[2] Intraday Data (5-minute bars)", example_intraday)

    # ------------------------------------------------------------------
    # Example 3: Different intraday timeframes
    # ------------------------------------------------------------------
    def example_timeframes():
        timeframes = ["15min", "1hour"]
        for tf in timeframes:
            df_tf = loader.get_historical_ohlcv_data(
                symbols="MSFT",
                start=start_date,
                end=end_date,
                timeframe=tf,
            )
            print(f"{tf}: Retrieved {len(df_tf)} bars")

    run_example_step("[3] Different Intraday Timeframes", example_timeframes)

    # ------------------------------------------------------------------
    # Example 4: Fetch analyst grades
    # ------------------------------------------------------------------
    def example_grades():
        grades = loader.get_historical_grades("AAPL")
        print(f"Retrieved {len(grades)} historical grades")
        print(f"Columns: {grades.columns.tolist()}")
        print(grades.head())
        print("\nGrade Frequencies:")
        for col in ["newGrade", "gradingCompany"]:
            if col in grades.columns:
                print(f"\nTop 5 {col}:")
                print(grades[col].value_counts().head())

    run_example_step("[4] Historical Analyst Grades", example_grades)

    # ------------------------------------------------------------------
    # Example 5: Fetch analyst ratings
    # ------------------------------------------------------------------
    def example_ratings():
        ratings = loader.get_historical_rating("AAPL", limit=500)
        print(f"Retrieved {len(ratings)} historical ratings")
        print(f"Columns: {ratings.columns.tolist()}")
        print(ratings.head())
        print("\nRating Frequencies:")
        for col in ["rating", "ratingRecommendation"]:
            if col in ratings.columns:
                print(f"\n{col} counts:")
                print(ratings[col].value_counts())

    run_example_step("[5] Historical Analyst Ratings", example_ratings)

    # ------------------------------------------------------------------
    # Example 6: Fetch company profile
    # ------------------------------------------------------------------
    def example_profile():
        profile = loader.get_company_profile("AAPL")
        if not profile.empty:
            print(f"Retrieved company profile for {profile.iloc[0].get('symbol', 'Unknown')}")
            print("\n--- Full Profile Data ---")
            for col, val in profile.iloc[0].items():
                print(f"{col}: {val}")
        else:
            print("No profile data found.")

    run_example_step("[6] Company Profile", example_profile)

    # ------------------------------------------------------------------
    # Example 7: Historical sector performance
    # ------------------------------------------------------------------
    def example_sector():
        sector_perf = loader.get_historical_sector_performance("Technology", start=start_date, end=end_date)
        print(f"Retrieved {len(sector_perf)} records for Technology sector")
        if not sector_perf.empty and "date" in sector_perf.columns:
            print(f"Date range: {sector_perf['date'].min()} to {sector_perf['date'].max()}")
            print(f"Columns: {sector_perf.columns.tolist()}")
        print(sector_perf.head())

    run_example_step("[7] Historical Sector Performance", example_sector)

    # ------------------------------------------------------------------
    # Example 8: Historical industry performance
    # ------------------------------------------------------------------
    def example_industry():
        industry_perf = loader.get_historical_industry_performance("Biotechnology", start=start_date, end=end_date)
        print(f"Retrieved {len(industry_perf)} records for Biotechnology industry")
        if not industry_perf.empty and "date" in industry_perf.columns:
            print(f"Date range: {industry_perf['date'].min()} to {industry_perf['date'].max()}")
            print(f"Columns: {industry_perf.columns.tolist()}")
        print(industry_perf.head())

    run_example_step("[8] Historical Industry Performance", example_industry)

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
