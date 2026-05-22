"""
Example: Demonstrating Protocol-Based Feature Detection.

This example shows how to use runtime protocol checking to detect which
capabilities a data source supports.
"""

from quantrl_lab.data.exceptions import AuthenticationError
from quantrl_lab.data.interface import (
    AnalystDataCapable,
    CompanyProfileCapable,
    FundamentalDataCapable,
    HistoricalDataCapable,
    LiveDataCapable,
    MacroDataCapable,
    NewsDataCapable,
    SectorDataCapable,
    StreamingCapable,
)
from quantrl_lab.data.sources import (
    AlpacaDataLoader,
    AlphaVantageDataLoader,
    FMPDataSource,
    YFinanceDataLoader,
)


def check_protocols(loader, name: str):
    """Check which protocols a data source implements."""
    print(f"\n{name}")
    print("=" * 60)

    protocols = [
        ("Historical OHLCV", HistoricalDataCapable),
        ("Live Data", LiveDataCapable),
        ("News Data", NewsDataCapable),
        ("Fundamental Data", FundamentalDataCapable),
        ("Macro Data", MacroDataCapable),
        ("Analyst Data", AnalystDataCapable),
        ("Sector Data", SectorDataCapable),
        ("Company Profile", CompanyProfileCapable),
        ("Streaming", StreamingCapable),
    ]

    for protocol_name, protocol in protocols:
        supported = isinstance(loader, protocol)
        status = "✓" if supported else "✗"
        print(f"  {status} {protocol_name}")

    # Use the supported_features property
    print("\nSupported Features (via supported_features property):")
    features = loader.supported_features
    for feature in features:
        print(f"  • {feature}")


def main():
    print("Protocol-Based Feature Detection Demonstration")
    print("=" * 60)

    # YFinance
    yf_loader = YFinanceDataLoader()
    check_protocols(yf_loader, "YFinance")

    # Alpaca (requires API keys in .env)
    try:
        alpaca_loader = AlpacaDataLoader()
        check_protocols(alpaca_loader, "Alpaca")
    except (ValueError, AuthenticationError):
        print("\nAlpaca: Skipped (API keys not configured)")

    # Alpha Vantage (requires API key in .env)
    try:
        av_loader = AlphaVantageDataLoader()
        check_protocols(av_loader, "Alpha Vantage")
    except (ValueError, AuthenticationError):
        print("\nAlpha Vantage: Skipped (API key not configured)")

    # FMP (requires API key in .env)
    try:
        fmp_loader = FMPDataSource()
        check_protocols(fmp_loader, "Financial Modeling Prep (FMP)")
    except (ValueError, AuthenticationError):
        print("\nFMP: Skipped (API key not configured)")

    # Demonstrate conditional usage based on protocol
    print("\n" + "=" * 60)
    print("Conditional Usage Example")
    print("=" * 60)

    # Example: Use sector data if available
    try:
        fmp_loader = FMPDataSource()
        if isinstance(fmp_loader, SectorDataCapable):
            print("\n✓ FMP supports sector data!")
            print("  Available methods:")
            print("  - get_historical_sector_performance()")
            print("  - get_historical_industry_performance()")

        if isinstance(fmp_loader, CompanyProfileCapable):
            print("\n✓ FMP supports company profiles!")
            print("  Available methods:")
            print("  - get_company_profile()")

        if isinstance(fmp_loader, AnalystDataCapable):
            print("\n✓ FMP supports analyst data!")
            print("  Available methods:")
            print("  - get_historical_grades()")
            print("  - get_historical_rating()")
    except (ValueError, AuthenticationError):
        print("\nFMP not configured")

    # Check using supports_feature() method
    print("\n" + "=" * 60)
    print("Using supports_feature() Method")
    print("=" * 60)

    yf = YFinanceDataLoader()
    print(f"\nYFinance supports 'historical_bars': {yf.supports_feature('historical_bars')}")
    print(f"YFinance supports 'sector_data': {yf.supports_feature('sector_data')}")

    try:
        fmp = FMPDataSource()
        print(f"\nFMP supports 'sector_data': {fmp.supports_feature('sector_data')}")
        print(f"FMP supports 'company_profile': {fmp.supports_feature('company_profile')}")
        print(f"FMP supports 'analyst_data': {fmp.supports_feature('analyst_data')}")
    except (ValueError, AuthenticationError):
        print("\nFMP not configured")

    print("\n" + "=" * 60)
    print("Protocol demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
