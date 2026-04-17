"""
Example: Automatic Indicator Selection Workflow.

This script demonstrates how to:
1. Fetch historical data for a stock.
2. Use AlphaSelector to test a grid of technical indicators.
3. Select the best performing indicators based on Sharpe Ratio.
4. Configure the DataProcessor with these optimized indicators.
"""

from datetime import datetime, timedelta

from rich.console import Console

from quantrl_lab.alpha_research import AlphaSelector
from quantrl_lab.data.processing.processor import DataProcessor
from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader

console = Console()


def main():
    # 1. Fetch Data
    console.rule("[bold blue]1. Fetching Data[/bold blue]")
    symbol = "AAPL"
    loader = YFinanceDataLoader()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years of data

    console.print(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}...")
    try:
        df = loader.get_historical_ohlcv_data(symbols=[symbol], start=start_date, end=end_date, timeframe="1d")
    except Exception as e:
        console.print(f"[red]Failed to fetch data: {e}[/red]")
        return

    if df.empty:
        console.print("[red]No data fetched. Exiting.[/red]")
        return

    console.print(f"Fetched {len(df)} rows.")

    # 2. Select Best Indicators
    console.rule("[bold blue]2. Running Alpha Research[/bold blue]")

    # Initialize selector with raw data
    selector = AlphaSelector(df)

    # Suggest indicators based on Sharpe Ratio
    # We test a default grid of RSI, MACD, SMA, BB, etc.
    console.print("Testing candidate indicators...")
    best_indicators = selector.suggest_indicators(
        metric="sharpe_ratio",
        threshold=0.0,
        top_k=3,
        selection_mode="strategy",
    )

    if not best_indicators:
        console.print("[yellow]No indicators met the threshold. Using default fallback.[/yellow]")
        best_indicators = [{"SMA": {"window": 50}}]
    else:
        console.print(f"\n[bold green]Selected Top {len(best_indicators)} Indicators:[/bold green]")
        for ind in best_indicators:
            console.print(f"  - {ind}")

    # 3. Process Data with Selected Indicators
    console.rule("[bold blue]3. Running Data Pipeline[/bold blue]")

    processor = DataProcessor(ohlcv_data=df)

    # The pipeline will now calculate ONLY the selected indicators
    processed_df, metadata = processor.data_processing_pipeline(indicators=best_indicators)

    console.print("[green]Processing Complete![/green]")
    console.print(f"Final Data Shape: {processed_df.shape}")
    console.print(f"Columns: {list(processed_df.columns)}")

    # Verify the selected indicators are in the columns
    # This confirms the integration worked: AlphaSelector output fed into DataProcessor.


if __name__ == "__main__":
    main()
