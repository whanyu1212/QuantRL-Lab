"""
Example: File-Based Data Processing Configuration.

This example demonstrates how to load indicator configurations from
external YAML files using the new DataProcessor.load_indicators utility.
"""

from quantrl_lab.data import DataProcessor, YFinanceDataLoader


def main():
    print("=" * 80)
    print("FILE-BASED DATA PROCESSING CONFIGURATION")
    print("=" * 80)

    # 1. Load data
    print("\n[1/3] Loading data from Yahoo Finance...")
    loader = YFinanceDataLoader()
    # Get 1 year of AAPL data
    from datetime import datetime, timedelta

    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    ohlcv_df = loader.get_historical_ohlcv_data(symbols="AAPL", start=start_date)

    # 2. Load indicators from files (supporting both YAML and JSON)
    yaml_config = "examples/pipelines/configs/indicators_sample.yaml"
    json_config = "examples/pipelines/configs/indicators_sample.json"

    print("\n[2/3] Loading configurations...")

    # Loading from YAML
    indicators_yaml = DataProcessor.load_indicators(yaml_config)
    print(f"✓ Loaded {len(indicators_yaml)} entries from YAML")

    # Loading from JSON
    indicators_json = DataProcessor.load_indicators(json_config)
    print(f"✓ Loaded {len(indicators_json)} entries from JSON")

    # For this run, we'll use the JSON one to demonstrate it works identically
    indicators = indicators_json
    print(f"\nApplying configurations from: {json_config}")
    for item in indicators:
        print(f"  - {item}")
    # 3. Process data
    print("\n[3/3] Running processing pipeline...")
    processor = DataProcessor(ohlcv_data=ohlcv_df)

    # The pipeline accepts the list of indicators loaded from YAML
    processed_data, metadata = processor.data_processing_pipeline(indicators=indicators)

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Original columns: {list(ohlcv_df.columns)}")
    print(f"Final Features: {len(processed_data.columns)}")

    # Preview some generated columns
    tech_cols = [c for c in processed_data.columns if c not in ohlcv_df.columns]
    print(f"\nSample of generated technical features ({len(tech_cols)} total):")
    for col in tech_cols[:10]:
        print(f"  - {col}")
    if len(tech_cols) > 10:
        print(f"  ... and {len(tech_cols) - 10} more")


if __name__ == "__main__":
    main()
