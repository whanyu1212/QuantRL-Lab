# Data Module

The `quantrl_lab.data` package covers the full data path from provider access to model-ready feature matrices.
It includes source loaders, capability-aware source registration, technical indicators, composable processing,
dataset partitioning, and shared utilities for request handling and dataframe normalization.

## Package Structure

### Core Entry Points
- **`source_registry.py`**: `DataSourceRegistry` for discovering and instantiating configured data sources.
- **`interface.py`**: source capability contracts and feature declarations.
- **`exceptions.py`**: typed data-source and validation exceptions.
- **`config.py`**: shared defaults and data-specific configuration constants.

### Data Sources
- **`sources/`**: provider loaders for Alpaca, YFinance, Alpha Vantage, and FMP.
- Capability metadata is explicit per loader, so registry filtering does not rely on runtime guessing.
- Request wrappers in `utils/request_utils.py` and `utils/async_request_utils.py` normalize provider failures into
  typed errors such as `AuthenticationError`, `RateLimitError`, and `APIConnectionError`.

### Feature Engineering
- **`indicators/`**: indicator registry plus built-in OHLCV indicators.
- **`processing/`**: `DataProcessor`, `DataPipeline`, typed pipeline config objects, metadata tracking,
  and pipeline steps for indicators, sentiment, analyst data, market context, cleanup, and cross-sectional features.
- **`partitioning/`**: ratio- and date-based splitters for train / validation / test workflows.

### Utilities
- **`utils/`**: date parsing, date alignment, symbol handling, response validation, async helpers,
  dataframe normalization, and request retry logic.

## Recommended Usage

### 1. Fetch raw market data

```python
from quantrl_lab.data import DataSourceRegistry

registry = DataSourceRegistry()
ohlcv_df = registry.get_historical_ohlcv_data(
    symbols="AAPL",
    start="2023-01-01",
    end="2023-12-31",
    timeframe="1d",
)
```

### 2. Build a typed processing config

```python
from quantrl_lab.data import DataProcessor
from quantrl_lab.data.processing import ProcessingPipelineConfig, SplitConfig

processor = DataProcessor(ohlcv_data=ohlcv_df)

pipeline_config = ProcessingPipelineConfig(
    indicators=["SMA", {"RSI": {"window": 14}}, "MACD"],
    split=SplitConfig(splits={"train": 0.7, "test": 0.3}),
    verbose=False,
)

split_data, metadata = processor.data_processing_pipeline(pipeline_config=pipeline_config)
```

### 3. Use alpha research outside the pipeline

Alpha selection is no longer performed inside `DataProcessor`. The recommended pattern is:

1. run alpha research on raw data
2. inspect or filter the suggested indicators
3. pass those indicators into `ProcessingPipelineConfig` or `DataProcessor`

```python
from quantrl_lab.alpha_research import build_processing_config_from_alpha_selection
from quantrl_lab.data import DataProcessor

processor = DataProcessor(ohlcv_data=ohlcv_df)
pipeline_config, selection_meta = build_processing_config_from_alpha_selection(
    raw_data=ohlcv_df,
    alpha_selection_config={"metric": "ic", "top_k": 5, "selection_mode": "feature"},
    split_config={"train": 0.7, "test": 0.3},
)

split_data, metadata = processor.data_processing_pipeline(pipeline_config=pipeline_config)
```

## Key Behaviors

- `DataProcessor` keeps the pipeline assembly high-level, but `build_pipeline()` lets you inspect the exact steps first.
- Optional enrichment data such as analyst estimates and market context is treated as optional features rather than
  forcing global row drops.
- Split outputs preserve `Symbol` by default for panel workflows when that identity is still needed downstream.
- `alpha_selection_config` on `DataProcessor` is deprecated and rejected; use `quantrl_lab.alpha_research.integration`
  helpers instead.
- Ratio splitting preserves all rows and requires the configured ratios to sum to `1.0`.

## Related READMEs

- [processing/README.md](./processing/README.md): pipeline assembly, steps, configs, and metadata
- [indicators/README.md](./indicators/README.md): built-in technical indicator coverage and rationale
