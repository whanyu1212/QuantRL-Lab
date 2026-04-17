# Data Processing Pipeline

This module implements a composable, builder-pattern data processing pipeline for financial time-series data. It transforms raw OHLCV (Open-High-Low-Close-Volume) data into feature-rich datasets ready for reinforcement learning or other analysis.

## Architecture

The system is built around three core concepts:

1.  **DataPipeline**: A container that manages a sequence of processing steps using a builder pattern (method chaining via `add_step()`).
2.  **ProcessingStep**: A protocol defining a single transformation unit.
3.  **DataProcessor**: A high-level facade that coordinates data loading, pipeline construction, splitting, and execution.

### Components

- **`pipeline.py`**: Contains `DataPipeline`, responsible for chaining steps and executing them sequentially.
- **`processor.py`**: Contains `DataProcessor`, the main entry point for most use cases. It handles data sources, splitting (train/test), and configuration. Also defines `ProcessingMetadata`.
- **`steps/`**: Directory containing concrete implementations of processing steps.
- **`features/`**: Logic for specific feature generation (e.g., technical indicators, sentiment) used by steps.

## Usage

### High-Level API (Recommended)

The `DataProcessor` class builds a `DataPipeline` internally based on the arguments you pass:

```python
from quantrl_lab.data.processing.processor import DataProcessor

processor = DataProcessor(
    ohlcv_data=df,
    news_data=news_df,           # optional
    analyst_grades=grades_df,    # optional
    analyst_ratings=ratings_df,  # optional
    sector_performance=sector_df,  # optional
    industry_performance=industry_df,  # optional
)

processed_data, metadata = processor.data_processing_pipeline(
    indicators=["SMA", "RSI"],
    fillna_strategy="neutral",
    split_config={"train": 0.7, "test": 0.3},
)
```

Steps are added conditionally — sentiment enrichment only runs if `news_data` is provided, analyst estimates only if `analyst_grades`/`analyst_ratings` are provided, etc.

For a more explicit assembly flow, use the typed config objects plus `build_pipeline()`:

```python
from quantrl_lab.data.processing import (
    CrossSectionalConfig,
    ProcessingPipelineConfig,
    SplitConfig,
)

pipeline_config = ProcessingPipelineConfig(
    indicators=["SMA", "RSI"],
    cross_sectional=CrossSectionalConfig(columns=["Close"], methods=["rank"]),
    split=SplitConfig(splits={"train": 0.7, "test": 0.3}),
)

pipeline = processor.build_pipeline(pipeline_config=pipeline_config)
print(pipeline.describe())
processed_data, metadata = processor.data_processing_pipeline(pipeline_config=pipeline_config)
```

### Low-Level Builder API (Custom Workflows)

For full control, construct a `DataPipeline` manually and chain steps using the builder pattern:

```python
from quantrl_lab.data.processing.pipeline import DataPipeline
from quantrl_lab.data.processing.steps import (
    TechnicalIndicatorStep,
    NumericConversionStep,
    ColumnCleanupStep,
)

pipeline = (
    DataPipeline()
    .add_step(TechnicalIndicatorStep(indicators=["SMA", {"RSI": {"window": 14}}]))
    .add_step(NumericConversionStep())
    .add_step(ColumnCleanupStep(columns_to_drop=["Date"]))
)

result_df, metadata = pipeline.execute(raw_df)
```

Each `add_step()` call returns the pipeline itself, so calls can be chained fluently. Steps execute in the order they were added.

### Integration with DataSourceRegistry

Fetch data from configured sources (Alpaca, YFinance, etc.) and feed it into the processor:

```python
from quantrl_lab.data.source_registry import DataSourceRegistry
from quantrl_lab.data.processing.processor import DataProcessor

# 1. Get data from registry
registry = DataSourceRegistry()

ohlcv_df = registry.get_historical_ohlcv_data(
    symbols="AAPL",
    start="2023-01-01",
    end="2023-12-31",
    timeframe="1d",
)

news_df = registry.get_news_data(
    symbols="AAPL",
    start="2023-01-01",
    end="2023-12-31",
)

# 2. Initialize Processor
processor = DataProcessor(ohlcv_data=ohlcv_df, news_data=news_df)

# 3. Run Pipeline
processed_data, metadata = processor.data_processing_pipeline(
    indicators=["SMA", "RSI"],
    split_config={"train": 0.7, "test": 0.3},
)
```

## Available Steps

| Step Class | Description |
|---|---|
| `TechnicalIndicatorStep` | Adds technical indicators (SMA, RSI, MACD, etc.) via `TechnicalFeatureGenerator`. |
| `AnalystEstimatesStep` | Merges analyst grade and rating data into the DataFrame. |
| `MarketContextStep` | Adds sector and industry relative performance features. |
| `SentimentEnrichmentStep` | Merges news data and computes sentiment scores via a configurable provider. |
| `NumericConversionStep` | Converts specified (or all object-type) columns to numeric, skipping date columns. |
| `ColumnCleanupStep` | Drops unwanted columns (e.g., raw dates, symbols) to prepare for model input. |

### Using Alpha Research to Choose Indicators (Recommended)

The `alpha_research` module can suggest the best indicators for your data. The recommended
pattern is to run `AlphaSelector` **outside** the pipeline so you can inspect, filter, or
augment the suggestions before they enter the processing pipeline:

```python
from quantrl_lab.alpha_research import AlphaSelector
from quantrl_lab.data.processing.pipeline import DataPipeline
from quantrl_lab.data.processing.steps import (
    TechnicalIndicatorStep,
    NumericConversionStep,
    ColumnCleanupStep,
)

# 1. Ask alpha research for suggestions (outside the pipeline)
selector = AlphaSelector(raw_df, verbose=True)
suggested = selector.suggest_indicators(metric="sharpe_ratio", top_k=5, selection_mode="strategy")
# Returns e.g.: [{"RSI": {"window": 14}}, {"SMA": {"window": 50}}]

# 2. User decision — inspect, filter, or mix with manual picks
indicators = suggested + [{"MACD": {"fast": 12, "slow": 26, "signal": 9}}]

# 3. Build pipeline with the chosen indicators
pipeline = (
    DataPipeline()
    .add_step(TechnicalIndicatorStep(indicators=indicators))
    .add_step(NumericConversionStep())
    .add_step(ColumnCleanupStep())
)

result_df, metadata = pipeline.execute(raw_df)
```

**Why decoupled?** The decoupled pattern gives you full control: skip indicators you don't trust,
add domain-specific ones, or skip alpha research entirely and just pass `["SMA", "RSI"]` directly.

### Default Pipeline Order (inside `DataProcessor`)

When using the high-level API, steps are added in this order:

1. `TechnicalIndicatorStep` — always added (no-ops if no indicators provided)
2. `AnalystEstimatesStep` — only if analyst data is provided
3. `MarketContextStep` — only if sector/industry data is provided
4. `SentimentEnrichmentStep` — only if `news_data` is provided
5. `NumericConversionStep` — always added
6. `CrossSectionalStep` — only if cross-sectional config is provided
7. `ColumnCleanupStep` — always added

After pipeline execution, NaN rows are dropped (indicator warm-up periods), and data is optionally split.

## Metadata Tracking

The pipeline tracks metadata through `ProcessingMetadata`, which records:
- Original and final data shapes
- Applied technical indicators
- Dropped columns
- Date ranges for splits
- Flags for analyst data, market context, and sentiment enrichment

This ensures reproducibility and allows downstream components to know how the data was transformed.

## Extending the Pipeline

To create a custom processing step, implement the `ProcessingStep` protocol defined in `steps/base.py`:

```python
from quantrl_lab.data.processing.processor import ProcessingMetadata
import pandas as pd

class MyCustomStep:
    def process(self, data: pd.DataFrame, metadata: ProcessingMetadata) -> pd.DataFrame:
        data["new_col"] = data["close"] * 2
        return data

    def get_step_name(self) -> str:
        return "My Custom Step"
```

Then add it to a pipeline:

```python
pipeline = (
    DataPipeline()
    .add_step(TechnicalIndicatorStep(indicators=["SMA"]))
    .add_step(MyCustomStep())
    .add_step(ColumnCleanupStep())
)

result_df, metadata = pipeline.execute(raw_df)
```
