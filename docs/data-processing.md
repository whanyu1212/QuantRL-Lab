# Data Processing Pipeline

QuantRL-Lab provides two ways to process raw OHLCV data before feeding it into a trading environment:

- **`DataProcessor.data_processing_pipeline()`** — high-level, batteries-included method that handles the full sequence automatically
- **`DataPipeline` builder** — low-level, step-by-step composition for full control

Both produce the same output: a cleaned, feature-enriched DataFrame (and optional train/test splits) along with a `ProcessingMetadata` object tracking every transformation applied.

---

## Quick Reference: Available Steps

| Step Class | What it does | Key parameters |
|---|---|---|
| `TechnicalIndicatorStep` | Computes and appends technical indicators | `indicators` (list of str or dict) |
| `NumericConversionStep` | Converts object columns to numeric types | `columns` (auto-detect if `None`) |
| `ColumnCleanupStep` | Drops unwanted columns | `columns_to_drop`, `keep_date` |
| `SentimentEnrichmentStep` | Merges news sentiment scores | `news_data`, `provider`, `fillna_strategy` |
| `AnalystEstimatesStep` | Merges analyst grades / ratings | `grades_df`, `ratings_df` |
| `MarketContextStep` | Merges sector / industry performance | `sector_perf_df`, `industry_perf_df` |
| `CrossSectionalStep` | Computes cross-sectional features across a basket | `columns`, `methods`, `date_column` |

---

## High-Level Usage: `DataProcessor`

Use `DataProcessor` when you want a single call to handle everything — indicators, sentiment, analyst data, numeric conversion, cleanup, and optional splitting.

### Typed config assembly

`DataProcessor` now supports typed pipeline config objects when you want a clearer, more inspectable assembly flow:

```python
from quantrl_lab.data.processing import (
    CleanupConfig,
    CrossSectionalConfig,
    ProcessingPipelineConfig,
    SplitConfig,
)

pipeline_config = ProcessingPipelineConfig(
    indicators=[{"SMA": {"window": 20}}, "RSI"],
    cross_sectional=CrossSectionalConfig(columns=["Close"], methods=["rank"]),
    split=SplitConfig(splits={"train": 0.7, "test": 0.3}),
    cleanup=CleanupConfig(),
    strict_indicators=True,
)

pipeline = processor.build_pipeline(pipeline_config=pipeline_config)
print(pipeline.get_step_names())
print(pipeline.describe())

processed, metadata = processor.data_processing_pipeline(pipeline_config=pipeline_config)
```

### Minimal example

```python
from quantrl_lab.data.sources import YFinanceDataLoader
from quantrl_lab.data.processing.processor import DataProcessor

loader = YFinanceDataLoader()
raw_df = loader.get_historical_ohlcv_data(
    symbols=["AAPL"], start="2022-01-01", end="2024-01-01"
)

processor = DataProcessor(ohlcv_data=raw_df)
processed_df, metadata = processor.data_processing_pipeline(
    indicators=["SMA", "RSI", "MACD", "CMF"]
)

print(processed_df.shape)
print(metadata["technical_indicators"])
```

### With train/test split (ratio-based)

```python
processed, metadata = processor.data_processing_pipeline(
    indicators=["SMA", "EMA", "RSI", "BB"],
    split_config={"train": 0.7, "val": 0.1, "test": 0.2},
)

train_df = processed["train"]
val_df   = processed["val"]
test_df  = processed["test"]

print(metadata["date_ranges"])
# {'train': {'start': '2022-01-03', 'end': '2023-03-15'}, ...}
```

### With train/test split (date-based)

```python
processed, metadata = processor.data_processing_pipeline(
    indicators=["RSI", "ATR"],
    split_config={
        "train": ("2021-01-01", "2022-12-31"),
        "test":  ("2023-01-01", "2023-12-31"),
    },
)
```

### With custom indicator parameters

Indicators can be specified as plain strings (use defaults) or dicts with explicit parameters:

```python
indicators = [
    "RSI",                             # default window=14
    {"SMA": {"window": 50}},           # custom window
    {"EMA": {"window": 20}},
    {"BB": {"window": 20, "num_std": 2.5}},
    {"MACD": {"fast": 8, "slow": 21, "signal": 9}},
    {"SUPERTREND": {"window": 10, "multiplier": 3.0}},
]

processed_df, metadata = processor.data_processing_pipeline(indicators=indicators)
```

### With news sentiment

Requires `news_data` to be passed to `DataProcessor`. The default sentiment provider is `HuggingFaceProvider` (FinBERT).

```python
from quantrl_lab.data.sources import AlpacaDataLoader

loader = AlpacaDataLoader()
raw_df   = loader.get_historical_ohlcv_data(["AAPL"], "2023-01-01", "2024-01-01")
news_df  = loader.get_news_data(symbols="AAPL", start="2023-01-01", end="2024-01-01")

processor = DataProcessor(ohlcv_data=raw_df, news_data=news_df)
processed_df, metadata = processor.data_processing_pipeline(
    indicators=["RSI", "SMA"],
    fillna_strategy="fill_forward",   # or "neutral" (default, fills with 0.0)
)

print(metadata["news_sentiment_applied"])   # True
```

### With analyst data (from FMP)

```python
from quantrl_lab.data.sources import FMPDataSource

fmp = FMPDataSource()
grades_df  = fmp.get_historical_grades("AAPL")
ratings_df = fmp.get_historical_rating("AAPL", limit=100)

processor = DataProcessor(
    ohlcv_data=raw_df,
    analyst_grades=grades_df,
    analyst_ratings=ratings_df,
)
processed_df, metadata = processor.data_processing_pipeline(indicators=["RSI"])

print(metadata["analyst_data_applied"])   # True
```

### With sector / industry context (from FMP)

```python
sector_df   = fmp.get_historical_sector_performance("Technology")
industry_df = fmp.get_historical_industry_performance("Software")

processor = DataProcessor(
    ohlcv_data=raw_df,
    sector_performance=sector_df,
    industry_performance=industry_df,
)
processed_df, metadata = processor.data_processing_pipeline(indicators=["RSI", "SMA"])

# New columns: sector_changesPercentage, industry_changesPercentage, etc.
```

### Loading indicators from a config file

For reproducible experiments, define indicators in YAML or JSON and load them:

=== "YAML"

    ```yaml
    # indicators.yaml
    indicators:
      - SMA
      - RSI
      - EMA:
          window: 20
      - BB:
          window: 20
          num_std: 2.0
      - ROC:
          window: 12
    ```

=== "JSON"

    ```json
    {
      "indicators": [
        "SMA",
        "RSI",
        {"EMA": {"window": 20}},
        {"BB": {"window": 20, "num_std": 2.0}},
        {"ROC": {"window": 12}}
      ]
    }
    ```

```python
indicators = DataProcessor.load_indicators("indicators.yaml")
processed_df, metadata = processor.data_processing_pipeline(indicators=indicators)
```

---

## Low-Level Usage: `DataPipeline`

Use `DataPipeline` when you need precise control over the order of steps, want to skip steps, or are building a custom pipeline outside of `DataProcessor`.

### Minimal pipeline

```python
from quantrl_lab.data.processing.pipeline import DataPipeline
from quantrl_lab.data.processing.steps import (
    TechnicalIndicatorStep,
    NumericConversionStep,
    ColumnCleanupStep,
)

pipeline = (
    DataPipeline()
    .add_step(TechnicalIndicatorStep(indicators=["SMA", "RSI"]))
    .add_step(NumericConversionStep())
    .add_step(ColumnCleanupStep())
)

processed_df, metadata = pipeline.execute(raw_df, symbol="AAPL")
print(processed_df.columns.tolist())
print(metadata.technical_indicators)   # ["SMA", "RSI"]
```

### Inspect the pipeline before running

```python
pipeline = (
    DataPipeline()
    .add_step(TechnicalIndicatorStep(["RSI", "MACD"]))
    .add_step(ColumnCleanupStep(columns_to_drop=["Symbol"]))
)

print(pipeline)
# DataPipeline(2 steps: ['Technical Indicators', 'Column Cleanup'])

print(len(pipeline))   # 2
```

`DataPipeline.describe()` returns a structured summary:

```python
{
    "step_count": 2,
    "steps": [
        {"index": 1, "name": "Technical Indicators", "type": "TechnicalIndicatorStep"},
        {"index": 2, "name": "Column Cleanup", "type": "ColumnCleanupStep"},
    ],
}
```

---

## Step Reference

### `TechnicalIndicatorStep`

Adds technical indicator columns to the DataFrame. Pass indicator names as strings for default parameters, or dicts for custom ones.

```python
from quantrl_lab.data.processing.steps import TechnicalIndicatorStep

step = TechnicalIndicatorStep(
    indicators=[
        "SMA",                             # SMA with default window
        "RSI",                             # RSI with default window=14
        {"EMA": {"window": 20}},
        {"BB": {"window": 20, "num_std": 2.0}},
        {"MACD": {"fast": 12, "slow": 26, "signal": 9}},
        {"CMF": {"window": 20}},
        {"SUPERTREND": {"window": 10, "multiplier": 3.0}},
        "ATR",
        "OBV",
    ]
)
```

To see all registered indicators:

```python
from quantrl_lab.data.indicators.registry import IndicatorRegistry

print(IndicatorRegistry.list_all())
# ['SMA', 'EMA', 'RSI', 'MACD', 'ATR', 'BB', 'ROC', 'PPO', 'CMF', 'SUPERTREND', ...]
```

---

### `NumericConversionStep`

Converts `object`-dtype columns to numeric. Automatically skips date columns. Useful after merging external data (analyst, sentiment) that may come in as strings.

```python
from quantrl_lab.data.processing.steps import NumericConversionStep

# Auto-detect all object columns (excluding date columns)
step = NumericConversionStep()

# Or convert specific columns only
step = NumericConversionStep(columns=["analyst_score", "rating_score"])
```

---

### `ColumnCleanupStep`

Drops columns before the DataFrame is passed to the environment. By default, drops `Date`, `Timestamp`, and `Symbol` — columns the RL agent should not see directly.

```python
from quantrl_lab.data.processing.steps import ColumnCleanupStep

# Default: drop Date, Timestamp, Symbol
step = ColumnCleanupStep()

# Custom columns
step = ColumnCleanupStep(columns_to_drop=["Date", "Symbol", "vwap"])

# Keep date (e.g., if you need to split afterwards)
step = ColumnCleanupStep(keep_date=True)
```

---

### `SentimentEnrichmentStep`

Merges per-day news sentiment scores into the OHLCV DataFrame. Missing days (no news) are filled using the chosen strategy.

```python
from quantrl_lab.data.processing.steps import SentimentEnrichmentStep
from quantrl_lab.data.processing.sentiment import HuggingFaceProvider, SentimentConfig

step = SentimentEnrichmentStep(
    news_data=news_df,
    provider=HuggingFaceProvider(),     # default; uses FinBERT
    config=SentimentConfig(),           # window size, aggregation, etc.
    fillna_strategy="neutral",          # "neutral" → 0.0, "fill_forward" → last known score
)
```

!!! note
    `SentimentEnrichmentStep` requires the `ml` optional dependency group:
    ```bash
    uv sync --extra ml
    ```

---

### `AnalystEstimatesStep`

Merges historical analyst grades and ratings into the DataFrame. Analyst updates are sparse (monthly), so values are forward-filled to represent the "current" consensus at each timestep. Matching is done at monthly granularity to handle timing mismatches between analyst reports and trading days.

```python
from quantrl_lab.data.processing.steps import AnalystEstimatesStep
from quantrl_lab.data.sources import FMPDataSource

fmp = FMPDataSource()

step = AnalystEstimatesStep(
    grades_df=fmp.get_historical_grades("AAPL"),
    ratings_df=fmp.get_historical_rating("AAPL", limit=100),
)
```

Output columns include analyst grade fields and rating scores merged on the closest month. Pass only one of `grades_df` / `ratings_df` if you only need one.

---

### `MarketContextStep`

Merges sector or industry performance into the DataFrame so the agent can observe broad market conditions alongside individual stock data.

```python
from quantrl_lab.data.processing.steps import MarketContextStep
from quantrl_lab.data.sources import FMPDataSource

fmp = FMPDataSource()

step = MarketContextStep(
    sector_perf_df=fmp.get_historical_sector_performance("Technology"),
    industry_perf_df=fmp.get_historical_industry_performance("Software"),
)
```

Added columns are prefixed with `sector_` and `industry_` respectively. Pass only the data you have — both arguments are optional.

---

### `CrossSectionalStep`

Computes cross-sectional features across a **basket of stocks** (panel data). Groups by date and normalises each feature relative to all stocks present on that day. Only meaningful with 2+ symbols; silently bypasses if a single symbol is detected.

The step now works with either:
- a `DatetimeIndex`
- a date-like column such as `Date` / `Timestamp`
- an explicit `date_column=...` override

```python
from quantrl_lab.data.processing.steps import CrossSectionalStep

# Supported methods: "zscore", "rank", "mean_centered"
step = CrossSectionalStep(
    columns=["RSI_14", "volume", "close"],
    methods=["zscore", "rank"],
)

# Output columns: RSI_14_cs_zscore, RSI_14_cs_rank, volume_cs_zscore, ...
```

| Method | Formula | Use when |
|---|---|---|
| `zscore` | `(x − μ) / σ` | You want mean-zero, unit-variance relative signal |
| `rank` | Percentile rank [0, 1] | You want a robust, outlier-resistant rank signal |
| `mean_centered` | `x − μ` | You want direction relative to the cross-section mean |

---

## Full Custom Pipeline Example

Combining all steps for a multi-source enrichment pipeline:

```python
from quantrl_lab.data.processing.pipeline import DataPipeline
from quantrl_lab.data.processing.steps import (
    TechnicalIndicatorStep,
    AnalystEstimatesStep,
    MarketContextStep,
    SentimentEnrichmentStep,
    NumericConversionStep,
    ColumnCleanupStep,
)
from quantrl_lab.data.sources import AlpacaDataLoader, FMPDataSource

alpaca = AlpacaDataLoader()
fmp    = FMPDataSource()

raw_df     = alpaca.get_historical_ohlcv_data(["AAPL"], "2022-01-01", "2024-01-01")
news_df    = alpaca.get_news_data("AAPL", "2022-01-01", "2024-01-01")
grades_df  = fmp.get_historical_grades("AAPL")
ratings_df = fmp.get_historical_rating("AAPL", limit=200)
sector_df  = fmp.get_historical_sector_performance("Technology")

pipeline = (
    DataPipeline()
    .add_step(TechnicalIndicatorStep(indicators=[
        "RSI",
        {"SMA": {"window": 50}},
        {"EMA": {"window": 20}},
        "ATR",
        "OBV",
    ]))
    .add_step(AnalystEstimatesStep(grades_df=grades_df, ratings_df=ratings_df))
    .add_step(MarketContextStep(sector_perf_df=sector_df))
    .add_step(SentimentEnrichmentStep(news_data=news_df, fillna_strategy="fill_forward"))
    .add_step(NumericConversionStep())
    .add_step(ColumnCleanupStep())
)

processed_df, metadata = pipeline.execute(raw_df, symbol="AAPL")

print(processed_df.shape)
print(f"Indicators: {metadata.technical_indicators}")
print(f"Analyst data: {metadata.analyst_data_applied}")
print(f"Sentiment: {metadata.news_sentiment_applied}")
print(f"Market context: {metadata.market_context_applied}")
```

---

## Inspecting `ProcessingMetadata`

Every pipeline execution returns a `ProcessingMetadata` object (or its dict form via `to_dict()`):

```python
processed_df, metadata = pipeline.execute(raw_df)

print(metadata.original_shape)          # (504, 7)
print(metadata.final_shapes)            # {'processed': (480, 23)}
print(metadata.technical_indicators)    # ['RSI', {'SMA': {'window': 50}}, ...]
print(metadata.columns_dropped)         # ['Date', 'Timestamp', 'Symbol']
print(metadata.news_sentiment_applied)  # True
print(metadata.analyst_data_applied)    # True
print(metadata.market_context_applied)  # True

# Serialize for logging / experiment tracking
import json
print(json.dumps(metadata.to_dict(), indent=2, default=str))
```

When using `DataProcessor.data_processing_pipeline()`, the metadata is returned as a plain dict:

```python
processed_df, metadata_dict = processor.data_processing_pipeline(indicators=["RSI"])
print(metadata_dict["date_ranges"])
print(metadata_dict["final_shapes"])
```
