# Alpha Research Module

The `alpha_research` module is a dedicated framework within QuantRL-Lab for discovering, validating, and analyzing trading signals (alphas) before they are deployed in Reinforcement Learning environments. It allows for fast, vectorized backtesting of technical indicators and statistical analysis of their predictive power.

## Overview

This module is designed to answer the question: *"Does this indicator have predictive power?"*

It decouples the signal discovery process from the full event-driven RL backtesting engine, allowing you to screen hundreds of potential features quickly. Validated signals can then be automatically converted into configuration for the `DataPipeline`.

Alpha selection supports two explicit modes:

- `selection_mode="feature"` ranks indicators by continuous score predictive power (`ic`, `rank_ic`, `mutual_info`).
- `selection_mode="strategy"` ranks indicators by discrete trading-rule behavior and backtest metrics (`sharpe_ratio`, `annual_return`, `ic`, etc.).

## Module Structure

The module is organized into the following components:

### Core Components
- **`models.py`**: Defines the core data structures:
    - `AlphaJob`: Configuration for a single research task (data, indicator, strategy parameters, and evaluation mode).
    - `AlphaResult`: The output of a job, containing performance metrics plus equity curves/signals for strategy mode or continuous scores for feature mode. The `error` field stores a traceback string (not an `Exception`) so results are safely picklable across `joblib` parallel workers.
- **`runner.py`**: Contains `AlphaRunner`, the execution engine that processes `AlphaJob`s (sequentially or in parallel) and calculates performance metrics. OHLCV columns are automatically normalised to title-case (`close` → `Close`) so data from lowercase sources is accepted without pre-processing.
- **`alpha_strategies.py`**: Implements vectorized trading logic (e.g., `MeanReversionStrategy`, `TrendFollowingStrategy`) used to translate raw indicator values into buy/sell signals for testing.

### Analysis & Optimization
- **`analysis.py`**: Contains `RobustnessTester` for performing parameter sensitivity analysis (grid search) and stability testing (sub-period analysis).
- **`ensemble.py`**: Contains `AlphaEnsemble` for combining multiple alpha signals into a single portfolio strategy using various weighting methods (Equal Weight, Inverse Volatility, Sharpe, IC).
- **`visualization.py`**: Contains `AlphaVisualizer` for generating professional-grade charts (Cumulative Returns, Drawdowns, IC Analysis) and HTML reports.

### Utilities
- **`indicator_research.py`**: Explicit research metadata for each built-in indicator, including default candidate grids, strategy wiring, and supported selection modes.
- **`integration.py`**: Adapters for converting alpha-selection output into `ProcessingPipelineConfig` objects for the data-processing layer.
- **`converters.py`**: Utilities to bridge the gap between this module and the rest of the system. Specifically, `results_to_pipeline_config` converts successful `AlphaResult` objects into a `DataPipeline` configuration.
- **`metrics.py`**: Signal quality functions: IC (Pearson), Rank IC (Spearman), autocorrelation, and turnover.
- **`registry.py`**: A registry system for managing available vectorized strategies.

## Usage Example

### 1. Define and Run Jobs

```python
import pandas as pd
from quantrl_lab.alpha_research import AlphaJob, AlphaRunner

# Load your data (DataFrame with Open, High, Low, Close, Volume)
# Column names are case-insensitive — lowercase OHLCV is auto-normalised.
data = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)

# Define research jobs
jobs = [
    AlphaJob(
        data=data,
        indicator_name="RSI",
        strategy_name="mean_reversion",
        indicator_params={"window": 14},
        strategy_params={"oversold": 30, "overbought": 70},
    ),
    AlphaJob(
        data=data,
        indicator_name="SMA",
        strategy_name="trend_following",
        indicator_params={"window": 50},
        strategy_params={},
    ),
]

# Run jobs (sequential or parallel via joblib)
runner = AlphaRunner(verbose=True)
results = runner.run_batch(jobs, n_jobs=-1)  # -1 = all cores

# Inspect results
for res in results:
    if res.status == "completed":
        print(f"{res.job.indicator_name}: Sharpe={res.metrics['sharpe_ratio']:.2f}, IC={res.metrics['ic']:.4f}")
    else:
        # res.error is a traceback string — safe to print or log
        print(f"{res.job.indicator_name}: FAILED\n{res.error}")
```

For feature-screening instead of backtest metrics, set `evaluation_mode="feature"` on each `AlphaJob`.

Strategy-mode and feature-mode results are both valid `AlphaResult`s, but not every downstream tool consumes both:

- `AlphaEnsemble` expects strategy-mode results with equity curves.
- cumulative-return and drawdown plots in `AlphaVisualizer` also expect strategy-mode results.
- IC-style ranking and pipeline configuration conversion work with either mode, as long as the selected metric exists.

### 2. Robustness Testing

Use the explicit `indicator_param_grid` / `strategy_param_grid` split to ensure parameters are always routed to the correct `AlphaJob` dict. The legacy flat `param_grid` is still accepted for backward compatibility but can mis-route new indicator params.

```python
from quantrl_lab.alpha_research import RobustnessTester

tester = RobustnessTester(runner)

# Explicit split (recommended) — no routing ambiguity
sensitivity_df = tester.parameter_sensitivity(
    jobs[0],
    indicator_param_grid={"window": [10, 14, 20]},   # → AlphaJob.indicator_params
    strategy_param_grid={"oversold": [20, 25, 30]},   # → AlphaJob.strategy_params
    n_jobs=-1,
)
print(sensitivity_df.sort_values("sharpe_ratio", ascending=False))
```

### 3. Visualization

`AlphaVisualizer` is available directly from the top-level package:

```python
from quantrl_lab.alpha_research import AlphaVisualizer

viz = AlphaVisualizer()
viz.generate_html_report(results, "alpha_report.html")
```

### 4. Integration with DataPipeline

```python
from quantrl_lab.alpha_research import build_processing_config_from_alpha_selection
from quantrl_lab.data.processing import DataProcessor

# Build a ready-to-run processing config using train-only alpha selection
processing_config, selection_meta = build_processing_config_from_alpha_selection(
    raw_data=data,
    alpha_selection_config={"metric": "ic", "threshold": 0.02, "top_k": 5, "selection_mode": "feature"},
    split_config={"train": 0.7, "test": 0.3},
)

# Execute the processing plan
processor = DataProcessor(ohlcv_data=data)
processed_data, metadata = processor.data_processing_pipeline(
    pipeline_config=processing_config
)
```

If you already have `AlphaResult` objects and just want indicator specs, `results_to_pipeline_config(...)`
still converts them into the `indicators=[...]` format expected by `DataProcessor`.

### 5. Williams %R / Custom Indicator Scales

`MeanReversionStrategy` accepts an explicit `indicator_scale` to avoid runtime guessing:

```python
from quantrl_lab.alpha_research.alpha_strategies import MeanReversionStrategy

# Default: RSI / MFI style (0-100 scale, center=50)
rsi_strat = MeanReversionStrategy(indicator_col="RSI_14")

# Williams %R (-100..0 scale, center=-50)
willr_strat = MeanReversionStrategy(indicator_col="WILLR_14", indicator_scale="williams_r")
```

| `indicator_scale` | Formula | Indicators |
|---|---|---|
| `"0_100"` (default) | `(50 - value) / 50` | RSI, MFI, Stochastic |
| `"williams_r"` | `(-50 - value) / 50` | Williams %R |

## Key Metrics

- **IC (Information Coefficient):** Pearson correlation between the signal and forward returns.
- **Rank IC:** Spearman rank correlation — more robust to outliers, generally preferred.
- **Sharpe Ratio:** Risk-adjusted return (annualised, 2% risk-free rate).
- **Sortino Ratio:** Risk-adjusted return penalising only downside volatility.
- **Calmar Ratio:** Annualised return divided by maximum drawdown.
- **Max Drawdown:** Maximum peak-to-trough loss.
- **Mutual Information:** Captures non-linear predictive relationships between signal and returns.
