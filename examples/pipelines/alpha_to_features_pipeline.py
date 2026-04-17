"""
Integration example: Alpha Research → Feature Pipeline.

Covers the full chain from raw OHLCV data to clean, split DataFrames that
are ready to hand off to BacktestRunner / ExperimentJob:

  1. Synthetic OHLCV data (no API keys needed)
  2. AlphaSelector.suggest_indicators()  — discover alpha-carrying indicators
  3. converters.results_to_pipeline_config() — convert results → indicator specs
  4. DataProcessor.data_processing_pipeline()
       • indicators from alpha research
       • ratio-based 70/15/15 train/val/test split
       • ProcessingMetadata inspection
  5. Per-split summary table — shapes, date ranges, feature count
  6. Column listing confirming readiness for RL experiments

Run with:
    uv run python examples/pipelines/alpha_to_features_pipeline.py
"""

import warnings

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from quantrl_lab.alpha_research import INDICATOR_STRATEGY_MAP
from quantrl_lab.alpha_research.converters import results_to_pipeline_config
from quantrl_lab.alpha_research.models import AlphaJob
from quantrl_lab.alpha_research.runner import AlphaRunner
from quantrl_lab.alpha_research.selector import AlphaSelector
from quantrl_lab.data.processing import DataProcessor

console = Console()


# ── Synthetic data ──────────────────────────────────────────────────────────


def make_ohlcv(n: int = 756, seed: int = 42) -> pd.DataFrame:
    """Three years of synthetic daily OHLCV (no API key needed)."""
    np.random.seed(seed)
    log_returns = np.random.randn(n) * 0.012 + 0.0003  # slight upward drift
    close = 100.0 * np.exp(np.cumsum(log_returns))
    noise = np.random.randn(n)
    return pd.DataFrame(
        {
            'Open': close * (1 + noise * 0.002),
            'High': close * (1 + np.abs(noise) * 0.006 + 0.002),
            'Low': close * (1 - np.abs(noise) * 0.006 - 0.002),
            'Close': close,
            'Volume': np.random.randint(1_000_000, 10_000_000, n),
        },
        index=pd.date_range('2021-01-04', periods=n, freq='B'),
    )


# ── Step 1 — Data ────────────────────────────────────────────────────────────


def step1_load_data() -> pd.DataFrame:
    console.rule('[bold cyan]Step 1 · Synthetic OHLCV data[/bold cyan]')
    data = make_ohlcv()
    console.print(
        f'  Rows: {len(data)}  |  '
        f'Date range: {data.index[0].date()} → {data.index[-1].date()}  |  '
        f'Columns: {list(data.columns)}'
    )
    return data


# ── Step 2 — Alpha Research ──────────────────────────────────────────────────


# Candidate indicators to screen (fast subset — runs in seconds)
CANDIDATES = [
    {'name': 'RSI', 'params': {'window': 14}},
    {'name': 'RSI', 'params': {'window': 21}},
    {'name': 'SMA', 'params': {'window': 20}},
    {'name': 'SMA', 'params': {'window': 50}},
    {'name': 'EMA', 'params': {'window': 12}},
    {'name': 'EMA', 'params': {'window': 26}},
    {'name': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    {'name': 'ATR', 'params': {'window': 14}},
    {'name': 'BB', 'params': {'window': 20}},
]


def step2_alpha_research(data: pd.DataFrame):
    """Run AlphaSelector to rank candidate indicators by feature-mode
    IC."""
    console.rule('[bold cyan]Step 2 · AlphaSelector — indicator ranking[/bold cyan]')

    selector = AlphaSelector(data, verbose=False)

    # suggest_indicators returns indicator specs sorted by `metric`
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        selected_specs = selector.suggest_indicators(
            candidates=CANDIDATES,
            metric='ic',
            threshold=0.0,  # keep all — we'll filter via top_n in step 3
            top_k=len(CANDIDATES),
            selection_mode='feature',
        )

    console.print(f'  Candidates evaluated: {len(CANDIDATES)}')
    console.print(f'  Specs returned by suggest_indicators: {len(selected_specs)}')
    for spec in selected_specs:
        console.print(f'    {spec}')

    # Also collect the underlying AlphaResult objects for results_to_pipeline_config
    runner = AlphaRunner(verbose=False)
    jobs = []
    for cand in CANDIDATES:
        strategy_config = INDICATOR_STRATEGY_MAP.get(cand['name'])
        if strategy_config is None:
            continue
        jobs.append(
            AlphaJob(
                data=data,
                indicator_name=cand['name'],
                indicator_params=cand['params'],
                strategy_name=strategy_config['name'],
                strategy_params=strategy_config.get('params', {}),
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        results = runner.run_batch(jobs, n_jobs=1)

    completed = [r for r in results if r.status == 'completed']
    console.print(f'\n  AlphaRunner: {len(jobs)} jobs → {len(completed)} completed')

    # Print IC ranking table
    table = Table(title='IC ranking (all candidates)')
    table.add_column('Indicator', style='magenta')
    table.add_column('Params')
    table.add_column('IC', justify='right')
    table.add_column('Sharpe', justify='right')

    for r in sorted(completed, key=lambda x: x.metrics.get('ic', 0), reverse=True):
        table.add_row(
            r.job.indicator_name,
            str(r.job.indicator_params),
            f"{r.metrics.get('ic', 0):.4f}",
            f"{r.metrics.get('sharpe_ratio', 0):.2f}",
        )
    console.print(table)

    return completed


# ── Step 3 — Convert Results → Pipeline Config ───────────────────────────────


def step3_build_indicator_specs(completed_results):
    """Convert AlphaResult list → deduplicated indicator specs for
    DataProcessor."""
    console.rule('[bold cyan]Step 3 · results_to_pipeline_config()[/bold cyan]')

    # top_n=5, best param per indicator name, ranked by IC
    indicator_specs = results_to_pipeline_config(
        completed_results,
        top_n=5,
        metric='ic',
        deduplicate=True,
    )

    console.print(f'  Selected {len(indicator_specs)} indicators (top-5, deduplicated by name):')
    for spec in indicator_specs:
        console.print(f'    {spec}')

    return indicator_specs


# ── Step 4 — DataProcessor Pipeline ─────────────────────────────────────────


def step4_process_and_split(data: pd.DataFrame, indicator_specs):
    """Run DataProcessor with the selected indicators and a 70/15/15
    split."""
    console.rule('[bold cyan]Step 4 · DataProcessor.data_processing_pipeline()[/bold cyan]')

    processor = DataProcessor(ohlcv_data=data)

    # Ratio-based train / val / test split
    split_config = {'train': 0.70, 'val': 0.15, 'test': 0.15}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        splits, metadata = processor.data_processing_pipeline(
            indicators=indicator_specs,
            split_config=split_config,
        )

    console.print(f'  Split config: {split_config}')
    return splits, metadata


# ── Step 5 — Metadata Inspection ─────────────────────────────────────────────


def step5_inspect_metadata(metadata: dict):
    """Print the ProcessingMetadata returned by the pipeline."""
    console.rule('[bold cyan]Step 5 · ProcessingMetadata[/bold cyan]')

    table = Table(title='Processing metadata')
    table.add_column('Key', style='magenta')
    table.add_column('Value')

    for key, value in metadata.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                table.add_row(f'{key}.{sub_key}', str(sub_value))
        else:
            table.add_row(key, str(value))

    console.print(table)


# ── Step 6 — Split Summary ───────────────────────────────────────────────────


def step6_split_summary(splits: dict):
    """Print shape, date range, and feature columns for each split."""
    console.rule('[bold cyan]Step 6 · Per-split summary[/bold cyan]')

    summary_table = Table(title='Train / Val / Test split summary')
    summary_table.add_column('Split', style='magenta')
    summary_table.add_column('Rows', justify='right')
    summary_table.add_column('Cols', justify='right')

    for split_name, df in splits.items():
        summary_table.add_row(split_name, str(len(df)), str(df.shape[1]))

    console.print(summary_table)

    # Feature columns (same across all splits)
    first_split = next(iter(splits.values()))
    feature_cols = list(first_split.columns)

    # Categorise by prefix
    ohlcv_cols = [c for c in feature_cols if c in ('Open', 'High', 'Low', 'Close', 'Volume')]
    indicator_cols = [c for c in feature_cols if c not in ohlcv_cols]

    console.print(f'\n  OHLCV columns   ({len(ohlcv_cols)}): {ohlcv_cols}')
    console.print(f'  Indicator cols  ({len(indicator_cols)}): {indicator_cols}')
    console.print(f'  Total features  : {len(feature_cols)}')

    console.print(
        '\n[dim]These DataFrames are ready to pass directly to '
        'BacktestRunner via BacktestEnvironmentBuilder.with_data(train_data=..., test_data=...).[/dim]'
    )

    return splits


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    console.print(
        '\n[bold]Alpha Research → Feature Pipeline Integration Example[/bold]\n'
        '[dim]No API keys required — all data is synthetic.[/dim]\n'
    )

    # 1. Raw data
    data = step1_load_data()

    # 2. Alpha research — rank indicators by IC
    completed_results = step2_alpha_research(data)

    # 3. Convert results to indicator specs
    if not completed_results:
        console.print('[red]No completed alpha results — cannot continue.[/red]')
        return
    indicator_specs = step3_build_indicator_specs(completed_results)

    # 4. DataProcessor: apply indicators + split
    splits, metadata = step4_process_and_split(data, indicator_specs)

    # 5. Inspect metadata
    step5_inspect_metadata(metadata)

    # 6. Per-split summary
    step6_split_summary(splits)

    console.print('\n[green]Integration pipeline complete.[/green]')


if __name__ == '__main__':
    main()
