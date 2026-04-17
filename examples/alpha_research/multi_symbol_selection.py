"""
Example: Multi-symbol alpha selection.

Demonstrates the two AlphaSelector APIs that are absent from other examples:

  AlphaSelector.suggest_for_universe()
    — runs suggest_indicators() on each symbol independently, then returns
      the deduplicated union so the shared feature set captures alpha from
      every symbol.

  converters.results_to_pipeline_config()
    — converts a list of AlphaResult objects directly into the indicator-spec
      format expected by DataProcessor.data_processing_pipeline(). Supports
      top_n filtering, metric ranking, and deduplicate=True to keep only the
      best parameterisation per indicator name.

Note: This example uses synthetic data so it runs without any API keys.
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from quantrl_lab.alpha_research import INDICATOR_STRATEGY_MAP
from quantrl_lab.alpha_research.converters import results_to_pipeline_config
from quantrl_lab.alpha_research.models import AlphaJob
from quantrl_lab.alpha_research.runner import AlphaRunner
from quantrl_lab.alpha_research.selector import AlphaSelector

console = Console()


# ── Synthetic universe ─────────────────────────────────────────────────────


def make_ohlcv(n: int = 252, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily OHLCV DataFrame."""
    np.random.seed(seed)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.015))
    return pd.DataFrame(
        {
            'Open': close * (1 + np.random.randn(n) * 0.003),
            'High': close * (1 + np.abs(np.random.randn(n) * 0.007)),
            'Low': close * (1 - np.abs(np.random.randn(n) * 0.007)),
            'Close': close,
            'Volume': np.random.randint(500_000, 5_000_000, n),
        },
        index=pd.date_range('2023-01-01', periods=n, freq='B'),
    )


# A small universe of three synthetic symbols
UNIVERSE: dict[str, pd.DataFrame] = {
    'SYM_A': make_ohlcv(seed=1),
    'SYM_B': make_ohlcv(seed=2),
    'SYM_C': make_ohlcv(seed=3),
}

# Restrict candidates to a fast subset so the example runs quickly
FAST_CANDIDATES = [
    {'name': 'RSI', 'params': {'window': 14}},
    {'name': 'SMA', 'params': {'window': 20}},
    {'name': 'EMA', 'params': {'window': 20}},
    {'name': 'MACD', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    {'name': 'ATR', 'params': {'window': 14}},
]


# ── 1. suggest_for_universe() ──────────────────────────────────────────────


def demo_suggest_for_universe():
    console.rule('[bold cyan]1. AlphaSelector.suggest_for_universe()[/bold cyan]')

    console.print(f'Universe: {list(UNIVERSE.keys())}  |  ' f'Candidates tested per symbol: {len(FAST_CANDIDATES)}\n')

    # suggest_for_universe is a classmethod — no instance needed.
    # It runs suggest_indicators() on each symbol, then returns the
    # deduplicated union so any alpha found for any symbol is included.
    selected = AlphaSelector.suggest_for_universe(
        raw_data=UNIVERSE,
        candidates=FAST_CANDIDATES,
        metric='ic',
        threshold=0.0,  # keep everything for demonstration
        top_k=3,  # top-3 per symbol before dedup
        verbose=True,
        selection_mode='feature',
    )

    console.print('\n[green]Deduplicated indicators selected across universe:[/green]')
    for spec in selected:
        console.print(f'  {spec}')

    console.print(
        '\n[dim]These specs are ready to pass directly to '
        'DataProcessor.data_processing_pipeline(indicators=selected)[/dim]\n'
    )
    return selected


# ── 2. results_to_pipeline_config() direct usage ──────────────────────────


def demo_results_to_pipeline_config():
    console.rule('[bold cyan]2. converters.results_to_pipeline_config()[/bold cyan]')

    # Build a small batch of jobs with multiple windows per indicator
    data = UNIVERSE['SYM_A']
    runner = AlphaRunner(verbose=False)

    jobs = []
    for indicator, windows in [
        ('RSI', [7, 14, 21]),
        ('SMA', [20, 50]),
        ('EMA', [12, 26]),
    ]:
        strategy_config = INDICATOR_STRATEGY_MAP[indicator]
        for w in windows:
            jobs.append(
                AlphaJob(
                    data=data,
                    indicator_name=indicator,
                    indicator_params={'window': w},
                    strategy_name=strategy_config['name'],
                    strategy_params=strategy_config['params'],
                )
            )

    results = runner.run_batch(jobs, n_jobs=1)
    completed = [r for r in results if r.status == 'completed']
    console.print(f'Ran {len(jobs)} jobs → {len(completed)} completed\n')

    # ── Without deduplication ────────────────────────────────────────────
    cfg_all = results_to_pipeline_config(completed, top_n=10, metric='ic', deduplicate=False)
    console.print(f'[yellow]Without deduplicate:[/yellow] {len(cfg_all)} specs')
    for spec in cfg_all:
        console.print(f'  {spec}')

    # ── With deduplication: best window per indicator name ───────────────
    console.print()
    cfg_dedup = results_to_pipeline_config(completed, top_n=10, metric='ic', deduplicate=True)
    console.print(f'[green]With deduplicate=True:[/green] {len(cfg_dedup)} specs (one per indicator)')
    for spec in cfg_dedup:
        console.print(f'  {spec}')

    # ── top_n applied after dedup ────────────────────────────────────────
    console.print()
    cfg_top1 = results_to_pipeline_config(completed, top_n=1, metric='ic', deduplicate=True)
    console.print(f'[green]top_n=1 after dedup:[/green] {cfg_top1}\n')

    # ── Show the DataProcessor call these specs feed into ─────────────────
    console.print(
        '[dim]Usage:\n'
        '  from quantrl_lab.data.processing import DataProcessor\n'
        '  processor = DataProcessor(ohlcv_data=df)\n'
        '  processed, metadata = processor.data_processing_pipeline(indicators=cfg_dedup)[/dim]\n'
    )

    return cfg_dedup


# ── 3. Per-symbol selection summary table ─────────────────────────────────


def demo_per_symbol_comparison():
    console.rule('[bold cyan]3. Per-symbol selection comparison[/bold cyan]')

    table = Table(title='Best indicator per symbol (IC metric)')
    table.add_column('Symbol', style='magenta')
    table.add_column('Indicator')
    table.add_column('Params')
    table.add_column('IC', justify='right')
    table.add_column('Sharpe', justify='right')

    runner = AlphaRunner(verbose=False)

    for symbol, df in UNIVERSE.items():
        jobs = [
            AlphaJob(
                data=df,
                indicator_name=cand['name'],
                indicator_params=cand['params'],
                strategy_name=INDICATOR_STRATEGY_MAP[cand['name']]['name'],
                strategy_params=INDICATOR_STRATEGY_MAP[cand['name']]['params'],
            )
            for cand in FAST_CANDIDATES
        ]
        results = runner.run_batch(jobs, n_jobs=1)
        completed = [r for r in results if r.status == 'completed']

        if not completed:
            continue

        best = max(completed, key=lambda r: r.metrics.get('ic', -999))
        table.add_row(
            symbol,
            best.job.indicator_name,
            str(best.job.indicator_params),
            f"{best.metrics.get('ic', 0):.4f}",
            f"{best.metrics.get('sharpe_ratio', 0):.2f}",
        )

    console.print(table)


def main():
    demo_suggest_for_universe()
    demo_results_to_pipeline_config()
    demo_per_symbol_comparison()

    console.print('[green]Multi-symbol selection demo complete.[/green]')


if __name__ == '__main__':
    main()
