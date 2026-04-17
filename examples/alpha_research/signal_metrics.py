"""
Example: Signal Metrics and Score Generation.

Demonstrates the signal-quality utilities that are distinct from a full
AlphaRunner job:

  - generate_scores()           — continuous alpha scores from any strategy
  - analyze_signal()            — one-call IC / Rank-IC / autocorr / turnover
  - calculate_pearson_ic()      — Pearson IC with p-value
  - calculate_rank_ic()         — Spearman Rank IC with p-value
  - calculate_forward_returns() — forward-return series construction
  - AlphaJob.tags               — tagging jobs for grouping / filtering
  - allow_short=False           — long-only mode on AlphaJob / AlphaRunner
"""

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from quantrl_lab.alpha_research.alpha_strategies import (
    MACDCrossoverStrategy,
    MeanReversionStrategy,
    TrendFollowingStrategy,
)
from quantrl_lab.alpha_research.metrics import (
    analyze_signal,
    calculate_forward_returns,
    calculate_pearson_ic,
    calculate_rank_ic,
)
from quantrl_lab.alpha_research.models import AlphaJob, AlphaResult
from quantrl_lab.alpha_research.runner import AlphaRunner

console = Console()


# ── Synthetic data fixture ─────────────────────────────────────────────────


def make_ohlcv(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """Return a year of synthetic daily OHLCV data."""
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


# ── 1. generate_scores() ──────────────────────────────────────────────────


def demo_generate_scores(data: pd.DataFrame):
    console.rule('[bold cyan]1. generate_scores()[/bold cyan]')

    # Add indicators manually so we can show scores from multiple strategies
    sma = data['Close'].rolling(20).mean()
    rsi_delta = data['Close'].diff()
    gain = rsi_delta.where(rsi_delta > 0, 0).rolling(14).mean()
    loss = (-rsi_delta.where(rsi_delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss))

    fast_ema = data['Close'].ewm(span=12, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=26, adjust=False).mean()

    df = data.copy()
    df['SMA_20'] = sma
    df['RSI_14'] = rsi
    df['MACD_line'] = fast_ema - slow_ema
    df['MACD_signal'] = df['MACD_line'].ewm(span=9, adjust=False).mean()
    df = df.dropna()

    strategies = {
        'TrendFollowing(SMA_20)': TrendFollowingStrategy(indicator_col='SMA_20'),
        'MeanReversion(RSI_14)': MeanReversionStrategy(indicator_col='RSI_14'),
        'MACDCrossover': MACDCrossoverStrategy(fast_col='MACD_line', slow_col='MACD_signal'),
    }

    table = Table(title='Score statistics (last 50 bars)')
    table.add_column('Strategy', style='magenta')
    table.add_column('Min', justify='right')
    table.add_column('Max', justify='right')
    table.add_column('Mean', justify='right')
    table.add_column('Std', justify='right')

    for label, strat in strategies.items():
        scores = strat.generate_scores(df).iloc[-50:]
        table.add_row(
            label,
            f'{scores.min():.3f}',
            f'{scores.max():.3f}',
            f'{scores.mean():.3f}',
            f'{scores.std():.3f}',
        )

    console.print(table)
    console.print('[dim]Scores are always clipped to [-1, 1].[/dim]\n')


# ── 2. analyze_signal() ───────────────────────────────────────────────────


def demo_analyze_signal(data: pd.DataFrame):
    console.rule('[bold cyan]2. analyze_signal() — one-call signal quality[/bold cyan]')

    sma = data['Close'].rolling(20).mean()
    signal = (data['Close'] - sma).fillna(0)  # simple price-vs-SMA distance

    # analyze_signal rank-normalises the signal by default and computes
    # IC, Rank-IC, autocorrelation, and turnover in one call.
    metrics = analyze_signal(
        signal=signal,
        prices=data['Close'],
        forward_periods=5,
        normalize=True,
    )

    table = Table(title='analyze_signal() output (SMA-distance signal, 5-day horizon)')
    table.add_column('Metric', style='magenta')
    table.add_column('Value', justify='right')

    for k, v in metrics.items():
        table.add_row(k, f'{v:.4f}' if isinstance(v, float) else str(v))

    console.print(table)
    console.print()


# ── 3. calculate_pearson_ic / calculate_rank_ic ───────────────────────────


def demo_ic_functions(data: pd.DataFrame):
    console.rule('[bold cyan]3. calculate_pearson_ic / calculate_rank_ic[/bold cyan]')

    fwd_returns = calculate_forward_returns(data['Close'], periods=5)

    sma20 = data['Close'].rolling(20).mean()
    sma_signal = (data['Close'] - sma20).dropna()

    # Align
    valid = sma_signal.index.intersection(fwd_returns.dropna().index)
    s = sma_signal.loc[valid]
    r = fwd_returns.loc[valid]

    ic, ic_p = calculate_pearson_ic(s, r)
    rank_ic, rank_p = calculate_rank_ic(s, r)

    console.print(f'  Pearson IC:  {ic:.4f}  (p={ic_p:.4f})')
    console.print(f'  Rank IC:     {rank_ic:.4f}  (p={rank_p:.4f})')
    console.print('[dim]IC near 0 is expected for synthetic random-walk data.[/dim]\n')

    # Edge-case: constant signal → returns 0.0 instead of NaN
    const_signal = pd.Series([1.0] * len(r), index=r.index)
    ic_const, p_const = calculate_pearson_ic(const_signal, r)
    console.print(f'  Constant signal → IC={ic_const}  p={p_const}  (no NaN propagation)\n')


# ── 4. AlphaJob.tags + allow_short=False ─────────────────────────────────


def demo_tags_and_long_only(data: pd.DataFrame):
    console.rule('[bold cyan]4. AlphaJob.tags and allow_short=False[/bold cyan]')

    runner = AlphaRunner(verbose=False)

    # Tags are free-form key-value strings — useful for grouping batch results
    jobs = [
        AlphaJob(
            data=data,
            indicator_name='RSI',
            strategy_name='mean_reversion',
            indicator_params={'window': w},
            allow_short=False,  # long-only: SELL signals are suppressed
            tags={'category': 'momentum', 'window': str(w)},
        )
        for w in (7, 14, 21)
    ]

    results = runner.run_batch(jobs, n_jobs=1)

    table = Table(title='Long-only RSI jobs (allow_short=False)')
    table.add_column('Window', justify='right')
    table.add_column('Tags')
    table.add_column('Status')
    table.add_column('Sharpe', justify='right')
    table.add_column('Total return', justify='right')

    for r in results:
        table.add_row(
            str(r.job.indicator_params.get('window', '?')),
            str(r.job.tags),
            r.status,
            f"{r.metrics.get('sharpe_ratio', 0):.2f}" if r.status == 'completed' else '-',
            f"{r.metrics.get('total_return', 0):.2%}" if r.status == 'completed' else '-',
        )

    console.print(table)

    # Demonstrate filtering by tag
    momentum_results = [r for r in results if r.job.tags.get('category') == 'momentum']
    console.print(f'\nFiltered by tag category=momentum: {len(momentum_results)} results\n')


# ── 5. Signals vs. Scores side-by-side from AlphaRunner ──────────────────


def demo_signals_vs_scores(data: pd.DataFrame):
    console.rule('[bold cyan]5. Signals vs. Scores from a completed AlphaResult[/bold cyan]')

    runner = AlphaRunner(verbose=False)
    strategy_job = AlphaJob(
        data=data,
        indicator_name='RSI',
        strategy_name='mean_reversion',
        indicator_params={'window': 14},
        allow_short=True,
    )
    feature_job = AlphaJob(
        data=data,
        indicator_name='RSI',
        strategy_name='mean_reversion',
        indicator_params={'window': 14},
        evaluation_mode='feature',
    )
    strategy_result: AlphaResult = runner.run_job(strategy_job)
    feature_result: AlphaResult = runner.run_job(feature_job)

    if strategy_result.status != 'completed':
        console.print(f'[red]Strategy job failed: {strategy_result.error}[/red]')
        return
    if feature_result.status != 'completed':
        console.print(f'[red]Feature job failed: {feature_result.error}[/red]')
        return

    sig_counts = strategy_result.signals.value_counts().to_dict()
    scores = feature_result.scores.dropna()

    console.print(f'  Strategy-mode signal distribution: {sig_counts}')
    console.print(f'  Feature-mode score range: [{scores.min():.3f}, {scores.max():.3f}]  ' f'mean={scores.mean():.3f}')
    console.print(
        f"  Feature-mode IC: {feature_result.metrics.get('ic', 0):.4f}  "
        f"Strategy-mode Sharpe: {strategy_result.metrics.get('sharpe_ratio', 0):.2f}\n"
    )


def main():
    data = make_ohlcv()

    demo_generate_scores(data)
    demo_analyze_signal(data)
    demo_ic_functions(data)
    demo_tags_and_long_only(data)
    demo_signals_vs_scores(data)

    console.print('[green]All signal-metrics demos complete.[/green]')


if __name__ == '__main__':
    main()
