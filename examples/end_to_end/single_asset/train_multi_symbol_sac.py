"""
Multi-symbol training with SAC (Soft Actor-Critic).

SAC is an off-policy algorithm and does not support vectorized environments
(``n_envs > 1``).  Instead, this script trains a *separate* SAC agent per
symbol and compares their performance via ``BacktestRunner.inspect_batch``.

This per-symbol approach is useful for:
- Identifying which tickers SAC learns most effectively.
- Comparing symbol-specific policy performance side-by-side.
- Serving as a baseline before moving to cross-symbol generalisation.

For cross-symbol generalisation with a *shared* policy see
``train_multi_symbol.py`` (PPO + SubprocVecEnv) or the A2C multi-symbol
variant (``train_multi_symbol_a2c.py``).

Workflow:
  1. Fetch OHLCV for each symbol concurrently (asyncio).
  2. Run AlphaSelector union across all symbols.
  3. Process each symbol through the DataPipeline.
  4. Train/test split per symbol.
  5. Run one SAC ExperimentJob per symbol via JobGenerator.
  6. Compare results with BacktestRunner.inspect_batch.

Shared data-loading logic lives in ``examples/end_to_end/shared/data_utils.py``.
"""

import asyncio
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiohttp  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402

load_dotenv()
warnings.filterwarnings("ignore")
console = Console()

from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import pandas as pd  # noqa: E402
from shared.data_utils import (  # noqa: E402
    DataSources,
    get_date_range,
    init_data_sources,
    process_symbol,
    select_alpha_indicators,
    train_test_split_by_date,
)

PERIOD_YEARS = 2
WINDOW_SIZE = 10
TOTAL_TIMESTEPS = 50_000

SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOG",
    "JPM",
    "JNJ",
]


# ── Async enrichment (reused from multi-symbol PPO example) ───────────────────


async def _fetch_enrichment_for_symbol(
    session: aiohttp.ClientSession,
    fmp: Optional[Any],
    alpaca: Optional[Any],
    symbol: str,
    start_date,
    end_date,
) -> Dict[str, Any]:
    """
    Concurrently fetch all optional enrichment data for a single symbol.

    Returns a dict compatible with ``process_symbol(enrichment=...)``.
    All values default to None on failure.
    """
    result: Dict[str, Any] = {
        "symbol": symbol,
        "ratings_df": None,
        "sector_perf_df": None,
        "industry_perf_df": None,
        "news_df": None,
    }

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    fmp_tasks = []
    if fmp:
        fmp_tasks = [
            fmp.async_fetch_ratings(session, symbol, limit=500),
            fmp.async_fetch_company_profile(session, symbol),
        ]

    alpaca_task = None
    if alpaca:
        alpaca_task = alpaca.async_fetch_news(session, symbol, start_date, end_date)

    all_tasks = fmp_tasks + ([alpaca_task] if alpaca_task else [])
    if not all_tasks:
        return result

    outcomes = await asyncio.gather(*all_tasks, return_exceptions=True)

    idx = 0
    if fmp:
        if not isinstance(outcomes[idx], Exception):
            _, result["ratings_df"] = outcomes[idx]
        idx += 1
        if not isinstance(outcomes[idx], Exception):
            _, profile_df = outcomes[idx]
            if not profile_df.empty:
                sector = profile_df.iloc[0].get("sector")
                industry = profile_df.iloc[0].get("industry")
                sector_tasks = []
                if sector:
                    sector_tasks.append(fmp.async_fetch_sector_perf(session, sector, start_str, end_str))
                if industry:
                    sector_tasks.append(fmp.async_fetch_industry_perf(session, industry, start_str, end_str))
                if sector_tasks:
                    sector_outcomes = await asyncio.gather(*sector_tasks, return_exceptions=True)
                    si = 0
                    if sector and not isinstance(sector_outcomes[si], Exception):
                        _, result["sector_perf_df"] = sector_outcomes[si]
                        si += 1
                    if industry and si < len(sector_outcomes) and not isinstance(sector_outcomes[si], Exception):
                        _, result["industry_perf_df"] = sector_outcomes[si]
        idx += 1

    if alpaca_task and not isinstance(outcomes[idx], Exception):
        _, result["news_df"] = outcomes[idx]

    return result


async def _fetch_all_symbol_data(
    symbols: List[str],
    sources: DataSources,
    start_date,
    end_date,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Concurrently fetch OHLCV and optional enrichment for all symbols.

    Returns:
        raw_data:   ``{symbol: ohlcv_df}``
        enrichment: ``{symbol: {ratings_df, sector_perf_df, ...}}``
    """

    async with aiohttp.ClientSession() as session:
        console.print(f"[bold blue]Fetching OHLCV for {len(symbols)} symbols concurrently...[/bold blue]")
        ohlcv_tasks = [sources.loader.async_fetch_ohlcv(sym, start_date, end_date) for sym in symbols]
        ohlcv_outcomes = await asyncio.gather(*ohlcv_tasks, return_exceptions=True)

        raw_data: Dict[str, pd.DataFrame] = {}
        for outcome in ohlcv_outcomes:
            if isinstance(outcome, Exception):
                console.print(f"[red]OHLCV fetch error: {outcome}[/red]")
                continue
            sym, df = outcome
            if "Date" in df.columns:
                df = df.set_index("Date")
            if not df.empty:
                raw_data[sym] = df

        if not raw_data:
            return {}, {}

        enrichment: Dict[str, Dict] = {}
        if sources.fmp or sources.alpaca:
            console.print("[bold blue]Fetching enrichment data concurrently...[/bold blue]")
            enrich_tasks = [
                _fetch_enrichment_for_symbol(session, sources.fmp, sources.alpaca, sym, start_date, end_date)
                for sym in raw_data
            ]
            enrich_outcomes = await asyncio.gather(*enrich_tasks, return_exceptions=True)
            for outcome in enrich_outcomes:
                if isinstance(outcome, Exception):
                    console.print(f"[yellow]Enrichment fetch error: {outcome}[/yellow]")
                    continue
                enrichment[outcome["symbol"]] = outcome

    return raw_data, enrichment


# ── Per-symbol data preparation ───────────────────────────────────────────────


def build_per_symbol_splits(
    symbols: List[str],
    period_years: float = 2.0,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fetch, process, and split data for each symbol independently.

    Args:
        symbols:      List of ticker symbols.
        period_years: Historical lookback window in years.

    Returns:
        Mapping of ``{symbol: (train_df, test_df)}``.
        Symbols with insufficient data are skipped.
    """
    sources = init_data_sources()
    start_date, end_date = get_date_range(period_years)

    raw_data, enrichment = asyncio.run(_fetch_all_symbol_data(symbols, sources, start_date, end_date))
    if not raw_data:
        raise ValueError("No data fetched for any symbol.")

    console.print("[bold purple]Running Alpha Selection across all stocks (union approach)...[/bold purple]")
    indicators = select_alpha_indicators(raw_data, metric="ic", threshold=0.015, top_k=2, verbose=False)
    console.print(f"[cyan]Shared indicator set ({len(indicators)}): {indicators}[/cyan]")

    splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for sym, raw_df in raw_data.items():
        console.print(f"Processing [cyan]{sym}[/cyan]...")
        try:
            processed_df = process_symbol(sym, raw_df, indicators, enrichment.get(sym), verbose=True)
            processed_df = processed_df.select_dtypes(include="number")
            train_df, test_df = train_test_split_by_date(processed_df, split_ratio=0.8)
            if len(train_df) <= WINDOW_SIZE or len(test_df) <= WINDOW_SIZE:
                console.print(f"[yellow]{sym}: split too small — skipping.[/yellow]")
                continue
            splits[sym] = (train_df, test_df)
        except Exception as exc:
            console.print(f"[yellow]{sym}: processing failed ({exc}) — skipping.[/yellow]")

    return splits


# ── Training ──────────────────────────────────────────────────────────────────


def main():
    console.rule("[bold blue]Multi-Symbol SAC Training (per-symbol)[/bold blue]")
    console.print(f"Symbols: {SYMBOLS}")

    # --- Phase 1: Data ---
    console.rule("[dim]Phase 1: Data[/dim]")
    splits = build_per_symbol_splits(SYMBOLS, period_years=PERIOD_YEARS)
    if not splits:
        console.print("[red]No usable data after processing. Exiting.[/red]")
        return
    console.print(f"[green]Symbols ready for training:[/green] {list(splits.keys())}")

    # --- Phase 2: Build one env config per symbol ---
    console.rule("[dim]Phase 2: Environment Configs[/dim]")
    from stable_baselines3 import SAC

    from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
    from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
    from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
    from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
    from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward
    from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder
    from quantrl_lab.experiments.backtesting.core import JobGenerator
    from quantrl_lab.experiments.backtesting.runner import BacktestRunner

    sac_config = dict(
        policy="MlpPolicy",
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        ent_coef="auto",
        verbose=0,
    )

    env_configs = {}
    for sym, (train_df, test_df) in splits.items():
        # Simplified reward: portfolio return is the primary signal.
        # Heavy penalties (drawdown, sortino) cause the agent to learn "do nothing"
        # as the optimal strategy. A small turnover penalty discourages churning.
        reward_strat = CompositeReward(
            strategies=[
                PortfolioValueChangeReward(),
                TurnoverPenaltyReward(penalty_factor=0.1),
            ],
            weights=[1.0, 0.05],
            auto_scale=False,  # raw % return is already well-scaled
        )
        env_config = (
            BacktestEnvironmentBuilder()
            .with_data(train_data=train_df, test_data=test_df)
            .with_env_params(
                initial_balance=100_000.0,
                transaction_cost_pct=0.001,
                window_size=WINDOW_SIZE,
            )
            .with_strategies(
                action=StandardActionStrategy(),
                reward=reward_strat,
                observation=FeatureAwareObservationStrategy(normalize_stationary=True),
            )
            .build()
        )
        env_configs[sym] = env_config

    # --- Phase 3: Generate and run jobs ---
    # SAC requires n_envs=1 (off-policy), so we pass it via job_kwargs.
    console.rule("[dim]Phase 3: Training (SAC — one job per symbol)[/dim]")
    jobs = JobGenerator.generate_grid(
        algorithms=[SAC],
        env_configs=env_configs,
        algorithm_configs=[sac_config],
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=1,  # SAC is off-policy — must be 1
    )

    runner = BacktestRunner(verbose=True)
    results = runner.run_batch(jobs)

    # --- Phase 4: Compare results ---
    console.rule("[dim]Phase 4: Results[/dim]")
    runner.inspect_batch(results)


if __name__ == "__main__":
    main()
