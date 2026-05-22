"""
Multi-symbol training with A2C (Advantage Actor-Critic).

A2C is synchronous and on-policy, so it supports vectorized environments
exactly like PPO.  This script trains a *single shared policy* across all
symbols simultaneously using ``SubprocVecEnv`` — one subprocess per symbol.

The shared-policy approach forces the agent to learn a generalised strategy
that works across different tickers, reducing overfitting to any one stock.

Key differences from the PPO multi-symbol script:
- Algorithm: A2C instead of PPO (no clipping, faster per-step update).
- Standard MLP policy instead of LSTM — A2C's simpler architecture is often
  competitive and much cheaper to train.
- Shorter rollout (``n_steps=5``) with more frequent gradient updates.

Workflow:
  1. Fetch OHLCV for each symbol concurrently (asyncio).
  2. Run AlphaSelector union across all symbols.
  3. Process each symbol through the shared DataPipeline.
  4. Cross-sectional post-processing (z-score volume).
  5. Train/test split on panel data.
  6. Train a single A2C agent via SubprocVecEnv.
  7. Evaluate and inspect results.

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

WINDOW_SIZE = 10
TOTAL_TIMESTEPS = 1_000_000

SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOG",
    "AMZN",
    "JPM",
    "JNJ",
    "PG",
    "XOM",
    "CAT",
]


# ── Async enrichment helpers ──────────────────────────────────────────────────


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
            else:
                console.print(f"[yellow]Empty OHLCV for {sym}, skipping.[/yellow]")

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


# ── Panel data construction ────────────────────────────────────────────────────


def get_multi_stock_data(symbols: List[str], period_years: float = 2.0) -> pd.DataFrame:
    """
    Fetch and process data for multiple stocks into a shared panel
    DataFrame.

    Uses a union of alpha-selected indicators across all symbols so that every
    symbol has the same feature set — required for a shared vectorized policy.

    Args:
        symbols:      List of ticker symbols.
        period_years: Historical lookback in years.

    Returns:
        Panel DataFrame (DatetimeIndex) with a ``Symbol`` column.
    """
    console.print("[bold blue]Starting Multi-Stock Data Acquisition...[/bold blue]")

    sources = init_data_sources()
    start_date, end_date = get_date_range(period_years)

    raw_data, enrichment = asyncio.run(_fetch_all_symbol_data(symbols, sources, start_date, end_date))
    if not raw_data:
        raise ValueError("No data fetched for any symbol.")

    console.print("[bold purple]Running Alpha Selection across all stocks (union approach)...[/bold purple]")
    indicators = select_alpha_indicators(raw_data, metric="ic", threshold=0.015, top_k=2, verbose=False)
    console.print(f"[cyan]Final indicator set ({len(indicators)}): {indicators}[/cyan]")

    processed_dfs = []
    for sym, raw_df in raw_data.items():
        console.print(f"Processing [cyan]{sym}[/cyan]...")
        try:
            processed_df = process_symbol(sym, raw_df, indicators, enrichment.get(sym), verbose=True)
            processed_dfs.append(processed_df)
        except Exception as exc:
            console.print(f"[yellow]{sym}: processing failed ({exc}) — skipping.[/yellow]")

    if not processed_dfs:
        raise ValueError("No data processed for any symbol.")

    panel_data = pd.concat(processed_dfs).sort_index()

    if "ratingScore" in panel_data.columns:
        panel_data["ratingScore"] = panel_data["ratingScore"].fillna(3.0)

    # Cross-sectional normalisation + cleanup
    from quantrl_lab.data.processing.pipeline import DataPipeline
    from quantrl_lab.data.processing.steps import ColumnCleanupStep, CrossSectionalStep, NumericConversionStep

    console.print("[bold purple]Running Cross-Sectional Processing...[/bold purple]")
    panel_pipeline = (
        DataPipeline()
        .add_step(CrossSectionalStep(columns=["Volume"], methods=["zscore", "rank"]))
        .add_step(NumericConversionStep())
        .add_step(ColumnCleanupStep(columns_to_drop=["Date", "Timestamp"]))
    )
    final_data, _ = panel_pipeline.execute(panel_data)
    return final_data


# ── Training ──────────────────────────────────────────────────────────────────


def main():
    console.rule("[bold blue]Multi-Symbol A2C Training (shared policy)[/bold blue]")
    console.print(f"Symbols: {SYMBOLS}")

    # --- Phase 1: Data ---
    console.rule("[dim]Phase 1: Data[/dim]")
    try:
        full_data = get_multi_stock_data(SYMBOLS)
    except Exception as exc:
        console.print(f"[red]Failed to fetch/process data: {exc}[/red]")
        import traceback

        traceback.print_exc()
        return

    # --- Phase 2: Train/Test split ---
    console.rule("[dim]Phase 2: Split[/dim]")
    train_data, test_data = train_test_split_by_date(full_data, split_ratio=0.8)

    # --- Phase 3: Environment ---
    console.rule("[dim]Phase 3: Environment[/dim]")
    from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
    from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
    from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
    from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
    from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward
    from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder

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

    builder = BacktestEnvironmentBuilder()
    builder.with_data(train_data=train_data, test_data=test_data)
    builder.with_env_params(
        initial_balance=100_000.0,
        transaction_cost_pct=0.001,
        window_size=WINDOW_SIZE,
    )
    builder.with_strategies(
        action=StandardActionStrategy(),
        reward=reward_strat,
        observation=FeatureAwareObservationStrategy(normalize_stationary=True),
    )
    # Builder detects the Symbol column and returns one factory per symbol
    env_config = builder.build()

    # --- Phase 4: Training ---
    # A2C with SubprocVecEnv: one subprocess per symbol.
    # n_envs is set to the number of unique symbols so each gets its own env.
    # A2C notes:
    #   - ``n_steps=5``: classic short rollout for A2C.
    #   - ``ent_coef=0.01``: mild exploration bonus.
    #   - ``max_grad_norm=0.5``: gradient clipping for stability across diverse symbols.
    console.rule("[dim]Phase 4: Training (A2C + SubprocVecEnv)[/dim]")
    from stable_baselines3 import A2C

    from quantrl_lab.experiments.backtesting.core import ExperimentJob
    from quantrl_lab.experiments.backtesting.runner import BacktestRunner

    job = ExperimentJob(
        algorithm_class=A2C,
        env_config=env_config,
        algorithm_config=dict(
            policy="MlpPolicy",
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            verbose=0,
        ),
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=len(full_data["Symbol"].unique()),
    )

    runner = BacktestRunner(verbose=True)
    result = runner.run_job(job)
    runner.inspect_result(result)


if __name__ == "__main__":
    main()
