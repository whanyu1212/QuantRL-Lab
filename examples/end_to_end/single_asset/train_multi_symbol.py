"""
Multi-Symbol Training — SingleStockTradingEnv.

Trains a single RL agent across MULTIPLE symbols simultaneously using
Vectorized Environments (SubprocVecEnv).  Each symbol runs in its own
``SingleStockTradingEnv`` instance — the agent learns a policy that
generalises across tickers rather than overfitting to one stock.

Note: This is still a *single-asset* environment (the agent manages one
position per symbol).  For joint portfolio management across assets see
``examples/end_to_end/multi_asset/`` (coming soon).

Workflow:
1.  Fetch OHLCV for each symbol (concurrent via asyncio).
2.  Enrich with optional FMP analyst ratings and Alpaca news sentiment.
3.  Run AlphaSelector on all symbols to pick the best indicators (union).
4.  Build a shared DataPipeline (same feature set for every symbol).
5.  Train RecurrentPPO on all symbols via SubprocVecEnv.
6.  Evaluate on Train vs. Test sets and report feature-action correlations.

Shared data-loading logic lives in ``examples/end_to_end/shared/data_utils.py``.
"""

import asyncio
import os
import sys
import warnings

# Allow ``python examples/end_to_end/single_asset/multi_symbol_training.py``
# to resolve the shared package regardless of working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import aiohttp  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import pandas as pd  # noqa: E402
from rich.console import Console  # noqa: E402

# ── Shared utilities ──────────────────────────────────────────────────────────
from shared.data_utils import (  # noqa: E402
    DataSources,
    get_date_range,
    init_data_sources,
    process_symbol,
    select_alpha_indicators,
    train_test_split_by_date,
)

warnings.filterwarnings("ignore")
console = Console()

WINDOW_SIZE = 10
TOTAL_TIMESTEPS = 1_000_000


# ---------------------------------------------------------------------------
# Async enrichment helpers (multi-symbol specific)
# ---------------------------------------------------------------------------


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

    Returns a dict compatible with ``process_symbol(enrichment=...)``:
    ``{symbol, ratings_df, sector_perf_df, industry_perf_df, news_df}``.
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
        # ratings
        if not isinstance(outcomes[idx], Exception):
            _, result["ratings_df"] = outcomes[idx]
        idx += 1
        # company profile → fetch sector/industry perf
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

    if alpaca_task:
        if not isinstance(outcomes[idx], Exception):
            _, result["news_df"] = outcomes[idx]

    return result


async def _fetch_all_symbol_data(
    symbols: List[str],
    sources: DataSources,
    start_date,
    end_date,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Concurrently fetch OHLCV and optional enrichment data for all
    symbols.

    Args:
        symbols:    List of ticker symbols.
        sources:    Initialised DataSources bundle.
        start_date: Start of the historical window.
        end_date:   End of the historical window.

    Returns:
        ``raw_data``:   ``{symbol: ohlcv_df}``
        ``enrichment``: ``{symbol: {ratings_df, sector_perf_df, industry_perf_df, news_df}}``
    """

    async with aiohttp.ClientSession() as session:
        # 1. Fetch all OHLCV concurrently
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
                console.print(f"[yellow]Empty OHLCV result for {sym}, skipping.[/yellow]")

        if not raw_data:
            return {}, {}

        # 2. Fetch enrichment for all fetched symbols concurrently
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


# ---------------------------------------------------------------------------
# Multi-symbol data acquisition
# ---------------------------------------------------------------------------


def get_multi_stock_data(symbols: List[str], period_years: float = 2.0) -> pd.DataFrame:
    """
    Fetch and process data for multiple stocks using concurrent data
    acquisition.

    OHLCV and optional enrichment data (FMP analyst ratings, sector/industry
    performance, Alpaca news) are fetched concurrently via asyncio.  The alpha
    research, per-symbol pipeline execution, and cross-sectional post-processing
    remain synchronous.

    Args:
        symbols:      List of ticker symbols to process.
        period_years: Historical lookback window in years.

    Returns:
        Panel DataFrame (DatetimeIndex) with a ``Symbol`` column and a shared
        feature set across all symbols.
    """
    console.print("[bold blue]Starting Multi-Stock Data Acquisition...[/bold blue]")

    # 1. Initialise sources and date range (shared utility)
    sources = init_data_sources()
    start_date, end_date = get_date_range(period_years)

    # 2. Fetch all data concurrently
    raw_data, enrichment = asyncio.run(_fetch_all_symbol_data(symbols, sources, start_date, end_date))

    if not raw_data:
        raise ValueError("No data fetched for any symbol.")

    # 3. Alpha Research — union of indicators across all symbols (shared utility)
    console.print("[bold purple]Running Alpha Selection across all stocks (union approach)...[/bold purple]")
    indicators = select_alpha_indicators(raw_data, metric="ic", threshold=0.015, top_k=2, verbose=False)
    console.print(f"[cyan]Final indicator set ({len(indicators)}): {indicators}[/cyan]")

    # 4. Build and run the processing pipeline for each stock (shared utility)
    processed_dfs = []
    for symbol, raw_df in raw_data.items():
        console.print(f"Processing [cyan]{symbol}[/cyan]...")
        processed_df = process_symbol(
            symbol,
            raw_df,
            indicators,
            enrichment.get(symbol),
            verbose=True,
        )
        processed_dfs.append(processed_df)

    if not processed_dfs:
        raise ValueError("No data processed for any symbol.")

    panel_data = pd.concat(processed_dfs).sort_index()

    if "ratingScore" in panel_data.columns:
        # Fill missing analyst scores with neutral (3.0 = hold)
        panel_data["ratingScore"] = panel_data["ratingScore"].fillna(3.0)

    # 5. Cross-Sectional Processing & Cleanup
    from quantrl_lab.data.processing.pipeline import DataPipeline
    from quantrl_lab.data.processing.steps import (
        ColumnCleanupStep,
        CrossSectionalStep,
        NumericConversionStep,
    )

    console.print("[bold purple]Running Cross-Sectional Processing (Panel Data)...[/bold purple]")
    panel_pipeline = (
        DataPipeline()
        .add_step(CrossSectionalStep(columns=["Volume"], methods=["zscore", "rank"]))
        .add_step(NumericConversionStep())
        .add_step(ColumnCleanupStep(columns_to_drop=["Date", "Timestamp"]))
    )
    final_data, _ = panel_pipeline.execute(panel_data)
    return final_data


# ---------------------------------------------------------------------------
# Main training workflow
# ---------------------------------------------------------------------------


def main():
    console.rule("[bold blue]QuantRL-Lab Multi-Symbol Training[/bold blue]")

    symbols = [
        "AAPL",
        "MSFT",
        "GOOG",
        "AMZN",  # Mega-cap tech
        "JPM",
        "JNJ",
        "PG",
        "XOM",
        "CAT",  # Diversified sector leaders
    ]
    console.print(f"Symbols: {symbols}")

    # --- Phase 1: Data ---
    console.rule("[dim]Phase 1: Data[/dim]")
    try:
        full_data = get_multi_stock_data(symbols)
    except Exception as exc:
        console.print(f"[red]Failed to fetch/process data: {exc}[/red]")
        import traceback

        traceback.print_exc()
        return

    # --- Phase 2: Train/Test split (shared utility) ---
    console.rule("[dim]Phase 2: Split[/dim]")
    train_data, test_data = train_test_split_by_date(full_data, split_ratio=0.8)

    # --- Phase 3: Environment setup ---
    console.rule("[dim]Phase 3: Environment[/dim]")
    from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
    from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
    from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
    from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
    from quantrl_lab.environments.stock.strategies.rewards.execution_bonus import LimitExecutionReward
    from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty
    from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
    from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward
    from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder

    reward_strat = CompositeReward(
        strategies=[
            DifferentialSortinoReward(),
            DrawdownPenaltyReward(penalty_factor=0.5),
            TurnoverPenaltyReward(penalty_factor=0.1),
            LimitExecutionReward(improvement_multiplier=2.0),
            InvalidActionPenalty(penalty=-0.1),
        ],
        weights=[1.0, 0.2, 0.1, 0.05, 0.05],
        auto_scale=True,
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
    # The builder detects multiple symbols and creates one env factory per symbol
    env_config = builder.build()

    # --- Phase 4: Training ---
    console.rule("[dim]Phase 4: Training[/dim]")
    from sb3_contrib import RecurrentPPO

    from quantrl_lab.experiments.backtesting.core import ExperimentJob
    from quantrl_lab.experiments.backtesting.runner import BacktestRunner

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        lstm_hidden_size=64,
        n_lstm_layers=1,
        enable_critic_lstm=False,
    )

    job = ExperimentJob(
        algorithm_class=RecurrentPPO,
        env_config=env_config,
        algorithm_config=dict(
            policy="MlpLstmPolicy",
            ent_coef=0.05,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=128,
            clip_range=0.1,
            max_grad_norm=0.5,
            gamma=0.95,
            gae_lambda=0.95,
            policy_kwargs=policy_kwargs,
        ),
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=len(full_data["Symbol"].unique()),
    )

    runner = BacktestRunner(verbose=True)
    result = runner.run_job(job)
    runner.inspect_result(result)


if __name__ == "__main__":
    main()
