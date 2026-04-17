"""
Shared-policy multi-symbol training script.

Trains a single shared policy across multiple symbols simultaneously,
so the agent learns to generalise across tickers rather than overfit to one.
Supports all algorithms via ``--algo``:
  SAC, PPO, A2C, TD3, TQC, TRPO, CrossQ, RecurrentPPO

On-policy algorithms (PPO, A2C, TRPO, RecurrentPPO) use SubprocVecEnv with
one subprocess per symbol — parallel rollouts are pooled into a single buffer.

Off-policy algorithms (SAC, TD3, TQC, CrossQ) use a single env with panel
data (all symbols concatenated), so the replay buffer samples transitions
across all tickers.

Usage::

    python train_multi_symbol.py --algo SAC
    python train_multi_symbol.py --algo RecurrentPPO --total-timesteps 1000000
    python train_multi_symbol.py --algo TD3 --period-years 3
    python train_multi_symbol.py --algo TQC --algorithm-config '{"learning_rate": 3e-4}'

Shared infrastructure (data-source init, date helpers) lives in
``examples/end_to_end/shared/data_utils.py``.
"""

import argparse
import asyncio
import json
import os
import sys
import warnings
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import aiohttp  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

load_dotenv()
warnings.filterwarnings('ignore')
console = Console()

import pandas as pd  # noqa: E402
from shared.data_utils import (  # noqa: E402
    DataSources,
    get_date_range,
    init_data_sources,
)

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Top-level experiment configuration."""

    algo: str = 'RecurrentPPO'
    symbols: List[str] = None  # defaults to DEFAULT_SYMBOLS
    period_years: int = 2
    window_size: int = 10
    total_timesteps: int = 1_000_000
    algorithm_config: Optional[Dict[str, Any]] = None  # overrides/merges with preset

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = list(DEFAULT_SYMBOLS)


DEFAULT_SYMBOLS = [
    'AAPL',
    'MSFT',
    'GOOG',
    'AMZN',
    'JPM',
    'JNJ',
    'PG',
    'XOM',
    'CAT',
]


@dataclass
class AlgorithmPreset:
    """
    Algorithm-specific configuration.

    ``n_envs_from_symbols``: if True, n_envs is set to the number of symbols
    (one subprocess per ticker) — valid for on-policy algorithms only.
    If False, n_envs=1 (off-policy: single env with panel data).
    """

    algo_class_factory: Any
    algorithm_config_factory: Any
    reward_factory: Any
    n_envs_from_symbols: bool = False  # True → on-policy; False → off-policy
    description: str = ''


def _make_presets() -> Dict[str, AlgorithmPreset]:

    # ── Off-policy ─────────────────────────────────────────────────────────────

    def sac_class():
        from stable_baselines3 import SAC

        return SAC

    def sac_config():
        return dict(
            policy='MlpPolicy',
            learning_rate=1e-4,
            buffer_size=500_000,
            learning_starts=10_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            gradient_steps=2,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
            verbose=0,
        )

    def sac_reward():
        from quantrl_lab.environments.stock.strategies.rewards.sharpe import DifferentialSharpeReward

        return DifferentialSharpeReward(risk_free_rate=0.0)

    # ──────────────────────────────────────────────────────────────────────────

    def td3_class():
        from stable_baselines3 import TD3

        return TD3

    def td3_config():
        import numpy as np
        from stable_baselines3.common.noise import NormalActionNoise

        action_noise = NormalActionNoise(mean=np.zeros(3), sigma=0.1 * np.ones(3))
        return dict(
            policy='MlpPolicy',
            learning_rate=1e-4,
            buffer_size=500_000,
            learning_starts=20_000,
            batch_size=256,
            tau=0.005,
            gamma=0.995,
            train_freq=8,
            gradient_steps=2,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            action_noise=action_noise,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
        )

    def td3_reward():
        from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
        from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
        from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

        return CompositeReward(
            strategies=[PortfolioValueChangeReward(), TurnoverPenaltyReward(penalty_factor=0.1)],
            weights=[1.0, 0.05],
            auto_scale=False,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def tqc_class():
        from sb3_contrib import TQC

        return TQC

    def tqc_config():
        return dict(
            policy='MlpPolicy',
            learning_rate=3e-4,
            buffer_size=500_000,
            learning_starts=20_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=8,
            gradient_steps=2,
            ent_coef='auto',
            top_quantiles_to_drop_per_net=2,
            policy_kwargs=dict(net_arch=[256, 256], n_quantiles=25),
            verbose=0,
        )

    def tqc_reward():
        from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
        from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
        from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

        return CompositeReward(
            strategies=[PortfolioValueChangeReward(), TurnoverPenaltyReward(penalty_factor=0.01)],
            weights=[1.0, 0.05],
            auto_scale=True,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def crossq_class():
        from sb3_contrib import CrossQ

        return CrossQ

    def crossq_config():
        return dict(
            policy='MlpPolicy',
            learning_rate=1e-4,
            buffer_size=500_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0,
        )

    def crossq_reward():
        from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
        from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
        from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

        return CompositeReward(
            strategies=[PortfolioValueChangeReward(), TurnoverPenaltyReward(penalty_factor=0.1)],
            weights=[1.0, 0.05],
            auto_scale=False,
        )

    # ── On-policy ──────────────────────────────────────────────────────────────

    def recurrent_ppo_class():
        from sb3_contrib import RecurrentPPO

        return RecurrentPPO

    def recurrent_ppo_config():
        return dict(
            policy='MlpLstmPolicy',
            ent_coef=0.05,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=128,
            clip_range=0.1,
            max_grad_norm=0.5,
            gamma=0.95,
            gae_lambda=0.95,
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
                lstm_hidden_size=64,
                n_lstm_layers=1,
                enable_critic_lstm=False,
            ),
            verbose=0,
        )

    def recurrent_ppo_reward():
        from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
        from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
        from quantrl_lab.environments.stock.strategies.rewards.execution_bonus import LimitExecutionReward
        from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty
        from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
        from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

        return CompositeReward(
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

    # ──────────────────────────────────────────────────────────────────────────

    def ppo_class():
        from stable_baselines3 import PPO

        return PPO

    def ppo_config():
        return dict(
            policy='MlpPolicy',
            ent_coef=0.01,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            gamma=0.98,
            clip_range=0.1,
            gae_lambda=0.95,
            verbose=0,
        )

    def ppo_reward():
        from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
        from quantrl_lab.environments.stock.strategies.rewards.invalid_action import InvalidActionPenalty
        from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward
        from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

        return CompositeReward(
            strategies=[
                PortfolioValueChangeReward(),
                TurnoverPenaltyReward(penalty_factor=0.01),
                InvalidActionPenalty(penalty=-0.1),
            ],
            weights=[1.0, 0.05, 0.5],
            auto_scale=True,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def a2c_class():
        from stable_baselines3 import A2C

        return A2C

    def a2c_config():
        return dict(
            policy='MlpPolicy',
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.1,
            vf_coef=0.5,
            max_grad_norm=0.5,
            rms_prop_eps=1e-5,
            verbose=0,
        )

    def a2c_reward():
        from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
        from quantrl_lab.environments.stock.strategies.rewards.portfolio_value import PortfolioValueChangeReward

        return CompositeReward(
            strategies=[PortfolioValueChangeReward()],
            weights=[1.0],
            auto_scale=False,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def trpo_class():
        from sb3_contrib import TRPO

        return TRPO

    def trpo_config():
        return dict(
            policy='MlpPolicy',
            learning_rate=1e-3,
            n_steps=2048,
            batch_size=128,
            gamma=0.99,
            gae_lambda=0.95,
            cg_max_steps=15,
            target_kl=0.01,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            verbose=0,
        )

    def trpo_reward():
        from quantrl_lab.environments.stock.strategies.rewards.composite import CompositeReward
        from quantrl_lab.environments.stock.strategies.rewards.drawdown import DrawdownPenaltyReward
        from quantrl_lab.environments.stock.strategies.rewards.sortino import DifferentialSortinoReward
        from quantrl_lab.environments.stock.strategies.rewards.turnover import TurnoverPenaltyReward

        return CompositeReward(
            strategies=[
                DifferentialSortinoReward(),
                DrawdownPenaltyReward(penalty_factor=0.5),
                TurnoverPenaltyReward(penalty_factor=0.1),
            ],
            weights=[1.0, 0.2, 0.1],
            auto_scale=True,
        )

    # ──────────────────────────────────────────────────────────────────────────

    return {
        'SAC': AlgorithmPreset(
            algo_class_factory=sac_class,
            algorithm_config_factory=sac_config,
            reward_factory=sac_reward,
            n_envs_from_symbols=False,
            description='Off-policy, max-entropy. Panel env, single replay buffer.',
        ),
        'TD3': AlgorithmPreset(
            algo_class_factory=td3_class,
            algorithm_config_factory=td3_config,
            reward_factory=td3_reward,
            n_envs_from_symbols=False,
            description='Off-policy, deterministic. Panel env, single replay buffer.',
        ),
        'TQC': AlgorithmPreset(
            algo_class_factory=tqc_class,
            algorithm_config_factory=tqc_config,
            reward_factory=tqc_reward,
            n_envs_from_symbols=False,
            description='Off-policy, distributional Q. Panel env, single replay buffer.',
        ),
        'CrossQ': AlgorithmPreset(
            algo_class_factory=crossq_class,
            algorithm_config_factory=crossq_config,
            reward_factory=crossq_reward,
            n_envs_from_symbols=False,
            description='Off-policy, batch-norm critic. Panel env, single replay buffer.',
        ),
        'RecurrentPPO': AlgorithmPreset(
            algo_class_factory=recurrent_ppo_class,
            algorithm_config_factory=recurrent_ppo_config,
            reward_factory=recurrent_ppo_reward,
            n_envs_from_symbols=True,
            description='On-policy, LSTM. One subprocess per symbol via SubprocVecEnv.',
        ),
        'PPO': AlgorithmPreset(
            algo_class_factory=ppo_class,
            algorithm_config_factory=ppo_config,
            reward_factory=ppo_reward,
            n_envs_from_symbols=True,
            description='On-policy, clipped surrogate. One subprocess per symbol.',
        ),
        'A2C': AlgorithmPreset(
            algo_class_factory=a2c_class,
            algorithm_config_factory=a2c_config,
            reward_factory=a2c_reward,
            n_envs_from_symbols=True,
            description='On-policy actor-critic. One subprocess per symbol.',
        ),
        'TRPO': AlgorithmPreset(
            algo_class_factory=trpo_class,
            algorithm_config_factory=trpo_config,
            reward_factory=trpo_reward,
            n_envs_from_symbols=True,
            description='On-policy, trust-region. One subprocess per symbol.',
        ),
    }


PRESETS: Dict[str, AlgorithmPreset] = _make_presets()


# ── Async data fetching ────────────────────────────────────────────────────────


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

    Returns a dict with keys: symbol, ratings_df, sector_perf_df,
    industry_perf_df, news_df. All values default to None on failure.
    """
    result: Dict[str, Any] = {
        'symbol': symbol,
        'ratings_df': None,
        'sector_perf_df': None,
        'industry_perf_df': None,
        'news_df': None,
    }

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

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
            _, result['ratings_df'] = outcomes[idx]
        idx += 1
        if not isinstance(outcomes[idx], Exception):
            _, profile_df = outcomes[idx]
            if not profile_df.empty:
                sector = profile_df.iloc[0].get('sector')
                industry = profile_df.iloc[0].get('industry')
                sector_tasks = []
                if sector:
                    sector_tasks.append(fmp.async_fetch_sector_perf(session, sector, start_str, end_str))
                if industry:
                    sector_tasks.append(fmp.async_fetch_industry_perf(session, industry, start_str, end_str))
                if sector_tasks:
                    sector_outcomes = await asyncio.gather(*sector_tasks, return_exceptions=True)
                    si = 0
                    if sector and not isinstance(sector_outcomes[si], Exception):
                        _, result['sector_perf_df'] = sector_outcomes[si]
                        si += 1
                    if industry and si < len(sector_outcomes) and not isinstance(sector_outcomes[si], Exception):
                        _, result['industry_perf_df'] = sector_outcomes[si]
        idx += 1

    if alpaca_task and not isinstance(outcomes[idx], Exception):
        _, result['news_df'] = outcomes[idx]

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
        console.print(f'[bold blue]Fetching OHLCV for {len(symbols)} symbols concurrently...[/bold blue]')
        ohlcv_tasks = [sources.loader.async_fetch_ohlcv(sym, start_date, end_date) for sym in symbols]
        ohlcv_outcomes = await asyncio.gather(*ohlcv_tasks, return_exceptions=True)

        raw_data: Dict[str, pd.DataFrame] = {}
        for outcome in ohlcv_outcomes:
            if isinstance(outcome, Exception):
                console.print(f'[red]OHLCV fetch error: {outcome}[/red]')
                continue
            sym, df = outcome
            if 'Date' in df.columns:
                df = df.set_index('Date')
            if not df.empty:
                raw_data[sym] = df
            else:
                console.print(f'[yellow]Empty OHLCV for {sym}, skipping.[/yellow]')

        if not raw_data:
            return {}, {}

        enrichment: Dict[str, Dict] = {}
        if sources.fmp or sources.alpaca:
            console.print('[bold blue]Fetching enrichment data concurrently...[/bold blue]')
            enrich_tasks = [
                _fetch_enrichment_for_symbol(session, sources.fmp, sources.alpaca, sym, start_date, end_date)
                for sym in raw_data
            ]
            enrich_outcomes = await asyncio.gather(*enrich_tasks, return_exceptions=True)
            for outcome in enrich_outcomes:
                if isinstance(outcome, Exception):
                    console.print(f'[yellow]Enrichment fetch error: {outcome}[/yellow]')
                    continue
                enrichment[outcome['symbol']] = outcome

    return raw_data, enrichment


# ── Data pipeline ─────────────────────────────────────────────────────────────


def get_multi_stock_data(cfg: RunConfig) -> pd.DataFrame:
    """
    Fetch and process panel data for multiple symbols with shared
    feature engineering.

    Args:
        cfg: RunConfig with symbols and period_years.

    Returns:
        Panel DataFrame (DatetimeIndex) with a ``Symbol`` column and a shared
        feature set across all symbols.
    """
    console.print('[bold blue]Starting Multi-Stock Data Acquisition...[/bold blue]')

    sources = init_data_sources()
    start_date, end_date = get_date_range(cfg.period_years)

    raw_data, enrichment = asyncio.run(_fetch_all_symbol_data(cfg.symbols, sources, start_date, end_date))
    if not raw_data:
        raise ValueError('No data fetched for any symbol.')

    from quantrl_lab.alpha_research import AlphaSelector

    console.print('[bold purple]Running Alpha Selection across all stocks (union approach)...[/bold purple]')
    indicators = AlphaSelector.suggest_for_universe(
        {sym: raw_df for sym, raw_df in raw_data.items()},
        metric='ic',
        threshold=0.02,
        top_k=4,
        verbose=True,
        selection_mode='feature',
    )
    console.print(f'[cyan]Shared indicator set ({len(indicators)}): {indicators}[/cyan]')

    from quantrl_lab.data.processing import DataProcessor

    processed_dfs = []
    for sym, raw_df in raw_data.items():
        console.print(f'Processing [cyan]{sym}[/cyan]...')
        try:
            enrich = enrichment.get(sym, {})
            processor = DataProcessor(
                ohlcv_data=raw_df,
                analyst_ratings=enrich.get('ratings_df'),
                sector_performance=enrich.get('sector_perf_df'),
                industry_performance=enrich.get('industry_perf_df'),
                news_data=enrich.get('news_df'),
            )
            processed_df, _ = processor.data_processing_pipeline(indicators=indicators, verbose=False)
            processed_df['Symbol'] = sym
            if len(processed_df) <= cfg.window_size:
                console.print(f'[yellow]{sym}: dataset too small after processing — skipping.[/yellow]')
                continue
            processed_dfs.append(processed_df)
        except Exception as exc:
            console.print(f'[yellow]{sym}: processing failed ({exc}) — skipping.[/yellow]')

    if not processed_dfs:
        raise ValueError('No data processed for any symbol.')

    return pd.concat(processed_dfs).sort_index()


# ── Training & Evaluation ─────────────────────────────────────────────────────


def main(cfg: RunConfig = RunConfig()):
    algo_key = cfg.algo
    # Normalise casing: accept 'recurrentppo' → 'RecurrentPPO', 'sac' → 'SAC', etc.
    for key in PRESETS:
        if algo_key.lower() == key.lower():
            algo_key = key
            break
    if algo_key not in PRESETS:
        console.print(f'[red]Unknown algo "{cfg.algo}". Choose from: {", ".join(PRESETS)}[/red]')
        return

    preset = PRESETS[algo_key]
    console.rule(f'[bold blue]Multi-Symbol {algo_key} Training[/bold blue]')
    console.print(f'[dim]{preset.description}[/dim]')
    console.print(f'Symbols: {cfg.symbols}')
    overall_start = perf_counter()

    # --- Phase 1: Data & Split ---
    console.rule('[dim]Phase 1: Data & Split[/dim]')
    t0 = perf_counter()
    try:
        full_data = get_multi_stock_data(cfg)
    except Exception as exc:
        console.print(f'[red]Failed to fetch/process data: {exc}[/red]')
        import traceback

        traceback.print_exc()
        return

    if 'Symbol' not in full_data.columns or len(full_data['Symbol'].unique()) < 2:
        console.print('[red]Need at least 2 symbols in panel data. Exiting.[/red]')
        return

    unique_dates = full_data.index.unique().sort_values()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_data = full_data[full_data.index < split_date]
    test_data = full_data[full_data.index >= split_date]
    data_sec = perf_counter() - t0
    console.print(f'[cyan]Train:[/cyan] {len(train_data)} rows  [cyan]Test:[/cyan] {len(test_data)} rows')
    console.print(f'[dim]Phase 1 duration:[/dim] {data_sec:.2f}s')

    # Convert Symbol to numeric IDs for the observation space
    symbol_to_id = {sym: float(i) for i, sym in enumerate(sorted(full_data['Symbol'].unique().tolist()))}
    train_env_data = train_data.copy()
    test_env_data = test_data.copy()
    train_env_data['Symbol'] = train_env_data['Symbol'].map(symbol_to_id).astype('float32')
    test_env_data['Symbol'] = test_env_data['Symbol'].map(symbol_to_id).astype('float32')

    # --- Phase 2: Environment ---
    console.rule('[dim]Phase 2: Environment[/dim]')
    t0 = perf_counter()
    from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
    from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
    from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder

    env_config = (
        BacktestEnvironmentBuilder()
        .with_data(train_data=train_env_data, test_data=test_env_data)
        .with_env_params(
            initial_balance=100_000.0,
            transaction_cost_pct=0.001,
            window_size=cfg.window_size,
        )
        .with_strategies(
            action=StandardActionStrategy(),
            reward=preset.reward_factory(),
            observation=FeatureAwareObservationStrategy(normalize_stationary=True),
        )
        .build()
    )
    env_sec = perf_counter() - t0
    console.print(f'[dim]Phase 2 duration:[/dim] {env_sec:.2f}s')

    # --- Phase 3: Training ---
    console.rule(f'[dim]Phase 3: Training ({algo_key})[/dim]')
    from quantrl_lab.experiments.backtesting.core import ExperimentJob
    from quantrl_lab.experiments.backtesting.runner import BacktestRunner

    n_envs = len(full_data['Symbol'].unique()) if preset.n_envs_from_symbols else 1

    algo_config = preset.algorithm_config_factory()
    if cfg.algorithm_config:
        algo_config.update(cfg.algorithm_config)

    job = ExperimentJob(
        algorithm_class=preset.algo_class_factory(),
        env_config=env_config,
        algorithm_config=algo_config,
        total_timesteps=cfg.total_timesteps,
        n_envs=n_envs,
    )

    runner = BacktestRunner(verbose=True)
    result = runner.run_job(job)

    # --- Phase 4: Results ---
    console.rule('[dim]Phase 4: Results[/dim]')
    runner.inspect_result(result)

    total_sec = perf_counter() - overall_start
    phase_table = Table(title='Script Runtime Summary', show_header=True, header_style='bold green')
    phase_table.add_column('Phase', style='cyan')
    phase_table.add_column('Duration (s)', justify='right')
    phase_table.add_row('1. Data', f'{data_sec:.2f}')
    phase_table.add_row('2. Environment', f'{env_sec:.2f}')
    phase_table.add_row('3. Training + Eval', f'{result.execution_time:.2f}')
    phase_table.add_row('[bold]Total[/bold]', f'[bold]{total_sec:.2f}[/bold]')
    console.print(phase_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-symbol RL training')
    parser.add_argument(
        '--algo',
        type=str,
        default=RunConfig.algo,
        choices=list(PRESETS),
        help='Algorithm to use (default: %(default)s)',
    )
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        default=None,
        help='Space-separated list of symbols (default: 9 diversified tickers)',
    )
    parser.add_argument('--period-years', type=int, default=RunConfig.period_years)
    parser.add_argument('--window-size', type=int, default=RunConfig.window_size)
    parser.add_argument('--total-timesteps', type=int, default=RunConfig.total_timesteps)
    parser.add_argument(
        '--algorithm-config',
        type=str,
        default=None,
        metavar='JSON',
        help='JSON string of hyperparameter overrides, e.g. \'{"learning_rate": 3e-4}\'',
    )
    args = parser.parse_args()

    algo_config_override = json.loads(args.algorithm_config) if args.algorithm_config else None

    main(
        RunConfig(
            algo=args.algo,
            symbols=args.symbols,
            period_years=args.period_years,
            window_size=args.window_size,
            total_timesteps=args.total_timesteps,
            algorithm_config=algo_config_override,
        )
    )
