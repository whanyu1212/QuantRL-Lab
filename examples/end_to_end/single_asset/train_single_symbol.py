"""
Unified single-stock training script.

Supports all algorithms via ``--algo``:
  SAC, PPO, A2C, TD3, TQC, TRPO, CrossQ

Each algorithm ships with a tuned preset (reward strategy + hyperparameters).
All other settings (symbol, timesteps, window) are controlled via flags or
by editing ``RunConfig``.

Usage::

    python train_single_symbol.py --algo SAC --symbol AAPL
    python train_single_symbol.py --algo PPO --symbol MU --total-timesteps 100000
    python train_single_symbol.py --algo TD3 --symbol NVDA --period-years 3

Shared infrastructure (data-source init, date helpers) lives in
``examples/end_to_end/shared/data_utils.py``.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

load_dotenv()
console = Console()

from shared.data_utils import (  # noqa: E402
    get_date_range,
    init_data_sources,
)

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Top-level experiment configuration."""

    algo: str = "SAC"
    symbol: str = "MU"
    period_years: int = 5
    window_size: int = 15
    total_timesteps: int = 50_000
    algorithm_config: Optional[Dict[str, Any]] = None  # overrides/merges with preset


@dataclass
class AlgorithmPreset:
    """
    Algorithm-specific configuration: class, hyperparameters, n_envs,
    and a factory for the reward strategy.

    ``reward_factory`` is a zero-argument callable so reward objects
    (which may import heavy deps) are created lazily at runtime.
    ``algorithm_config_factory`` is used instead of a plain dict for
    algorithms like TD3 that need runtime-constructed objects
    (e.g. ``NormalActionNoise``). For all others it just returns the
    static dict.
    """

    algo_class_factory: Any  # zero-arg callable → algorithm class
    algorithm_config_factory: Any  # zero-arg callable → Dict[str, Any]
    reward_factory: Any  # zero-arg callable → reward strategy instance
    n_envs: int = 1
    description: str = ""


def _make_presets() -> Dict[str, AlgorithmPreset]:
    """Build the preset registry lazily so heavy imports only happen
    when a preset is actually requested."""

    def sac_class():
        from stable_baselines3 import SAC

        return SAC

    def sac_config():
        return dict(
            policy='MlpPolicy',
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=5_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], qf=[128, 128])),
            verbose=0,
        )

    def sac_reward():
        from quantrl_lab.environments.stock.strategies.rewards.sharpe import DifferentialSharpeReward

        return DifferentialSharpeReward(risk_free_rate=0.0)

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

    def td3_class():
        from stable_baselines3 import TD3

        return TD3

    def td3_config():
        import numpy as np
        from stable_baselines3.common.noise import NormalActionNoise

        # Action space for StandardActionStrategy is Box(1,) in [-1, 1]
        action_noise = NormalActionNoise(mean=np.zeros(1), sigma=0.1 * np.ones(1))
        return dict(
            policy='MlpPolicy',
            learning_rate=1e-4,
            buffer_size=300_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
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
            strategies=[
                PortfolioValueChangeReward(),
                TurnoverPenaltyReward(penalty_factor=0.1),
            ],
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
            buffer_size=300_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            top_quantiles_to_drop_per_net=1,
            policy_kwargs=dict(net_arch=[256, 256], n_quantiles=25),
            verbose=0,
        )

    def tqc_reward():
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

    def crossq_class():
        from sb3_contrib import CrossQ

        return CrossQ

    def crossq_config():
        return dict(
            policy='MlpPolicy',
            learning_rate=1e-4,
            buffer_size=300_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
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
            strategies=[
                PortfolioValueChangeReward(),
                TurnoverPenaltyReward(penalty_factor=0.1),
            ],
            weights=[1.0, 0.05],
            auto_scale=False,
        )

    # ──────────────────────────────────────────────────────────────────────────

    return {
        'SAC': AlgorithmPreset(
            algo_class_factory=sac_class,
            algorithm_config_factory=sac_config,
            reward_factory=sac_reward,
            n_envs=1,
            description='Off-policy, max-entropy. Best sample efficiency.',
        ),
        'PPO': AlgorithmPreset(
            algo_class_factory=ppo_class,
            algorithm_config_factory=ppo_config,
            reward_factory=ppo_reward,
            n_envs=1,
            description='On-policy, clipped surrogate. Stable and widely used.',
        ),
        'A2C': AlgorithmPreset(
            algo_class_factory=a2c_class,
            algorithm_config_factory=a2c_config,
            reward_factory=a2c_reward,
            n_envs=4,
            description='On-policy actor-critic. Fast, lower variance than REINFORCE.',
        ),
        'TD3': AlgorithmPreset(
            algo_class_factory=td3_class,
            algorithm_config_factory=td3_config,
            reward_factory=td3_reward,
            n_envs=1,
            description='Off-policy, deterministic. Double Q + delayed policy update.',
        ),
        'TQC': AlgorithmPreset(
            algo_class_factory=tqc_class,
            algorithm_config_factory=tqc_config,
            reward_factory=tqc_reward,
            n_envs=1,
            description='Off-policy, distributional Q. Often outperforms SAC.',
        ),
        'TRPO': AlgorithmPreset(
            algo_class_factory=trpo_class,
            algorithm_config_factory=trpo_config,
            reward_factory=trpo_reward,
            n_envs=4,
            description='On-policy, trust-region. Monotonic improvement guarantee.',
        ),
        'CrossQ': AlgorithmPreset(
            algo_class_factory=crossq_class,
            algorithm_config_factory=crossq_config,
            reward_factory=crossq_reward,
            n_envs=1,
            description='Off-policy, batch-norm critic. High sample efficiency.',
        ),
    }


PRESETS: Dict[str, AlgorithmPreset] = _make_presets()


# ── Data ──────────────────────────────────────────────────────────────────────


def build_processed_data(cfg: RunConfig):
    """
    Fetch and process data for a single symbol.

    Returns ``(train_df, test_df)`` ready for the trading environment.
    Alpha indicator selection is performed explicitly via
    ``alpha_research`` before the resulting indicator specs are passed
    into ``DataProcessor``.
    """
    from quantrl_lab.alpha_research import build_processing_config_from_alpha_selection
    from quantrl_lab.data.processing import DataProcessor

    sources = init_data_sources()
    start_date, end_date = get_date_range(cfg.period_years)

    raw_df = sources.loader.get_historical_ohlcv_data(
        symbols=[cfg.symbol], start=start_date, end=end_date, timeframe='1d'
    )
    if 'Date' in raw_df.columns:
        raw_df = raw_df.set_index('Date')
    console.print(f'[cyan]Raw OHLCV:[/cyan] {raw_df.shape}')

    processor = DataProcessor(ohlcv_data=raw_df)
    processing_config, _ = build_processing_config_from_alpha_selection(
        raw_df,
        {'metric': 'ic', 'threshold': 0.02, 'top_k': 4},
        split_config={'train': 0.8, 'test': 0.2},
        verbose=True,
    )
    splits, _ = processor.data_processing_pipeline(
        pipeline_config=processing_config,
    )

    train_df = splits['train'].select_dtypes(include='number')
    test_df = splits['test'].select_dtypes(include='number')
    console.print(f'[green]Train:[/green] {train_df.shape}  [green]Test:[/green] {test_df.shape}')
    console.print(f'[green]Features:[/green] {list(train_df.columns)}')
    return train_df, test_df


# ── Training & Evaluation ─────────────────────────────────────────────────────


def main(cfg: RunConfig = RunConfig()):
    algo_key = cfg.algo.upper()
    if algo_key not in PRESETS:
        console.print(f'[red]Unknown algo "{cfg.algo}". Choose from: {", ".join(PRESETS)}[/red]')
        return

    preset = PRESETS[algo_key]
    console.rule(f'[bold blue]Single-Stock {algo_key} Training — {cfg.symbol}[/bold blue]')
    console.print(f'[dim]{preset.description}[/dim]')
    overall_start = perf_counter()

    # --- Phase 1: Data ---
    console.rule('[dim]Phase 1: Data[/dim]')
    t0 = perf_counter()
    train_df, test_df = build_processed_data(cfg)
    data_sec = perf_counter() - t0
    console.print(f'[green]Train date range:[/green] {train_df.index.min().date()} → {train_df.index.max().date()}')
    console.print(f'[green]Test date range:[/green]  {test_df.index.min().date()} → {test_df.index.max().date()}')
    console.print(f'[dim]Phase 1 duration:[/dim] {data_sec:.2f}s')

    if len(train_df) <= cfg.window_size:
        console.print(f'[red]Train set too small ({len(train_df)} rows ≤ window_size={cfg.window_size})[/red]')
        return
    if len(test_df) <= cfg.window_size:
        console.print(f'[red]Test set too small ({len(test_df)} rows ≤ window_size={cfg.window_size})[/red]')
        return

    # --- Phase 2: Environment ---
    console.rule('[dim]Phase 2: Environment[/dim]')
    t0 = perf_counter()
    from quantrl_lab.environments.stock.strategies.actions.standard import StandardActionStrategy
    from quantrl_lab.environments.stock.strategies.observations.feature_aware import FeatureAwareObservationStrategy
    from quantrl_lab.experiments.backtesting.builder import BacktestEnvironmentBuilder

    builder = BacktestEnvironmentBuilder()
    builder.with_data(train_data=train_df, test_data=test_df)
    builder.with_env_params(
        initial_balance=100_000.0,
        transaction_cost_pct=0.001,
        window_size=cfg.window_size,
        random_start=True,
    )
    builder.with_strategies(
        action=StandardActionStrategy(),
        reward=preset.reward_factory(),
        observation=FeatureAwareObservationStrategy(normalize_stationary=True),
    )
    env_config = builder.build()
    env_sec = perf_counter() - t0
    console.print(f'[dim]Phase 2 duration:[/dim] {env_sec:.2f}s')

    # --- Phase 3: Training ---
    console.rule(f'[dim]Phase 3: Training ({algo_key})[/dim]')
    from quantrl_lab.experiments.backtesting.core import ExperimentJob
    from quantrl_lab.experiments.backtesting.runner import BacktestRunner

    algo_config = preset.algorithm_config_factory()
    if cfg.algorithm_config:
        algo_config.update(cfg.algorithm_config)

    job = ExperimentJob(
        algorithm_class=preset.algo_class_factory(),
        env_config=env_config,
        algorithm_config=algo_config,
        total_timesteps=cfg.total_timesteps,
        n_envs=preset.n_envs,
    )

    runner = BacktestRunner(verbose=True)
    result = runner.run_job(job)

    # --- Phase 4: Results ---
    console.rule('[dim]Phase 4: Results[/dim]')
    from quantrl_lab.experiments.benchmarks import BenchmarkComparison

    # Detect the close price column (handles 'close', 'Close', 'adj_close', etc.)
    close_col = next((c for c in test_df.columns if c.lower() in ('close', 'adj_close')), test_df.columns[3])
    benchmarks = BenchmarkComparison(
        result,
        index_symbol=cfg.symbol,
        index_start=test_df.index.min().strftime('%Y-%m-%d'),
        index_end=test_df.index.max().strftime('%Y-%m-%d'),
        price_series=test_df[close_col],
    )
    runner.inspect_result(result, benchmarks=benchmarks)

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
    parser = argparse.ArgumentParser(description='Single-stock RL training')
    parser.add_argument(
        '--algo',
        type=str,
        default=RunConfig.algo,
        choices=list(PRESETS),
        help='Algorithm to use (default: %(default)s)',
    )
    parser.add_argument('--symbol', type=str, default=RunConfig.symbol)
    parser.add_argument('--period-years', type=int, default=RunConfig.period_years)
    parser.add_argument('--window-size', type=int, default=RunConfig.window_size)
    parser.add_argument('--total-timesteps', type=int, default=RunConfig.total_timesteps)
    parser.add_argument(
        '--algorithm-config',
        type=str,
        default=None,
        metavar='JSON',
        help='JSON string of hyperparameter overrides, e.g. \'{"learning_rate": 1e-3}\'',
    )
    args = parser.parse_args()

    algo_config_override = json.loads(args.algorithm_config) if args.algorithm_config else None

    main(
        RunConfig(
            algo=args.algo,
            symbol=args.symbol,
            period_years=args.period_years,
            window_size=args.window_size,
            total_timesteps=args.total_timesteps,
            algorithm_config=algo_config_override,
        )
    )
