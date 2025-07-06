import numpy as np
import pytest

from quantrl_lab.custom_envs.stock import SingleStockEnvConfig, SingleStockTradingEnv
from quantrl_lab.custom_envs.stock.strategies.actions.types.standard_market_action_strategy import (
    StandardMarketActionStrategy,
)
from quantrl_lab.custom_envs.stock.strategies.observations import (
    PortfolioWithTrendObservation,
)
from quantrl_lab.custom_envs.stock.strategies.rewards import PortfolioValueChangeReward


@pytest.fixture
def sample_data() -> tuple[np.ndarray, int]:
    """
    Create sample price data for testing.

    Returns:
        tuple[np.ndarray, int]: The generated price data
        and the index of the price column.
    """
    data_size = 100
    price_column = 3
    data = np.random.rand(data_size, 5).astype(np.float32)
    data[:, price_column] = 50 + np.arange(data_size) * 0.2 + np.random.randn(data_size) * 0.5
    return data, price_column


@pytest.fixture
def stock_env_config() -> SingleStockEnvConfig:
    """
    Create a standard config for testing.

    Returns:
        SingleStockEnvConfig: The generated environment config.
    """
    return SingleStockEnvConfig(
        price_column_index=3, window_size=10, initial_balance=10000.0, transaction_cost_pct=0.001, slippage=0.0005
    )


@pytest.fixture
def standard_env(sample_data: tuple[np.ndarray, int], stock_env_config: SingleStockEnvConfig) -> SingleStockTradingEnv:
    """
    Create a standard environment for testing.

    Args:
        sample_data (tuple[np.ndarray, int]): The generated price data and the index of the price column.
        stock_env_config (SingleStockEnvConfig): The environment configuration.

    Returns:
        SingleStockTradingEnv: The created trading environment.
    """
    data, _ = sample_data
    env = SingleStockTradingEnv(
        data=data,
        config=stock_env_config,
        action_strategy=StandardMarketActionStrategy(),
        reward_strategy=PortfolioValueChangeReward(),
        observation_strategy=PortfolioWithTrendObservation(),
    )
    return env
