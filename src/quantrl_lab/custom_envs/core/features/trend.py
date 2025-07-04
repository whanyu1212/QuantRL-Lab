from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.stock.stock_env import StockTradingEnv


class TrendCalculable(Protocol):
    """Protocol for environments that support trend calculation."""

    current_step: int
    data: np.ndarray
    price_column_index: int


def calculate_trend_strength(env: Union[TrendCalculable, gym.Env, 'StockTradingEnv'], lookback: int = 10) -> float:
    """
    Calculates the normalized slope of the price trend over a lookback
    period.

    Args:
        env (Union[TrendCalculable, gym.Env, 'StockTradingEnv']): The environment to calculate the trend from.
        lookback (int, optional): The number of steps to look back for the trend calculation. Defaults to 10.

    Raises:
        AttributeError: If the environment is missing required attributes.

    Returns:
        float: The normalized trend strength, typically between -1 and 1.
    """
    # Ensure env has the required attributes
    if not hasattr(env, 'current_step') or not hasattr(env, 'data') or not hasattr(env, 'price_column_index'):
        raise AttributeError("Environment must have current_step, data, and price_column_index attributes")

    if env.current_step < lookback or len(env.data) < lookback:
        return 0.0

    end_idx = env.current_step + 1
    start_idx = end_idx - lookback
    recent_prices = env.data[start_idx:end_idx, env.price_column_index]

    if len(recent_prices) < 2:
        return 0.0

    x = np.arange(len(recent_prices))
    slope, _ = np.polyfit(x, recent_prices, 1)

    max_price = np.max(recent_prices)
    if max_price > 1e-9:
        return slope / max_price

    return 0.0
