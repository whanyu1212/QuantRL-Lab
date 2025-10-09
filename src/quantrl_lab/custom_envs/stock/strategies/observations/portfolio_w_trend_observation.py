from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Use TYPE_CHECKING to prevent circular imports
if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol

from quantrl_lab.custom_envs.utils.trend import calculate_trend_strength

from .base_observation import BaseObservationStrategy


class PortfolioWithTrendObservation(BaseObservationStrategy):
    """Constructs a detailed observation including a market window,
    portfolio state, and various engineered features like trend,
    volatility, and risk metrics."""

    NUM_PORTFOLIO_FEATURES = 9  # The number of features related to the portfolio state

    def __init__(self, volatility_lookback: int = 10, trend_lookback: int = 10):
        """
        Initialize the observation strategy with configurable lookback
        periods.

        Args:
            volatility_lookback (int): Lookback period for volatility calculation.
            trend_lookback (int): Lookback period for trend calculation.
        """
        super().__init__()
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback

    def define_observation_space(self, env: TradingEnvProtocol) -> gym.spaces.Box:
        """
        Defines the observation space based on the environment's
        parameters.

        Args:
            env (TradingEnvProtocol): The trading environment.

        Returns:
            gym.spaces.Box: The observation space.
        """
        obs_market_shape = env.window_size * env.num_features
        total_obs_dim = obs_market_shape + self.NUM_PORTFOLIO_FEATURES

        return spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

    def build_observation(self, env: TradingEnvProtocol) -> np.ndarray:
        """
        Builds the observation vector for the current state.

        Args:
            env (TradingEnvProtocol): The trading environment.

        Returns:
            np.ndarray: The observation vector.

        This observation includes:
            - A normalized market window of the last `window_size` steps.
            - Portfolio information such as balance ratio, position size ratio,
              unrealized profit/loss percentage, risk/reward ratio, and distances to stop and target
              prices.
            - Trend strength calculated from the market data.
            - Volatility based on recent price movements.
            - Recent high and low prices for additional context.
        """

        # === 1. Market Window Processing ===
        start_idx = max(0, env.current_step - env.window_size + 1)
        end_idx = env.current_step + 1
        market_window = env.data[start_idx:end_idx, :]

        actual_len = market_window.shape[0]
        if actual_len < env.window_size:
            if actual_len > 0:
                padding = np.repeat(market_window[0, :][np.newaxis, :], env.window_size - actual_len, axis=0)
            else:
                padding = np.zeros((env.window_size - actual_len, env.num_features), dtype=env.data.dtype)
            market_window = np.concatenate((padding, market_window), axis=0)

        first_step_values = market_window[0, :]
        denominator = np.where(np.abs(first_step_values) < 1e-9, 1.0, first_step_values)
        normalized_market_window = market_window / denominator
        normalized_market_window[:, np.abs(first_step_values) < 1e-9] = 0.0

        # === 2. Enhanced Portfolio Information ===
        current_price = env._get_current_price()  # The env is still the source of price
        total_shares = env.portfolio.total_shares

        # Default values
        position_size_ratio, unrealized_pl_pct, risk_reward_ratio, distance_to_stop, distance_to_target = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        if total_shares > 0:
            portfolio_value = env.portfolio.get_value(current_price)
            total_position_value = total_shares * current_price
            position_size_ratio = total_position_value / portfolio_value if portfolio_value > 1e-9 else 0.0

            entry_prices = [
                o["price"]
                for o in env.portfolio.executed_orders_history
                if o["type"] in ["market_buy", "limit_buy_executed"]
            ]
            avg_entry_price = np.mean(entry_prices) if entry_prices else current_price
            unrealized_pl_pct = (current_price - avg_entry_price) / avg_entry_price if avg_entry_price > 1e-9 else 0.0

            sl_prices = [o["price"] for o in env.portfolio.stop_loss_orders]
            tp_prices = [o["price"] for o in env.portfolio.take_profit_orders]
            if sl_prices and tp_prices:
                avg_stop_price = np.mean(sl_prices)
                avg_target_price = np.mean(tp_prices)
                if abs(current_price - avg_stop_price) > 1e-9:
                    risk_reward_ratio = (avg_target_price - current_price) / (current_price - avg_stop_price)
                distance_to_stop = (current_price - avg_stop_price) / current_price if current_price > 1e-9 else 0.0
                distance_to_target = (avg_target_price - current_price) / current_price if current_price > 1e-9 else 0.0

        # === 3. Feature Engineering ===
        recent_slice = env.data[max(0, env.current_step - self.volatility_lookback + 1) : end_idx]  # noqa: E203
        recent_high = np.max(recent_slice[:, env.price_column_index])
        recent_low = np.min(recent_slice[:, env.price_column_index])

        if len(recent_slice) > 1:
            returns = np.diff(recent_slice[:, env.price_column_index]) / recent_slice[:-1, env.price_column_index]
            volatility = np.std(returns)
        else:
            volatility = 0.0

        trend = calculate_trend_strength(env, lookback=self.trend_lookback)

        # === 4. Combine into Final Observation Vector ===
        portfolio_info = np.array(
            [
                env.portfolio.balance / env.portfolio.initial_balance,
                position_size_ratio,
                unrealized_pl_pct,
                (current_price - recent_low) / (recent_high - recent_low + 1e-9),  # prevent division by zero
                volatility,
                trend,
                risk_reward_ratio,
                distance_to_stop,
                distance_to_target,
            ],
            dtype=np.float32,
        )

        flattened_market_obs = normalized_market_window.flatten()
        return np.concatenate((flattened_market_obs, portfolio_info))
