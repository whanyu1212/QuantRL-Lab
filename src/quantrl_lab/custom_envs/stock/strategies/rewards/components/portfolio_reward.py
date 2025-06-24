from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.stock.env_single_stock import StockTradingEnv


class PortfolioValueChangeReward(BaseRewardStrategy):
    """Calculates reward based on the % change in portfolio value."""

    def calculate_reward(self, env: StockTradingEnv) -> float:
        """
        Calculate the reward based on the percentage change in portfolio
        value. This reward is calculated as the percentage change in
        portfolio value from the previous step to the current step.

        Args:
            env (StockTradingEnv): StockTradingEnv instance

        Returns:
            float: percentage change in portfolio value
        """
        prev_val = env.prev_portfolio_value
        current_val = env._get_portfolio_value()

        if prev_val > 1e-9:
            return (current_val - prev_val) / prev_val
        return 0.0
