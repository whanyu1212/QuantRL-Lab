from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class PortfolioValueChangeReward(BaseRewardStrategy):
    """Calculates reward based on the % change in portfolio value."""

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculates the reward based on the percentage change in
        portfolio value.

        This method now correctly interacts with the environment's portfolio
        component to get the current value.

        Args:
            env (TradingEnvProtocol): The environment instance.

        Returns:
            float: The percentage change in portfolio value since the previous step.
        """
        prev_val = env.prev_portfolio_value

        current_price = env._get_current_price()

        current_val = env.portfolio.get_value(current_price)

        if prev_val > 1e-9:
            return (current_val - prev_val) / prev_val

        return 0.0
