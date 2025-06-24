from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.stock.env_single_stock import StockTradingEnv


class InvalidActionPenalty(BaseRewardStrategy):
    """
    Applies a fixed penalty for attempting an invalid action.

    E.g. Sell or Limit Sell when no shares are held.
    """

    def __init__(self, penalty: float = -1.0):
        self.penalty = penalty

    def calculate_reward(self, env: StockTradingEnv) -> float:
        """
        Calculate the reward based on the action taken in the
        environment. If an invalid action is attempted, a penalty is
        applied.

        Args:
            env (StockTradingEnv): StockTradingEnv instance

        Returns:
            float: penalty for invalid action attempt
        """
        if env.decoded_action_info.get("invalid_action_attempt", False):
            return self.penalty
        return 0.0
