from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class InvalidActionPenalty(BaseRewardStrategy):
    """
    Applies a fixed penalty for attempting an invalid action.

    E.g. Sell or Limit Sell when no shares are held.
    """

    def __init__(self, penalty: float = -1.0):
        self.penalty = penalty

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the reward based on the action taken in the
        environment. If an invalid action is attempted, a penalty is
        applied.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The penalty for invalid action attempt.
        """
        if env.decoded_action_info.get("invalid_action_attempt", False):
            return self.penalty
        return 0.0
