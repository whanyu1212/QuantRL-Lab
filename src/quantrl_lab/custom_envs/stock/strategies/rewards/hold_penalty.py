from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


# TODO: Think of a more robust way to penalize holding actions.


class HoldPenalty(BaseRewardStrategy):
    """Applies a small penalty for holding to encourage action."""

    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the reward based on the action taken in the
        environment. In this case, it applies a penalty if the action is
        a hold action.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The penalty for holding action.
        """
        if env.action_type == env.Actions.Hold:
            return self.penalty
        return 0.0
