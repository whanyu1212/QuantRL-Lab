from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol

from quantrl_lab.custom_envs.core.calculations.trend import calculate_trend_strength
from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)


# TODO: Think of a more robust way to reward trend following actions.
class TrendFollowingReward(BaseRewardStrategy):
    """
    Rewards/penalizes actions based on the current price trend.

    - Rewards taking profit in an uptrend.
    - Rewards stop-loss in a downtrend.

    Parameters:
        reward_multiplier (float): Multiplier to scale the trend-based rewards.
            Default is 0.05.
    """

    def __init__(self, reward_multiplier: float = 0.05):
        super().__init__()
        self.reward_multiplier = reward_multiplier

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the reward based on the action taken in the
        environment. This reward is based on the current price trend. It
        rewards taking profit in an uptrend and stop-loss in a
        downtrend. If the action is not aligned with the trend, it does
        not apply a reward.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: reward based on the trend following actions
        """
        action_type = env.action_type
        trend = calculate_trend_strength(env, lookback=10)
        trend_strength = abs(trend)
        is_uptrend = trend > 0
        is_downtrend = trend < 0

        reward = 0.0
        if action_type == env.Actions.TakeProfit and is_uptrend:
            reward += self.reward_multiplier * (1 + trend_strength)
        elif action_type == env.Actions.StopLoss and is_downtrend:
            reward += self.reward_multiplier * (1 + trend_strength)

        return reward
