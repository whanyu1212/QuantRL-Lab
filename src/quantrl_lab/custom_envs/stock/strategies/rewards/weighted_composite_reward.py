from __future__ import annotations

from typing import TYPE_CHECKING, List

from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class WeightedCompositeReward(BaseRewardStrategy):
    """
    A composite strategy that combines multiple reward strategies with
    weights.

    This class implements the Composite design pattern.
    """

    def __init__(self, strategies: List[BaseRewardStrategy], weights: List[float]):
        if len(strategies) != len(weights):
            raise ValueError("The number of strategies and weights must be equal.")

        self.strategies = strategies
        self.weights = weights

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the weighted composite reward based on the child
        strategies.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The weighted composite reward based on the child strategies.
        """
        total_reward = 0.0
        for strategy, weight in zip(self.strategies, self.weights):
            # Calculate the reward from the child strategy and apply the weight
            component_reward = strategy.calculate_reward(env)
            total_reward += weight * component_reward

        return total_reward

    def on_step_end(self, env: TradingEnvProtocol):
        """Optional: A hook to update any internal state if needed.
        This method is called at the end of each step in the environment.

        Args:
            env (TradingEnvProtocol): The trading environment instance.
        """
        for strategy in self.strategies:
            strategy.on_step_end(env)
