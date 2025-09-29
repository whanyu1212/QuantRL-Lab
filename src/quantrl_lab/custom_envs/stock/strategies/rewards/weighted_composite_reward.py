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

    def __init__(self, strategies: List[BaseRewardStrategy], weights: List[float], normalize_weights: bool = True):
        if len(strategies) != len(weights):
            raise ValueError("The number of strategies and weights must be equal.")

        self.strategies = strategies
        self.weights = weights
        self.normalize_weights = normalize_weights

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the weighted composite reward based on the child
        strategies.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The weighted composite reward based on the child strategies.
        """
        weights_to_use = self.weights
        if self.normalize_weights:
            total_weight = sum(self.weights)
            if total_weight == 0:
                raise ValueError("Sum of weights must not be zero when normalize_weights is True.")
            weights_to_use = [w / total_weight for w in self.weights]
        total_reward = 0.0
        for strategy, weight in zip(self.strategies, weights_to_use):
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
