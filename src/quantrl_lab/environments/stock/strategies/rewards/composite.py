from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np

from quantrl_lab.environments.core.interfaces import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.environments.core.interfaces import TradingEnvProtocol


class _RunningStat:
    """
    Internal helper to track running Mean and Variance using Welford's
    algorithm.

    Keeps the main class clean.
    """

    def __init__(self, clip: float = 5.0):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squares of differences
        self.var = 1.0
        self.clip = clip

    def update_and_normalize(self, value: float) -> float:
        # 1. Update stats
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

        if self.count > 1:
            self.var = self.M2 / (self.count - 1)

        # 2. Normalize (if we have enough data)
        if self.count < 2:
            return value

        sigma = np.sqrt(self.var) + 1e-8
        normalized = (value - self.mean) / sigma

        # 3. Clip
        return float(np.clip(normalized, -self.clip, self.clip))


class CompositeReward(BaseRewardStrategy):
    """
    A composite strategy that combines multiple reward strategies with
    weights.

    This class implements the Composite design pattern.

    Features:
    - Weight Normalization: Ensures weights sum to 1.0.
    - Auto-Scaling: Optionally normalizes each component strategy to N(0,1)
      before weighting, preventing one strategy from dominating others due to scale.
    """

    def __init__(
        self,
        strategies: List[BaseRewardStrategy],
        weights: List[float],
        normalize_weights: bool = True,
        auto_scale: bool = False,
    ):
        if len(strategies) != len(weights):
            raise ValueError("The number of strategies and weights must be equal.")

        self.strategies = strategies
        self.weights = weights
        self.normalize_weights = normalize_weights
        self.auto_scale = auto_scale

        # Initialize stats trackers if auto-scaling is enabled
        self._stats = [_RunningStat() for _ in strategies] if auto_scale else []

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the composite reward based on the child strategies.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The composite reward based on the child strategies.
        """
        weights_to_use = self.weights
        if self.normalize_weights:
            total_weight = sum(self.weights)
            if total_weight == 0:
                raise ValueError("Sum of weights must not be zero when normalize_weights is True.")
            weights_to_use = [w / total_weight for w in self.weights]

        total_reward = 0.0
        for i, (strategy, weight) in enumerate(zip(self.strategies, weights_to_use)):
            # Calculate the reward from the child strategy
            component_reward = strategy.calculate_reward(env)

            # Auto-scale if enabled
            if self.auto_scale:
                component_reward = self._stats[i].update_and_normalize(component_reward)

            total_reward += weight * component_reward

        return total_reward

    def on_step_end(self, env: TradingEnvProtocol):
        """
        Optional: A hook to update any internal state if needed. This
        method is called at the end of each step in the environment.

        Args:
            env (TradingEnvProtocol): The trading environment instance.
        """
        for strategy in self.strategies:
            strategy.on_step_end(env)

    def reset(self):
        """
        Reset child strategies.

        Note: We do NOT reset running stats (if auto_scale=True) because they
        represent the global distribution of the environment rewards, which
        should persist across episodes for stability.
        """
        for strategy in self.strategies:
            if hasattr(strategy, "reset"):
                strategy.reset()
