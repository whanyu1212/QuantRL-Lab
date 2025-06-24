from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.stock.env_single_stock import StockTradingEnv


class BaseRewardStrategy(ABC):
    @abstractmethod
    def calculate_reward(self, env: StockTradingEnv) -> float:
        """
        Calculate the reward based on the action taken in the
        environment. This method should be implemented by subclasses to
        define how the reward is calculated based on the current state
        of the environment and the action taken.

        Args:
            env (StockTradingEnv): StockTradingEnv instance

        Returns:
            float: reward based on the action taken in the environment
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def on_step_end(self, env: StockTradingEnv):
        """Optional: A hook to update any internal state if needed."""
        pass
