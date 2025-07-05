from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class BaseRewardStrategy(ABC):
    @abstractmethod
    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate the reward based on the action taken in the
        environment.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.

        Returns:
            float: The calculated reward.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def on_step_end(self, env: TradingEnvProtocol):
        """Optional: A hook to update any internal state if needed."""
        pass
