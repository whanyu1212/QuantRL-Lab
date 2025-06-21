from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

import gymnasium as gym

# Solves circular import issues by using TYPE_CHECKING
if TYPE_CHECKING:
    from quantrl_lab.custom_envs.stock.envs.stock_trading_env import StockTradingEnv


class BaseActionStrategy(ABC):
    """
    An abstract base class for defining action spaces and handling agent
    actions.

    This class is not intended to be used directly. Instead, create a
    subclass that implements the abstract methods.
    """

    @abstractmethod
    def define_action_space(self) -> gym.spaces.Space:
        """Defines and returns the action space for the environment."""
        pass

    @abstractmethod
    def handle_action(self, env_self: StockTradingEnv, action: Any) -> Tuple[Any, Dict[str, Any]]:
        """Processes the raw action from the agent and executes the
        trade."""
        pass
