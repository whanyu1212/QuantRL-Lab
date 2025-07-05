from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

import gymnasium as gym

# Solves circular import issues by using TYPE_CHECKING
if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class BaseActionStrategy(ABC):
    """
    An abstract base class for defining action spaces and handling agent
    actions.

    This class is not intended to be used directly. Instead, create a
    subclass that implements the abstract methods.
    """

    @abstractmethod
    def define_action_space(self) -> gym.spaces.Space:
        """
        Defines the action space for the environment.

        Returns:
            gym.spaces.Space: The action space for the environment.
        """
        pass

    @abstractmethod
    def handle_action(self, env_self: TradingEnvProtocol, action: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Handles the action taken by the agent in the environment.

        Args:
            env_self (TradingEnvProtocol): The environment instance where the action is taken.
            action (Any): The action taken by the agent,
            which should be compatible with the defined action space.

        Returns:
            Tuple[Any, Dict[str, Any]]: The outcome of the action taken in the environment
        """
        pass
