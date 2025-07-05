from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class BaseObservationStrategy(ABC):
    """Abstract base class for defining how an agent perceives the
    environment."""

    @abstractmethod
    def define_observation_space(self, env: TradingEnvProtocol) -> gym.spaces.Space:
        """
        Defines and returns the observation space for the environment.

        Args:
            env (TradingEnvProtocol): The trading environment.

        Returns:
            gym.spaces.Space: The observation space.
        """
        pass

    @abstractmethod
    def build_observation(self, env: TradingEnvProtocol) -> np.ndarray:
        """
        Builds the observation vector for the current state.

        Args:
            env (TradingEnvProtocol): The trading environment.

        Returns:
            np.ndarray: The observation vector.
        """
        pass
