from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.stock.stock_env import StockTradingEnv


class BaseObservationStrategy(ABC):
    """Abstract base class for defining how an agent perceives the
    environment."""

    @abstractmethod
    def define_observation_space(self, env: StockTradingEnv) -> gym.spaces.Space:
        """
        Defines and returns the observation space for the environment.

        Args:
            env (StockTradingEnv): The stock trading environment.

        Returns:
            gym.spaces.Space: The observation space.
        """
        pass

    @abstractmethod
    def build_observation(self, env: StockTradingEnv) -> np.ndarray:
        """
        Constructs and returns the observation vector based on the
        current state of the environment.

        Args:
            env (StockTradingEnv): The stock trading environment.

        Returns:
            np.ndarray: The observation vector.
        """
        pass
