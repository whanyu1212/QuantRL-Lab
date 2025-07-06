from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import gymnasium as gym
import numpy as np


class TradingEnvProtocol(Protocol):
    """Protocol defining the interface for trading environments."""

    # Compulsory attributes for trading environments
    data: np.ndarray
    current_step: int
    price_column_index: int
    window_size: int
    action_space: gym.Space
    observation_space: gym.Space

    # Compulsory methods for trading environments
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]: ...
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]: ...
    def render(self, mode: str = "human"): ...
    def close(self): ...
