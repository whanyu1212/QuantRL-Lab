import numpy as np
from stable_baselines3.common.env_checker import check_env

from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


def test_env_creation(standard_env: TradingEnvProtocol):
    """Test that environment can be created without errors."""
    assert standard_env is not None
    assert standard_env.action_space is not None
    assert standard_env.observation_space is not None


def test_env_reset(standard_env: TradingEnvProtocol):
    """Test that environment can be reset."""
    obs, info = standard_env.reset()
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert info is not None
    assert isinstance(info, dict)


def test_env_step(standard_env: TradingEnvProtocol):
    """Test that environment can take a step."""
    standard_env.reset()
    action = standard_env.action_space.sample()
    obs, reward, terminated, truncated, info = standard_env.step(action)

    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_sb3_compatibility(standard_env: TradingEnvProtocol):
    """Test compatibility with Stable Baselines 3."""
    check_env(standard_env)
