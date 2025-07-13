from .algorithm_config import A2CConfig, PPOConfig, SACConfig
from .config_registry import AlgorithmConfigRegistry

# For backward compatibility, expose the preset config method at module level
get_preset_config = AlgorithmConfigRegistry.get_preset_config

__all__ = [
    'PPOConfig',
    'A2CConfig',
    'SACConfig',
    'AlgorithmConfigRegistry',
    'get_preset_config',
]  # allows you to import * from quantrl_lab.backtesting.config
