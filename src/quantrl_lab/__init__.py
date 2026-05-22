"""
QuantRL-Lab: A modular reinforcement learning framework for quantitative
trading.

Main modules:
- alpha_research: Signal discovery and validation
- environments: Trading environments with pluggable strategies
- data: Data acquisition and feature engineering
- experiments: Backtesting, feature engineering, hyperparameter tuning, and hedge pair screening
- deployment: Live trading
- utils: Shared utilities
"""

__version__ = "0.1.0"

from quantrl_lab import (
    alpha_research,
    data,
    deployment,
    environments,
    experiments,
    utils,
)

__all__ = [
    "alpha_research",
    "data",
    "deployment",
    "environments",
    "experiments",
    "utils",
]
