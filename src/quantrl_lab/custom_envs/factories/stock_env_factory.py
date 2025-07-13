import numpy as np

from quantrl_lab.custom_envs.stock import SingleStockEnvConfig, SingleStockTradingEnv


def create_stock_env_factories(
    train_data: np.ndarray,
    test_data: np.ndarray,
    action_strategy,
    reward_strategy,
    observation_strategy,
    config_overrides: dict = None,
) -> dict:
    """
    Create train and test environment factories for stock trading.

    Args:
        train_data (np.ndarray): Training data for the environment.
        test_data (np.ndarray): Testing data for the environment.
        action_strategy (_type_): Action strategy to use.
        reward_strategy (_type_): Reward strategy to use.
        observation_strategy (_type_): Observation strategy to use.
        config_overrides (dict, optional): Configuration overrides. Defaults to None.

    Returns:
        dict: A dictionary containing train and test environment factories.
    """

    base_config = SingleStockEnvConfig(
        initial_balance=100000.0,
        transaction_cost_pct=0.001,
        slippage=0.0005,
        window_size=10,
        price_column_index=3,
        order_expiration_steps=5,
        **(config_overrides or {}),
    )

    return {
        'train_env_factory': lambda: SingleStockTradingEnv(
            data=train_data,
            config=base_config,
            action_strategy=action_strategy,
            reward_strategy=reward_strategy,
            observation_strategy=observation_strategy,
        ),
        'test_env_factory': lambda: SingleStockTradingEnv(
            data=test_data,
            config=base_config,
            action_strategy=action_strategy,
            reward_strategy=reward_strategy,
            observation_strategy=observation_strategy,
        ),
    }
