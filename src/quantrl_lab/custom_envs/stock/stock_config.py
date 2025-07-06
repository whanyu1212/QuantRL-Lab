from typing import Tuple

from quantrl_lab.custom_envs.core.config import CoreEnvConfig


class SingleStockEnvConfig(CoreEnvConfig):
    """
    Stock environment configuration, extending the core environment
    configuration.

    Providing default values for the stock trading environment.
    """

    initial_balance: float = 100000.0
    window_size: int = 20
    price_column_index: int = 0
    transaction_cost_pct: float = 0.001  # 0.1% transaction cost
    slippage: float = 0.001  # 0.1% slippage
    order_expiration_steps: int = 5
    reward_clip_range: Tuple[float, float] = (-5.0, 5.0)

    class Config:
        from_attributes = True  # "ORM Mode"
