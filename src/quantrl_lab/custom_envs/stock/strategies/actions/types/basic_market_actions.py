from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from quantrl_lab.custom_envs.stock.strategies.actions.base_action import (
    BaseActionStrategy,
)

if TYPE_CHECKING:  # Solves circular import issues
    from quantrl_lab.custom_envs.stock.envs.stock_trading_env import StockTradingEnv


class Actions(IntEnum):
    Hold = 0
    Buy = 1
    Sell = 2
    LimitBuy = 3
    LimitSell = 4
    StopLoss = 5
    TakeProfit = 6


class BasicActionStrategy(BaseActionStrategy):
    """
    Implements the original, full-featured action space with a 3-part
    Box space.

    Action: [action_type, amount, price_modifier]
    """

    def define_action_space(self) -> gym.spaces.Box:
        action_type_low = 0
        action_type_high = len(Actions) - 1
        amount_low = 0.0
        amount_high = 1.0
        price_mod_low = 0.9
        price_mod_high = 1.1

        return gym.spaces.Box(
            low=np.array([action_type_low, amount_low, price_mod_low], dtype=np.float32),
            high=np.array([action_type_high, amount_high, price_mod_high], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def handle_action(self, env_self: StockTradingEnv, action: np.ndarray) -> Tuple[Any, Dict[str, Any]]:

        # 1. Decode the flattened Box action
        action_type_raw = np.clip(action[0], 0, len(Actions) - 1)
        amount_pct = np.clip(action[1], 0.0, 1.0)
        price_modifier = np.clip(action[2], 0.9, 1.1)

        action_type_int = int(np.round(action_type_raw))
        try:
            action_type = Actions(action_type_int)
        except ValueError:
            action_type = Actions.Hold  # Default to Hold on invalid type

        current_price = env_self._get_current_price()
        if current_price <= 1e-9:
            action_type = Actions.Hold

        # 2. Execute the agent's chosen action by calling methods on the env instance
        had_no_shares = env_self._get_total_shares() <= 0
        invalid_action_attempt = False

        if action_type == Actions.Hold:
            pass
        elif action_type == Actions.Buy:
            env_self._execute_market_order(action_type, current_price, amount_pct)
        elif action_type == Actions.Sell:
            if had_no_shares:
                invalid_action_attempt = True
            env_self._execute_market_order(action_type, current_price, amount_pct)
        elif action_type == Actions.LimitBuy:
            env_self._place_limit_order(action_type, current_price, amount_pct, price_modifier)
        elif action_type == Actions.LimitSell:
            if had_no_shares:
                invalid_action_attempt = True
            env_self._place_limit_order(action_type, current_price, amount_pct, price_modifier)
        elif action_type == Actions.StopLoss or action_type == Actions.TakeProfit:
            if had_no_shares:
                invalid_action_attempt = True
            env_self._place_risk_management_order(action_type, current_price, amount_pct, price_modifier)

        # 3. Return decoded info for the `step` function to use
        decoded_info = {
            "type": action_type.name,
            "amount_pct": amount_pct,
            "price_modifier": price_modifier,
            "raw_input": action,
            "invalid_action_attempt": invalid_action_attempt,
        }

        return action_type, decoded_info
