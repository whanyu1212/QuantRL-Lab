from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.stock.strategies.actions.base_action import (
    BaseActionStrategy,
)

if TYPE_CHECKING:  # Solves circular import issues
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class StandardMarketActionStrategy(BaseActionStrategy):
    """
    Implements the full-featured action space with a 3-part Box space.

    Action: [action_type, amount, price_modifier]
    """

    def define_action_space(self) -> gym.spaces.Box:
        """
        Defines the action space for the trading environment.

        Returns:
            gym.spaces.Box: The action space as a Box space.
        """
        action_type_low = 0  # Assuming Actions is an IntEnum, the first action type is 0
        action_type_high = len(Actions) - 1
        amount_low = 0.0  # Minimum amount percentage to trade
        amount_high = 1.0  # Maximum amount percentage to trade
        # Price modifier for limit orders, typically between 0.9 and 1.1
        # 0.9 means 10% below current price, 1.1 means 10% above current price
        # This allows for a range of limit prices around the current price
        # For example, if current price is $100, a price modifier of 0.9 means a limit buy at $90,
        # and 1.1 means a limit sell at $110
        # This gives flexibility in setting limit orders around the current market price
        price_mod_low = 0.9
        price_mod_high = 1.1

        # np.float32 is used to ensure compatibility with most RL libraries
        # and to save memory compared to np.float64
        # The action space is a Box with 3 dimensions:
        # 1. action_type: Integer representing the type of action (Buy, Sell,
        #    Hold, LimitBuy, LimitSell, StopLoss, TakeProfit)
        # 2. amount: Float representing the percentage of the portfolio to trade
        # 3. price_modifier: Float representing the price modifier for limit orders
        #    (0.9 means 10% below current price, 1.1 means 10% above current price)
        # The action space is defined as a Box with low and high values for each dimension,
        # and a shape of (3,) indicating it is a 3-dimensional action space.
        return gym.spaces.Box(
            low=np.array([action_type_low, amount_low, price_mod_low], dtype=np.float32),
            high=np.array([action_type_high, amount_high, price_mod_high], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

    def handle_action(self, env_self: TradingEnvProtocol, action: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """
        Handles the action by decoding it and instructing the
        environment's portfolio.

        Args:
            env_self (TradingEnvProtocol): The environment instance.
            action (np.ndarray): The raw action from the agent.

        Returns:
            Tuple[Any, Dict[str, Any]]: The decoded action type and a dictionary of details.
        """
        # --- 1. Decode the action (No changes needed here) ---
        action_type_raw = np.clip(action[0], 0, len(Actions) - 1)
        amount_pct = np.clip(action[1], 0.0, 1.0)
        price_modifier = np.clip(action[2], 0.9, 1.1)
        action_type_int = int(np.round(action_type_raw))

        try:
            action_type = Actions(action_type_int)
        except ValueError:
            action_type = Actions.Hold

        # The environment is still responsible for providing the current price
        current_price = env_self._get_current_price()
        if current_price <= 1e-9:
            action_type = Actions.Hold

        # --- 2. Execute the action by calling methods on the PORTFOLIO ---

        # CORRECTED: Get total shares from the portfolio
        had_no_shares = env_self.portfolio.total_shares <= 0
        invalid_action_attempt = False

        # Get current_step from the environment, as the portfolio methods need it
        current_step = env_self.current_step

        if action_type == Actions.Hold:
            pass
        elif action_type == Actions.Buy:
            # CORRECTED: Call the portfolio's method, passing current_step
            env_self.portfolio.execute_market_order(action_type, current_price, amount_pct, current_step)
        elif action_type == Actions.Sell:
            if had_no_shares:
                invalid_action_attempt = True
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.execute_market_order(action_type, current_price, amount_pct, current_step)
        elif action_type == Actions.LimitBuy:
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.place_limit_order(action_type, current_price, amount_pct, price_modifier, current_step)
        elif action_type == Actions.LimitSell:
            if had_no_shares:
                invalid_action_attempt = True
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.place_limit_order(action_type, current_price, amount_pct, price_modifier, current_step)
        elif action_type in [Actions.StopLoss, Actions.TakeProfit]:
            if had_no_shares:
                invalid_action_attempt = True
            # CORRECTED: Call the portfolio's method
            env_self.portfolio.place_risk_management_order(
                action_type, current_price, amount_pct, price_modifier, current_step
            )

        # --- 3. Return decoded info (No changes needed here) ---
        decoded_info = {
            "type": action_type.name,
            "amount_pct": amount_pct,
            "price_modifier": price_modifier,
            "raw_input": action,
            "invalid_action_attempt": invalid_action_attempt,
        }

        return action_type, decoded_info
