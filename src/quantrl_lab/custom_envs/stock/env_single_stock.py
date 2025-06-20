from enum import IntEnum
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from quantrl_lab.custom_envs.stock.config import EnvConfig


# Define the actions using IntEnum
class Actions(IntEnum):
    Hold = 0
    Buy = 1  # Market Buy
    Sell = 2  # Market Sell
    LimitBuy = 3  # Limit Order Buy
    LimitSell = 4  # Limit Order Sell
    StopLoss = 5  # Stop loss order
    TakeProfit = 6  # Take Profit order


class StockTradingEnv(gym.Env):
    # Added metadata for Gymnasium compatibility
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        data: np.ndarray,  # Multidimensional array of market data + additional features
        config: EnvConfig,  # Configuration object for environment settings
    ):
        super().__init__()

        # === Runtime error handling===
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array (num_steps, num_features).")
        if data.shape[0] <= config.window_size:
            raise ValueError("Data length must be greater than window_size.")
        if not (0 <= config.price_column_index < data.shape[1]):
            raise ValueError(f"price_column_index ({config.price_column_index}) is out of bounds.")

        # === Attributes ===
        self.data = data.astype(np.float32)  # float32 for more efficiency(?)
        self.num_steps, self.num_features = self.data.shape
        self.price_column_index = config.price_column_index
        self.window_size = config.window_size
        self.initial_balance = config.initial_balance
        self.transaction_cost_pct = config.transaction_cost_pct
        self.slippage = config.slippage
        self.order_expiration_steps = config.order_expiration_steps
        self._max_steps = self.num_steps - 1  # Max indexable step

        # --- State variables (will be reset in reset()) ---
        self.balance = 0.0
        self.shares_held = 0  # Shares held freely (not reserved in SL/TP/LimitSell)
        self.current_step = 0
        self.pending_orders = []  # Limit Buy/Sell orders not yet executed/expired
        self.stop_loss_orders = []  # Active Stop Loss orders
        self.take_profit_orders = []  # Active Take Profit orders
        self.executed_orders_history = []  # Log of executed/expired/placed orders

        # === Define Box action space ===
        # Shape: (3,) corresponding to [action_type, amount, price_modifier]
        action_type_low = 0
        action_type_high = len(Actions) - 1
        amount_low = 0.0
        amount_high = 1.0
        price_mod_low = 0.9  # effective stop loss range is 0.9 to 0.999
        price_mod_high = 1.1  # effective take profit range is 1.001 to 1.1

        # Define the action space as a Box space
        self.action_space = spaces.Box(
            low=np.array([action_type_low, amount_low, price_mod_low], dtype=np.float32),
            high=np.array([action_type_high, amount_high, price_mod_high], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        # === Example action space values:
        # Market Buy 50% of available balance
        # [1.0, 0.5, 1.0]  # Action type 1, 50% amount, price modifier ignored

        # Limit Sell 75% of shares at 5% above market price
        # [4.0, 0.75, 1.05]  # Action type 4, 75% amount, 5% above price

        # Stop Loss 100% of shares at 10% below market price
        # [5.0, 1.0, 0.9]  # Action type 5, 100% amount, 10% below price
        # ================================================================

        # Some private attributes for internal use
        # Store bounds for easy clipping/scaling in step
        self._action_type_bounds = (action_type_low, action_type_high)
        self._amount_bounds = (amount_low, amount_high)
        self._price_mod_bounds = (price_mod_low, price_mod_high)

        # --- Define Observation Space ---
        # Shape: (window_size * num_features) for market data + 6 for portfolio info
        obs_market_shape = self.window_size * self.num_features
        obs_portfolio_shape = 9  # 6 portfolio features + 3 for price/position
        total_obs_dim = obs_market_shape + obs_portfolio_shape

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)

    def _get_current_price(self) -> float:
        """
        Get the current price from the data array based on the current
        step.

        Returns:
            float: The current price at the current step.
        """
        if 0 <= self.current_step < self.num_steps:
            return float(self.data[self.current_step, self.price_column_index])
        else:
            # If step is out of bounds (e.g., after done), return the last known price
            if self.num_steps > 0:
                last_valid_step = min(self.current_step, self.num_steps - 1)
                return float(self.data[last_valid_step, self.price_column_index])
            else:
                return 0.0  # or maybe raise an error here
                # raise ValueError("Current step is out of bounds and no valid price available.")

    def _get_reserved_shares(self) -> int:
        """
        Get the total number of shares being reserved in active orders
        such as stop loss, take profit, and pending limit orders. This
        is used to calculate the total shares held. This is important
        for calculating the portfolio value and for determining the
        available shares for trading.

        Returns:
            int: The total number of shares reserved in active orders.
        """
        reserved_in_stop_loss = sum(order["shares"] for order in self.stop_loss_orders)
        reserved_in_take_profit = sum(order["shares"] for order in self.take_profit_orders)
        reserved_in_limit_sell = sum(order["shares"] for order in self.pending_orders if order["type"] == "limit_sell")
        return reserved_in_stop_loss + reserved_in_take_profit + reserved_in_limit_sell

    def _get_total_shares(self) -> int:
        """
        Get the total number of shares held, including reserved shares.

        Returns:
            int: The total number of shares held (free + reserved).
        """
        return self.shares_held + self._get_reserved_shares()

    def _get_portfolio_value(self) -> float:
        """
        Get current total portfolio value. This includes the balance and
        the value of shares held. The value of shares is calculated
        using the current price. This is important for calculating the
        reward and for determining the overall performance of the
        trading strategy.

        Returns:
            float: The total portfolio value (balance + shares value).
        """
        current_price = self._get_current_price()
        total_shares = self._get_total_shares()
        portfolio_value = self.balance + (total_shares * current_price)
        return portfolio_value

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation for the agent with enhanced
        features.

        Returns:
            np.ndarray: Normalized and enhanced observation space
        """
        # 1. Market window processing (keep existing code)
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        market_window = self.data[start_idx:end_idx, :]

        # 2. Padding (keep existing padding logic)
        actual_len = market_window.shape[0]
        if actual_len < self.window_size:
            if actual_len > 0:
                padding = np.repeat(
                    market_window[0, :][np.newaxis, :],
                    self.window_size - actual_len,
                    axis=0,
                )
            else:
                padding = np.zeros(
                    (self.window_size - actual_len, self.num_features),
                    dtype=self.data.dtype,
                )
            market_window = np.concatenate((padding, market_window), axis=0)

        # 3. Normalization (keep existing normalization)
        first_step_values = market_window[0, :]
        denominator = np.where(np.abs(first_step_values) < 1e-9, 1.0, first_step_values)
        normalized_market_window = market_window / denominator
        normalized_market_window[:, np.abs(first_step_values) < 1e-9] = 0.0

        # 4. Enhanced Portfolio Information
        current_price = self._get_current_price()
        total_shares = self._get_total_shares()
        total_position_value = total_shares * current_price

        # Calculate recent high/low (last 5 periods)
        lookback = 5
        recent_slice = self.data[max(0, self.current_step - lookback + 1) : self.current_step + 1]  # noqa: E203
        recent_high = np.max(recent_slice[:, self.price_column_index])
        recent_low = np.min(recent_slice[:, self.price_column_index])

        # Calculate position metrics
        position_info = []  # noqa: F841
        if total_shares > 0:
            # Position size relative to account
            position_size_ratio = total_position_value / self._get_portfolio_value()

            # Get average entry price (from executed orders)
            entry_prices = [
                order["price"]
                for order in self.executed_orders_history
                if order["type"] in ["market_buy", "limit_buy_executed"]
            ]
            avg_entry_price = np.mean(entry_prices) if entry_prices else current_price

            # Unrealized P&L percentage
            unrealized_pl_pct = (current_price - avg_entry_price) / avg_entry_price

            # Risk management metrics
            stop_loss_prices = [order["price"] for order in self.stop_loss_orders]
            take_profit_prices = [order["price"] for order in self.take_profit_orders]

            if stop_loss_prices and take_profit_prices:
                avg_stop_price = np.mean(stop_loss_prices)
                avg_target_price = np.mean(take_profit_prices)
                risk_reward_ratio = (avg_target_price - current_price) / (current_price - avg_stop_price)
                distance_to_stop = (current_price - avg_stop_price) / current_price
                distance_to_target = (avg_target_price - current_price) / current_price
            else:
                risk_reward_ratio = 0.0
                distance_to_stop = 0.0
                distance_to_target = 0.0
        else:
            # No position metrics
            position_size_ratio = 0.0
            unrealized_pl_pct = 0.0
            risk_reward_ratio = 0.0
            distance_to_stop = 0.0
            distance_to_target = 0.0

        # Calculate trend and volatility
        trend = self.calculate_trend_strength(lookback=5)

        # Calculate recent volatility (using std of returns)
        returns = np.diff(recent_slice[:, self.price_column_index]) / recent_slice[:-1, self.price_column_index]
        volatility = np.std(returns) if len(returns) > 0 else 0.0

        # 5. Combine enhanced portfolio metrics
        portfolio_info = np.array(
            [
                self.balance / self.initial_balance,  # Normalized balance
                position_size_ratio,  # Position size relative to portfolio
                unrealized_pl_pct,  # Unrealized P&L percentage
                (current_price - recent_low) / (recent_high - recent_low + 1e-9),  # Price position in recent range
                volatility,  # Recent volatility
                trend,  # Trend strength
                risk_reward_ratio,  # Current R:R ratio
                distance_to_stop,  # Normalized distance to stop loss
                distance_to_target,  # Normalized distance to take profit
            ],
            dtype=np.float32,
        )

        # 6. Combine and Flatten
        flattened_market_obs = normalized_market_window.flatten()
        observation = np.concatenate((flattened_market_obs, portfolio_info))

        return observation.astype(np.float32)

    def _process_pending_orders(self, current_price: float) -> None:
        """
        Processing pending limit orders and checking for execution or
        expiration.

        Args:
            current_price (float): The current market price at the current step.
        """
        remaining_orders = []
        executed_order_details = []  # Log executions/expirations from this step

        for order in self.pending_orders:
            executed = False
            # We assume that the pending orders will be GTC (Good Till Cancelled) after 5 days/steps
            expired = self.current_step - order["placed_at"] > self.order_expiration_steps

            if expired:
                if order["type"] == "limit_buy":
                    # Return reserved funds if expired
                    self.balance += order["cost_reserved"]
                elif order["type"] == "limit_sell":
                    # Return reserved shares if expired
                    self.shares_held += order["shares"]  # Add back to freely held shares

                executed_order_details.append(
                    {
                        "step": self.current_step,
                        "type": f"{order['type']}_expired",
                        "shares": order["shares"],
                        "price": order["price"],
                        "reason": "Expired",
                    }
                )
                executed = True  # Mark as processed

            # Check for execution if not expired
            elif order["type"] == "limit_buy" and current_price <= order["price"]:
                # Execute limit buy: Use limit price or better (current price). For simplicity, use limit price.
                execution_price = order["price"]
                # Cost was already subtracted when placing order, just add shares
                self.shares_held += order["shares"]
                executed = True
                executed_order_details.append(
                    {
                        "step": self.current_step,
                        "type": "limit_buy_executed",
                        "shares": order["shares"],
                        "price": execution_price,
                        "cost": order["cost_reserved"],
                    }
                )

            elif order["type"] == "limit_sell" and current_price >= order["price"]:
                # Execute limit sell: Use limit price or better (current price). For simplicity, use limit price.
                execution_price = order["price"]
                # Shares were already reserved, calculate revenue and add to balance
                revenue = order["shares"] * execution_price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                executed = True
                executed_order_details.append(
                    {
                        "step": self.current_step,
                        "type": "limit_sell_executed",
                        "shares": order["shares"],
                        "price": execution_price,
                        "revenue": revenue,
                    }
                )

            # If order cannot be fulfilled or are not expired, keep it in the list
            if not executed:
                remaining_orders.append(order)

        # Update the list of pending orders
        self.pending_orders = remaining_orders
        # Add any executions/expirations from this step to the main history
        if executed_order_details:
            self.executed_orders_history.extend(executed_order_details)

    def _process_risk_management_orders(self, current_price: float) -> None:
        """
        Process stop loss and take profit orders, aka stop orders.

        Args:
            current_price (float): The current market price at the current step.
        """
        executed_order_details = []  # Log executions from this step

        # Process Stop Loss
        remaining_stop_loss = []
        for order in self.stop_loss_orders:
            if current_price <= order["price"]:
                # Triggered: Sell at current price with slippage
                adjusted_price = current_price * (1 - self.slippage)
                revenue = order["shares"] * adjusted_price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                # Shares were already reserved (subtracted from self.shares_held)
                executed_order_details.append(
                    {
                        "step": self.current_step,
                        "type": "stop_loss_executed",
                        "shares": order["shares"],
                        "trigger_price": order["price"],
                        "execution_price": adjusted_price,
                        "revenue": revenue,
                    }
                )
            else:
                # Not triggered, keep the order
                remaining_stop_loss.append(order)
        self.stop_loss_orders = remaining_stop_loss

        # Process Take Profit
        remaining_take_profit = []
        for order in self.take_profit_orders:
            if current_price >= order["price"]:
                # Triggered: Sell at current price with slippage
                adjusted_price = current_price * (1 - self.slippage)  # Note: Slippage might not apply to TP in reality
                revenue = order["shares"] * adjusted_price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                # Shares were already reserved (subtracted from self.shares_held)
                executed_order_details.append(
                    {
                        "step": self.current_step,
                        "type": "take_profit_executed",
                        "shares": order["shares"],
                        "trigger_price": order["price"],
                        "execution_price": adjusted_price,
                        "revenue": revenue,
                    }
                )
            else:
                # Not triggered, keep the order
                remaining_take_profit.append(order)
        self.take_profit_orders = remaining_take_profit

        # Add any executions from this step to the main history
        if executed_order_details:
            self.executed_orders_history.extend(executed_order_details)

    def _execute_market_order(self, action_type: Actions, current_price: float, amount_pct: float) -> None:
        """
        Execute a market buy or sell order at the current price.

        Args:
            action_type (Actions): Buy or Sell action
            current_price (float): The current market price at the current step.
            amount_pct (float): Percentage of shares to buy/sell with respect to the available balance
        """
        order_placed = False
        if action_type == Actions.Buy:
            # Apply slippage (price increases for buyer)
            adjusted_price = current_price * (1 + self.slippage)
            cost_per_share = adjusted_price * (1 + self.transaction_cost_pct)
            # Avoid division by zero if price is effectively zero
            # wont happen in reality, but just in case
            if cost_per_share <= 1e-9:
                return

            # Calculate max shares affordable with current balance
            max_shares_affordable = self.balance / cost_per_share
            # Determine shares to buy based on percentage of affordable amount
            shares_to_buy = int(max_shares_affordable * amount_pct)

            if shares_to_buy > 0:
                actual_cost = shares_to_buy * cost_per_share
                # Double check affordability after converting to int shares
                if actual_cost <= self.balance:
                    self.balance -= actual_cost
                    self.shares_held += shares_to_buy
                    order_placed = True
                    self.executed_orders_history.append(
                        {
                            "step": self.current_step,
                            "type": "market_buy",
                            "shares": shares_to_buy,
                            "price": adjusted_price,
                            "cost": actual_cost,
                        }
                    )

        elif action_type == Actions.Sell:
            if self.shares_held <= 0:
                return  # Cannot sell if holding nothing

            # Determine shares to sell based on percentage of *freely held* shares
            shares_to_sell = int(self.shares_held * amount_pct)

            if shares_to_sell > 0:
                # Apply slippage (price decreases for seller)
                adjusted_price = current_price * (1 - self.slippage)
                revenue_per_share = adjusted_price * (1 - self.transaction_cost_pct)
                actual_revenue = shares_to_sell * revenue_per_share

                self.shares_held -= shares_to_sell
                self.balance += actual_revenue
                order_placed = True  # noqa: F841
                self.executed_orders_history.append(
                    {
                        "step": self.current_step,
                        "type": "market_sell",
                        "shares": shares_to_sell,
                        "price": adjusted_price,
                        "revenue": actual_revenue,
                    }
                )
        # Return True if an order was logged, False otherwise (optional)
        # return order_placed

    def _place_limit_order(
        self,
        action_type: Actions,
        current_price: float,
        amount_pct: float,
        price_modifier: float,
    ) -> None:
        """
        Place a limit order (buy/sell) for a percentage of shares.

        Args:
            action_type (Actions): LimitBuy or LimitSell action
            current_price (float): The current market price at the current step.
            amount_pct (float): Percentage of shares to buy/sell with respect to the available balance
            price_modifier (float): Modifier for the limit price (e.g., 0.9 for 10% below current price)
        """
        limit_price = current_price * price_modifier
        order_placed = False

        if action_type == Actions.LimitBuy:
            # Calculate cost if order executes at limit price
            cost_per_share = limit_price * (1 + self.transaction_cost_pct)
            if cost_per_share <= 1e-9:
                return  # Avoid zero cost

            # Calculate max shares affordable with current balance *if* the order were to fill
            max_shares_affordable = self.balance / cost_per_share
            # Determine shares based on percentage of affordable amount
            shares_to_buy = int(max_shares_affordable * amount_pct)

            if shares_to_buy > 0:
                cost_reserved = shares_to_buy * cost_per_share
                # Check if we have enough balance to reserve the funds
                if cost_reserved <= self.balance:
                    self.balance -= cost_reserved  # Reserve funds
                    self.pending_orders.append(
                        {
                            "type": "limit_buy",
                            "shares": shares_to_buy,
                            "price": limit_price,
                            "placed_at": self.current_step,
                            "cost_reserved": cost_reserved,  # Store cost for potential expiration refund
                        }
                    )
                    order_placed = True
                    self.executed_orders_history.append(
                        {
                            "step": self.current_step,
                            "type": "limit_buy_placed",
                            "shares": shares_to_buy,
                            "price": limit_price,
                        }
                    )

        elif action_type == Actions.LimitSell:
            if self.shares_held <= 0:
                return  # Cannot place sell order if holding nothing

            # Determine shares based on percentage of *freely held* shares
            shares_to_sell = int(self.shares_held * amount_pct)

            if shares_to_sell > 0:
                # Ensure we don't try to sell more than we have freely available
                shares_to_sell = min(shares_to_sell, self.shares_held)
                if shares_to_sell > 0:
                    self.shares_held -= shares_to_sell  # Reserve shares
                    self.pending_orders.append(
                        {
                            "type": "limit_sell",
                            "shares": shares_to_sell,
                            "price": limit_price,
                            "placed_at": self.current_step,
                            # No cost reserved, revenue calculated on execution
                        }
                    )
                    order_placed = True  # noqa: F841
                    self.executed_orders_history.append(
                        {
                            "step": self.current_step,
                            "type": "limit_sell_placed",
                            "shares": shares_to_sell,
                            "price": limit_price,
                        }
                    )
        # return order_placed

    def _place_risk_management_order(
        self,
        action_type: Actions,
        current_price: float,
        amount_pct: float,
        price_modifier: float,
    ) -> None:
        """
        _summary_

        Args:
            action_type (Actions): StopLoss or TakeProfit action
            current_price (float): The current market price at the current step.
            amount_pct (float): percentage of shares to do SL/TP with respect to the available balance
            price_modifier (float): Modifier for the stop/take profit price
        """
        if self.shares_held <= 0:
            return  # Cannot place SL/TP if holding nothing

        # Determine shares based on percentage of *freely held* shares
        shares_to_cover = int(self.shares_held * amount_pct)
        order_placed = False

        if shares_to_cover > 0:
            # Ensure we don't try to cover more shares than freely available
            shares_to_cover = min(shares_to_cover, self.shares_held)
            if shares_to_cover > 0:
                if action_type == Actions.StopLoss:
                    # Stop price should be below current price
                    stop_price = current_price * min(0.999, price_modifier)  # Ensure <= current price
                    # Ensure it's strictly below current price if possible
                    if stop_price >= current_price and current_price > 1e-9:
                        stop_price = current_price * 0.999
                    elif stop_price >= current_price:  # Handle case where current price is near zero
                        stop_price = 0.0

                    self.shares_held -= shares_to_cover  # Reserve shares
                    self.stop_loss_orders.append(
                        {
                            "shares": shares_to_cover,
                            "price": stop_price,
                            "placed_at": self.current_step,
                        }
                    )
                    order_placed = True
                    self.executed_orders_history.append(
                        {
                            "step": self.current_step,
                            "type": "stop_loss_placed",
                            "shares": shares_to_cover,
                            "price": stop_price,
                        }
                    )

                elif action_type == Actions.TakeProfit:
                    # Take profit price should be above current price
                    take_profit_price = current_price * max(1.001, price_modifier)  # Ensure >= current price
                    # Ensure it's strictly above current price
                    if take_profit_price <= current_price:
                        take_profit_price = current_price * 1.001

                    self.shares_held -= shares_to_cover  # Reserve shares
                    self.take_profit_orders.append(
                        {
                            "shares": shares_to_cover,
                            "price": take_profit_price,
                            "placed_at": self.current_step,
                        }
                    )
                    order_placed = True  # noqa: F841
                    self.executed_orders_history.append(
                        {
                            "step": self.current_step,
                            "type": "take_profit_placed",
                            "shares": shares_to_cover,
                            "price": take_profit_price,
                        }
                    )
        # return order_placed

    def calculate_trend_strength(self, lookback=5):
        """Calculate recent price trend strength."""
        if self.current_step < lookback:
            return 0.0

        # Get recent prices
        end_idx = self.current_step + 1
        start_idx = end_idx - lookback
        recent_prices = self.data[start_idx:end_idx, self.price_column_index]

        # Calculate trend using linear regression slope
        x = np.arange(lookback)
        slope, _ = np.polyfit(x, recent_prices, 1)

        # Normalize slope to be between -1 and 1
        max_price = np.max(recent_prices)
        if max_price > 0:
            normalized_slope = slope / max_price
        else:
            normalized_slope = 0.0

        return normalized_slope

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment using a flattened
        Box action.

        Args:
            action: A numpy array (Box action) with shape (3,) representing
                    [action_type, amount, price_modifier].

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]:
            observation, reward, terminated, truncated, info
        """
        if not isinstance(action, np.ndarray) or action.shape != self.action_space.shape:
            raise ValueError(
                f"Invalid action format received in step: {action}. Expected shape {self.action_space.shape}"
            )

        # 1. Decode the flattened Box action
        action_type_raw = np.clip(action[0], self._action_type_bounds[0], self._action_type_bounds[1])
        amount_pct = np.clip(action[1], self._amount_bounds[0], self._amount_bounds[1])
        price_modifier = np.clip(action[2], self._price_mod_bounds[0], self._price_mod_bounds[1])

        action_type_int = int(np.round(action_type_raw))
        action_type_int = max(
            int(self._action_type_bounds[0]),
            min(action_type_int, int(self._action_type_bounds[1])),
        )
        try:
            action_type = Actions(action_type_int)
        except ValueError:
            print(f"Warning: Invalid action type integer {action_type_int} after rounding. Defaulting to Hold.")
            action_type = Actions.Hold

        # 2. Store previous portfolio value and price for reward calculation
        prev_portfolio_value = self._get_portfolio_value()

        # This is not a redundant step, we will be comparing the price to assess if a hold is worth it
        prev_price = self._get_current_price()

        current_price = prev_price
        if current_price <= 1e-9:
            print(f"Warning: Near-zero price ({current_price}) at step {self.current_step}. Holding.")
            action_type = Actions.Hold

        # 4. Process existing orders based on current price (whether limit orders or stop orders can be fulfilled)
        self._process_pending_orders(current_price)
        self._process_risk_management_orders(current_price)

        # 5. Track if an invalid risk management order was attempted
        invalid_action_attempt = False
        had_no_shares = self._get_total_shares() <= 0

        # 6. Execute the agent's chosen action
        if action_type == Actions.Hold:
            pass
        elif action_type == Actions.Buy:
            self._execute_market_order(action_type, current_price, amount_pct)
        elif action_type == Actions.Sell:
            if had_no_shares:
                invalid_action_attempt = True
            self._execute_market_order(action_type, current_price, amount_pct)
        elif action_type == Actions.LimitBuy:
            self._place_limit_order(action_type, current_price, amount_pct, price_modifier)
        elif action_type == Actions.LimitSell:
            if had_no_shares:
                invalid_action_attempt = True
            self._place_limit_order(action_type, current_price, amount_pct, price_modifier)
        elif action_type == Actions.StopLoss or action_type == Actions.TakeProfit:
            # Check if there are no shares held before attempting a risk management order
            if had_no_shares:
                invalid_action_attempt = True
                # Still call the function because it has its own validation
                # (it will return early if no shares are held)
            self._place_risk_management_order(action_type, current_price, amount_pct, price_modifier)

        # 7. Move to the next time step
        self.current_step += 1

        # 8. Get new price after step update (for reward calculation and next step)
        current_price = self._get_current_price()  # Now this is the updated price

        # 9. Check termination condition
        terminated = self.current_step >= self._max_steps

        # 10. Calculate reward
        current_portfolio_value = self._get_portfolio_value()

        # Calculate basic portfolio value change reward
        if prev_portfolio_value > 1e-9:
            reward = 0.5 * (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        elif self.initial_balance > 1e-9:
            reward = 0.5 * (current_portfolio_value - self.initial_balance) / self.initial_balance
        else:
            reward = 0.0

        trend = self.calculate_trend_strength(lookback=10)
        trend_strength = abs(trend)
        is_uptrend = trend > 0
        is_downtrend = trend < 0

        # Incentivize risk management based on trend
        if invalid_action_attempt:
            reward -= 1.0  # Keep harsh penalty for invalid attempts
        elif action_type == Actions.TakeProfit:
            if is_uptrend:
                # Encourage take profit in uptrend
                reward += 0.05 * (1 + trend_strength)
            else:
                # Discourage take profit in downtrend
                reward -= 0.05 * (1 - trend_strength)
        elif action_type == Actions.StopLoss:
            if is_downtrend:
                reward += 0.05 * (1 + trend_strength)
            else:
                # Discourage stop loss in uptrend
                reward -= 0.05 * (1 - trend_strength)
        elif action_type in [Actions.LimitBuy, Actions.LimitSell]:
            reward += 0.01  # Keep moderate bonus for limit orders
        elif action_type == Actions.Hold:
            reward -= 0.01  # penalize overly aggressive actions

        # Add trend-aware holding bonus
        if self._get_total_shares() > 0:
            position_size = (self._get_total_shares() * current_price) / self.initial_balance
            if is_uptrend:
                # Bigger bonus for holding during uptrend
                reward += 0.1 * position_size * (1 + trend_strength)
            elif is_downtrend:
                # Penalty for holding during downtrend without stop loss
                if len(self.stop_loss_orders) == 0:
                    reward -= 0.1 * position_size * trend_strength

        # Optional: Clip reward to prevent extreme values
        reward = np.clip(reward, -5.0, 5.0).item()

        # 11. Get next observation and prepare return values
        observation = self._get_observation()
        truncated = False

        # 12. Info dictionary
        info = {
            "step": self.current_step,
            "portfolio_value": current_portfolio_value,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "total_shares": self._get_total_shares(),
            "current_price": current_price,
            "prev_price": prev_price,  # Include previous price in info for analysis
            "price_change": current_price - prev_price,  # Add price change for easier analysis
            "reward": reward,
            "action_decoded": {
                "type": action_type.name,
                "amount_pct": amount_pct,
                "price_modifier": price_modifier,
                "raw_input": action,
            },
            "orders_info": {
                "pending_count": len(self.pending_orders),
                "stop_loss_count": len(self.stop_loss_orders),
                "take_profit_count": len(self.take_profit_orders),
            },
            "invalid_action_attempt": invalid_action_attempt,  # Add this flag to the info dict
            "last_order_event": (self.executed_orders_history[-1] if self.executed_orders_history else None),
        }

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to an initial state and returns the
        initial observation.

        Args:
            seed (Optional[int], optional): Defaults to None.
            options (Optional[Dict], optional): Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict]: initial observation and info dictionary.
        """
        super().reset(seed=seed)

        # Reset state variables
        self.balance = self.initial_balance
        self.shares_held = 0
        # Start at index `window_size` to ensure the first observation has a full window
        self.current_step = self.window_size
        self.pending_orders = []
        self.stop_loss_orders = []
        self.take_profit_orders = []
        self.executed_orders_history = []  # Clear history on reset

        # Get the initial observation
        initial_observation = self._get_observation()  # Uses the updated method

        # Initial info dictionary
        initial_info = {
            "initial_balance": self.initial_balance,
            "starting_step": self.current_step,
            "window_size": self.window_size,
            "num_features": self.num_features,
            "price_column_index": self.price_column_index,
            "message": "Multi-feature environment reset.",
        }
        return initial_observation, initial_info

    def render(self, mode="human"):
        """Renders the environment state."""
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "human":
            self._render_human()

    def _render_human(self):
        """Prints state information to the console."""
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        total_shares = self._get_total_shares()

        print("-" * 40)
        print(f"Step:         {self.current_step}/{self._max_steps}")
        print(f"Current Price:{current_price:>15.2f}")
        print(f"Balance:      {self.balance:>15.2f}")
        print(f"Shares Held:  {self.shares_held:>15} (Free)")
        print(f"Total Shares: {total_shares:>15} (Free + Reserved)")
        print(f"Portfolio Val:{portfolio_value:>15.2f}")
        print("-" * 40)
        print("Active Orders:")
        print(f"  Pending Limit:{len(self.pending_orders):>5}")
        print(f"  Stop Loss:    {len(self.stop_loss_orders):>5}")
        print(f"  Take Profit:  {len(self.take_profit_orders):>5}")

        if self.executed_orders_history:
            last_event = self.executed_orders_history[-1]
            print("-" * 40)
            price_value = last_event.get("price")

            # Check if price is a number before applying format
            if isinstance(price_value, (int, float)):
                price_str = f"{price_value:.2f}"
            else:
                price_str = str(price_value)

            print(
                f"Last Event:   {last_event['type']} "
                f"(Shares: {last_event.get('shares', 'N/A')}, "
                f"Price: {price_str})"
            )

        print("-" * 40)

    def _render_ansi(self) -> str:
        """Returns state information as a string."""
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        total_shares = self._get_total_shares()
        last_event_str = "None"

        if self.executed_orders_history:
            last_event = self.executed_orders_history[-1]
            price_value = last_event.get("price", "N/A")

            # Check if price is a number before applying format
            if isinstance(price_value, (int, float)):
                price_str = f"{price_value:.2f}"
            else:
                price_str = str(price_value)

            last_event_str = f"{last_event['type']} (S:{last_event.get('shares', 'N/A')}, P:{price_str})"

        return (
            f"Step: {self.current_step}/{self._max_steps} | "
            f"Price: {current_price:.2f} | "
            f"Balance: {self.balance:.2f} | "
            f"Shares(F/T): {self.shares_held}/{total_shares} | "
            f"Value: {portfolio_value:.2f} | "
            f"Orders(P/SL/TP): {len(self.pending_orders)}/"
            f"{len(self.stop_loss_orders)}/"
            f"{len(self.take_profit_orders)} | "
            f"Last Event: {last_event_str}"
        )

    def close(self):
        print("StockTradingEnvMultiFeature closed.")
        pass
