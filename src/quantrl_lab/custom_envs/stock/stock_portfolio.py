from typing import Any, Dict, List

from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.core.portfolio import Portfolio


class StockPortfolio(Portfolio):
    """
    A portfolio for stock trading that handles complex order types,
    fees, and slippage.

    It extends the simple Portfolio with stock-specific logic and state.
    """

    def __init__(
        self,
        initial_balance: float,
        transaction_cost_pct: float,
        slippage: float,
        order_expiration_steps: int,
    ):
        # === Initialize the parent class with the part it cares about ===
        super().__init__(initial_balance=initial_balance)

        # === Transaction cost and slippage can be adjusted to reflect difficulties in trading ===
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage = slippage
        self.order_expiration_steps = order_expiration_steps

        # === Stock-specific state ===
        self.pending_orders: List[Dict[str, Any]] = []
        self.stop_loss_orders: List[Dict[str, Any]] = []
        self.take_profit_orders: List[Dict[str, Any]] = []
        self.executed_orders_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset the portfolio to its initial state."""
        super().reset()
        self.pending_orders = []
        self.stop_loss_orders = []
        self.take_profit_orders = []
        self.executed_orders_history = []

    @property
    def shares_held(self) -> int:
        """
        Returns the number of shares currently held in the portfolio.

        Returns:
            int: The number of shares held.
        """
        return self.units_held

    @property
    def total_shares(self) -> int:
        """
        Returns the total number of shares held, including those
        reserved in orders.

        Returns:
            int: The total number of shares held.
        """
        return self.units_held + self._get_reserved_shares()

    def get_value(self, current_price: float) -> float:
        """
        Calculate the total value of the portfolio including unfilled
        orders and reserved money.

        This overrides the parent's get_value method to account for:
        - Current balance (free cash not tied up in orders)
        - Value of currently held shares (free shares not reserved in orders)
        - Reserved cash in pending buy orders (money locked up waiting for execution)
        - Value of shares reserved in pending sell/stop/take-profit orders

        The accounting works as follows:
        - When a limit buy is placed: cash is moved from balance to cost_reserved
        - When a limit sell is placed: shares are moved from units_held to the order
        - When risk management orders are placed: shares are moved from units_held to respective order lists
        - This method sums all these components to get the true portfolio value

        Args:
            current_price (float): The current market price of the asset.

        Returns:
            float: The total portfolio value including all positions and reserved amounts.
        """
        # Base value: free balance + value of free shares
        total_value = self.balance + (self.units_held * current_price)

        # Add reserved cash from pending buy orders
        for order in self.pending_orders:
            if order["type"] == "limit_buy":
                total_value += order["cost_reserved"]

        # Add value of shares reserved in pending sell orders
        for order in self.pending_orders:
            if order["type"] == "limit_sell":
                total_value += order["shares"] * current_price

        # Add value of shares reserved in stop loss orders
        for order in self.stop_loss_orders:
            total_value += order["shares"] * current_price

        # Add value of shares reserved in take profit orders
        for order in self.take_profit_orders:
            total_value += order["shares"] * current_price

        return total_value

    def process_open_orders(self, current_step: int, current_price: float) -> None:
        """
        Process all open orders at the current market price.

        Args:
            current_step (int): The current step in the trading environment.
            current_price (float): The current market price of the asset.
        """
        self._process_pending_orders(current_step, current_price)
        self._process_risk_management_orders(current_step, current_price)

    def execute_market_order(
        self, action_type: Actions, current_price: float, amount_pct: float, current_step: int
    ) -> None:
        """
        Execute a market order.

        Args:
            action_type (Actions): The type of action (buy/sell).
            current_price (float): The current market price.
            amount_pct (float): The percentage of the portfolio to use for the order.
            current_step (int): The current step in the trading environment.

        Returns:
            None
        """
        # Clip amount_pct to valid range
        # During training, the env.action_space.sample() function can generate values:
        # - Slightly below 0 due to floating-point precision
        # - Slightly above 1 due to the same reason
        amount_pct = max(0.0, min(1.0, amount_pct))

        # === Runtime error checks ===
        if self.balance <= 0 and action_type == Actions.Buy:
            raise ValueError("Insufficient balance to execute buy order")
        if action_type not in [Actions.Buy, Actions.Sell]:
            raise ValueError("Invalid action type for market order")

        # === Buy Logic ===
        # Adjust the current price for slippage and transaction costs
        # and calculate the number of shares that we can afford
        if action_type == Actions.Buy:
            adjusted_price = current_price * (1 + self.slippage)
            cost_per_share = adjusted_price * (1 + self.transaction_cost_pct)
            if cost_per_share <= 1e-9:
                return  # Avoid division by zero

            # TODO: Consider buying using notional value instead of balance
            shares_to_buy = int((self.balance / cost_per_share) * amount_pct)
            if shares_to_buy > 0:
                actual_cost = shares_to_buy * cost_per_share
                if actual_cost <= self.balance:
                    self.balance -= actual_cost
                    self.units_held += shares_to_buy
                    self.executed_orders_history.append(
                        {
                            "step": current_step,
                            "type": "market_buy",
                            "shares": shares_to_buy,
                            "price": adjusted_price,
                            "cost": actual_cost,
                        }
                    )

        # === Sell Logic ===
        # Adjust the current price for slippage and transaction costs
        # and calculate the number of shares to sell
        elif action_type == Actions.Sell:
            if self.units_held <= 0:
                return
            shares_to_sell = int(self.units_held * amount_pct)
            if shares_to_sell > 0:
                adjusted_price = current_price * (1 - self.slippage)
                revenue = shares_to_sell * adjusted_price * (1 - self.transaction_cost_pct)
                self.units_held -= shares_to_sell
                self.balance += revenue
                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "market_sell",
                        "shares": shares_to_sell,
                        "price": adjusted_price,
                        "revenue": revenue,
                    }
                )

    def place_limit_order(
        self, action_type: Actions, current_price: float, amount_pct: float, price_modifier: float, current_step: int
    ) -> None:
        """
        Place a limit order for buying or selling an asset.

        Args:
            action_type (Actions): The type of action (LimitBuy/LimitSell).
            current_price (float): The current market price.
            amount_pct (float): The percentage of the portfolio to use for the order.
            price_modifier (float): The price modifier to apply to the current price.
            current_step (int): The current step in the trading environment.

        Returns:
            None
        """

        # TODO: Think about more robust way to set limit price
        limit_price = current_price * price_modifier

        # === Limit Buy Logic ===
        if action_type == Actions.LimitBuy:
            cost_per_share = limit_price * (1 + self.transaction_cost_pct)
            if cost_per_share <= 1e-9:
                return
            shares_to_buy = int((self.balance / cost_per_share) * amount_pct)
            if shares_to_buy > 0:
                cost_reserved = shares_to_buy * cost_per_share
                if cost_reserved <= self.balance:
                    self.balance -= cost_reserved
                    self.pending_orders.append(
                        {
                            "type": "limit_buy",
                            "shares": shares_to_buy,
                            "price": limit_price,
                            "placed_at": current_step,
                            "cost_reserved": cost_reserved,
                        }
                    )
                    self.executed_orders_history.append(
                        {
                            "step": current_step,
                            "type": "limit_buy_placed",
                            "shares": shares_to_buy,
                            "price": limit_price,
                        }
                    )

        # === Limit Sell Logic ===
        elif action_type == Actions.LimitSell:
            if self.units_held <= 0:
                return
            shares_to_sell = int(self.units_held * amount_pct)
            if shares_to_sell > 0:
                self.units_held -= shares_to_sell
                self.pending_orders.append(
                    {
                        "type": "limit_sell",
                        "shares": shares_to_sell,
                        "price": limit_price,
                        "placed_at": current_step,
                    }
                )
                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "limit_sell_placed",
                        "shares": shares_to_sell,
                        "price": limit_price,
                    }
                )

    def place_risk_management_order(
        self, action_type: Actions, current_price: float, amount_pct: float, price_modifier: float, current_step: int
    ) -> None:
        """
        Place a risk management order (stop loss or take profit).

        Args:
            action_type (Actions): The type of action (StopLoss/TakeProfit).
            current_price (float): The current market price.
            amount_pct (float): The percentage of the portfolio to use for the order.
            price_modifier (float): The price modifier to apply to the current price.
            current_step (int): The current step in the trading environment.

        Returns:
            None
        """
        if self.units_held <= 0:
            return
        shares_to_cover = int(self.units_held * amount_pct)
        if shares_to_cover > 0:
            # === Stop Loss Logic ===
            if action_type == Actions.StopLoss:
                stop_price = current_price * min(0.999, price_modifier)
                if stop_price >= current_price:
                    stop_price = current_price * 0.999
                self.units_held -= shares_to_cover
                self.stop_loss_orders.append(
                    {"shares": shares_to_cover, "price": stop_price, "placed_at": current_step}
                )
                self.executed_orders_history.append(
                    {"step": current_step, "type": "stop_loss_placed", "shares": shares_to_cover, "price": stop_price}
                )
            # === Take Profit Logic ===
            elif action_type == Actions.TakeProfit:
                take_profit_price = current_price * max(1.001, price_modifier)
                if take_profit_price <= current_price:
                    take_profit_price = current_price * 1.001
                self.units_held -= shares_to_cover
                self.take_profit_orders.append(
                    {"shares": shares_to_cover, "price": take_profit_price, "placed_at": current_step}
                )
                self.executed_orders_history.append(
                    {
                        "step": current_step,
                        "type": "take_profit_placed",
                        "shares": shares_to_cover,
                        "price": take_profit_price,
                    }
                )

    # === Private Helper Methods ===
    def _get_reserved_shares(self) -> int:
        """
        Get the total number of shares reserved for open orders.

        Returns:
            int: The total number of shares reserved.
        """
        reserved_sl = sum(order["shares"] for order in self.stop_loss_orders)
        reserved_tp = sum(order["shares"] for order in self.take_profit_orders)
        reserved_limit_sell = sum(order["shares"] for order in self.pending_orders if order["type"] == "limit_sell")
        return reserved_sl + reserved_tp + reserved_limit_sell

    def _process_pending_orders(self, current_step: int, current_price: float) -> None:
        """
        Process all pending limit orders against the current market
        price.

        Args:
            current_step (int): The current step of the environment.
            current_price (float): The current market price.
        """
        remaining_orders = []
        executed_order_details = []

        for order in self.pending_orders:
            executed = False  # Initialize execution flag

            # Check for expiration using the passed-in current_step
            # TODO: Consider other ways to handle expiration, e.g., GTC etc.
            expired = current_step - order["placed_at"] > self.order_expiration_steps

            if expired:
                if order["type"] == "limit_buy":
                    # Refund the reserved balance if a buy order expires
                    self.balance += order["cost_reserved"]
                elif order["type"] == "limit_sell":
                    # Return the reserved shares to the free pool if a sell order expires
                    self.units_held += order["shares"]

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": f"{order['type']}_expired",
                        "shares": order["shares"],
                        "price": order["price"],
                        "reason": "Expired",
                    }
                )
                executed = True

            # Check for execution if not expired
            elif order["type"] == "limit_buy" and current_price <= order["price"]:
                # A limit buy executes at or below the limit price.
                # For simplicity, we execute at the limit price.
                execution_price = order["price"]

                # The cost was already subtracted. We just add the shares to our free pool.
                self.units_held += order["shares"]
                executed = True

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "limit_buy_executed",
                        "shares": order["shares"],
                        "price": execution_price,
                        "cost": order["cost_reserved"],
                    }
                )

            elif order["type"] == "limit_sell" and current_price >= order["price"]:
                # A limit sell executes at or above the limit price.
                # For simplicity, we execute at the limit price.
                execution_price = order["price"]

                # The shares were already reserved. We calculate revenue and add to balance.
                revenue = order["shares"] * execution_price * (1 - self.transaction_cost_pct)
                self.balance += revenue
                executed = True

                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "limit_sell_executed",
                        "shares": order["shares"],
                        "price": execution_price,
                        "revenue": revenue,
                    }
                )

            if not executed:
                remaining_orders.append(order)

        # Update the list of pending orders and log any events
        self.pending_orders = remaining_orders
        if executed_order_details:
            self.executed_orders_history.extend(executed_order_details)

    def _process_risk_management_orders(self, current_step: int, current_price: float) -> None:
        """
        Process all stop-loss and take-profit orders against the current
        market price.

        Args:
            current_step (int): The current step of the environment.
            current_price (float): The current market price.
        """
        executed_order_details = []

        # === Process Stop Loss Orders ===
        # If the current price is equal or below the stop loss price, execute the order.
        # If the stop loss is triggered, we sell at the current price with slippage.
        # The shares were already reserved, so we just adjust the balance.
        # If the stop loss is not triggered, we keep the order for the next step.
        # We also log the executed orders for later analysis.
        remaining_stop_loss = []
        for order in self.stop_loss_orders:
            if current_price <= order["price"]:
                # Triggered: Sell at the current price with slippage
                adjusted_price = current_price * (1 - self.slippage)
                revenue = order["shares"] * adjusted_price * (1 - self.transaction_cost_pct)
                self.balance += revenue

                # Shares were already reserved, so no change to self.units_held is needed.
                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "stop_loss_executed",
                        "shares": order["shares"],
                        "trigger_price": order["price"],
                        "execution_price": adjusted_price,
                        "revenue": revenue,
                    }
                )
            else:
                remaining_stop_loss.append(order)
        self.stop_loss_orders = remaining_stop_loss

        # === Process Take Profit Orders ===
        # If the current price is equal or above the take profit price, execute the order.
        # If the take profit is triggered, we sell at the current price with slippage.
        # The shares were already reserved, so we just adjust the balance.
        # If the take profit is not triggered, we keep the order for the next step.
        # We also log the executed orders for later analysis.
        remaining_take_profit = []
        for order in self.take_profit_orders:
            if current_price >= order["price"]:
                # Triggered: Sell at the current price with slippage
                adjusted_price = current_price * (1 - self.slippage)
                revenue = order["shares"] * adjusted_price * (1 - self.transaction_cost_pct)
                self.balance += revenue

                # Shares were already reserved.
                executed_order_details.append(
                    {
                        "step": current_step,
                        "type": "take_profit_executed",
                        "shares": order["shares"],
                        "trigger_price": order["price"],
                        "execution_price": adjusted_price,
                        "revenue": revenue,
                    }
                )
            else:
                remaining_take_profit.append(order)
        self.take_profit_orders = remaining_take_profit

        # Log any events
        if executed_order_details:
            self.executed_orders_history.extend(executed_order_details)
