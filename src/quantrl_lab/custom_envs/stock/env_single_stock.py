from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.stock.stock_config import SingleStockEnvConfig
from quantrl_lab.custom_envs.stock.stock_portfolio import StockPortfolio
from quantrl_lab.custom_envs.stock.strategies.actions.base_action import (
    BaseActionStrategy,
)
from quantrl_lab.custom_envs.stock.strategies.observations.base_observation import (
    BaseObservationStrategy,
)
from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)


class SingleStockTradingEnv(gym.Env):
    # Added metadata for Gymnasium compatibility
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # TODO: Switch to use DataSourceRegistry for data loading and management

    def __init__(
        self,
        data: np.ndarray,  # Multidimensional array of market data + additional features
        config: SingleStockEnvConfig,  # Configuration object for environment settings
        action_strategy: BaseActionStrategy,  # Strategy for defining action space and handling actions,
        reward_strategy: BaseRewardStrategy,  # Strategy for calculating rewards
        observation_strategy: BaseObservationStrategy,
    ):
        super().__init__()

        # === Runtime error handling ===
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Data must be a 2D numpy array (num_steps, num_features).")
        if data.shape[0] <= config.window_size:
            raise ValueError("Data length must be greater than window_size.")
        if not (0 <= config.price_column_index < data.shape[1]):
            raise ValueError(f"price_column_index ({config.price_column_index}) is out of bounds.")

        # === Attributes ===
        self.Actions = Actions  # reference to the Actions class for easy access
        self.data = data.astype(np.float32)  # float32 for more efficiency and better compatibility
        self.num_steps, self.num_features = self.data.shape
        self.price_column_index = config.price_column_index
        self.window_size = config.window_size
        self._max_steps = self.num_steps - 1  # Max indexable step

        # Initialize the portfolio
        self.portfolio = StockPortfolio(
            initial_balance=config.initial_balance,
            transaction_cost_pct=config.transaction_cost_pct,
            slippage=config.slippage,
            order_expiration_steps=config.order_expiration_steps,
        )
        # TODO: consider other ways to handle expiration, e.g., GTC etc.

        # === Define the strategies for action, reward, and observation ===
        self.action_strategy = action_strategy
        self.reward_strategy = reward_strategy
        self.observation_strategy = observation_strategy
        # === Delegate the action space and observation space definitions to the strategies ===
        # This allows for more modular and flexible design, where each strategy can define its own logic
        # for actions and observations without cluttering the environment class.
        self.action_space = self.action_strategy.define_action_space()
        self.observation_space = self.observation_strategy.define_observation_space(self)

        # === Example action space values:
        # Market Buy 50% of available balance
        # [1.0, 0.5, 1.0]  # Action type 1, 50% amount, price modifier ignored

        # Limit Sell 75% of shares at 5% above market price
        # [4.0, 0.75, 1.05]  # Action type 4, 75% amount, 5% above price

        # Stop Loss 100% of shares at 10% below market price
        # [5.0, 1.0, 0.9]  # Action type 5, 100% amount, 10% below price
        # ================================================================

        # === Initialize some environment state variables ===
        self.reward_clip_range = config.reward_clip_range
        self.prev_portfolio_value = 0.0
        self.action_type = None
        self.decoded_action_info = {}
        self.current_step = 0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.

        Args:
            action (np.ndarray): The action to execute.

        Raises:
            ValueError: If the action is not valid.

        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: The observation, reward, terminated, truncated, and info.
        """
        # === Input validation ===
        if not isinstance(action, np.ndarray) or action.shape != self.action_space.shape:
            raise ValueError(
                f"Invalid action format received in step: {action}. Expected shape {self.action_space.shape}"
            )

        # === Step Execution ===

        # 1. Get the current price and store the portfolio value BEFORE any changes happen in this step
        # This is important for reward calculations, as we need to know the previous value
        # of the portfolio before applying the new action.
        current_price = self._get_current_price()
        self.prev_portfolio_value = self.portfolio.get_value(current_price)

        # 2. Process any open orders that might be pending from previous steps.
        self.portfolio.process_open_orders(self.current_step, current_price)

        # 3. Handle the new action and STORE the results on `self`.
        # The reward strategies will access these via `env.action_type` and `env.decoded_action_info`.
        self.action_type, self.decoded_action_info = self.action_strategy.handle_action(self, action)

        # 4. Advance time and get new state (no change here).
        # Increment the current step and check if we have reached the end of the episode.
        # This is done before calculating the reward to ensure the reward is based on the state after
        # the action has been applied.
        if self.current_step >= self._max_steps:
            raise ValueError("Cannot step beyond the maximum number of steps in the environment.")
        # Increment the step count
        self.current_step += 1
        current_price = self._get_current_price()
        terminated = self.current_step >= self._max_steps
        truncated = False

        # 5. Reward Calculation. This is delegated to the reward strategy.
        # We pass `self` so the strategy has full access to the environment's state.
        reward = self.reward_strategy.calculate_reward(self)

        # 6. Clip the final, combined reward (good practice to keep this).
        reward = np.clip(reward, *self.reward_clip_range).item()

        # 7. Call the 'on_step_end' hook for stateful strategies to update their internal memory.
        self.reward_strategy.on_step_end(self)

        # 8. Get next observation (no change here).
        observation = self.observation_strategy.build_observation(self)

        # 9. Build the info dictionary.
        # This contains useful information about the current state of the environment,
        # including portfolio value, balance, shares held, and the last executed order.
        # This is useful for debugging and analysis.
        # It can also be used by the reward strategy to provide additional context for reward calculation.
        info = self._build_info_dict()

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resets the environment to an initial state and returns the
        initial observation.

        Args:
            seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
            options (Optional[Dict], optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            Tuple[np.ndarray, Dict]: Initial observation and info dictionary.
        """
        super().reset(seed=seed)

        # 1. Reset the current step to the initial state.
        # This is typically the first step after the initial observation.
        # We set it to the window size to ensure we have enough data for the first observation
        # and to avoid index errors.
        self.current_step = self.window_size

        # 2. Reset the portfolio to its initial state.
        # This clears any pending orders, resets the balance, and prepares the portfolio
        # for a new episode.
        # Note: This does not reset the portfolio's initial balance, which is set in the
        # StockPortfolio constructor. It only clears the current state.
        self.portfolio.reset()

        # 3. Reset the action type and decoded action info.
        # This is important to ensure that the environment starts fresh without any
        # lingering state from previous episodes.
        initial_observation = self.observation_strategy.build_observation(self)
        info = {
            "initial_balance": self.portfolio.initial_balance,
            "starting_step": self.current_step,
            "message": "Environment reset.",
        }
        return initial_observation, info

    def render(self, mode="human"):
        """Renders the environment state."""
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "human":
            self._render_human()

    def _render_human(self):
        """Prints state information to the console."""
        current_price = self._get_current_price()
        portfolio_value = self.portfolio.get_value(current_price)
        total_shares = self.portfolio.total_shares

        print("-" * 40)
        print(f"Step:         {self.current_step}/{self._max_steps}")
        print(f"Current Price:{current_price:>15.2f}")
        print(f"Balance:      {self.portfolio.balance:>15.2f}")
        print(f"Shares Held:  {self.portfolio.shares_held:>15} (Free)")
        print(f"Total Shares: {total_shares:>15} (Free + Reserved)")
        print(f"Portfolio Val:{portfolio_value:>15.2f}")
        print("-" * 40)
        print("Active Orders:")
        print(f" Pending Limit:{len(self.portfolio.pending_orders):>5}")
        print(f"  Stop Loss:    {len(self.portfolio.stop_loss_orders):>5}")
        print(f"  Take Profit:  {len(self.portfolio.take_profit_orders):>5}")

        if self.portfolio.executed_orders_history:
            last_event = self.portfolio.executed_orders_history[-1]
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
        portfolio_value = self.portfolio.get_value(current_price)
        total_shares = self.portfolio.total_shares
        last_event_str = "None"

        if self.portfolio.executed_orders_history:
            last_event = self.portfolio.executed_orders_history[-1]
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
            f"Balance: {self.portfolio.balance:.2f} | "
            f"Shares(F/T): {self.portfolio.shares_held}/{total_shares} | "
            f"Value: {portfolio_value:.2f} | "
            f"Orders(P/SL/TP): {len(self.portfolio.pending_orders)}/"
            f"{len(self.portfolio.stop_loss_orders)}/"
            f"{len(self.portfolio.take_profit_orders)} | "
            f"Last Event: {last_event_str}"
        )

    def close(self):
        print("SingleStockTradingEnv closed.")
        pass

    # === Private Methods ===

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
                raise ValueError(
                    f"No valid price data available at step {self.current_step} (data length: {self.num_steps})"
                )

    def _build_info_dict(self) -> Dict[str, Any]:
        """
        Builds an information dictionary for the current environment
        state.

        Returns:
            Dict[str, Any]: A dictionary containing relevant information about the environment state.
        """
        current_price = self._get_current_price()
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio.get_value(current_price),
            "balance": self.portfolio.balance,
            "shares_held": self.portfolio.shares_held,
            "total_shares": self.portfolio.total_shares,
            "current_price": current_price,
            "reward": self.reward_strategy.calculate_reward(self),  # Re-calculate for info or store from step
            "action_decoded": self.decoded_action_info,
            "orders_info": {
                "pending_count": len(self.portfolio.pending_orders),
                "stop_loss_count": len(self.portfolio.stop_loss_orders),
                "take_profit_count": len(self.portfolio.take_profit_orders),
            },
            "last_order_event": (
                self.portfolio.executed_orders_history[-1] if self.portfolio.executed_orders_history else None
            ),
        }
