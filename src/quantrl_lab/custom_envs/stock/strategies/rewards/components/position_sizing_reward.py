from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class PositionSizingRiskReward(BaseRewardStrategy):
    """
    A reward strategy focused on proper position sizing and risk
    management.

    Encourages:
    1. Appropriate position sizes relative to portfolio
    2. Using stop losses for downside protection
    3. Taking profits at reasonable levels
    4. Maintaining liquidity for opportunities

    Discourages:
    1. Over-leveraging (using too much of portfolio in single position)
    2. All-in trades without protection
    3. Ignoring risk management tools available
    """

    def __init__(
        self,
        optimal_position_size: float = 0.2,  # Optimal position size (20% of portfolio)
        max_position_size: float = 0.4,  # Maximum acceptable position size (40%)
        risk_management_reward: float = 0.02,  # Reward for using stop losses/take profits
        position_size_penalty: float = -0.05,  # Penalty for oversized positions
        liquidity_bonus: float = 0.01,  # Small bonus for maintaining cash
    ):
        self.optimal_position_size = optimal_position_size
        self.max_position_size = max_position_size
        self.risk_management_reward = risk_management_reward
        self.position_size_penalty = position_size_penalty
        self.liquidity_bonus = liquidity_bonus

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate reward based on position sizing and risk management.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The calculated reward/penalty.
        """
        reward = 0.0
        portfolio = env.portfolio
        current_price = env._get_current_price()
        total_value = portfolio.get_value(current_price)

        if total_value <= 1e-9:
            return 0.0

        # 1. Evaluate current action if it's a buy order
        if hasattr(env, 'action_type') and hasattr(env, 'decoded_action_info'):
            reward += self._evaluate_position_sizing(env, total_value)

        # 2. Evaluate existing risk management
        reward += self._evaluate_risk_management(env)

        # 3. Small liquidity bonus for maintaining reasonable cash levels
        cash_ratio = portfolio.balance / total_value
        if 0.1 <= cash_ratio <= 0.3:  # 10-30% cash is good for opportunities
            reward += self.liquidity_bonus

        return reward

    def _evaluate_position_sizing(self, env: TradingEnvProtocol, total_value: float) -> float:
        """Evaluate the position sizing of the current action."""
        if env.action_type not in [Actions.Buy, Actions.LimitBuy]:
            return 0.0

        amount_pct = env.decoded_action_info.get('amount_pct', 0.0)
        portfolio = env.portfolio

        # Calculate the position size relative to total portfolio value
        trade_value = portfolio.balance * amount_pct
        position_ratio = trade_value / total_value if total_value > 0 else 0.0

        # Reward/penalty based on position size
        if position_ratio <= self.optimal_position_size:
            # Good position sizing - small positive reward
            return 0.01
        elif position_ratio <= self.max_position_size:
            # Acceptable but not optimal - neutral
            return 0.0
        else:
            # Oversized position - penalty that scales with excess
            excess_ratio = position_ratio - self.max_position_size
            penalty_multiplier = excess_ratio / self.optimal_position_size
            return self.position_size_penalty * penalty_multiplier

    def _evaluate_risk_management(self, env: TradingEnvProtocol) -> float:
        """Evaluate the use of risk management tools."""
        portfolio = env.portfolio

        if portfolio.total_shares <= 0:
            return 0.0  # No positions, no risk management needed

        reward = 0.0

        # Reward for having stop losses
        if len(portfolio.stop_loss_orders) > 0:
            # Scale reward by coverage (more stop losses for larger positions)
            stop_loss_coverage = min(1.0, len(portfolio.stop_loss_orders) / max(1, portfolio.total_shares // 100))
            reward += self.risk_management_reward * stop_loss_coverage

        # Reward for having take profit orders
        if len(portfolio.take_profit_orders) > 0:
            take_profit_coverage = min(1.0, len(portfolio.take_profit_orders) / max(1, portfolio.total_shares // 100))
            reward += self.risk_management_reward * 0.5 * take_profit_coverage  # Half weight vs stop loss

        # Bonus for having both types of risk management
        if len(portfolio.stop_loss_orders) > 0 and len(portfolio.take_profit_orders) > 0:
            reward += self.risk_management_reward * 0.25  # Small bonus for comprehensive risk management

        return reward

    def on_step_end(self, env: TradingEnvProtocol):
        """No persistent state to update."""
        pass
