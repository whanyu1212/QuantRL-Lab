from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class CashFlowRiskManagementReward(BaseRewardStrategy):
    """
    A reward strategy that penalizes excessive cash flow usage without
    proper risk management and rewards prudent cash flow management with
    appropriate risk controls.

    Key Principles:
    1. Penalize using too much cash (>90%) in a single trade without stop losses
    2. Penalize having too many open positions without risk management orders
    3. Reward balanced portfolio allocation (cash vs invested)
    4. Reward setting stop losses and take profit orders
    5. Penalize margin-like behavior (trying to use more cash than available)
    """

    def __init__(
        self,
        max_single_trade_pct: float = 0.3,  # Max % of portfolio value in single trade
        max_cash_usage_pct: float = 0.9,  # Max % of cash to use without risk management
        min_cash_reserve_pct: float = 0.1,  # Min cash reserve to maintain
        risk_management_bonus: float = 0.05,  # Bonus for using stop loss/take profit
        excessive_exposure_penalty: float = -0.1,  # Penalty for excessive exposure
        no_risk_management_penalty: float = -0.05,  # Penalty for no risk management
        balanced_allocation_bonus: float = 0.02,  # Bonus for balanced allocation
    ):
        self.max_single_trade_pct = max_single_trade_pct
        self.max_cash_usage_pct = max_cash_usage_pct
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.risk_management_bonus = risk_management_bonus
        self.excessive_exposure_penalty = excessive_exposure_penalty
        self.no_risk_management_penalty = no_risk_management_penalty
        self.balanced_allocation_bonus = balanced_allocation_bonus

        # Track state for reward calculation
        self.previous_cash_ratio = None
        self.previous_total_value = None

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate reward/penalty based on cash flow risk management.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The calculated reward (positive for good practices, negative for poor ones).
        """
        current_price = env._get_current_price()
        portfolio = env.portfolio
        total_value = portfolio.get_value(current_price)

        if total_value <= 1e-9:
            return 0.0

        cash_ratio = portfolio.balance / total_value
        invested_ratio = 1 - cash_ratio

        reward = 0.0

        # 1. Check current action for excessive single trade exposure
        if hasattr(env, 'action_type') and hasattr(env, 'decoded_action_info'):
            reward += self._evaluate_current_action(env, total_value)

        # 2. Evaluate overall portfolio risk management
        reward += self._evaluate_portfolio_risk_management(env, cash_ratio, invested_ratio)

        # 3. Evaluate cash flow patterns
        reward += self._evaluate_cash_flow_patterns(env, cash_ratio, total_value)

        # 4. Check for balanced allocation
        reward += self._evaluate_allocation_balance(cash_ratio)

        # Update state for next step
        self.previous_cash_ratio = cash_ratio
        self.previous_total_value = total_value

        return reward

    def _evaluate_current_action(self, env: TradingEnvProtocol, total_value: float) -> float:
        """Evaluate the current action for excessive exposure."""
        if env.action_type not in [Actions.Buy, Actions.LimitBuy]:
            return 0.0

        amount_pct = env.decoded_action_info.get('amount_pct', 0.0)
        current_price = env._get_current_price()  # noqa F841
        portfolio = env.portfolio

        # Calculate the value of this trade relative to total portfolio
        if env.action_type == Actions.Buy:
            trade_value = portfolio.balance * amount_pct
        else:  # LimitBuy
            # Estimate based on available cash
            trade_value = portfolio.balance * amount_pct

        trade_ratio = trade_value / total_value if total_value > 0 else 0.0

        penalty = 0.0

        # Penalize excessive single trade exposure
        if trade_ratio > self.max_single_trade_pct:
            excess_ratio = trade_ratio - self.max_single_trade_pct
            penalty += self.excessive_exposure_penalty * (excess_ratio / self.max_single_trade_pct)

        # Extra penalty if using high % of cash without any risk management orders
        has_risk_management = len(portfolio.stop_loss_orders) > 0 or len(portfolio.take_profit_orders) > 0

        if amount_pct > self.max_cash_usage_pct and not has_risk_management:
            penalty += self.no_risk_management_penalty * 2  # Double penalty

        return penalty

    def _evaluate_portfolio_risk_management(
        self, env: TradingEnvProtocol, cash_ratio: float, invested_ratio: float
    ) -> float:
        """Evaluate overall portfolio risk management practices."""
        portfolio = env.portfolio
        reward = 0.0

        # Reward having risk management orders when holding positions
        if portfolio.total_shares > 0:
            total_risk_orders = len(portfolio.stop_loss_orders) + len(portfolio.take_profit_orders)
            if total_risk_orders > 0:
                # Bonus for having risk management, scaled by position size
                risk_coverage_ratio = min(1.0, total_risk_orders / max(1, portfolio.total_shares / 100))
                reward += self.risk_management_bonus * risk_coverage_ratio
            else:
                # Penalty for having significant positions without risk management
                if invested_ratio > 0.5:  # If more than 50% invested
                    reward += self.no_risk_management_penalty * invested_ratio

        return reward

    def _evaluate_cash_flow_patterns(self, env: TradingEnvProtocol, cash_ratio: float, total_value: float) -> float:
        """Evaluate cash flow usage patterns."""
        reward = 0.0

        # Penalize maintaining too little cash reserve
        if cash_ratio < self.min_cash_reserve_pct:
            shortage = self.min_cash_reserve_pct - cash_ratio
            reward += self.excessive_exposure_penalty * (shortage / self.min_cash_reserve_pct)

        # Penalize erratic cash flow behavior (rapid swings in cash ratio)
        if self.previous_cash_ratio is not None:
            cash_ratio_change = abs(cash_ratio - self.previous_cash_ratio)
            if cash_ratio_change > 0.5:  # More than 50% change in cash ratio
                reward += self.excessive_exposure_penalty * 0.5  # Moderate penalty for volatility

        return reward

    def _evaluate_allocation_balance(self, cash_ratio: float) -> float:
        """Reward balanced allocation between cash and investments."""
        # Optimal range: 10-40% cash, 60-90% invested
        optimal_cash_min = 0.1
        optimal_cash_max = 0.4

        if optimal_cash_min <= cash_ratio <= optimal_cash_max:
            # Reward being in the optimal range
            return self.balanced_allocation_bonus
        elif cash_ratio > 0.9:
            # Penalize being mostly in cash (not investing)
            return -self.balanced_allocation_bonus * 0.5
        elif cash_ratio < 0.05:
            # Penalize being over-invested
            return -self.balanced_allocation_bonus

        return 0.0

    def on_step_end(self, env: TradingEnvProtocol):
        """Update any internal state if needed."""
        # State is already updated in calculate_reward
        pass
