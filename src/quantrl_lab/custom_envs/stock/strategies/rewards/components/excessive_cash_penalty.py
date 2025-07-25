from __future__ import annotations

from typing import TYPE_CHECKING

from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.stock.strategies.rewards.base_reward import (
    BaseRewardStrategy,
)

if TYPE_CHECKING:
    from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol


class ExcessiveCashUsagePenalty(BaseRewardStrategy):
    """
    A focused penalty strategy that specifically targets unreasonable
    cash flow usage.

    This strategy penalizes:
    1. Using too much cash in a single trade (e.g., >80% of available cash)
    2. Making large trades without any risk management orders (stop loss/take profit)
    3. Depleting cash reserves below a minimum threshold
    4. Attempting to use more cash than available (invalid actions)
    """

    def __init__(
        self,
        max_cash_usage_pct: float = 0.8,  # Max % of cash to use in single trade
        min_cash_reserve_pct: float = 0.05,  # Min cash reserve to maintain (5%)
        large_trade_threshold: float = 0.5,  # Trades using >50% cash are "large"
        base_penalty: float = -0.1,  # Base penalty for violations
        severe_penalty: float = -0.2,  # Severe penalty for major violations
    ):
        self.max_cash_usage_pct = max_cash_usage_pct
        self.min_cash_reserve_pct = min_cash_reserve_pct
        self.large_trade_threshold = large_trade_threshold
        self.base_penalty = base_penalty
        self.severe_penalty = severe_penalty

    def calculate_reward(self, env: TradingEnvProtocol) -> float:
        """
        Calculate penalties for excessive cash usage.

        Args:
            env (TradingEnvProtocol): The trading environment instance.

        Returns:
            float: The penalty (negative value) for poor cash management, 0 for acceptable behavior.
        """
        # Only evaluate if an action was taken this step
        if not hasattr(env, 'action_type') or not hasattr(env, 'decoded_action_info'):
            return 0.0

        action_type = env.action_type
        decoded_info = env.decoded_action_info

        # Only evaluate buy actions (where cash is used)
        if action_type not in [Actions.Buy, Actions.LimitBuy]:
            return 0.0

        penalty = 0.0
        portfolio = env.portfolio
        current_price = env._get_current_price()
        total_value = portfolio.get_value(current_price)

        if total_value <= 1e-9 or portfolio.balance <= 1e-9:
            return 0.0

        amount_pct = decoded_info.get('amount_pct', 0.0)
        invalid_action = decoded_info.get('invalid_action_attempt', False)

        # 1. Severe penalty for invalid actions (trying to use unavailable cash)
        if invalid_action:
            penalty += self.severe_penalty

        # 2. Penalty for using too much cash in a single trade
        if amount_pct > self.max_cash_usage_pct:
            excess_ratio = amount_pct - self.max_cash_usage_pct
            # Scale penalty by how much they exceeded the limit
            penalty += self.base_penalty * (excess_ratio / (1.0 - self.max_cash_usage_pct))

        # 3. Extra penalty for large trades without risk management
        if amount_pct > self.large_trade_threshold:
            has_risk_management = len(portfolio.stop_loss_orders) > 0 or len(portfolio.take_profit_orders) > 0

            if not has_risk_management:
                # Penalty scales with trade size
                risk_penalty = self.base_penalty * (amount_pct / self.large_trade_threshold)
                penalty += risk_penalty

        # 4. Check if this trade would violate minimum cash reserve
        cash_after_trade = portfolio.balance * (1.0 - amount_pct)
        cash_ratio_after = cash_after_trade / total_value if total_value > 0 else 0.0

        if cash_ratio_after < self.min_cash_reserve_pct:
            # Penalty for depleting cash reserves
            reserve_violation = self.min_cash_reserve_pct - cash_ratio_after
            penalty += self.base_penalty * (reserve_violation / self.min_cash_reserve_pct)

        return penalty

    def on_step_end(self, env: TradingEnvProtocol):
        """No state to update for this penalty strategy."""
        pass
