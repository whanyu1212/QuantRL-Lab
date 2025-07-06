from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.core.trading_env import TradingEnvProtocol
from quantrl_lab.custom_envs.stock.strategies.rewards import (  # InvalidActionPenalty,; WeightedCompositeReward,
    PortfolioValueChangeReward,
)


def test_portfolio_value_reward(standard_env: TradingEnvProtocol):
    """Test portfolio value change reward calculation."""
    standard_env.reset()

    # Set up test conditions
    standard_env.prev_portfolio_value = 10000.0
    current_price = standard_env._get_current_price()
    # Use the correct enum value for Buy
    standard_env.portfolio.execute_market_order(
        action_type=Actions.Buy,  # Use proper enum instead of 0
        current_price=current_price,
        amount_pct=0.1,
        current_step=0,
    )

    # Calculate reward
    reward_strategy = PortfolioValueChangeReward()
    reward = reward_strategy.calculate_reward(standard_env)

    assert isinstance(reward, float)
