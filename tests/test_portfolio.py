from quantrl_lab.custom_envs.core.actions import Actions
from quantrl_lab.custom_envs.stock.stock_portfolio import StockPortfolio


def test_portfolio_creation():
    """Test that portfolio can be created."""
    portfolio = StockPortfolio(
        initial_balance=10000.0, transaction_cost_pct=0.001, slippage=0.0005, order_expiration_steps=5
    )
    assert portfolio.balance == 10000.0
    assert portfolio.shares_held == 0


def test_portfolio_reset():
    """Test that portfolio can be reset."""
    portfolio = StockPortfolio(
        initial_balance=10000.0, transaction_cost_pct=0.001, slippage=0.0005, order_expiration_steps=5
    )
    portfolio.balance = 5000.0
    portfolio.units_held = 10
    portfolio.reset()
    assert portfolio.balance == 10000.0
    assert portfolio.shares_held == 0


def test_market_buy():
    """Test market buy functionality."""
    portfolio = StockPortfolio(
        initial_balance=10000.0, transaction_cost_pct=0.001, slippage=0.0005, order_expiration_steps=5
    )
    portfolio.execute_market_order(action_type=Actions.Buy, current_price=100.0, amount_pct=0.5, current_step=0)
    assert portfolio.balance < 10000.0
    assert portfolio.shares_held > 0


def test_market_sell():
    """Test market sell functionality."""
    portfolio = StockPortfolio(
        initial_balance=10000.0, transaction_cost_pct=0.001, slippage=0.0005, order_expiration_steps=5
    )
    # First buy some shares
    portfolio.execute_market_order(action_type=Actions.Buy, current_price=100.0, amount_pct=0.5, current_step=0)
    initial_shares = portfolio.shares_held
    initial_balance = portfolio.balance

    # Then sell half of them
    portfolio.execute_market_order(action_type=Actions.Sell, current_price=100.0, amount_pct=0.5, current_step=1)

    assert portfolio.balance > initial_balance
    assert portfolio.shares_held < initial_shares
