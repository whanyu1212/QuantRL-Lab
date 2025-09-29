from dataclasses import dataclass

# Removed unused imports
from typing import Any, Dict

import numpy as np

# Only importing what's used
import yfinance as yf


@dataclass
class PerformanceMetrics:
    """Standard performance metrics for trading strategies."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sortino_ratio: float
    calmar_ratio: float


class BenchmarkAnalyzer:
    """Analyze RL agent performance against standard benchmarks."""

    def __init__(self, symbol: str, start_date: str, end_date: str):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

    def get_buy_and_hold_performance(self, initial_capital: float = 100000) -> PerformanceMetrics:
        """Calculate buy-and-hold performance for the same period."""
        # Fetch price data
        ticker = yf.Ticker(self.symbol)
        price_data = ticker.history(start=self.start_date, end=self.end_date)

        # Calculate returns
        start_price = price_data["Close"].iloc[0]
        end_price = price_data["Close"].iloc[-1]
        shares_bought = initial_capital / start_price
        final_value = shares_bought * end_price

        return self._calculate_metrics(price_data["Close"].values, initial_capital, final_value)

    def get_index_performance(self, index_symbol: str = "^GSPC") -> PerformanceMetrics:
        """Get performance of market index (default S&P 500)."""
        ticker = yf.Ticker(index_symbol)
        index_data = ticker.history(start=self.start_date, end=self.end_date)

        start_value = index_data["Close"].iloc[0]
        end_value = index_data["Close"].iloc[-1]

        return self._calculate_metrics(index_data["Close"].values, start_value, end_value)

    def get_dollar_cost_averaging_performance(self, initial_capital: float = 100000) -> PerformanceMetrics:
        """Calculate dollar-cost averaging performance."""
        ticker = yf.Ticker(self.symbol)
        price_data = ticker.history(start=self.start_date, end=self.end_date)

        # Simulate weekly DCA
        weekly_investment = initial_capital / len(price_data)
        total_shares = 0

        for price in price_data["Close"]:
            shares_bought = weekly_investment / price
            total_shares += shares_bought

        final_value = total_shares * price_data["Close"].iloc[-1]
        return self._calculate_metrics(price_data["Close"].values, initial_capital, final_value)

    def compare_with_benchmarks(self, agent_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare agent performance with multiple benchmarks."""
        initial_capital = agent_results.get("initial_value", 100000)

        benchmarks = {
            "RL_Agent": self._extract_agent_metrics(agent_results),
            "Buy_and_Hold": self.get_buy_and_hold_performance(initial_capital),
            "S&P_500": self.get_index_performance("^GSPC"),
            "NASDAQ": self.get_index_performance("^IXIC"),
            "DJI": self.get_index_performance("^DJI"),
            "Dollar_Cost_Averaging": self.get_dollar_cost_averaging_performance(initial_capital),
        }

        return benchmarks

    def _calculate_metrics(self, prices: np.ndarray, initial_value: float, final_value: float) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        total_return = (final_value - initial_value) / initial_value

        # Annualized return
        trading_days = len(prices)
        years = trading_days / 252  # Assume 252 trading days per year
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)

        # Sharpe ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)

        # Profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        profit_factor = (
            np.sum(positive_returns) / abs(np.sum(negative_returns)) if len(negative_returns) > 0 else float("inf")
        )

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
        )

    def _extract_agent_metrics(self, agent_results: Dict[str, Any]) -> PerformanceMetrics:
        """Extract metrics from agent results."""
        # This would process your agent's episode results
        # and calculate the same metrics
        episodes = agent_results.get("test_episodes", [])

        if not episodes:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Calculate portfolio values over time
        portfolio_values = []
        for episode in episodes:
            if "detailed_actions" in episode:
                for action in episode["detailed_actions"]:
                    portfolio_values.append(action["portfolio_value"])

        if len(portfolio_values) < 2:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)

        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]

        return self._calculate_metrics(np.array(portfolio_values), initial_value, final_value)

    def create_comparison_report(self, benchmarks: Dict[str, PerformanceMetrics]) -> str:
        """Create a formatted comparison report."""
        report = ["=" * 80]
        report.append("PERFORMANCE COMPARISON REPORT")
        report.append("=" * 80)

        # Header
        report.append(
            f"{'Strategy':<20} {'Total Ret':<10} {'Annual Ret':<10} {'Sharpe':<8} {'Max DD':<10} {'Win Rate':<10}"
        )
        report.append("-" * 80)

        # Sort by Sharpe ratio
        sorted_strategies = sorted(benchmarks.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)

        for strategy, metrics in sorted_strategies:
            report.append(
                f"{strategy:<20} {metrics.total_return*100:>8.2f}% {metrics.annualized_return*100:>8.2f}% "
                f"{metrics.sharpe_ratio:>6.2f} {metrics.max_drawdown*100:>8.2f}% {metrics.win_rate*100:>8.2f}%"
            )

        return "\n".join(report)


def run_comprehensive_analysis(
    agent_results: Dict[str, Any], symbol: str, start_date: str, end_date: str
) -> Dict[str, Any]:
    """Run comprehensive analysis comparing agent to benchmarks."""
    analyzer = BenchmarkAnalyzer(symbol, start_date, end_date)
    benchmarks = analyzer.compare_with_benchmarks(agent_results)

    # Create detailed comparison
    comparison_report = analyzer.create_comparison_report(benchmarks)

    # Calculate relative performance
    agent_sharpe = benchmarks["RL_Agent"].sharpe_ratio
    buy_hold_sharpe = benchmarks["Buy_and_Hold"].sharpe_ratio
    sp500_sharpe = benchmarks["S&P_500"].sharpe_ratio

    relative_performance = {
        "vs_buy_hold": agent_sharpe - buy_hold_sharpe,
        "vs_sp500": agent_sharpe - sp500_sharpe,
        "outperformed_buy_hold": agent_sharpe > buy_hold_sharpe,
        "outperformed_sp500": agent_sharpe > sp500_sharpe,
    }

    return {
        "benchmarks": benchmarks,
        "comparison_report": comparison_report,
        "relative_performance": relative_performance,
        "analysis_summary": {
            "best_strategy": max(benchmarks.keys(), key=lambda k: benchmarks[k].sharpe_ratio),
            "agent_rank": sorted(benchmarks.keys(), key=lambda k: benchmarks[k].sharpe_ratio, reverse=True).index(
                "RL_Agent"
            )
            + 1,
            "total_strategies": len(benchmarks),
        },
    }
