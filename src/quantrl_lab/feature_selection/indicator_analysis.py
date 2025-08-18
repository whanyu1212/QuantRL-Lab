from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.indicators.indicator_registry import IndicatorRegistry
from .base_vectorized_strategy import SignalType
from .vectorized_strategy_factory import VectorizedStrategyFactory


@dataclass
class IndicatorAnalysisConfig:
    """Configuration for indicator analysis."""

    initial_capital: float = 100000
    transaction_cost: float = 0.001  # 0.1% per trade
    risk_free_rate: float = 0.02  # 2% annual
    min_holding_period: int = 1  # Minimum holding period in days
    position_sizing: str = 'full'  # 'full', 'half', 'third', 'kelly (based on Kelly Criterion)'

    def __post_init__(self):
        """Validate configuration."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")
        if self.position_sizing not in ['full', 'half', 'third', 'kelly']:
            raise ValueError("position_sizing must be 'full', 'half', 'third', or 'kelly'")


class IndicatorAnalysis:
    """
    Comprehensive analysis of technical indicator trading signals.

    This class generates trading signals from technical indicators,
    simulates portfolio performance, and calculates various risk/return
    metrics.
    """

    def __init__(self, config: Optional[IndicatorAnalysisConfig] = None):

        # user can override default config by defining their own
        self.config = config or IndicatorAnalysisConfig()

        # Factory for creating vectorized trading strategies
        self.strategy_factory = VectorizedStrategyFactory()

    @staticmethod
    def create_indicator_configs(*configs) -> Dict[str, Dict[str, Any]]:
        """
        Generate indicator configs from simplified definitions.

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of indicator names
            to their configurations.
        """
        result = {}

        for config in configs:
            indicator_name = config['name']
            param_sets = config['params']
            strategy_params = config.get('strategy_params', {})

            for i, params in enumerate(param_sets):
                # Use base name for single configs, descriptive names for multiple
                if len(param_sets) == 1:
                    config_name = indicator_name  # Just 'SMA', 'RSI', etc.
                else:
                    # param_str = "_".join([f"{v}" if k == 'window' else f"{k}{v}" for k, v in params.items()])
                    param_str = "_".join([f"{v}" for v in params.values()])
                    config_name = f"{indicator_name}_{param_str}"  # 'SMA_20', etc.

                result[config_name] = {'indicator_params': params, 'strategy_params': strategy_params}

        return result

    def analyze_indicator(
        self,
        data: pd.DataFrame,
        indicator_name: str,
        indicator_params: Dict[str, Any],
        allow_short: bool = True,
        **strategy_params,
    ) -> Dict[str, Any]:
        """
        Analyze a technical indicator based on vectorized backtesting.

        Args:
            data (pd.DataFrame): OHLCV Dataframe
            indicator_name (str): indicator name
            indicator_params (Dict[str, Any]): params applied to it
            allow_short (bool, optional): Whether to allow short positions.
            Defaults to True.

        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        # Validate input data by checking the OHLCV columns
        self._validate_data(data)

        # Calculate technical indicators
        data_with_indicators = self._calculate_indicators(data, indicator_name, indicator_params)

        # Create strategy and generate signals
        strategy = self.strategy_factory.create_strategy(
            indicator_name=indicator_name, allow_short=allow_short, **indicator_params, **strategy_params
        )

        signals = strategy.generate_signals(data_with_indicators)

        # Simulate portfolio performance
        portfolio_results = self._simulate_portfolio(data_with_indicators, signals)

        # Calculate performance metrics
        metrics = self._calculate_metrics(portfolio_results)

        # Combine all results
        return {
            'indicator_name': indicator_name,
            'indicator_params': indicator_params,
            'strategy_params': strategy_params,
            'allow_short': allow_short,
            'total_return': metrics['total_return'],
            'annual_return': metrics['annual_return'],
            'volatility': metrics['volatility'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': metrics['max_drawdown'],
            'win_rate': metrics['win_rate'],
            'profit_factor': metrics['profit_factor'],
            'sortino_ratio': metrics['sortino_ratio'],
            'calmar_ratio': metrics['calmar_ratio'],
            'final_value': portfolio_results['portfolio_values'].iloc[-1],
            'portfolio_values': portfolio_results['portfolio_values'],
            'strategy_returns': portfolio_results['strategy_returns'],
            'positions': portfolio_results['positions'],
            'signals': signals,
            'trades_executed': portfolio_results['trades_executed'],
            'total_trades': portfolio_results['total_trades'],
            'transaction_costs': portfolio_results['transaction_costs'],
            'data_with_indicators': data_with_indicators,
        }

    def batch_analyze_indicators(
        self, data: pd.DataFrame, indicator_configs: Dict[str, Dict[str, Any]], allow_short: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze multiple indicators in batch.

        Args:
            data (pd.DataFrame): OHLCV Dataframe
            indicator_configs (Dict[str, Dict[str, Any]]): Dictionary mapping
            indicator names to their configurations
            allow_short (bool, optional): Whether to allow short positions.
            Defaults to True.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping indicator names to
            their analysis results
        """
        results = {}

        for config_name, config in indicator_configs.items():
            indicator_params = config.get('indicator_params', {})
            strategy_params = config.get('strategy_params', {})

            # Extract base indicator name from config name
            base_indicator_name = config_name.split('_')[0]  # 'SMA_10' -> 'SMA'

            try:
                results[config_name] = self.analyze_indicator(
                    data=data,
                    indicator_name=base_indicator_name,  # Use base name here
                    indicator_params=indicator_params,
                    allow_short=allow_short,
                    **strategy_params,
                )
            except Exception as e:
                print(f"Error analyzing {config_name}: {str(e)}")  # Add debugging
                results[config_name] = {
                    'error': str(e),
                    'indicator_name': base_indicator_name,
                    'indicator_params': indicator_params,
                }

        return results

    def compare_indicators(
        self,
        data: pd.DataFrame,
        indicator_configs: Dict[str, Dict[str, Any]],
        sort_by: str = 'sharpe_ratio',
        ascending: bool = False,
    ) -> pd.DataFrame:
        """
        Compare multiple indicators and return ranked results.

        Args:
            data (pd.DataFrame): OHLCV DataFrame
            indicator_configs (Dict[str, Dict[str, Any]]): Dictionary mapping
            indicator names to their configurations
            sort_by (str, optional): Metric to sort by. Defaults to 'sharpe_ratio'.
            ascending (bool, optional): Sort order. Defaults to False.

        Returns:
            pd.DataFrame: DataFrame with comparison results
        """
        results = self.batch_analyze_indicators(data, indicator_configs)

        comparison_data = []
        for name, result in results.items():
            if 'error' not in result:
                comparison_data.append(
                    {
                        'indicator': name,
                        'total_return': result['total_return'],
                        'annual_return': result['annual_return'],
                        'volatility': result['volatility'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'max_drawdown': result['max_drawdown'],
                        'win_rate': result['win_rate'],
                        'profit_factor': result['profit_factor'],
                        'sortino_ratio': result['sortino_ratio'],
                        'calmar_ratio': result['calmar_ratio'],
                        'final_value': result['final_value'],
                        'total_trades': result['total_trades'],
                    }
                )

        comparison_df = pd.DataFrame(comparison_data)

        if not comparison_df.empty and sort_by in comparison_df.columns:
            comparison_df = comparison_df.sort_values(sort_by, ascending=ascending)
        else:
            print(f"Warning: {sort_by} not found in comparison DataFrame columns.")

        return comparison_df

    # === Private Methods ===

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data.

        Args:
            data (pd.DataFrame): Input data to validate.

        Raises:
            ValueError: If the data is not a DataFrame.
            ValueError: If the DataFrame is empty.
            ValueError: If the DataFrame does not contain the required columns.
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if data.empty:
            raise ValueError("Data cannot be empty")

        if data.isnull().any().any():
            raise ValueError("Data contains null values")

    def _calculate_indicators(
        self, data: pd.DataFrame, indicator_name: str, indicator_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Private method to calculate technical indicators for the data.

        Args:
            data (pd.DataFrame): OHLCV DataFrame
            indicator_name (str): Name of the indicator (e.g., 'SMA', 'RSI', 'MACD')
            indicator_params (Dict[str, Any]): Parameters for the indicator calculation

        Raises:
            ValueError: If the indicator calculation fails

        Returns:
            pd.DataFrame: DataFrame with calculated indicator values
        """
        data_copy = data.copy()

        # Use the IndicatorRegistry to apply the indicator
        try:
            data_with_indicators = IndicatorRegistry.apply(name=indicator_name, df=data_copy, **indicator_params)
            return data_with_indicators
        except Exception as e:
            raise ValueError(f"Failed to calculate {indicator_name}: {str(e)}")

    def _simulate_portfolio(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        """
        Private method to simulate portfolio performance based on
        signals.

        Args:
            data (pd.DataFrame): Input OHLCV data.
            signals (pd.Series): Generated trading signals.

        Returns:
            Dict[str, Any]: Simulated portfolio performance metrics.
        """
        portfolio_value = self.config.initial_capital
        position = 0  # 0 = no position, 1 = long, -1 = short
        shares = 0
        cash = self.config.initial_capital

        portfolio_values = []
        strategy_returns = []
        positions = []
        trades_executed = []
        total_trades = 0
        transaction_costs = 0

        previous_signal = SignalType.HOLD.value

        for i, (idx, row) in enumerate(data.iterrows()):
            current_price = row['Close']
            current_signal = signals.iloc[i] if i < len(signals) else SignalType.HOLD.value

            # Position sizing
            position_size = self._calculate_position_size(cash, current_price, trades_executed)

            # Execute trades based on signal changes
            if current_signal != previous_signal:

                # Close existing position
                if position != 0:
                    trade_value = shares * current_price
                    cost = trade_value * self.config.transaction_cost
                    transaction_costs += cost

                    if position == 1:  # Close long position
                        cash = trade_value - cost
                    else:  # Close short position (position == -1)
                        cash = cash - (trade_value + cost)

                    trades_executed.append(
                        {
                            'date': idx,
                            'action': 'close',
                            'position_type': 'long' if position == 1 else 'short',
                            'shares': shares,
                            'price': current_price,
                            'value': trade_value,
                            'cost': cost,
                        }
                    )

                    shares = 0
                    position = 0
                    total_trades += 1

                # Open new position
                if current_signal == SignalType.BUY.value:
                    shares = position_size // current_price
                    if shares > 0:
                        trade_value = shares * current_price
                        cost = trade_value * self.config.transaction_cost
                        transaction_costs += cost
                        cash -= trade_value + cost
                        position = 1

                        trades_executed.append(
                            {
                                'date': idx,
                                'action': 'buy',
                                'position_type': 'long',
                                'shares': shares,
                                'price': current_price,
                                'value': trade_value,
                                'cost': cost,
                            }
                        )
                        total_trades += 1

                elif current_signal == SignalType.SELL.value and self.config.position_sizing in [
                    'full',
                    'half',
                    'third',
                ]:
                    # Short selling
                    shares = position_size // current_price
                    if shares > 0:
                        trade_value = shares * current_price
                        cost = trade_value * self.config.transaction_cost
                        transaction_costs += cost
                        cash += trade_value - cost
                        position = -1

                        trades_executed.append(
                            {
                                'date': idx,
                                'action': 'sell_short',
                                'position_type': 'short',
                                'shares': shares,
                                'price': current_price,
                                'value': trade_value,
                                'cost': cost,
                            }
                        )
                        total_trades += 1

            # Calculate current portfolio value
            if position == 1:  # Long position
                portfolio_value = cash + (shares * current_price)
            elif position == -1:  # Short position
                portfolio_value = cash - (shares * current_price)
            else:  # No position
                portfolio_value = cash

            # Calculate strategy returns
            if i > 0:
                strategy_return = (portfolio_value - portfolio_values[-1]) / portfolio_values[-1]
                strategy_returns.append(strategy_return)

            portfolio_values.append(portfolio_value)
            positions.append(position)
            previous_signal = current_signal

        return {
            'portfolio_values': pd.Series(portfolio_values, index=data.index),
            'strategy_returns': pd.Series(strategy_returns, index=data.index[1:]),
            'positions': pd.Series(positions, index=data.index),
            'trades_executed': trades_executed,
            'total_trades': total_trades,
            'transaction_costs': transaction_costs,
        }

    def _calculate_position_size(self, available_cash: float, price: float, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate position size based on configuration.

        Args:
            available_cash (float): Amount of cash available for trading.
            price (float): Current price of the asset.
            trades (List[Dict[str, Any]]): List of executed trades.

        Returns:
            float: Calculated position size.
        """
        if self.config.position_sizing == 'full':
            return available_cash
        elif self.config.position_sizing == 'half':
            return available_cash * 0.5
        elif self.config.position_sizing == 'third':
            return available_cash * 0.33
        elif self.config.position_sizing == 'kelly':
            kelly_fraction = self._calculate_kelly_fraction(trades)
            return available_cash * kelly_fraction
        else:
            return available_cash  # Default to full investment

    def _get_trade_pnl(self, trades: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate profit and loss for a list of trades.

        Args:
            trades (List[Dict[str, Any]]): List of executed trades.

        Returns:
            List[float]: List of profit and loss values for each trade.
        """
        pnl = []
        for i in range(0, len(trades) - 1, 2):
            trade_open = trades[i]
            trade_close = trades[i + 1]

            if trade_open['position_type'] == 'long':
                pnl.append(trade_close['value'] - trade_open['value'])
            else:  # short
                pnl.append(trade_open['value'] - trade_close['value'])
        return pnl

    def _calculate_kelly_fraction(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculates the Kelly Criterion fraction based on historical
        trades.

        Args:
            trades (List[Dict[str, Any]]): List of executed trades.

        Returns:
            float: The Kelly fraction (0 to 1).
        """
        # Need at least a few trades to get a meaningful statistic
        if len(trades) < 10:
            return 0.25  # Default to a conservative fraction

        pnl = self._get_trade_pnl(trades)
        if not pnl:
            return 0.25

        wins = [p for p in pnl if p > 0]
        losses = [p for p in pnl if p < 0]

        if not wins or not losses:
            return 0.25  # Cannot calculate with no wins or no losses

        win_probability = len(wins) / len(pnl)
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

        if win_loss_ratio == 0:
            return 0.0

        # Kelly Criterion formula
        kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)

        # Use a fractional Kelly (e.g., half) to be more conservative
        # and ensure the fraction is between 0 and 1.
        return max(0, min(1, kelly_fraction * 0.5))

    def _calculate_metrics(self, portfolio_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Private method to calculate comprehensive performance metrics.

        Args:
            portfolio_results (Dict[str, Any]): Portfolio simulation results.

        Returns:
            Dict[str, float]: Calculated performance metrics.
        """
        portfolio_values = portfolio_results['portfolio_values']
        strategy_returns = portfolio_results['strategy_returns']

        if len(strategy_returns) == 0:
            return self._empty_metrics()

        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - self.config.initial_capital) / self.config.initial_capital

        # Annualized return
        trading_days = len(portfolio_values)
        years = trading_days / 252
        if years > 0 and (1 + total_return) >= 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
        else:
            annual_return = 0

        # Volatility (annualized)
        volatility = strategy_returns.std() * np.sqrt(252)

        # Sharpe ratio
        excess_return = annual_return - self.config.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        positive_returns = strategy_returns[strategy_returns > 0]
        win_rate = len(positive_returns) / len(strategy_returns)

        # Profit factor
        positive_sum = positive_returns.sum()
        negative_sum = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = positive_sum / negative_sum if negative_sum > 0 else float('inf')

        # Sortino ratio
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
        }

    def _empty_metrics(self) -> Dict[str, float]:
        """
        Return empty metrics for failed calculations.

        Returns:
            Dict[str, float]: Dictionary of empty metrics.
        """
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
        }


# Convenience functions for quick analysis
def analyze_single_indicator(
    data: pd.DataFrame,
    indicator_name: str,
    indicator_params: Dict[str, Any],
    config: Optional[IndicatorAnalysisConfig] = None,
    allow_short: bool = True,
    **strategy_params,
) -> Dict[str, Any]:
    """
    Analyze a single indicator.

    Args:
        data (pd.DataFrame): OHLCV DataFrame
        indicator_name (str): Name of the indicator
        indicator_params (Dict[str, Any]): Parameters for indicator calculation
        config (Optional[IndicatorAnalysisConfig], optional): Analysis configuration. Defaults to None.
        allow_short (bool, optional): Whether to allow short positions. Defaults to True.

    Returns:
        Dict[str, Any]: Analysis results dictionary
    """
    analyzer = IndicatorAnalysis(config)
    return analyzer.analyze_indicator(
        data=data,
        indicator_name=indicator_name,
        indicator_params=indicator_params,
        allow_short=allow_short,
        **strategy_params,
    )


def rank_indicator_performance(
    data: pd.DataFrame, indicators: Dict[str, Dict[str, Any]], config: Optional[IndicatorAnalysisConfig] = None
) -> pd.DataFrame:
    """
    Compare and rank the performance of multiple indicator strategies.

    Args:
        data: OHLCV DataFrame
        indicators: Dictionary mapping indicator names to their configs
        config: Analysis configuration

    Returns:
        Comparison DataFrame sorted by Sharpe ratio
    """
    analyzer = IndicatorAnalysis(config)
    return analyzer.compare_indicators(data, indicators, sort_by='sharpe_ratio')
