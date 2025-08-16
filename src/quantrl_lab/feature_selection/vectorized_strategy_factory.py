from .base_vectorized_strategy import VectorizedTradingStrategy
from .vectorized_strategy_implementations import (
    BollingerBandsStrategy,
    MACDCrossoverStrategy,
    MACDHistogramStrategy,
    MeanReversionStrategy,
    OnBalanceVolumeStrategy,
    StochasticStrategy,
    TrendFollowingStrategy,
    VolatilityBreakoutStrategy,
)


class VectorizedStrategyFactory:
    """Factory to create appropriate strategy for each indicator
    type."""

    @staticmethod
    def create_strategy(indicator_name: str, allow_short: bool = True, **params) -> VectorizedTradingStrategy:
        """
        Create appropriate strategy based on indicator type.

        Args:
            indicator_name: Name of the indicator ('SMA', 'EMA', 'RSI', etc.)
            allow_short: Whether to allow short positions (default: True)
            **params: Indicator-specific parameters

        Returns:
            VectorizedTradingStrategy: The appropriate strategy instance

        Raises:
            ValueError: If indicator_name is not supported
        """

        if indicator_name == 'SMA':
            window = params['window']
            col_name = f'SMA_{window}'
            return TrendFollowingStrategy(col_name, allow_short)

        elif indicator_name == 'EMA':
            window = params['window']
            col_name = f'EMA_{window}'
            return TrendFollowingStrategy(col_name, allow_short)

        elif indicator_name == 'RSI':
            window = params['window']
            col_name = f'RSI_{window}'
            oversold = params.get('oversold', 30)
            overbought = params.get('overbought', 70)
            return MeanReversionStrategy(col_name, oversold, overbought, allow_short)

        elif indicator_name == 'MACD':
            fast, slow, signal = params['fast'], params['slow'], params['signal']
            line_col = f'MACD_line_{fast}_{slow}'
            signal_col = f'MACD_signal_{signal}'
            return MACDCrossoverStrategy(line_col, signal_col, allow_short)

        elif indicator_name == 'MACD_HISTOGRAM':
            # Note: MACD histogram column name doesn't include parameters in the actual implementation
            histogram_col = 'MACD_histogram'
            momentum_threshold = params.get('momentum_threshold', 0.0)
            return MACDHistogramStrategy(histogram_col, allow_short, momentum_threshold)

        elif indicator_name == 'ATR':
            window = params.get('window', 14)
            col_name = f'ATR_{window}'
            threshold_percentile = params.get('threshold_percentile', 0.7)
            return VolatilityBreakoutStrategy(col_name, threshold_percentile, allow_short)

        elif indicator_name == 'BB':
            window = params['window']
            lower_col = f'BB_lower_{window}_2.0'
            middle_col = f'BB_middle_{window}'
            upper_col = f'BB_upper_{window}_2.0'
            return BollingerBandsStrategy(lower_col, middle_col, upper_col, allow_short)

        elif indicator_name == 'STOCH':
            k_window = params.get('k_window', 14)
            d_window = params.get('d_window', 3)
            k_col = f'STOCH_%K_{k_window}'
            d_col = f'STOCH_%D_{d_window}'
            oversold = params.get('oversold', 20)
            overbought = params.get('overbought', 80)
            return StochasticStrategy(k_col, d_col, oversold, overbought, allow_short)

        elif indicator_name == 'OBV':
            col_name = 'OBV'
            lookback_period = params.get('lookback_period', 10)
            return OnBalanceVolumeStrategy(col_name, lookback_period, allow_short)

        else:
            raise ValueError(
                f"Unknown indicator: {indicator_name}. Supported indicators: "
                f"['SMA', 'EMA', 'RSI', 'MACD', 'MACD_HISTOGRAM', 'ATR', 'BB', 'STOCH', 'OBV']"
            )

    @staticmethod
    def get_supported_indicators() -> list:
        """
        Get list of supported indicator names.

        Returns:
            list: List of supported indicator names
        """
        return ['SMA', 'EMA', 'RSI', 'MACD', 'MACD_HISTOGRAM', 'ATR', 'BB', 'STOCH', 'OBV']

    @staticmethod
    def get_indicator_requirements(indicator_name: str) -> dict:
        """
        Get required parameters for a specific indicator.

        Args:
            indicator_name: Name of the indicator

        Returns:
            dict: Required and optional parameters for the indicator
        """
        requirements = {
            'SMA': {'required': ['window'], 'optional': []},
            'EMA': {'required': ['window'], 'optional': []},
            'RSI': {'required': ['window'], 'optional': ['oversold', 'overbought']},
            'MACD': {'required': ['fast', 'slow', 'signal'], 'optional': []},
            'MACD_HISTOGRAM': {'required': ['fast', 'slow', 'signal'], 'optional': ['momentum_threshold']},
            'ATR': {'required': [], 'optional': ['window', 'threshold_percentile']},
            'BB': {'required': ['window'], 'optional': []},
            'STOCH': {'required': [], 'optional': ['k_window', 'd_window', 'oversold', 'overbought']},
            'OBV': {'required': [], 'optional': ['lookback_period']},
        }

        if indicator_name not in requirements:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        return requirements[indicator_name]
