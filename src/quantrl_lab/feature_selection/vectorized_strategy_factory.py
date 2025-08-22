from .base_vectorized_strategy import VectorizedTradingStrategy
from .vectorized_strategy_implementations import (
    BollingerBandsStrategy,
    MACDCrossoverStrategy,
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
        """Create appropriate strategy based on indicator type."""

        if indicator_name == "SMA":
            window = params["window"]
            col_name = f"SMA_{window}"
            return TrendFollowingStrategy(col_name, allow_short)

        elif indicator_name == "EMA":
            window = params["window"]
            col_name = f"EMA_{window}"
            return TrendFollowingStrategy(col_name, allow_short)

        elif indicator_name == "RSI":
            window = params["window"]
            col_name = f"RSI_{window}"
            oversold = params.get("oversold", 30)
            overbought = params.get("overbought", 70)
            return MeanReversionStrategy(col_name, oversold, overbought, allow_short)

        elif indicator_name == "MACD":
            fast, slow, signal = params["fast"], params["slow"], params["signal"]
            line_col = f"MACD_line_{fast}_{slow}"
            signal_col = f"MACD_signal_{signal}"
            return MACDCrossoverStrategy(line_col, signal_col, allow_short)

        elif indicator_name == "ATR":
            window = params.get("window", 14)
            col_name = f"ATR_{window}"
            threshold_percentile = params.get("threshold_percentile", 0.7)
            return VolatilityBreakoutStrategy(col_name, threshold_percentile, allow_short)

        elif indicator_name == "BB":
            window = params["window"]
            # Handle both 'std_dev' and 'num_std' parameter names
            num_std = params.get("num_std", params.get("std_dev", 2.0))
            lower_col = f"BB_lower_{window}_{num_std}"
            middle_col = f"BB_middle_{window}"
            upper_col = f"BB_upper_{window}_{num_std}"
            return BollingerBandsStrategy(lower_col, middle_col, upper_col, allow_short)

        elif indicator_name == "STOCH":
            k_window = params["k_window"]
            d_window = params["d_window"]
            smooth_k = params["smooth_k"]
            k_col = f"STOCH_%K_{k_window}_{smooth_k}"
            d_col = f"STOCH_%D_{d_window}"
            oversold = params.get("oversold", 20)
            overbought = params.get("overbought", 80)
            return StochasticStrategy(k_col, d_col, oversold, overbought, allow_short)

        elif indicator_name == "OBV":
            col_name = "OBV"
            return OnBalanceVolumeStrategy(col_name, allow_short)

        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")

    @staticmethod
    def get_supported_indicators() -> list:
        """
        Get list of supported indicator names.

        Returns:
            list: List of supported indicator names
        """
        return ["SMA", "EMA", "RSI", "MACD", "ATR", "BB", "STOCH", "OBV"]

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
            "SMA": {"required": ["window"], "optional": []},
            "EMA": {"required": ["window"], "optional": []},
            "RSI": {"required": ["window"], "optional": ["oversold", "overbought"]},
            "MACD": {"required": ["fast", "slow", "signal"], "optional": []},
            "MACD_HISTOGRAM": {"required": ["fast", "slow", "signal"], "optional": ["momentum_threshold"]},
            "ATR": {"required": [], "optional": ["window", "threshold_percentile"]},
            "BB": {"required": ["window"], "optional": []},
            "STOCH": {"required": ["k_window", "d_window", "smooth_k"], "optional": ["oversold", "overbought"]},
            "OBV": {"required": [], "optional": []},
        }

        if indicator_name not in requirements:
            raise ValueError(f"Unknown indicator: {indicator_name}")

        return requirements[indicator_name]
