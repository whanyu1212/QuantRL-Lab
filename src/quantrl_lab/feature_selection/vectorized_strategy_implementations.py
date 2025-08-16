import pandas as pd

from .base_vectorized_strategy import SignalType, VectorizedTradingStrategy


class TrendFollowingStrategy(VectorizedTradingStrategy):
    """Strategy for trend-following indicators like SMA, EMA."""

    def __init__(self, indicator_col: str, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on trend-following strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.indicator_col not in data.columns:
            return signals

        # Buy when price > indicator
        signals[data['Close'] > data[self.indicator_col]] = SignalType.BUY.value

        # Sell when price < indicator (if shorting allowed)
        if self.allow_short:
            signals[data['Close'] < data[self.indicator_col]] = SignalType.SELL.value

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.indicator_col, 'Close']


class MeanReversionStrategy(VectorizedTradingStrategy):
    """Strategy for mean-reversion indicators like RSI."""

    def __init__(self, indicator_col: str, oversold: float = 30, overbought: float = 70, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on mean-reversion strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.indicator_col not in data.columns:
            return signals

        # Buy when oversold
        buy_condition = data[self.indicator_col] < self.oversold
        signals[buy_condition] = SignalType.BUY.value

        # Sell when overbought (if shorting allowed)
        if self.allow_short:
            sell_condition = data[self.indicator_col] > self.overbought
            signals[sell_condition] = SignalType.SELL.value

        # Forward fill to maintain positions
        signals = signals.replace(SignalType.HOLD.value, pd.NA).fillna(method='ffill').fillna(SignalType.HOLD.value)

        return signals

    def get_required_columns(self) -> list:
        return [self.indicator_col]


class MACDCrossoverStrategy(VectorizedTradingStrategy):
    """Strategy for crossover indicators from MACD line."""

    def __init__(self, fast_col: str, slow_col: str, allow_short: bool = True):
        self.fast_col = fast_col
        self.slow_col = slow_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD crossover strategy.

        Args:
            data (pd.DataFrame): input OHLCV df

        Returns:
            pd.Series: generated trading signals
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.fast_col not in data.columns or self.slow_col not in data.columns:
            return signals

        # Buy when fast > slow
        signals[data[self.fast_col] > data[self.slow_col]] = SignalType.BUY.value

        # Sell when fast < slow (if shorting allowed)
        if self.allow_short:
            signals[data[self.fast_col] < data[self.slow_col]] = SignalType.SELL.value

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.fast_col, self.slow_col]


class VolatilityBreakoutStrategy(VectorizedTradingStrategy):
    """Strategy for volatility indicators like ATR."""

    def __init__(self, indicator_col: str, threshold_percentile: float = 0.7, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.threshold_percentile = threshold_percentile
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on volatility breakout strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.indicator_col not in data.columns:
            return signals

        # Buy when volatility is high (breakout)
        high_threshold = data[self.indicator_col].quantile(self.threshold_percentile)
        signals[data[self.indicator_col] > high_threshold] = SignalType.BUY.value

        # Sell when volatility is low (if shorting allowed)
        if self.allow_short:
            low_threshold = data[self.indicator_col].quantile(1 - self.threshold_percentile)
            signals[data[self.indicator_col] < low_threshold] = SignalType.SELL.value
        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.indicator_col]


class BollingerBandsStrategy(VectorizedTradingStrategy):
    """Strategy for Bollinger Bands - Mean reversion at bands"""

    def __init__(self, lower_col: str, middle_col: str, upper_col: str, allow_short: bool = True):
        self.lower_col = lower_col
        self.middle_col = middle_col
        self.upper_col = upper_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        missing_cols = [col for col in self.get_required_columns() if col not in data.columns]
        if missing_cols:
            return signals

        # Mean reversion strategy with state management
        # Buy when price touches or goes below lower band
        # Exit long when price reaches middle band
        # Short when price touches or goes above upper band (if allowed)
        # Exit short when price reaches middle band

        position = SignalType.HOLD.value
        for i in range(len(data)):
            current_price = data['Close'].iloc[i]
            lower_band = data[self.lower_col].iloc[i]
            middle_band = data[self.middle_col].iloc[i]
            upper_band = data[self.upper_col].iloc[i]

            if position == SignalType.HOLD.value:
                if current_price <= lower_band:
                    position = SignalType.BUY.value
                elif self.allow_short and current_price >= upper_band:
                    position = SignalType.SELL.value
            elif position == SignalType.BUY.value:
                if current_price >= middle_band:
                    position = SignalType.HOLD.value
            elif position == SignalType.SELL.value and self.allow_short:
                if current_price <= middle_band:
                    position = SignalType.HOLD.value

            signals.iloc[i] = position

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.lower_col, self.middle_col, self.upper_col, 'Close']


class StochasticStrategy(VectorizedTradingStrategy):
    """Strategy for Stochastic Oscillator - Mean reversion"""

    def __init__(
        self, k_col: str, d_col: str = None, oversold: float = 20, overbought: float = 80, allow_short: bool = True
    ):
        self.k_col = k_col
        self.d_col = d_col  # Optional %D line
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Stochastic Oscillator
        strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.k_col not in data.columns:
            return signals

        if self.d_col and self.d_col in data.columns:
            # Use both %K and %D for more robust signals
            buy_condition = (data[self.k_col] < self.oversold) & (data[self.d_col] < self.oversold)

            if self.allow_short:
                sell_condition = (data[self.k_col] > self.overbought) & (data[self.d_col] > self.overbought)
        else:
            # Use only %K
            buy_condition = data[self.k_col] < self.oversold

            if self.allow_short:
                sell_condition = data[self.k_col] > self.overbought

        # Apply signals
        signals[buy_condition] = SignalType.BUY.value

        if self.allow_short:
            signals[sell_condition] = SignalType.SELL.value

        # Forward fill to maintain positions
        signals = signals.replace(SignalType.HOLD.value, pd.NA).fillna(method='ffill').fillna(SignalType.HOLD.value)

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        required = [self.k_col]
        if self.d_col:
            required.append(self.d_col)
        return required


class OnBalanceVolumeStrategy(VectorizedTradingStrategy):
    """Strategy for On-Balance Volume - Trend following based on volume"""

    def __init__(self, obv_col: str, lookback_period: int = 10, allow_short: bool = True):
        self.obv_col = obv_col
        self.lookback_period = lookback_period
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on On-Balance Volume strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.obv_col not in data.columns:
            return signals

        # Calculate OBV momentum (rate of change over lookback period)
        obv_momentum = data[self.obv_col].pct_change(periods=self.lookback_period)

        # Alternative: Use OBV trend (SMA of OBV)
        obv_sma = data[self.obv_col].rolling(window=self.lookback_period).mean()
        obv_trend = data[self.obv_col] > obv_sma

        # Buy when OBV is increasing (positive momentum) and trending up
        buy_condition = (obv_momentum > 0) & obv_trend
        signals[buy_condition] = SignalType.BUY.value

        # Sell when OBV is decreasing (negative momentum) and trending down
        if self.allow_short:
            sell_condition = (obv_momentum < 0) & (~obv_trend)
            signals[sell_condition] = SignalType.SELL.value

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.obv_col]


class MACDHistogramStrategy(VectorizedTradingStrategy):
    """Strategy specifically for MACD Histogram - Zero line crossovers and momentum"""

    def __init__(self, histogram_col: str, allow_short: bool = True, momentum_threshold: float = 0.0):
        self.histogram_col = histogram_col
        self.allow_short = allow_short
        self.momentum_threshold = momentum_threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD Histogram strategy.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.histogram_col not in data.columns:
            return signals

        # Strategy 1: Zero Line Crossover
        # Buy when histogram > 0 (MACD line above signal line)
        signals[data[self.histogram_col] > self.momentum_threshold] = SignalType.BUY.value

        # Sell when histogram < 0 (MACD line below signal line)
        if self.allow_short:
            signals[data[self.histogram_col] < -self.momentum_threshold] = SignalType.SELL.value

        return signals

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.histogram_col]
