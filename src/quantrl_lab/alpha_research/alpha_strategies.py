import numpy as np
import pandas as pd

from .base import SignalType, VectorizedTradingStrategy
from .registry import VectorizedStrategyRegistry


@VectorizedStrategyRegistry.register("trend_following")
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
        signals[data["Close"] > data[self.indicator_col]] = SignalType.BUY.value

        # Sell when price < indicator (if shorting allowed)
        if self.allow_short:
            signals[data["Close"] < data[self.indicator_col]] = SignalType.SELL.value

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores.

        Score = (Price - Indicator) / Price
        Measures the percentage distance from the trend line.

        Positive score = Price above trend (Bullish)
        Negative score = Price below trend (Bearish)
        """
        if self.indicator_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        scores = (data["Close"] - data[self.indicator_col]) / data["Close"]

        # Normalize roughly to [-1, 1]. A 5% deviation is huge for daily data.
        # Let's scale by a factor of 20 (so 5% becomes 1.0)
        scores = scores * 20.0

        return scores.clip(-1.0, 1.0)

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.indicator_col, "Close"]


@VectorizedStrategyRegistry.register("mean_reversion")
class MeanReversionStrategy(VectorizedTradingStrategy):
    """Strategy for mean-reversion indicators like RSI, MFI, Williams
    %R."""

    def __init__(
        self,
        indicator_col: str,
        oversold: float = 30,
        overbought: float = 70,
        allow_short: bool = True,
        indicator_scale: str = "0_100",
    ):
        self.indicator_col = indicator_col
        self.oversold = oversold
        self.overbought = overbought
        self.allow_short = allow_short
        # Explicit scale of the underlying indicator — avoids runtime heuristics.
        # "0_100"     : RSI / MFI style  (range 0-100,   center=50)
        # "williams_r": Williams %R style (range -100..0, center=-50)
        self.indicator_scale = indicator_scale

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
        signals = (
            signals.astype(float)
            .replace(SignalType.HOLD.value, float("nan"))
            .ffill()
            .fillna(SignalType.HOLD.value)
            .astype(int)
        )

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores.

        Uses the explicit ``indicator_scale`` constructor parameter instead of
        guessing from runtime data values, which was unreliable for indicators
        like CCI that can go negative but follow a different formula.

        Scales:
            ``"0_100"``      – RSI / MFI style (range 0-100, center=50).
                               Score = (50 - value) / 50 → +1 at 0, -1 at 100.
            ``"williams_r"`` – Williams %R style (range -100..0, center=-50).
                               Score = (-50 - value) / 50 → +1 at -100, -1 at 0.
        """
        if self.indicator_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        val = data[self.indicator_col]

        if self.indicator_scale == "williams_r":
            # -100 is oversold (Buy → +1), 0 is overbought (Sell → -1)
            scores = (-50.0 - val) / 50.0
        else:
            # Default: RSI / MFI (0-100) — center=50
            # 30 → (50-30)/50 = +0.4 (Buy), 70 → (50-70)/50 = -0.4 (Sell)
            scores = (50.0 - val) / 50.0

        return scores.clip(-1.0, 1.0)

    def get_required_columns(self) -> list:
        return [self.indicator_col]


@VectorizedStrategyRegistry.register("macd_crossover")
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

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores.

        Score = MACD Histogram (Fast - Slow).
        """
        if self.fast_col not in data.columns or self.slow_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        hist = data[self.fast_col] - data[self.slow_col]

        # Normalize histogram roughly. It's absolute price difference.
        # Divide by Close price to make it percentage-based
        scores = hist / data["Close"]

        # Scale: 1% divergence is strong.
        scores = scores * 100.0

        return scores.clip(-1.0, 1.0)

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.fast_col, self.slow_col, "Close"]


@VectorizedStrategyRegistry.register("volatility_breakout")
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

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores.

        Score = Z-Score of the volatility indicator.
        (Value - Mean) / StdDev

        High volatility = High Score (Breakout potential)
        """
        if self.indicator_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        return self._rolling_zscore(data[self.indicator_col], window=60)

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.indicator_col]


@VectorizedStrategyRegistry.register("bollinger_bands")
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

        Uses an explicit O(n) state machine so that exit conditions
        (price crossing the middle band) correctly override the current
        position on the same bar and persist until the next entry
        signal. The previous vectorised+ffill approach had a bug where
        exits would not "stick" — the subsequent ffill would re-fill
        those bars with the old position.
        """
        missing_cols = [col for col in self.get_required_columns() if col not in data.columns]
        if missing_cols:
            return pd.Series(SignalType.HOLD.value, index=data.index)

        close = data["Close"].values
        lower = data[self.lower_col].values
        middle = data[self.middle_col].values
        upper = data[self.upper_col].values

        n = len(close)
        result = np.full(n, SignalType.HOLD.value, dtype=np.int8)

        pos = SignalType.HOLD.value  # current held position

        for i in range(n):
            c, lo, mid, hi = close[i], lower[i], middle[i], upper[i]

            # Apply exits first so a simultaneous entry+exit resolves cleanly
            if pos == SignalType.BUY.value and c >= mid:
                pos = SignalType.HOLD.value
            elif pos == SignalType.SELL.value and c <= mid:
                pos = SignalType.HOLD.value

            # Entry conditions (only enter when flat)
            if pos == SignalType.HOLD.value:
                if c <= lo:
                    pos = SignalType.BUY.value
                elif self.allow_short and c >= hi:
                    pos = SignalType.SELL.value

            result[i] = pos

        return pd.Series(result, index=data.index)

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores using %B (Percent B).

        %B = (Price - Lower) / (Upper - Lower)

        - %B < 0: Price below lower band (Oversold -> Buy)
        - %B > 1: Price above upper band (Overbought -> Sell)
        - %B = 0.5: Price at mean

        Score = (0.5 - %B) * 2  => +1 at 0 (Buy), -1 at 1 (Sell)
        """
        if self.lower_col not in data.columns or self.upper_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        lower = data[self.lower_col]
        upper = data[self.upper_col]
        close = data["Close"]

        bandwidth = upper - lower
        # Avoid division by zero
        percent_b = (close - lower) / bandwidth.replace(0, 1e-9)

        # Invert so Low %B is Positive Score (Buy)
        scores = (0.5 - percent_b) * 2.0

        return scores.clip(-1.0, 1.0)

    def get_required_columns(self) -> list:
        return [self.lower_col, self.middle_col, self.upper_col, "Close"]


@VectorizedStrategyRegistry.register("stochastic")
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
        signals = (
            signals.astype(float)
            .replace(SignalType.HOLD.value, float("nan"))
            .ffill()
            .fillna(SignalType.HOLD.value)
            .astype(int)
        )

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores.

        Same as RSI: (50 - K) / 50.
        """
        if self.k_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        scores = (50.0 - data[self.k_col]) / 50.0
        return scores.clip(-1.0, 1.0)

    def get_required_columns(self) -> list:
        required = [self.k_col]
        if self.d_col:
            required.append(self.d_col)
        return required


@VectorizedStrategyRegistry.register("obv_trend")
class OnBalanceVolumeStrategy(VectorizedTradingStrategy):
    """Strategy for On-Balance Volume - Trend following based on volume"""

    def __init__(self, obv_col: str, allow_short: bool = True):
        self.obv_col = obv_col
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on On-Balance Volume strategy.

        A simple strategy is to buy when the OBV is rising and sell when
        it's falling. We can use a moving average of OBV to determine
        the trend.

        Args:
            data (pd.DataFrame): Input OHLCV data.

        Returns:
            pd.Series: Generated trading signals.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.obv_col not in data.columns:
            return signals

        # Use a short-term moving average to identify the trend of OBV
        obv_sma = data[self.obv_col].rolling(window=20).mean()

        # Buy when OBV is above its moving average (upward trend)
        buy_condition = data[self.obv_col] > obv_sma
        signals[buy_condition] = SignalType.BUY.value

        # Sell when OBV is below its moving average (downward trend)
        if self.allow_short:
            sell_condition = data[self.obv_col] < obv_sma
            signals[sell_condition] = SignalType.SELL.value

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate continuous alpha scores.

        Score = (OBV - OBV_SMA) / OBV_STD
        Z-Score of OBV Trend.
        """
        if self.obv_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        return self._rolling_zscore(data[self.obv_col], window=20)

    def get_required_columns(self) -> list:
        return [self.obv_col]


@VectorizedStrategyRegistry.register("adx_trend")
class ADXTrendStrategy(VectorizedTradingStrategy):
    """Strategy for ADX Trend Strength."""

    def __init__(self, adx_col: str, pdi_col: str, mdi_col: str, threshold: float = 25.0, allow_short: bool = True):
        self.adx_col = adx_col
        self.pdi_col = pdi_col
        self.mdi_col = mdi_col
        self.threshold = threshold
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Buy if ADX > Threshold AND +DI > -DI.

        Sell if ADX > Threshold AND -DI > +DI.
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        cols = [self.adx_col, self.pdi_col, self.mdi_col]
        if not all(c in data.columns for c in cols):
            return signals

        strong_trend = data[self.adx_col] > self.threshold
        bullish = data[self.pdi_col] > data[self.mdi_col]

        signals[strong_trend & bullish] = SignalType.BUY.value

        if self.allow_short:
            signals[strong_trend & ~bullish] = SignalType.SELL.value

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Score = Trend Strength * Direction.

        Strength = ADX / 100
        Direction = (+DI - -DI) / (+DI + -DI)  [Bounded -1 to 1]

        Score = Strength * Direction
        """
        cols = [self.adx_col, self.pdi_col, self.mdi_col]
        if not all(c in data.columns for c in cols):
            return pd.Series(0.0, index=data.index)

        strength = data[self.adx_col] / 100.0

        pdi = data[self.pdi_col]
        mdi = data[self.mdi_col]

        direction = (pdi - mdi) / (pdi + mdi + 1e-9)

        scores = strength * direction
        return scores.clip(-1.0, 1.0)

    def get_required_columns(self) -> list:
        return [self.adx_col, self.pdi_col, self.mdi_col]


@VectorizedStrategyRegistry.register("cci_reversal")
class CCIStrategy(VectorizedTradingStrategy):
    """Strategy for Commodity Channel Index."""

    def __init__(self, indicator_col: str, threshold: float = 100.0, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.threshold = threshold
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        CCI Mean Reversion:
        Buy if CCI < -100 (Oversold).
        Sell if CCI > 100 (Overbought).
        """
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        if self.indicator_col not in data.columns:
            return signals

        signals[data[self.indicator_col] < -self.threshold] = SignalType.BUY.value
        if self.allow_short:
            signals[data[self.indicator_col] > self.threshold] = SignalType.SELL.value

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """
        Score based on CCI value.

        CCI is theoretically unbounded but usually +/- 200.

        Mean Reversion Logic:
        High CCI -> Sell (-Score)
        Low CCI -> Buy (+Score)
        """
        if self.indicator_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        # Normalize assuming range +/- 200 covers most events
        scores = -data[self.indicator_col] / 200.0
        return scores.clip(-1.0, 1.0)

    def get_required_columns(self) -> list:
        return [self.indicator_col]
