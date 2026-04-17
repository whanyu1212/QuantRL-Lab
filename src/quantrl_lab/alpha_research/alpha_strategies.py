from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import SignalType, VectorizedTradingStrategy
from .indicator_research import INDICATOR_STRATEGY_MAP  # noqa: F401
from .registry import VectorizedStrategyRegistry


@VectorizedStrategyRegistry.register("trend_following", description="Buy when price > indicator, sell when below")
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

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Prefer line-like outputs over direction/helper columns."""
        if "indicator_col" in current_params or not new_cols:
            return {}

        preferred = [
            c
            for c in new_cols
            if all(token not in c.lower() for token in ["dir", "signal", "hist", "upper", "lower", "middle"])
        ]
        chosen = preferred[0] if preferred else new_cols[0]
        return {"indicator_col": chosen}


@VectorizedStrategyRegistry.register(
    "mean_reversion", description="Buy when oversold, sell when overbought (RSI, MFI, Williams %R)"
)
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


@VectorizedStrategyRegistry.register("macd_crossover", description="Buy when MACD line crosses above signal line")
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

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Wire MACD line → fast_col, signal line → slow_col."""

        def _find(substring: str) -> Any:
            matches = [c for c in new_cols if substring in c]
            return matches[0] if matches else None

        resolved: Dict[str, Any] = {}
        if "fast_col" not in current_params:
            resolved["fast_col"] = _find("MACD_line")
        if "slow_col" not in current_params:
            resolved["slow_col"] = _find("MACD_signal")
        return resolved

    def get_required_columns(self) -> list:
        """
        Get the list of required columns for the strategy.

        Returns:
            list: List of required column names.
        """
        return [self.fast_col, self.slow_col, "Close"]


@VectorizedStrategyRegistry.register("crossover", description="Buy when fast line crosses above slow line")
class CrossoverStrategy(VectorizedTradingStrategy):
    """Generic crossover strategy for paired indicator lines."""

    def __init__(
        self,
        fast_col: str,
        slow_col: str,
        allow_short: bool = True,
        score_normalization: str = "zscore",
    ):
        self.fast_col = fast_col
        self.slow_col = slow_col
        self.allow_short = allow_short
        self.score_normalization = score_normalization

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate crossover signals from fast and slow indicator
        lines."""
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        if self.fast_col not in data.columns or self.slow_col not in data.columns:
            return signals

        signals[data[self.fast_col] > data[self.slow_col]] = SignalType.BUY.value
        if self.allow_short:
            signals[data[self.fast_col] < data[self.slow_col]] = SignalType.SELL.value

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """Generate continuous crossover strength scores."""
        if self.fast_col not in data.columns or self.slow_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        spread = data[self.fast_col] - data[self.slow_col]

        if self.score_normalization == "close_pct":
            close = data["Close"].replace(0, np.nan)
            scores = (spread / close) * 100.0
            return scores.clip(-1.0, 1.0).fillna(0.0)

        return self._rolling_zscore(spread, window=60).fillna(0.0)

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve fast and slow columns from generic crossover
        outputs."""
        resolved: Dict[str, Any] = {}
        signal_cols = [c for c in new_cols if "signal" in c.lower()]
        line_cols = [
            c for c in new_cols if "signal" not in c.lower() and "hist" not in c.lower() and "dir" not in c.lower()
        ]
        positive_cols = [c for c in new_cols if any(token in c.lower() for token in ["_pos", "_up"])]
        negative_cols = [c for c in new_cols if any(token in c.lower() for token in ["_neg", "_down"])]

        if "fast_col" not in current_params:
            if line_cols:
                resolved["fast_col"] = line_cols[0]
            elif positive_cols:
                resolved["fast_col"] = positive_cols[0]
            elif new_cols:
                resolved["fast_col"] = new_cols[0]

        if "slow_col" not in current_params:
            if signal_cols:
                resolved["slow_col"] = signal_cols[0]
            elif negative_cols:
                resolved["slow_col"] = negative_cols[0]
            elif len(new_cols) > 1:
                resolved["slow_col"] = new_cols[1]

        return resolved

    def get_required_columns(self) -> list:
        """Return required columns for the crossover strategy."""
        return [self.fast_col, self.slow_col]


@VectorizedStrategyRegistry.register("zero_line", description="Buy when oscillator is above zero, sell when below")
class ZeroLineStrategy(VectorizedTradingStrategy):
    """Strategy for zero-centered oscillators such as ROC or CMF."""

    def __init__(
        self,
        indicator_col: str,
        threshold: float = 0.0,
        allow_short: bool = True,
        score_scale: float = None,
    ):
        self.indicator_col = indicator_col
        self.threshold = threshold
        self.allow_short = allow_short
        self.score_scale = score_scale

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate long/short signals from a zero-line oscillator."""
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        if self.indicator_col not in data.columns:
            return signals

        signals[data[self.indicator_col] > self.threshold] = SignalType.BUY.value
        if self.allow_short:
            signals[data[self.indicator_col] < -self.threshold] = SignalType.SELL.value

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """Generate continuous scores from a centered oscillator."""
        if self.indicator_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        values = data[self.indicator_col]
        if self.score_scale is not None and self.score_scale != 0:
            return (values / self.score_scale).clip(-1.0, 1.0).fillna(0.0)

        return self._rolling_zscore(values, window=60).fillna(0.0)

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Prefer oscillator-like columns when an indicator adds
        multiple outputs."""
        resolved: Dict[str, Any] = {}
        if "indicator_col" in current_params or not new_cols:
            return resolved

        preferred = [c for c in new_cols if "_osc" in c.lower()]
        if preferred:
            resolved["indicator_col"] = preferred[0]
        else:
            resolved["indicator_col"] = new_cols[0]
        return resolved

    def get_required_columns(self) -> list:
        """Return required columns for the zero-line strategy."""
        return [self.indicator_col]


@VectorizedStrategyRegistry.register("volatility_breakout", description="Buy on high-volatility breakouts (ATR)")
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


@VectorizedStrategyRegistry.register(
    "bollinger_bands", description="Mean reversion at Bollinger Band extremes via state machine"
)
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

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Wire BB_upper / BB_lower / BB_middle columns."""

        def _find(substring: str) -> Any:
            matches = [c for c in new_cols if substring in c]
            return matches[0] if matches else None

        resolved: Dict[str, Any] = {}
        if "upper_col" not in current_params:
            resolved["upper_col"] = _find("BB_upper")
        if "lower_col" not in current_params:
            resolved["lower_col"] = _find("BB_lower")
        if "middle_col" not in current_params:
            resolved["middle_col"] = _find("BB_middle")
        return resolved

    def get_required_columns(self) -> list:
        return [self.lower_col, self.middle_col, self.upper_col, "Close"]


@VectorizedStrategyRegistry.register("stochastic", description="Mean reversion via %K/%D stochastic crossover")
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

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Wire STOCH_%K and optional STOCH_%D columns."""

        def _find(substring: str) -> Any:
            matches = [c for c in new_cols if substring in c]
            return matches[0] if matches else None

        resolved: Dict[str, Any] = {}
        if "k_col" not in current_params:
            resolved["k_col"] = _find("STOCH_%K")
        if "d_col" not in current_params:
            resolved["d_col"] = _find("STOCH_%D")
        return resolved

    def get_required_columns(self) -> list:
        required = [self.k_col]
        if self.d_col:
            required.append(self.d_col)
        return required


@VectorizedStrategyRegistry.register(
    "flow_trend",
    description="Trend following using a cumulative flow line versus its moving average",
)
class FlowTrendStrategy(VectorizedTradingStrategy):
    """Strategy for cumulative flow lines such as OBV and ADL."""

    def __init__(self, indicator_col: str, ma_window: int = 20, allow_short: bool = True):
        self.indicator_col = indicator_col
        self.ma_window = ma_window
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals from a cumulative flow line relative to its
        trend."""
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        if self.indicator_col not in data.columns:
            return signals

        baseline = data[self.indicator_col].rolling(window=self.ma_window).mean()
        signals[data[self.indicator_col] > baseline] = SignalType.BUY.value
        if self.allow_short:
            signals[data[self.indicator_col] < baseline] = SignalType.SELL.value

        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """Generate flow-trend scores by normalizing distance from the
        rolling mean."""
        if self.indicator_col not in data.columns:
            return pd.Series(0.0, index=data.index)

        values = data[self.indicator_col]
        baseline = values.rolling(window=self.ma_window).mean()
        spread = values - baseline
        scale = values.rolling(window=self.ma_window).std().replace(0, np.nan)
        scores = (spread / (scale + 1e-9)) / 3.0
        return scores.clip(-1.0, 1.0).fillna(0.0)

    def get_required_columns(self) -> list:
        """Return required columns for the flow-trend strategy."""
        return [self.indicator_col]


@VectorizedStrategyRegistry.register("obv_trend", description="Trend following via OBV vs its 20-period moving average")
class OnBalanceVolumeStrategy(FlowTrendStrategy):
    """Strategy for On-Balance Volume - Trend following based on volume"""

    def __init__(self, obv_col: str, allow_short: bool = True):
        super().__init__(indicator_col=obv_col, ma_window=20, allow_short=allow_short)
        self.obv_col = obv_col

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Wire OBV column."""
        resolved: Dict[str, Any] = {}
        if "obv_col" not in current_params:
            matches = [c for c in new_cols if "OBV" in c]
            resolved["obv_col"] = matches[0] if matches else (new_cols[0] if new_cols else None)
        return resolved

    def get_required_columns(self) -> list:
        return [self.obv_col]


@VectorizedStrategyRegistry.register(
    "channel_breakout",
    description="Buy on channel breakouts above the upper band and sell below the lower band",
)
class ChannelBreakoutStrategy(VectorizedTradingStrategy):
    """Strategy for breakout channels such as Donchian and Keltner."""

    def __init__(
        self,
        upper_col: str,
        lower_col: str,
        middle_col: str = None,
        breakout_buffer: float = 0.0,
        allow_short: bool = True,
    ):
        self.upper_col = upper_col
        self.lower_col = lower_col
        self.middle_col = middle_col
        self.breakout_buffer = breakout_buffer
        self.allow_short = allow_short

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate persistent breakout signals from upper/lower channel
        bands."""
        signals = pd.Series(SignalType.HOLD.value, index=data.index)
        required = [self.upper_col, self.lower_col, "Close"]
        if not all(col in data.columns for col in required):
            return signals

        upper = data[self.upper_col] * (1.0 + self.breakout_buffer)
        lower = data[self.lower_col] * (1.0 - self.breakout_buffer)
        long_entries = data["Close"] > upper
        short_entries = data["Close"] < lower

        signals[long_entries] = SignalType.BUY.value
        if self.allow_short:
            signals[short_entries] = SignalType.SELL.value

        signals = (
            signals.astype(float)
            .replace(SignalType.HOLD.value, float("nan"))
            .ffill()
            .fillna(SignalType.HOLD.value)
            .astype(int)
        )
        return signals

    def generate_scores(self, data: pd.DataFrame) -> pd.Series:
        """Generate normalized channel-position scores relative to the
        channel midpoint."""
        required = [self.upper_col, self.lower_col, "Close"]
        if not all(col in data.columns for col in required):
            return pd.Series(0.0, index=data.index)

        upper = data[self.upper_col]
        lower = data[self.lower_col]
        middle = data[self.middle_col] if self.middle_col and self.middle_col in data.columns else (upper + lower) / 2.0
        half_width = ((upper - lower) / 2.0).replace(0, np.nan)
        scores = (data["Close"] - middle) / (half_width + 1e-9)
        return scores.clip(-1.0, 1.0).fillna(0.0)

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve channel band columns from indicator outputs."""
        resolved: Dict[str, Any] = {}
        if "upper_col" not in current_params:
            upper = [c for c in new_cols if "upper" in c.lower()]
            resolved["upper_col"] = upper[0] if upper else None
        if "lower_col" not in current_params:
            lower = [c for c in new_cols if "lower" in c.lower()]
            resolved["lower_col"] = lower[0] if lower else None
        if "middle_col" not in current_params:
            middle = [c for c in new_cols if "mid" in c.lower() or "middle" in c.lower()]
            if middle:
                resolved["middle_col"] = middle[0]
        return resolved

    def get_required_columns(self) -> list:
        """Return required columns for the breakout strategy."""
        required = [self.upper_col, self.lower_col, "Close"]
        if self.middle_col:
            required.append(self.middle_col)
        return required


@VectorizedStrategyRegistry.register(
    "adx_trend", description="Trade in DI direction only when ADX confirms strong trend"
)
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

    @classmethod
    def resolve_columns(cls, new_cols: List[str], current_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wire ADX / ADX_pos / ADX_neg columns.

        The ADX indicator writes columns named "ADX_{window}",
        "ADX_pos_{window}", and "ADX_neg_{window}" — not "PDI"/"MDI".
        """

        def _find(substring: str) -> Any:
            matches = [c for c in new_cols if substring in c]
            return matches[0] if matches else None

        resolved: Dict[str, Any] = {}
        if "adx_col" not in current_params:
            # "ADX_14" matches "ADX" but also "ADX_pos_14" — prefer exact prefix
            adx_matches = [c for c in new_cols if c.startswith("ADX_") and "pos" not in c and "neg" not in c]
            resolved["adx_col"] = adx_matches[0] if adx_matches else _find("ADX")
        if "pdi_col" not in current_params:
            resolved["pdi_col"] = _find("ADX_pos")
        if "mdi_col" not in current_params:
            resolved["mdi_col"] = _find("ADX_neg")
        return resolved

    def get_required_columns(self) -> list:
        return [self.adx_col, self.pdi_col, self.mdi_col]


@VectorizedStrategyRegistry.register("cci_reversal", description="Mean reversion when CCI crosses +/-100 threshold")
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
        Score based on CCI value. CCI is theoretically unbounded but
        usually +/- 200.

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
