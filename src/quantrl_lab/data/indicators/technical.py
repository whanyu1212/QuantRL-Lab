import numpy as np
import pandas as pd

from quantrl_lab.data.indicators._shared import (
    apply_grouped_indicator,
)
from quantrl_lab.data.indicators._shared import ema as ema_series
from quantrl_lab.data.indicators._shared import (
    money_flow_multiplier,
    true_range,
    wilder_smooth,
)
from quantrl_lab.data.indicators.registry import IndicatorRegistry


@IndicatorRegistry.register(
    name="SMA",
    required_columns={"close"},
    output_columns=["SMA_20"],
    description="Simple Moving Average - smooths price data by averaging over a rolling window",
)
def sma(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Add Simple Moving Average to dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 20.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with SMA column added.
    """
    result = df.copy()

    # Handle multiple symbols — "Symbol" column is added by the YFinance loader
    if "Symbol" in result.columns:
        result[f"SMA_{window}"] = result.groupby("Symbol")[column].transform(lambda x: x.rolling(window=window).mean())
    else:
        result[f"SMA_{window}"] = result[column].rolling(window=window).mean()

    return result


@IndicatorRegistry.register(
    name="EMA",
    required_columns={"close"},
    output_columns=["EMA_20"],
    description="Exponential Moving Average - gives more weight to recent prices",
)
def ema(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Add Exponential Moving Average to dataframe.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 20.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with EMA column added.
    """
    result = df.copy()

    if "Symbol" in result.columns:
        result[f"EMA_{window}"] = result.groupby("Symbol")[column].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
    else:
        result[f"EMA_{window}"] = result[column].ewm(span=window, adjust=False).mean()

    return result


@IndicatorRegistry.register(
    name="RSI",
    required_columns={"close"},
    output_columns=["RSI_14"],
    description="Relative Strength Index - momentum oscillator measuring speed and magnitude of price changes (0-100)",
)
def rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.DataFrame:
    """
    Calculate Relative Strength Index using Wilder's smoothing.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 14.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with RSI column added.
    """
    result = df.copy()

    def _calculate_rsi(prices):
        prices = prices.astype(float)
        deltas = np.zeros_like(prices)
        deltas[1:] = np.diff(prices)

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.full_like(prices, np.nan, dtype=float)
        avg_losses = np.full_like(prices, np.nan, dtype=float)
        rsi_values = np.full_like(prices, np.nan, dtype=float)

        if len(prices) > window:
            avg_gains[window] = np.mean(gains[1 : window + 1])  # noqa: E203
            avg_losses[window] = np.mean(losses[1 : window + 1])  # noqa: E203

            if avg_losses[window] != 0:
                rs = avg_gains[window] / avg_losses[window]
                rsi_values[window] = 100 - (100 / (1 + rs))
            else:
                rsi_values[window] = 100

            # Wilder's smoothing uses (n-1) multiplier for subsequent values
            for i in range(window + 1, len(prices)):
                avg_gains[i] = (avg_gains[i - 1] * (window - 1) + gains[i]) / window
                avg_losses[i] = (avg_losses[i - 1] * (window - 1) + losses[i]) / window

                if avg_losses[i] != 0:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi_values[i] = 100 - (100 / (1 + rs))
                else:
                    rsi_values[i] = 100
        return rsi_values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"RSI_{window}"] = _calculate_rsi(group[column].values)
    else:
        result[f"RSI_{window}"] = _calculate_rsi(result[column].values)

    return result


@IndicatorRegistry.register(
    name="MACD",
    required_columns={"close"},
    output_columns=["MACD_line_12_26", "MACD_signal_9"],
    description="Moving Average Convergence Divergence - trend-following momentum indicator",
)
def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD) crossover
    indicator.

    This implementation focuses on the crossover strategy using MACD line and signal line.
    Trading signals are generated when MACD line crosses above/below the signal line.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        fast (int, optional): Short term EMA period. Defaults to 12.
        slow (int, optional): Long term EMA period. Defaults to 26.
        signal (int, optional): EMA of MACD line period. Defaults to 9.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with MACD line and signal line added.
    """
    result = df.copy()

    if "Symbol" in result.columns:
        for _, group in result.groupby("Symbol"):
            fast_ema = group[column].ewm(span=fast, adjust=False).mean()
            slow_ema = group[column].ewm(span=slow, adjust=False).mean()
            macd_line = fast_ema - slow_ema
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()

            result.loc[group.index, f"MACD_line_{fast}_{slow}"] = macd_line
            result.loc[group.index, f"MACD_signal_{signal}"] = signal_line
    else:
        fast_ema = result[column].ewm(span=fast, adjust=False).mean()
        slow_ema = result[column].ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        result[f"MACD_line_{fast}_{slow}"] = macd_line
        result[f"MACD_signal_{signal}"] = signal_line

    return result


@IndicatorRegistry.register(
    name="ATR",
    required_columns={"high", "low", "close"},
    output_columns=["ATR_14"],
    description="Average True Range - measures market volatility by decomposing the entire range of prices",
)
def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with ATR column added.
    """
    result = df.copy()

    def _calculate_atr(high, low, close):
        high_low = high - low
        high_close_prev = np.abs(high - np.append(np.nan, close[:-1]))
        low_close_prev = np.abs(low - np.append(np.nan, close[:-1]))

        tr = np.maximum(high_low, high_close_prev)
        tr = np.maximum(tr, low_close_prev)

        atr_values = np.full_like(close, np.nan, dtype=float)

        # First ATR value is the simple average of the first n periods
        if len(close) > window:
            atr_values[window - 1] = np.nanmean(tr[:window])

            # Subsequent values use Wilder's smoothing
            for i in range(window, len(close)):
                atr_values[i] = (atr_values[i - 1] * (window - 1) + tr[i]) / window

        return atr_values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"ATR_{window}"] = _calculate_atr(
                group["High"].values, group["Low"].values, group["Close"].values
            )
    else:
        result[f"ATR_{window}"] = _calculate_atr(result["High"].values, result["Low"].values, result["Close"].values)

    return result


@IndicatorRegistry.register(
    name="BB",
    required_columns={"close"},
    output_columns=["BB_middle_20", "BB_upper_20_2.0", "BB_lower_20_2.0", "BB_bandwidth_20"],
    description="Bollinger Bands - volatility bands placed above and below a moving average",
)
def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "Close") -> pd.DataFrame:
    """
    Calculate Bollinger Bands indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Window size for moving average. Defaults to 20.
        num_std (float, optional): Number of standard deviations. Defaults to 2.0.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with Bollinger Bands columns added.
    """
    result = df.copy()

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            middle_band = group[column].rolling(window=window).mean()
            std = group[column].rolling(window=window).std()
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            bandwidth = (upper_band - lower_band) / middle_band

            result.loc[group.index, f"BB_middle_{window}"] = middle_band
            result.loc[group.index, f"BB_upper_{window}_{num_std}"] = upper_band
            result.loc[group.index, f"BB_lower_{window}_{num_std}"] = lower_band
            result.loc[group.index, f"BB_bandwidth_{window}"] = bandwidth
    else:
        middle_band = result[column].rolling(window=window).mean()
        std = result[column].rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        bandwidth = (upper_band - lower_band) / middle_band

        result[f"BB_middle_{window}"] = middle_band
        result[f"BB_upper_{window}_{num_std}"] = upper_band
        result[f"BB_lower_{window}_{num_std}"] = lower_band
        result[f"BB_bandwidth_{window}"] = bandwidth

    return result


@IndicatorRegistry.register(
    name="STOCH",
    required_columns={"high", "low", "close"},
    output_columns=["STOCH_%K_14_1", "STOCH_%D_3"],
    description="Stochastic Oscillator - momentum indicator comparing closing price to price range over time (0-100)",
)
def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3, smooth_k: int = 1) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        k_window (int, optional): Window for %K calculation. Defaults to 14.
        d_window (int, optional): Window for %D calculation. Defaults to 3.
        smooth_k (int, optional): Smoothing period for %K. Defaults to 1.

    Returns:
        pd.DataFrame: Dataframe with Stochastic Oscillator columns added.
    """
    result = df.copy()

    def _calculate_stochastic(high, low, close):
        lowest_low = pd.Series(low).rolling(window=k_window).min()
        highest_high = pd.Series(high).rolling(window=k_window).max()

        k_fast = 100 * ((pd.Series(close) - lowest_low) / (highest_high - lowest_low))

        if smooth_k > 1:
            k = k_fast.rolling(window=smooth_k).mean()
        else:
            k = k_fast

        d = k.rolling(window=d_window).mean()

        return k.values, d.values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            k_values, d_values = _calculate_stochastic(group["High"].values, group["Low"].values, group["Close"].values)
            result.loc[group.index, f"STOCH_%K_{k_window}_{smooth_k}"] = k_values
            result.loc[group.index, f"STOCH_%D_{d_window}"] = d_values
    else:
        k_values, d_values = _calculate_stochastic(result["High"].values, result["Low"].values, result["Close"].values)
        result[f"STOCH_%K_{k_window}_{smooth_k}"] = k_values
        result[f"STOCH_%D_{d_window}"] = d_values

    return result


@IndicatorRegistry.register(
    name="OBV",
    required_columns={"close", "volume"},
    output_columns=["OBV"],
    description="On-Balance Volume - cumulative volume indicator showing buying/selling pressure",
)
def on_balance_volume(df: pd.DataFrame, close_col: str = "Close", volume_col: str = "Volume") -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV) indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        close_col (str, optional): Column name for close prices. Defaults to "Close".
        volume_col (str, optional): Column name for volume. Defaults to "Volume".

    Returns:
        pd.DataFrame: Dataframe with OBV column added.
    """
    result = df.copy()

    def _calculate_obv(close, volume):
        close_diff = np.diff(close, prepend=close[0])
        obv = np.zeros_like(close)
        obv[0] = volume[0]

        for i in range(1, len(close)):
            if close_diff[i] > 0:
                obv[i] = obv[i - 1] + volume[i]
            elif close_diff[i] < 0:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        return obv

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, "OBV"] = _calculate_obv(group[close_col].values, group[volume_col].values)
    else:
        result["OBV"] = _calculate_obv(result[close_col].values, result[volume_col].values)

    return result


@IndicatorRegistry.register(
    name="WILLR",
    required_columns={"high", "low", "close"},
    output_columns=["WILLR_14"],
    description="Williams %R - momentum indicator measuring overbought/oversold levels (-100 to 0)",
)
def williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Williams %R indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with Williams %R column added.
    """
    result = df.copy()

    def _calculate_willr(high, low, close):
        highest_high = pd.Series(high).rolling(window=window).max()
        lowest_low = pd.Series(low).rolling(window=window).min()

        willr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return willr.values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"WILLR_{window}"] = _calculate_willr(
                group["High"].values, group["Low"].values, group["Close"].values
            )
    else:
        result[f"WILLR_{window}"] = _calculate_willr(
            result["High"].values, result["Low"].values, result["Close"].values
        )

    return result


@IndicatorRegistry.register(
    name="CCI",
    required_columns={"high", "low", "close"},
    output_columns=["CCI_20"],
    description="Commodity Channel Index - measures deviation from average price to identify cyclical trends",
)
def cci(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Commodity Channel Index (CCI).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 20.

    Returns:
        pd.DataFrame: Dataframe with CCI column added.
    """
    result = df.copy()

    def _calculate_cci(high, low, close):
        tp = (high + low + close) / 3
        tp_series = pd.Series(tp)

        sma_tp = tp_series.rolling(window=window).mean()

        # Rolling MAD via apply is slower but correct; pandas has no vectorized rolling MAD centered on rolling mean
        def mean_deviation(x):
            return np.mean(np.abs(x - np.mean(x)))

        mad = tp_series.rolling(window=window).apply(mean_deviation, raw=True)

        # 0.015 is the Lambert constant used in the CCI formula
        cci_val = (tp_series - sma_tp) / (0.015 * mad)

        return cci_val.values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"CCI_{window}"] = _calculate_cci(
                group["High"].values, group["Low"].values, group["Close"].values
            )
    else:
        result[f"CCI_{window}"] = _calculate_cci(result["High"].values, result["Low"].values, result["Close"].values)

    return result


@IndicatorRegistry.register(
    name="MFI",
    required_columns={"high", "low", "close", "volume"},
    output_columns=["MFI_14"],
    description="Money Flow Index - volume-weighted RSI measuring buying and selling pressure (0-100)",
)
def mfi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Money Flow Index (MFI).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with MFI column added.
    """
    result = df.copy()

    def _calculate_mfi(high, low, close, volume):
        tp = (high + low + close) / 3
        rmf = tp * volume

        tp_diff = np.diff(tp, prepend=tp[0])

        pos_flow = np.where(tp_diff > 0, rmf, 0)
        neg_flow = np.where(tp_diff < 0, rmf, 0)

        pos_mf_sum = pd.Series(pos_flow).rolling(window=window).sum()
        neg_mf_sum = pd.Series(neg_flow).rolling(window=window).sum()

        money_ratio = pos_mf_sum / neg_mf_sum
        mfi_calc = 100 - (100 / (1 + money_ratio))

        # When all flow is positive (neg_mf_sum == 0), MFI is 100; when both are 0 (no volume), use 50
        mfi_calc = np.where(neg_mf_sum == 0, 100, mfi_calc)
        mfi_calc = np.where((neg_mf_sum == 0) & (pos_mf_sum == 0), 50, mfi_calc)

        return mfi_calc

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"MFI_{window}"] = _calculate_mfi(
                group["High"].values, group["Low"].values, group["Close"].values, group["Volume"].values
            )
    else:
        result[f"MFI_{window}"] = _calculate_mfi(
            result["High"].values, result["Low"].values, result["Close"].values, result["Volume"].values
        )

    return result


@IndicatorRegistry.register(
    name="ADX",
    required_columns={"high", "low", "close"},
    output_columns=["ADX_14", "ADX_pos_14", "ADX_neg_14"],
    description="Average Directional Index - measures trend strength and direction (0-100)",
)
def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX).

    Includes +DI and -DI.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with ADX, +DI, and -DI columns added.
    """
    result = df.copy()

    def _wilder_smooth(data, window):
        """Wilder's Smoothing (RMA): first value is SMA, subsequent are
        (prev * (n-1) + curr) / n."""
        smoothed = np.full_like(data, np.nan, dtype=float)
        if len(data) > window:
            # Use nanmean to handle initial NaNs from TR calculation
            smoothed[window - 1] = np.nanmean(data[:window])
            for i in range(window, len(data)):
                if np.isnan(smoothed[i - 1]):
                    smoothed[i] = data[i]
                else:
                    smoothed[i] = (smoothed[i - 1] * (window - 1) + data[i]) / window
        return smoothed

    def _calculate_adx(high, low, close):
        high_low = high - low
        high_close_prev = np.abs(high - np.append(np.nan, close[:-1]))
        low_close_prev = np.abs(low - np.append(np.nan, close[:-1]))
        tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))

        up_move = high - np.append(np.nan, high[:-1])
        down_move = np.append(np.nan, low[:-1]) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = _wilder_smooth(tr, window)
        plus_dm_smooth = _wilder_smooth(plus_dm, window)
        minus_dm_smooth = _wilder_smooth(minus_dm, window)

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)

        sum_di = plus_di + minus_di
        diff_di = np.abs(plus_di - minus_di)

        with np.errstate(divide="ignore", invalid="ignore"):
            dx = 100 * (diff_di / sum_di)

        adx_val = _wilder_smooth(dx, window)

        return adx_val, plus_di, minus_di

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            adx_val, p_di, m_di = _calculate_adx(group["High"].values, group["Low"].values, group["Close"].values)
            result.loc[group.index, f"ADX_{window}"] = adx_val
            result.loc[group.index, f"ADX_pos_{window}"] = p_di
            result.loc[group.index, f"ADX_neg_{window}"] = m_di
    else:
        adx_val, p_di, m_di = _calculate_adx(result["High"].values, result["Low"].values, result["Close"].values)
        result[f"ADX_{window}"] = adx_val
        result[f"ADX_pos_{window}"] = p_di
        result[f"ADX_neg_{window}"] = m_di

    return result


@IndicatorRegistry.register(
    name="ROC",
    required_columns={"close"},
    output_columns=["ROC_12"],
    description="Rate of Change - percentage momentum over a configurable lookback window",
)
def rate_of_change(df: pd.DataFrame, window: int = 12, column: str = "Close") -> pd.DataFrame:
    """
    Calculate Rate of Change (ROC).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 12.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with ROC column added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        close = group[column].astype(float)
        roc = ((close / close.shift(window)) - 1.0) * 100.0
        return {f"ROC_{window}": roc}

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="PPO",
    required_columns={"close"},
    output_columns=["PPO_line_12_26", "PPO_signal_12_26_9", "PPO_hist_12_26_9"],
    description="Percentage Price Oscillator - percentage spread between fast and slow EMAs",
)
def ppo(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Calculate Percentage Price Oscillator (PPO).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        fast (int, optional): Fast EMA span. Defaults to 12.
        slow (int, optional): Slow EMA span. Defaults to 26.
        signal (int, optional): Signal-line EMA span. Defaults to 9.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with PPO columns added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        close = group[column].astype(float)
        fast_ema = ema_series(close, fast)
        slow_ema = ema_series(close, slow)

        with np.errstate(divide="ignore", invalid="ignore"):
            ppo_line = 100.0 * (fast_ema - slow_ema) / slow_ema

        ppo_line = pd.Series(ppo_line, index=group.index)
        signal_line = ema_series(ppo_line, signal)
        histogram = ppo_line - signal_line

        return {
            f"PPO_line_{fast}_{slow}": ppo_line,
            f"PPO_signal_{fast}_{slow}_{signal}": signal_line,
            f"PPO_hist_{fast}_{slow}_{signal}": histogram,
        }

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="TRIX",
    required_columns={"close"},
    output_columns=["TRIX_15", "TRIX_signal_15_9"],
    description="TRIX - triple-smoothed rate-of-change oscillator for trend momentum",
)
def trix(df: pd.DataFrame, window: int = 15, signal: int = 9, column: str = "Close") -> pd.DataFrame:
    """
    Calculate TRIX.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): EMA span used in each smoothing stage. Defaults to 15.
        signal (int, optional): Signal-line EMA span. Defaults to 9.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with TRIX columns added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        close = group[column].astype(float)
        ema1 = ema_series(close, window)
        ema2 = ema_series(ema1, window)
        ema3 = ema_series(ema2, window)
        trix_line = ema3.pct_change() * 100.0
        signal_line = ema_series(trix_line, signal)

        return {
            f"TRIX_{window}": trix_line,
            f"TRIX_signal_{window}_{signal}": signal_line,
        }

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="TSI",
    required_columns={"close"},
    output_columns=["TSI_25_13", "TSI_signal_25_13_13"],
    description="True Strength Index - double-smoothed momentum oscillator",
)
def tsi(
    df: pd.DataFrame,
    slow: int = 25,
    fast: int = 13,
    signal: int = 13,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Calculate True Strength Index (TSI).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        slow (int, optional): First EMA span applied to momentum. Defaults to 25.
        fast (int, optional): Second EMA span applied to momentum. Defaults to 13.
        signal (int, optional): Signal-line EMA span. Defaults to 13.
        column (str, optional): Column used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with TSI columns added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        close = group[column].astype(float)
        momentum = close.diff()
        smoothed_momentum = ema_series(ema_series(momentum, slow), fast)
        smoothed_abs_momentum = ema_series(ema_series(momentum.abs(), slow), fast)

        with np.errstate(divide="ignore", invalid="ignore"):
            tsi_line = 100.0 * smoothed_momentum / smoothed_abs_momentum

        tsi_line = pd.Series(tsi_line, index=group.index)
        signal_line = ema_series(tsi_line, signal)

        return {
            f"TSI_{slow}_{fast}": tsi_line,
            f"TSI_signal_{slow}_{fast}_{signal}": signal_line,
        }

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="AROON",
    required_columns={"high", "low"},
    output_columns=["AROON_up_25", "AROON_down_25", "AROON_osc_25"],
    description="Aroon - measures how recently highs and lows occurred within a rolling window",
)
def aroon(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """
    Calculate the Aroon indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 25.

    Returns:
        pd.DataFrame: Dataframe with Aroon columns added.
    """

    def _aroon_recent_high(values: np.ndarray) -> float:
        periods_since_high = int(np.argmax(values[::-1]))
        return 100.0 * (window - periods_since_high) / window

    def _aroon_recent_low(values: np.ndarray) -> float:
        periods_since_low = int(np.argmin(values[::-1]))
        return 100.0 * (window - periods_since_low) / window

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        aroon_up = high.rolling(window=window).apply(_aroon_recent_high, raw=True)
        aroon_down = low.rolling(window=window).apply(_aroon_recent_low, raw=True)
        oscillator = aroon_up - aroon_down

        return {
            f"AROON_up_{window}": aroon_up,
            f"AROON_down_{window}": aroon_down,
            f"AROON_osc_{window}": oscillator,
        }

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="VORTEX",
    required_columns={"high", "low", "close"},
    output_columns=["VORTEX_pos_14", "VORTEX_neg_14"],
    description="Vortex Indicator - compares upward and downward price movement over a rolling window",
)
def vortex(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate the Vortex Indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with Vortex columns added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        close = group["Close"].astype(float)
        tr_sum = true_range(high, low, close).rolling(window=window).sum()
        positive_vm = (high - low.shift(1)).abs().rolling(window=window).sum()
        negative_vm = (low - high.shift(1)).abs().rolling(window=window).sum()

        with np.errstate(divide="ignore", invalid="ignore"):
            positive_vi = positive_vm / tr_sum
            negative_vi = negative_vm / tr_sum

        return {
            f"VORTEX_pos_{window}": positive_vi,
            f"VORTEX_neg_{window}": negative_vi,
        }

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="DONCHIAN",
    required_columns={"high", "low"},
    output_columns=["DONCHIAN_upper_20", "DONCHIAN_lower_20", "DONCHIAN_mid_20"],
    description="Donchian Channels - rolling breakout bands based on recent highs and lows",
)
def donchian_channels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Donchian Channels.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 20.

    Returns:
        pd.DataFrame: Dataframe with Donchian channel columns added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        upper = group["High"].astype(float).rolling(window=window).max()
        lower = group["Low"].astype(float).rolling(window=window).min()
        middle = (upper + lower) / 2.0

        return {
            f"DONCHIAN_upper_{window}": upper,
            f"DONCHIAN_lower_{window}": lower,
            f"DONCHIAN_mid_{window}": middle,
        }

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="KELTNER",
    required_columns={"high", "low", "close"},
    output_columns=["KC_middle_20", "KC_upper_20_2.0", "KC_lower_20_2.0"],
    description="Keltner Channels - EMA centerline with ATR-based upper and lower envelopes",
)
def keltner_channels(
    df: pd.DataFrame,
    window: int = 20,
    atr_mult: float = 2.0,
    column: str = "Close",
) -> pd.DataFrame:
    """
    Calculate Keltner Channels.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): EMA and ATR window. Defaults to 20.
        atr_mult (float, optional): ATR multiplier for channel width. Defaults to 2.0.
        column (str, optional): Column used for the centerline EMA. Defaults to "Close".

    Returns:
        pd.DataFrame: Dataframe with Keltner channel columns added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        close = group[column].astype(float)
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        middle = ema_series(close, window)
        atr_values = wilder_smooth(true_range(high, low, close), window)
        upper = middle + (atr_mult * atr_values)
        lower = middle - (atr_mult * atr_values)

        return {
            f"KC_middle_{window}": middle,
            f"KC_upper_{window}_{atr_mult}": upper,
            f"KC_lower_{window}_{atr_mult}": lower,
        }

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="NATR",
    required_columns={"high", "low", "close"},
    output_columns=["NATR_14"],
    description="Normalized ATR - ATR expressed as a percentage of closing price",
)
def natr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Normalized Average True Range (NATR).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: Dataframe with NATR column added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        close = group["Close"].astype(float)
        atr_values = wilder_smooth(true_range(high, low, close), window)

        with np.errstate(divide="ignore", invalid="ignore"):
            normalized_atr = 100.0 * atr_values / close

        return {f"NATR_{window}": normalized_atr}

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="CMF",
    required_columns={"high", "low", "close", "volume"},
    output_columns=["CMF_20"],
    description="Chaikin Money Flow - rolling accumulation and distribution pressure normalised by volume",
)
def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculate Chaikin Money Flow (CMF).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): Lookback period. Defaults to 20.

    Returns:
        pd.DataFrame: Dataframe with CMF column added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        close = group["Close"].astype(float)
        volume = group["Volume"].astype(float)
        mfv = money_flow_multiplier(high, low, close) * volume
        volume_sum = volume.rolling(window=window).sum()

        with np.errstate(divide="ignore", invalid="ignore"):
            cmf_values = mfv.rolling(window=window).sum() / volume_sum

        return {f"CMF_{window}": cmf_values}

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="ADL",
    required_columns={"high", "low", "close", "volume"},
    output_columns=["ADL"],
    description="Accumulation/Distribution Line - cumulative money flow volume",
)
def accumulation_distribution_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Accumulation/Distribution Line (ADL).

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.

    Returns:
        pd.DataFrame: Dataframe with ADL column added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        close = group["Close"].astype(float)
        volume = group["Volume"].astype(float)
        adl_values = (money_flow_multiplier(high, low, close) * volume).cumsum()
        return {"ADL": adl_values}

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="CHO",
    required_columns={"high", "low", "close", "volume"},
    output_columns=["CHO_3_10"],
    description="Chaikin Oscillator - fast and slow EMA spread of the accumulation/distribution line",
)
def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.DataFrame:
    """
    Calculate the Chaikin Oscillator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        fast (int, optional): Fast EMA span. Defaults to 3.
        slow (int, optional): Slow EMA span. Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe with Chaikin Oscillator column added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        close = group["Close"].astype(float)
        volume = group["Volume"].astype(float)
        adl_values = (money_flow_multiplier(high, low, close) * volume).cumsum()
        oscillator = ema_series(adl_values, fast) - ema_series(adl_values, slow)
        return {f"CHO_{fast}_{slow}": oscillator}

    return apply_grouped_indicator(df, _calculate)


@IndicatorRegistry.register(
    name="SUPERTREND",
    required_columns={"high", "low", "close"},
    output_columns=["SUPERTREND_10_3.0", "SUPERTREND_dir_10_3.0"],
    description="SuperTrend - ATR-based trailing trend line with directional state",
)
def supertrend(df: pd.DataFrame, window: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate the SuperTrend indicator.

    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data.
        window (int, optional): ATR lookback period. Defaults to 10.
        multiplier (float, optional): ATR multiplier for band width. Defaults to 3.0.

    Returns:
        pd.DataFrame: Dataframe with SuperTrend line and direction columns added.
    """

    def _calculate(group: pd.DataFrame) -> dict[str, pd.Series]:
        high = group["High"].astype(float)
        low = group["Low"].astype(float)
        close = group["Close"].astype(float)

        atr_values = wilder_smooth(true_range(high, low, close), window)
        hl2 = (high + low) / 2.0
        basic_upper = hl2 + (multiplier * atr_values)
        basic_lower = hl2 - (multiplier * atr_values)

        final_upper = pd.Series(np.nan, index=group.index, dtype=float)
        final_lower = pd.Series(np.nan, index=group.index, dtype=float)
        supertrend_line = pd.Series(np.nan, index=group.index, dtype=float)
        direction = pd.Series(np.nan, index=group.index, dtype=float)

        valid_positions = np.flatnonzero(atr_values.notna().to_numpy())
        if len(valid_positions) == 0:
            return {
                f"SUPERTREND_{window}_{multiplier}": supertrend_line,
                f"SUPERTREND_dir_{window}_{multiplier}": direction,
            }

        first_valid = int(valid_positions[0])
        final_upper.iloc[first_valid] = basic_upper.iloc[first_valid]
        final_lower.iloc[first_valid] = basic_lower.iloc[first_valid]
        direction.iloc[first_valid] = 1.0
        supertrend_line.iloc[first_valid] = final_lower.iloc[first_valid]

        for i in range(first_valid + 1, len(group)):
            current_upper = basic_upper.iloc[i]
            current_lower = basic_lower.iloc[i]
            prev_upper = final_upper.iloc[i - 1]
            prev_lower = final_lower.iloc[i - 1]
            prev_close = close.iloc[i - 1]

            if np.isnan(current_upper) or np.isnan(current_lower):
                continue

            if np.isnan(prev_upper) or current_upper < prev_upper or prev_close > prev_upper:
                final_upper.iloc[i] = current_upper
            else:
                final_upper.iloc[i] = prev_upper

            if np.isnan(prev_lower) or current_lower > prev_lower or prev_close < prev_lower:
                final_lower.iloc[i] = current_lower
            else:
                final_lower.iloc[i] = prev_lower

            previous_direction = direction.iloc[i - 1]
            if np.isnan(previous_direction):
                previous_direction = 1.0

            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1.0
            elif close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1.0
            else:
                direction.iloc[i] = previous_direction
                if direction.iloc[i] > 0 and not np.isnan(prev_lower) and final_lower.iloc[i] < prev_lower:
                    final_lower.iloc[i] = prev_lower
                if direction.iloc[i] < 0 and not np.isnan(prev_upper) and final_upper.iloc[i] > prev_upper:
                    final_upper.iloc[i] = prev_upper

            supertrend_line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] > 0 else final_upper.iloc[i]

        return {
            f"SUPERTREND_{window}_{multiplier}": supertrend_line,
            f"SUPERTREND_dir_{window}_{multiplier}": direction,
        }

    return apply_grouped_indicator(df, _calculate)
