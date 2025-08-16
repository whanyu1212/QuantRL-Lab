import numpy as np
import pandas as pd

from quantrl_lab.data.indicators.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register(name="SMA")
def sma(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Add Simple Moving Average to dataframe.

    Args:
        df (pd.DataFrame): input dataframe with OHLCV data
        window (int, optional): window size. Defaults to 20.
        column (str, optional): col used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: dataframe with SMA column added
    """
    result = df.copy()

    # Handle multiple symbols if present
    # In the YFinance loader, I created a column called "Symbol"
    # to handle multiple symbols.

    if "Symbol" in result.columns:
        result[f"SMA_{window}"] = result.groupby("Symbol")[column].transform(lambda x: x.rolling(window=window).mean())
    else:
        result[f"SMA_{window}"] = result[column].rolling(window=window).mean()

    return result


@IndicatorRegistry.register(name="EMA")
def ema(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
    """
    Add Exponential Moving Average to dataframe.

    Args:
        df (pd.DataFrame): input dataframe with OHLCV data
        window (int, optional): window size. Defaults to 20.
        column (str, optional): col used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: dataframe with EMA column added
    """
    result = df.copy()

    if "Symbol" in result.columns:
        result[f"EMA_{window}"] = result.groupby("Symbol")[column].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
    else:
        result[f"EMA_{window}"] = result[column].ewm(span=window, adjust=False).mean()

    return result


@IndicatorRegistry.register(name="RSI")
def rsi(df: pd.DataFrame, window: int = 14, column: str = "Close") -> pd.DataFrame:
    """
    Calculate Relative Strength Index using Wilder's smoothing.

    Args:
        df (pd.DataFrame): input dataframe with OHLCV data
        window (int, optional): window size. Defaults to 14.
        column (str, optional): col used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: dataframe with RSI column added
    """
    result = df.copy()

    def _calculate_rsi(prices):
        # Ensure prices are float for accurate division
        prices = prices.astype(float)
        # Initialize deltas with zeros and compute differences
        deltas = np.zeros_like(prices)
        deltas[1:] = np.diff(prices)

        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Initialize arrays for averages and RSI values
        avg_gains = np.full_like(prices, np.nan, dtype=float)
        avg_losses = np.full_like(prices, np.nan, dtype=float)
        rsi_values = np.full_like(prices, np.nan, dtype=float)

        if len(prices) > window:
            # Compute initial simple averages for the first window
            avg_gains[window] = np.mean(gains[1 : window + 1])  # noqa: E203
            avg_losses[window] = np.mean(losses[1 : window + 1])  # noqa: E203

            # Calculate RSI at the first full window
            if avg_losses[window] != 0:
                rs = avg_gains[window] / avg_losses[window]
                rsi_values[window] = 100 - (100 / (1 + rs))
            else:
                rsi_values[window] = 100

            # Use Wilder's smoothing for subsequent values
            for i in range(window + 1, len(prices)):
                avg_gains[i] = (avg_gains[i - 1] * (window - 1) + gains[i]) / window
                avg_losses[i] = (avg_losses[i - 1] * (window - 1) + losses[i]) / window

                if avg_losses[i] != 0:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi_values[i] = 100 - (100 / (1 + rs))
                else:
                    rsi_values[i] = 100
        return rsi_values

    # Handle multiple symbols if present in the dataframe
    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"RSI_{window}"] = _calculate_rsi(group[column].values)
    else:
        result[f"RSI_{window}"] = _calculate_rsi(result[column].values)

    return result


@IndicatorRegistry.register(name="MACD")
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
        df (pd.DataFrame): input dataframe with OHLCV data
        fast (int, optional): short term EMA. Defaults to 12.
        slow (int, optional): long term EMA. Defaults to 26.
        signal (int, optional): EMA of MACD itself. Defaults to 9.
        column (str, optional): col used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: dataframe with MACD line and signal line added
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


@IndicatorRegistry.register(name="ATR")
def atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) indicator.

    Args:
        df (pd.DataFrame): input dataframe with OHLCV data
        window (int, optional): window size. Defaults to 14.

    Returns:
        pd.DataFrame: dataframe with ATR column added
    """
    result = df.copy()

    def _calculate_atr(high, low, close):
        # Calculate True Range
        high_low = high - low
        high_close_prev = np.abs(high - np.append(np.nan, close[:-1]))
        low_close_prev = np.abs(low - np.append(np.nan, close[:-1]))

        # Get maximum of the three
        tr = np.maximum(high_low, high_close_prev)
        tr = np.maximum(tr, low_close_prev)

        # Calculate ATR using Wilder's smoothing
        atr_values = np.full_like(close, np.nan, dtype=float)

        # First ATR value is just the average of first n periods
        if len(close) > window:
            atr_values[window - 1] = np.nanmean(tr[:window])

            # Subsequent values use Wilder's smoothing
            for i in range(window, len(close)):
                atr_values[i] = (atr_values[i - 1] * (window - 1) + tr[i]) / window

        return atr_values

    # Handle multiple symbols if present
    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            result.loc[group.index, f"ATR_{window}"] = _calculate_atr(
                group["High"].values, group["Low"].values, group["Close"].values
            )
    else:
        result[f"ATR_{window}"] = _calculate_atr(result["High"].values, result["Low"].values, result["Close"].values)

    return result


@IndicatorRegistry.register(name="BB")
def bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0, column: str = "Close") -> pd.DataFrame:
    """
    Calculate Bollinger Bands indicator.

    Args:
        df (pd.DataFrame): input dataframe with OHLCV data
        window (int, optional): window size for moving average. Defaults to 20.
        num_std (float, optional): number of standard deviations. Defaults to 2.0.
        column (str, optional): col used for calculation. Defaults to "Close".

    Returns:
        pd.DataFrame: dataframe with Bollinger Bands columns added
    """
    result = df.copy()

    if "Symbol" in result.columns:
        # Process each symbol separately
        for symbol, group in result.groupby("Symbol"):
            # Calculate middle band (SMA)
            middle_band = group[column].rolling(window=window).mean()

            # Calculate standard deviation
            std = group[column].rolling(window=window).std()

            # Calculate upper and lower bands
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)

            # Calculate bandwidth
            bandwidth = (upper_band - lower_band) / middle_band

            # Add to result dataframe
            result.loc[group.index, f"BB_middle_{window}"] = middle_band
            result.loc[group.index, f"BB_upper_{window}_{num_std}"] = upper_band
            result.loc[group.index, f"BB_lower_{window}_{num_std}"] = lower_band
            result.loc[group.index, f"BB_bandwidth_{window}"] = bandwidth
    else:
        # Calculate middle band (SMA)
        middle_band = result[column].rolling(window=window).mean()

        # Calculate standard deviation
        std = result[column].rolling(window=window).std()

        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / middle_band

        # Add to result dataframe
        result[f"BB_middle_{window}"] = middle_band
        result[f"BB_upper_{window}_{num_std}"] = upper_band
        result[f"BB_lower_{window}_{num_std}"] = lower_band
        result[f"BB_bandwidth_{window}"] = bandwidth

    return result


@IndicatorRegistry.register(name="STOCH")
def stochastic(df: pd.DataFrame, k_window: int = 14, d_window: int = 3, smooth_k: int = 1) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Args:
        df (pd.DataFrame): input dataframe with OHLCV data
        k_window (int, optional): window for %K calculation. Defaults to 14.
        d_window (int, optional): window for %D calculation. Defaults to 3.
        smooth_k (int, optional): smoothing for %K. Defaults to 1.

    Returns:
        pd.DataFrame: dataframe with Stochastic Oscillator columns added
    """
    result = df.copy()

    def _calculate_stochastic(high, low, close):
        # Calculate %K
        lowest_low = pd.Series(low).rolling(window=k_window).min()
        highest_high = pd.Series(high).rolling(window=k_window).max()

        # Fast %K
        k_fast = 100 * ((pd.Series(close) - lowest_low) / (highest_high - lowest_low))

        # Smooth %K if requested
        if smooth_k > 1:
            k = k_fast.rolling(window=smooth_k).mean()
        else:
            k = k_fast

        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_window).mean()

        return k.values, d.values

    if "Symbol" in result.columns:
        for symbol, group in result.groupby("Symbol"):
            k_values, d_values = _calculate_stochastic(group["High"].values, group["Low"].values, group["Close"].values)
            result.loc[group.index, f"STOCH_%K_{k_window}"] = k_values
            result.loc[group.index, f"STOCH_%D_{d_window}"] = d_values
    else:
        k_values, d_values = _calculate_stochastic(result["High"].values, result["Low"].values, result["Close"].values)
        result[f"STOCH_%K_{k_window}"] = k_values
        result[f"STOCH_%D_{d_window}"] = d_values

    return result


@IndicatorRegistry.register(name="OBV")
def on_balance_volume(df: pd.DataFrame, close_col: str = "Close", volume_col: str = "Volume") -> pd.DataFrame:
    """
    Calculate On-Balance Volume (OBV) indicator.

    Args:
        df (pd.DataFrame): input dataframe with OHLCV data
        close_col (str, optional): column name for close prices. Defaults to "Close".
        volume_col (str, optional): column name for volume. Defaults to "Volume".

    Returns:
        pd.DataFrame: dataframe with OBV column added
    """
    result = df.copy()

    def _calculate_obv(close, volume):
        close_diff = np.diff(close, prepend=close[0])
        obv = np.zeros_like(close)

        # Set first OBV value to first volume value
        obv[0] = volume[0]

        # Calculate OBV
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
