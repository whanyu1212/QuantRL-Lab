"""Internal helpers for technical indicator calculations."""

from typing import Callable, Dict, Iterator, Tuple

import numpy as np
import pandas as pd


def iter_symbol_groups(df: pd.DataFrame) -> Iterator[Tuple[pd.Index, pd.DataFrame]]:
    """
    Yield symbol-partitioned groups while preserving input order.

    Args:
        df (pd.DataFrame): Input dataframe.

    Yields:
        Iterator[Tuple[pd.Index, pd.DataFrame]]: Index and grouped dataframe.
    """
    if "Symbol" in df.columns:
        for _, group in df.groupby("Symbol", sort=False):
            yield group.index, group
    else:
        yield df.index, df


def apply_grouped_indicator(
    df: pd.DataFrame,
    calculator: Callable[[pd.DataFrame], Dict[str, pd.Series]],
) -> pd.DataFrame:
    """
    Apply an indicator calculator independently for each symbol group.

    Args:
        df (pd.DataFrame): Input OHLCV dataframe.
        calculator (Callable[[pd.DataFrame], Dict[str, pd.Series]]):
            Function returning output columns for a single time series.

    Returns:
        pd.DataFrame: Dataframe with indicator columns added.
    """
    result = df.copy()

    for index, group in iter_symbol_groups(result):
        outputs = calculator(group.copy())
        for column_name, values in outputs.items():
            result.loc[index, column_name] = pd.Series(values, index=group.index)

    return result


def ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate an exponential moving average.

    Args:
        series (pd.Series): Input values.
        span (int): EWM span.

    Returns:
        pd.Series: Exponentially weighted moving average.
    """
    return series.ewm(span=span, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Calculate true range.

    Args:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Close prices.

    Returns:
        pd.Series: True range series.
    """
    prev_close = close.shift(1)
    return pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """
    Apply Wilder's smoothing.

    Args:
        series (pd.Series): Input series.
        window (int): Smoothing window.

    Returns:
        pd.Series: Smoothed values with NaN warm-up.
    """
    values = series.to_numpy(dtype=float, copy=False)
    smoothed = np.full(len(values), np.nan, dtype=float)

    if len(values) < window:
        return pd.Series(smoothed, index=series.index)

    smoothed[window - 1] = np.nanmean(values[:window])
    for i in range(window, len(values)):
        previous = smoothed[i - 1]
        current = values[i]
        if np.isnan(previous):
            smoothed[i] = current
        else:
            smoothed[i] = (previous * (window - 1) + current) / window

    return pd.Series(smoothed, index=series.index)


def money_flow_multiplier(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Calculate the Chaikin money flow multiplier.

    Args:
        high (pd.Series): High prices.
        low (pd.Series): Low prices.
        close (pd.Series): Close prices.

    Returns:
        pd.Series: Money flow multiplier in the range [-1, 1].
    """
    denominator = high - low
    numerator = ((close - low) - (high - close)).astype(float)
    multiplier = np.where(denominator == 0, 0.0, numerator / denominator)
    return pd.Series(multiplier, index=close.index)
