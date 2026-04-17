from typing import Dict, Tuple

import pandas as pd
from scipy.stats import pearsonr, spearmanr


def calculate_forward_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate forward returns for a given period.

    Args:
        prices (pd.Series): Price series.
        periods (int): Number of periods forward (e.g., 1 for next day return).

    Returns:
        pd.Series: Forward returns.
    """
    return prices.pct_change(periods).shift(-periods)


def calculate_information_coefficient(signal: pd.Series, forward_returns: pd.Series) -> float:
    """
    Calculate the Information Coefficient (IC).

    IC is the Pearson correlation between the signal and future returns.
    A positive IC indicates the signal correctly predicts price direction.

    Args:
        signal (pd.Series): Alpha signal values.
        forward_returns (pd.Series): Forward returns aligned with signal dates.

    Returns:
        float: Information Coefficient.
    """
    # Align indices and drop NaNs
    df = pd.concat([signal, forward_returns], axis=1).dropna()
    if len(df) < 2:
        return 0.0

    # A-8: constant signal or returns → correlation is undefined (NaN) → treat as 0
    if df.iloc[:, 0].nunique() <= 1 or df.iloc[:, 1].nunique() <= 1:
        return 0.0

    return df.iloc[:, 0].corr(df.iloc[:, 1])


def calculate_pearson_ic(signal: pd.Series, forward_returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate Pearson IC and its two-tailed p-value.

    Args:
        signal (pd.Series): Alpha signal values.
        forward_returns (pd.Series): Forward returns aligned with signal dates.

    Returns:
        Tuple[float, float]: (IC, p-value)
    """
    df = pd.concat([signal, forward_returns], axis=1).dropna()
    if len(df) < 2:
        return 0.0, 1.0

    # A-8: constant signal or returns → correlation is undefined (NaN) → treat as 0
    if df.iloc[:, 0].nunique() <= 1 or df.iloc[:, 1].nunique() <= 1:
        return 0.0, 1.0

    ic, p_value = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    return float(ic), float(p_value)


def calculate_rank_ic(signal: pd.Series, forward_returns: pd.Series) -> Tuple[float, float]:
    """
    Calculate the Rank Information Coefficient (Rank IC).

    Rank IC is the Spearman correlation between the signal rank and
    future return rank. It is more robust to outliers than standard IC
    and is generally preferred for alpha research.

    Args:
        signal (pd.Series): Alpha signal values.
        forward_returns (pd.Series): Forward returns.

    Returns:
        Tuple[float, float]: (Rank IC, p-value)
    """
    df = pd.concat([signal, forward_returns], axis=1).dropna()
    if len(df) < 2:
        return 0.0, 1.0

    # A-8: constant signal or returns → correlation is undefined (NaN) → treat as 0
    if df.iloc[:, 0].nunique() <= 1 or df.iloc[:, 1].nunique() <= 1:
        return 0.0, 1.0

    correlation, p_value = spearmanr(df.iloc[:, 0], df.iloc[:, 1])
    return correlation, p_value


def calculate_autocorrelation(signal: pd.Series, lag: int = 1) -> float:
    """
    Calculate signal autocorrelation (stability).

    High autocorrelation means the signal changes slowly (low turnover).
    Low autocorrelation means the signal changes rapidly (high turnover, higher trading costs).

    Args:
        signal (pd.Series): Alpha signal values.
        lag (int): Lag period.

    Returns:
        float: Autocorrelation.
    """
    return signal.autocorr(lag=lag)


def calculate_turnover(signal: pd.Series) -> float:
    """
    Calculate signal turnover proxy.

    Turnover is approximated as the mean absolute difference between consecutive signal values,
    normalized by the mean absolute signal value.

    Args:
        signal (pd.Series): Alpha signal values.

    Returns:
        float: Turnover metric.
    """
    diff = signal.diff().abs().mean()
    avg_abs = signal.abs().mean()

    if avg_abs == 0 or pd.isna(avg_abs):
        return 0.0

    return diff / avg_abs


def analyze_signal(
    signal: pd.Series,
    prices: pd.Series,
    forward_periods: int = 1,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Perform comprehensive analysis of a signal.

    Args:
        signal (pd.Series): The alpha signal.
        prices (pd.Series): Price series to calculate returns against.
        forward_periods (int): Horizon for forward returns.
        normalize (bool): Whether to rank-normalize the signal to [-1, 1] before analysis.

    Returns:
        Dict[str, float]: Dictionary of metrics.
    """
    if normalize:
        # Rank normalize to [-1, 1] to simulate a trading portfolio position
        # (rank - 0.5) * 2 centers around 0
        signal = (signal.rank(pct=True) - 0.5) * 2

    fwd_ret = calculate_forward_returns(prices, forward_periods)

    ic = calculate_information_coefficient(signal, fwd_ret)
    rank_ic, p_val = calculate_rank_ic(signal, fwd_ret)
    autocorr = calculate_autocorrelation(signal)
    turnover = calculate_turnover(signal)

    return {
        "IC": ic,
        "Rank_IC": rank_ic,
        "IC_p_value": p_val,
        "Autocorrelation": autocorr,
        "Turnover": turnover,
        "Horizon": forward_periods,
    }
