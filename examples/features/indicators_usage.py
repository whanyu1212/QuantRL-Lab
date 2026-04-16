"""
Example script demonstrating how to use the Indicator Registry.

This script shows:
1. How to fetch data (using yfinance for this example).
2. How to apply and validate the expanded built-in indicator set.
3. How to create and register a custom indicator on the fly.
4. How to use the custom indicator just like a built-in one.
"""

import pandas as pd
from loguru import logger

from quantrl_lab.data.indicators import IndicatorRegistry
from quantrl_lab.data.sources.yfinance_loader import YFinanceDataLoader

BUILT_IN_INDICATOR_RUNS = [
    {"name": "SMA", "params": {"window": 20}, "expected": ["SMA_20"]},
    {"name": "EMA", "params": {"window": 20}, "expected": ["EMA_20"]},
    {"name": "RSI", "params": {"window": 14}, "expected": ["RSI_14"]},
    {"name": "MACD", "params": {"fast": 12, "slow": 26, "signal": 9}, "expected": ["MACD_line_12_26", "MACD_signal_9"]},
    {"name": "ATR", "params": {"window": 14}, "expected": ["ATR_14"]},
    {"name": "BB", "params": {"window": 20, "num_std": 2.0}, "expected": ["BB_upper_20_2.0", "BB_lower_20_2.0"]},
    {"name": "STOCH", "params": {"k_window": 14, "d_window": 3}, "expected": ["STOCH_%K_14_1", "STOCH_%D_3"]},
    {"name": "OBV", "params": {}, "expected": ["OBV"]},
    {"name": "WILLR", "params": {"window": 14}, "expected": ["WILLR_14"]},
    {"name": "CCI", "params": {"window": 20}, "expected": ["CCI_20"]},
    {"name": "MFI", "params": {"window": 14}, "expected": ["MFI_14"]},
    {"name": "ADX", "params": {"window": 14}, "expected": ["ADX_14", "ADX_pos_14", "ADX_neg_14"]},
    {"name": "ROC", "params": {"window": 12}, "expected": ["ROC_12"]},
    {
        "name": "PPO",
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "expected": ["PPO_line_12_26", "PPO_signal_12_26_9", "PPO_hist_12_26_9"],
    },
    {"name": "TRIX", "params": {"window": 15, "signal": 9}, "expected": ["TRIX_15", "TRIX_signal_15_9"]},
    {
        "name": "TSI",
        "params": {"slow": 25, "fast": 13, "signal": 13},
        "expected": ["TSI_25_13", "TSI_signal_25_13_13"],
    },
    {"name": "AROON", "params": {"window": 25}, "expected": ["AROON_up_25", "AROON_down_25", "AROON_osc_25"]},
    {"name": "VORTEX", "params": {"window": 14}, "expected": ["VORTEX_pos_14", "VORTEX_neg_14"]},
    {
        "name": "DONCHIAN",
        "params": {"window": 20},
        "expected": ["DONCHIAN_upper_20", "DONCHIAN_lower_20", "DONCHIAN_mid_20"],
    },
    {
        "name": "KELTNER",
        "params": {"window": 20, "atr_mult": 2.0},
        "expected": ["KC_middle_20", "KC_upper_20_2.0", "KC_lower_20_2.0"],
    },
    {"name": "NATR", "params": {"window": 14}, "expected": ["NATR_14"]},
    {"name": "CMF", "params": {"window": 20}, "expected": ["CMF_20"]},
    {"name": "ADL", "params": {}, "expected": ["ADL"]},
    {"name": "CHO", "params": {"fast": 3, "slow": 10}, "expected": ["CHO_3_10"]},
    {
        "name": "SUPERTREND",
        "params": {"window": 10, "multiplier": 3.0},
        "expected": ["SUPERTREND_10_3.0", "SUPERTREND_dir_10_3.0"],
    },
]

NEW_INDICATOR_NAMES = {
    "ROC",
    "PPO",
    "TRIX",
    "TSI",
    "AROON",
    "VORTEX",
    "DONCHIAN",
    "KELTNER",
    "NATR",
    "CMF",
    "ADL",
    "CHO",
    "SUPERTREND",
}


def apply_indicator_runs(df: pd.DataFrame, indicator_runs: list[dict]) -> pd.DataFrame:
    """
    Apply a sequence of built-in indicators and validate expected
    outputs.

    Args:
        df (pd.DataFrame): Input OHLCV dataframe.
        indicator_runs (list[dict]): Indicator configs with names, params, and expected columns.

    Returns:
        pd.DataFrame: Dataframe with indicator columns added.

    Raises:
        ValueError: If an indicator does not add the expected columns.
    """
    result = df.copy()

    for indicator_run in indicator_runs:
        name = indicator_run["name"]
        params = indicator_run["params"]
        expected_columns = indicator_run["expected"]

        logger.info(f"Applying {name} with params={params}...")
        result = IndicatorRegistry.apply(name, result, **params)

        missing_columns = [column for column in expected_columns if column not in result.columns]
        if missing_columns:
            raise ValueError(f"Indicator {name} did not add expected columns: {missing_columns}")

    return result


def main():
    # 1. Fetch some sample data
    logger.info("Fetching sample data (AAPL) from YFinance...")
    loader = YFinanceDataLoader()
    df = loader.get_historical_ohlcv_data(symbols=["AAPL"], start="2023-01-01", end="2023-06-01")

    # Basic data check
    logger.info(f"Data loaded: {len(df)} rows")
    logger.info(f"Columns: {df.columns.tolist()}")

    # ---------------------------------------------------------
    # 2. Apply Built-in Indicators
    # ---------------------------------------------------------
    logger.info("\n--- Applying Built-in Indicators ---")

    # You can list all available indicators
    available_indicators = sorted(IndicatorRegistry.list_all())
    logger.info(f"Available indicators: {available_indicators}")

    built_in_names = {indicator_run["name"] for indicator_run in BUILT_IN_INDICATOR_RUNS}
    missing_built_ins = built_in_names - set(available_indicators)
    if missing_built_ins:
        raise ValueError(f"Example configuration references unregistered indicators: {sorted(missing_built_ins)}")

    df = apply_indicator_runs(df, BUILT_IN_INDICATOR_RUNS)

    built_in_feature_columns = [
        column for indicator_run in BUILT_IN_INDICATOR_RUNS for column in indicator_run["expected"]
    ]
    new_indicator_runs = [run for run in BUILT_IN_INDICATOR_RUNS if run["name"] in NEW_INDICATOR_NAMES]

    logger.info(
        f"Applied {len(BUILT_IN_INDICATOR_RUNS)} built-in indicators "
        f"including all {len(new_indicator_runs)} new additions."
    )
    logger.info(f"Built-in feature column count: {len(built_in_feature_columns)}")
    logger.info(
        "Last row preview:\n{}",
        df.iloc[-1][
            [
                "Close",
                "SMA_20",
                "RSI_14",
                "ROC_12",
                "PPO_line_12_26",
                "AROON_osc_25",
                "CMF_20",
                "SUPERTREND_dir_10_3.0",
            ]
        ],
    )

    # ---------------------------------------------------------
    # 3. Register a Custom Indicator on the Fly
    # ---------------------------------------------------------
    logger.info("\n--- Registering Custom Indicator ---")

    # Let's define a custom indicator: rolling close-price z-score.

    # The decorator registers it automatically!
    @IndicatorRegistry.register(name="CLOSE_ZSCORE")
    def close_zscore(df: pd.DataFrame, window: int = 20, column: str = "Close") -> pd.DataFrame:
        """
        Calculate a rolling z-score for closing prices.

        Args:
            df (pd.DataFrame): Input dataframe.
            window (int): Lookback period.
            column (str): Column to calculate on.

        Returns:
            pd.DataFrame: Dataframe with z-score column added.
        """
        result = df.copy()

        def calc_zscore(x):
            rolling_mean = x.rolling(window=window).mean()
            rolling_std = x.rolling(window=window).std()
            return (x - rolling_mean) / rolling_std

        if "Symbol" in result.columns:
            result[f"CLOSE_ZSCORE_{window}"] = result.groupby("Symbol")[column].transform(calc_zscore)
        else:
            result[f"CLOSE_ZSCORE_{window}"] = calc_zscore(result[column])

        return result

    logger.info("Successfully registered 'CLOSE_ZSCORE' indicator.")
    logger.info(f"Updated registry: {IndicatorRegistry.list_all()}")

    # ---------------------------------------------------------
    # 4. Use the Custom Indicator
    # ---------------------------------------------------------
    logger.info("Applying custom close z-score indicator (window=20)...")

    df = IndicatorRegistry.apply("CLOSE_ZSCORE", df, window=20)

    logger.info(f"Total columns after custom indicator: {len(df.columns)}")

    # verify values exist
    last_val = df.iloc[-1]["CLOSE_ZSCORE_20"]
    logger.info(f"Calculated CLOSE_ZSCORE_20 for last row: {last_val:.4f}")


if __name__ == "__main__":
    main()
