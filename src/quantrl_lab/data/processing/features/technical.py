"""Technical feature generator."""

from typing import Dict, List, Union

import pandas as pd

from quantrl_lab.data.indicators.registry import IndicatorRegistry


class TechnicalFeatureGenerator:
    """
    Generator for adding technical indicators to OHLCV data.

    This generator delegates to the IndicatorRegistry to apply technical
    indicators like SMA, RSI, MACD, etc.

    Example:
        >>> from quantrl_lab.data.processing.features.technical import TechnicalFeatureGenerator
        >>> generator = TechnicalFeatureGenerator(["SMA", "RSI", {"MACD": {"fast": 12}}])
        >>> enriched_df = generator.generate(ohlcv_df)
    """

    def __init__(self, indicators: List[Union[str, Dict]]):
        """
        Initialize TechnicalFeatureGenerator.

        Args:
            indicators (List[Union[str, Dict]]): List of indicators to apply.
                Can be strings (use defaults) or dicts with custom parameters.
                Examples:
                    - ["SMA", "RSI"]
                    - [{"SMA": {"window": 20}}, {"RSI": {"window": 14}}]
                    - ["SMA", {"MACD": {"fast": 12, "slow": 26}}]
        """
        if not indicators:
            raise ValueError("indicators list cannot be empty")

        self.indicators = indicators
        self.registry = IndicatorRegistry

    def _generate_single(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Internal method to generate indicators for a single asset."""
        result = data.copy()
        available_indicators = set(self.registry.list_all())

        for indicator_config in self.indicators:
            # Handle both string and dictionary formats
            if isinstance(indicator_config, str):
                indicator_name = indicator_config
                custom_params = kwargs.get(f"{indicator_name}_params", {})
            elif isinstance(indicator_config, dict):
                if len(indicator_config) != 1:
                    continue
                indicator_name = list(indicator_config.keys())[0]
                custom_params = indicator_config[indicator_name]
            else:
                continue

            if indicator_name not in available_indicators:
                import warnings

                warnings.warn(
                    f"Unknown indicator '{indicator_name}' — not registered in IndicatorRegistry. Skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            try:
                if isinstance(custom_params, list):
                    for param_set in custom_params:
                        if isinstance(param_set, dict):
                            result = self.registry.apply(indicator_name, result, **param_set)
                elif isinstance(custom_params, dict) and any(isinstance(v, list) for v in custom_params.values()):
                    import itertools

                    param_names = list(custom_params.keys())
                    param_values = list(custom_params.values())
                    param_values = [v if isinstance(v, list) else [v] for v in param_values]

                    for combination in itertools.product(*param_values):
                        params_dict = dict(zip(param_names, combination))
                        result = self.registry.apply(indicator_name, result, **params_dict)
                elif isinstance(custom_params, dict):
                    result = self.registry.apply(indicator_name, result, **custom_params)
                else:
                    result = self.registry.apply(indicator_name, result)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to apply indicator '{indicator_name}': {e}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

        return result

    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate DataFrame with technical indicators. Automatically
        handles panel data by grouping by Symbol to prevent time-series
        crossover between different assets.

        Args:
            data (pd.DataFrame): Input OHLCV DataFrame.
            **kwargs: Additional parameters passed to indicator functions.

        Returns:
            pd.DataFrame: DataFrame with technical indicators added.

        Raises:
            ValueError: If data is empty or missing required columns.
        """
        if data.empty:
            raise ValueError("Input DataFrame is empty. Technical indicators cannot be added.")

        # Check for required columns (case-insensitive)
        column_check = {col.lower(): col for col in data.columns}
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = []

        for req_col in required_cols:
            if req_col not in column_check and req_col.upper() not in column_check:
                missing_cols.append(req_col)

        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {', '.join(missing_cols)}")

        # If panel data (multiple symbols), group by Symbol before calculating
        # rolling indicators to prevent data bleeding across assets.
        if "Symbol" in data.columns and len(data["Symbol"].unique()) > 1:
            # apply() on groupby might alter the index or order depending on pandas version.
            # We sort the final result to maintain chronological index order.
            result = data.groupby("Symbol", group_keys=False).apply(
                lambda df_group: self._generate_single(df_group, **kwargs)
            )
            return result.sort_index()
        else:
            return self._generate_single(data, **kwargs)

    def get_metadata(self) -> Dict:
        """
        Return metadata about technical indicators applied.

        Returns:
            Dict: Dictionary containing indicator information.
        """
        return {
            "type": "technical_indicators",
            "indicators": self.indicators.copy() if isinstance(self.indicators, list) else [self.indicators],
        }
