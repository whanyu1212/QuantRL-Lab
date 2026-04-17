"""Technical feature generator."""

import itertools
import warnings
from typing import Any, Dict, List, Union

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

    def __init__(self, indicators: List[Union[str, Dict]], strict: bool = False):
        """
        Initialize TechnicalFeatureGenerator.

        Args:
            indicators (List[Union[str, Dict]]): List of indicators to apply.
                Can be strings (use defaults) or dicts with custom parameters.
                Examples:
                    - ["SMA", "RSI"]
                    - [{"SMA": {"window": 20}}, {"RSI": {"window": 14}}]
                    - ["SMA", {"MACD": {"fast": 12, "slow": 26}}]
            strict (bool): If True, fail fast when an indicator is unknown or
                cannot be applied.
        """
        if not indicators:
            raise ValueError("indicators list cannot be empty")

        self.indicators = indicators
        self.strict = strict
        self.registry = IndicatorRegistry
        self._last_run_report: Dict[str, Any] = {}
        self._seen_report_keys: set[str] = set()

    def _reset_run_report(self) -> None:
        """Initialize per-run indicator execution metadata."""
        self._last_run_report = {
            "requested": self.indicators.copy(),
            "applied": [],
            "skipped": [],
            "failed": [],
            "strict": self.strict,
        }
        self._seen_report_keys = set()

    def _report_once(
        self, bucket: str, indicator_config: Union[str, Dict], indicator_name: str, **details: Any
    ) -> None:
        """Record a run event once even when panel data is processed per
        symbol."""
        key = (bucket, repr(indicator_config), repr(sorted(details.items())))
        if repr(key) in self._seen_report_keys:
            return

        self._seen_report_keys.add(repr(key))
        entry = {
            "indicator": indicator_name,
            "config": indicator_config,
            **details,
        }
        self._last_run_report[bucket].append(entry)

    def _record_applied(self, indicator_config: Union[str, Dict], indicator_name: str) -> None:
        """Track successfully applied indicators once per run."""
        key = ("applied", repr(indicator_config))
        if repr(key) in self._seen_report_keys:
            return

        self._seen_report_keys.add(repr(key))
        self._last_run_report["applied"].append(indicator_config)

    def _handle_skipped(self, indicator_config: Union[str, Dict], indicator_name: str, reason: str) -> None:
        """Handle unknown or malformed indicators according to
        strictness."""
        self._report_once("skipped", indicator_config, indicator_name, reason=reason)
        message = f"Skipping indicator '{indicator_name}': {reason}"
        if self.strict:
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)

    def _handle_failure(self, indicator_config: Union[str, Dict], indicator_name: str, error: Exception) -> None:
        """Handle indicator execution failures according to
        strictness."""
        error_message = str(error)
        self._report_once("failed", indicator_config, indicator_name, error=error_message)
        message = f"Failed to apply indicator '{indicator_name}': {error_message}"
        if self.strict:
            raise ValueError(message) from error
        warnings.warn(message, UserWarning, stacklevel=3)

    @staticmethod
    def _expand_parameter_sets(custom_params: Any) -> List[Dict[str, Any]]:
        """Expand list/grid-style indicator params into concrete
        dictionaries."""
        if isinstance(custom_params, list):
            return [param_set for param_set in custom_params if isinstance(param_set, dict)]

        if isinstance(custom_params, dict) and any(isinstance(v, list) for v in custom_params.values()):
            param_names = list(custom_params.keys())
            param_values = [value if isinstance(value, list) else [value] for value in custom_params.values()]
            return [dict(zip(param_names, combination)) for combination in itertools.product(*param_values)]

        if isinstance(custom_params, dict):
            return [custom_params]

        return [{}]

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
                    self._handle_skipped(indicator_config, "<invalid>", "indicator config must contain exactly one key")
                    continue
                indicator_name = list(indicator_config.keys())[0]
                custom_params = indicator_config[indicator_name]
            else:
                self._handle_skipped(
                    indicator_config, "<invalid>", "indicator config must be a string or single-key dict"
                )
                continue

            if indicator_name not in available_indicators:
                self._handle_skipped(indicator_config, indicator_name, "not registered in IndicatorRegistry")
                continue

            try:
                for params_dict in self._expand_parameter_sets(custom_params):
                    result = self.registry.apply(indicator_name, result, **params_dict)
                self._record_applied(indicator_config, indicator_name)
            except Exception as e:
                self._handle_failure(indicator_config, indicator_name, e)
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
        self._reset_run_report()

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
            "strict": self.strict,
            "last_run": {
                "requested": self._last_run_report.get("requested", []).copy(),
                "applied": self._last_run_report.get("applied", []).copy(),
                "skipped": self._last_run_report.get("skipped", []).copy(),
                "failed": self._last_run_report.get("failed", []).copy(),
            },
        }
