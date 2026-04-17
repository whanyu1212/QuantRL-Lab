"""Research metadata for technical indicators used by alpha
selection."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from quantrl_lab.data.indicators import IndicatorRegistry

FEATURE_SELECTION_MODE = "feature"
STRATEGY_SELECTION_MODE = "strategy"
SUPPORTED_SELECTION_MODES = (FEATURE_SELECTION_MODE, STRATEGY_SELECTION_MODE)


@dataclass
class IndicatorResearchMetadata:
    """
    Metadata describing how an indicator participates in alpha research.

    Attributes:
        name: Indicator registry key.
        strategy_name: Strategy registry key used to evaluate this indicator.
        strategy_params: Static constructor kwargs for the strategy.
        default_candidates: Default parameter grid for selector sweeps.
        supported_modes: Supported selection modes (feature and/or strategy).
        notes: Short rationale or caveat for research usage.
    """

    name: str
    strategy_name: Optional[str] = None
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    default_candidates: List[Dict[str, Any]] = field(default_factory=lambda: [{}])
    supported_modes: Tuple[str, ...] = SUPPORTED_SELECTION_MODES
    notes: str = ""

    def supports_mode(self, selection_mode: str) -> bool:
        """Return whether this indicator is supported for the requested
        mode."""
        return self.strategy_name is not None and selection_mode in self.supported_modes


def _meta(
    name: str,
    strategy_name: str,
    *,
    strategy_params: Optional[Dict[str, Any]] = None,
    default_candidates: Optional[List[Dict[str, Any]]] = None,
    supported_modes: Tuple[str, ...] = SUPPORTED_SELECTION_MODES,
    notes: str = "",
) -> IndicatorResearchMetadata:
    """Construct indicator research metadata with concise defaults."""
    return IndicatorResearchMetadata(
        name=name,
        strategy_name=strategy_name,
        strategy_params=strategy_params or {},
        default_candidates=default_candidates or [{}],
        supported_modes=supported_modes,
        notes=notes,
    )


INDICATOR_RESEARCH_METADATA: Dict[str, IndicatorResearchMetadata] = {
    "SMA": _meta(
        "SMA",
        "trend_following",
        default_candidates=[{"window": 20}, {"window": 50}, {"window": 200}],
        notes="Price-relative trend filter.",
    ),
    "EMA": _meta(
        "EMA",
        "trend_following",
        default_candidates=[{"window": 12}, {"window": 26}, {"window": 50}],
        notes="Faster trend filter than SMA.",
    ),
    "RSI": _meta(
        "RSI",
        "mean_reversion",
        strategy_params={"oversold": 30, "overbought": 70},
        default_candidates=[{"window": 7}, {"window": 14}, {"window": 21}],
    ),
    "MACD": _meta(
        "MACD",
        "crossover",
        strategy_params={"score_normalization": "close_pct"},
        default_candidates=[{"fast": 12, "slow": 26, "signal": 9}, {"fast": 5, "slow": 35, "signal": 5}],
    ),
    "ATR": _meta(
        "ATR",
        "volatility_breakout",
        default_candidates=[{"window": 14}],
        notes="Volatility regime screen rather than direct directional feature.",
    ),
    "BB": _meta(
        "BB",
        "bollinger_bands",
        default_candidates=[{"window": 20, "num_std": 2.0}],
    ),
    "STOCH": _meta(
        "STOCH",
        "stochastic",
        default_candidates=[{"k_window": 14, "d_window": 3}],
    ),
    "OBV": _meta(
        "OBV",
        "flow_trend",
        strategy_params={"ma_window": 20},
        default_candidates=[{}],
    ),
    "WILLR": _meta(
        "WILLR",
        "mean_reversion",
        strategy_params={"oversold": -80, "overbought": -20, "indicator_scale": "williams_r"},
        default_candidates=[{"window": 14}],
    ),
    "CCI": _meta(
        "CCI",
        "cci_reversal",
        strategy_params={"threshold": 100.0},
        default_candidates=[{"window": 20}],
    ),
    "MFI": _meta(
        "MFI",
        "mean_reversion",
        strategy_params={"oversold": 20, "overbought": 80},
        default_candidates=[{"window": 14}],
    ),
    "ADX": _meta(
        "ADX",
        "adx_trend",
        strategy_params={"threshold": 25.0},
        default_candidates=[{"window": 14}],
    ),
    "ROC": _meta(
        "ROC",
        "zero_line",
        default_candidates=[{"window": 12}, {"window": 20}],
    ),
    "PPO": _meta(
        "PPO",
        "crossover",
        strategy_params={"score_normalization": "zscore"},
        default_candidates=[{"fast": 12, "slow": 26, "signal": 9}],
    ),
    "TRIX": _meta(
        "TRIX",
        "crossover",
        strategy_params={"score_normalization": "zscore"},
        default_candidates=[{"window": 15, "signal": 9}, {"window": 30, "signal": 9}],
    ),
    "TSI": _meta(
        "TSI",
        "crossover",
        strategy_params={"score_normalization": "zscore"},
        default_candidates=[{"slow": 25, "fast": 13, "signal": 13}],
    ),
    "AROON": _meta(
        "AROON",
        "zero_line",
        strategy_params={"score_scale": 100.0},
        default_candidates=[{"window": 25}],
    ),
    "VORTEX": _meta(
        "VORTEX",
        "crossover",
        strategy_params={"score_normalization": "zscore"},
        default_candidates=[{"window": 14}],
    ),
    "DONCHIAN": _meta(
        "DONCHIAN",
        "channel_breakout",
        default_candidates=[{"window": 20}, {"window": 55}],
    ),
    "KELTNER": _meta(
        "KELTNER",
        "channel_breakout",
        default_candidates=[{"window": 20, "atr_mult": 2.0}],
    ),
    "NATR": _meta(
        "NATR",
        "volatility_breakout",
        default_candidates=[{"window": 14}],
        notes="Normalized volatility regime screen.",
    ),
    "CMF": _meta(
        "CMF",
        "zero_line",
        strategy_params={"score_scale": 0.5},
        default_candidates=[{"window": 20}],
    ),
    "ADL": _meta(
        "ADL",
        "flow_trend",
        strategy_params={"ma_window": 20},
        default_candidates=[{}],
    ),
    "CHO": _meta(
        "CHO",
        "zero_line",
        default_candidates=[{"fast": 3, "slow": 10}],
    ),
    "SUPERTREND": _meta(
        "SUPERTREND",
        "trend_following",
        default_candidates=[{"window": 10, "multiplier": 3.0}],
    ),
}


def validate_selection_mode(selection_mode: str) -> str:
    """Validate and normalize a selection mode string."""
    if selection_mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(
            f"Unknown selection_mode: {selection_mode}. " f"Supported modes: {list(SUPPORTED_SELECTION_MODES)}"
        )
    return selection_mode


def get_indicator_research_metadata(indicator_name: str) -> IndicatorResearchMetadata:
    """Return research metadata for one indicator."""
    name = indicator_name.upper()
    if name not in INDICATOR_RESEARCH_METADATA:
        raise KeyError(f"Indicator '{indicator_name}' does not have research metadata.")
    return INDICATOR_RESEARCH_METADATA[name]


def list_supported_indicators(selection_mode: Optional[str] = None) -> List[str]:
    """List indicators supported by alpha research, optionally by
    mode."""
    if selection_mode is None:
        return list(INDICATOR_RESEARCH_METADATA.keys())

    validated_mode = validate_selection_mode(selection_mode)
    return [name for name, metadata in INDICATOR_RESEARCH_METADATA.items() if metadata.supports_mode(validated_mode)]


def get_default_candidates(selection_mode: str = FEATURE_SELECTION_MODE) -> List[Dict[str, Any]]:
    """Return the default candidate sweep for the requested selection
    mode."""
    validated_mode = validate_selection_mode(selection_mode)
    candidates: List[Dict[str, Any]] = []
    for name, metadata in INDICATOR_RESEARCH_METADATA.items():
        if not metadata.supports_mode(validated_mode):
            continue
        for params in metadata.default_candidates:
            candidates.append({"name": name, "params": deepcopy(params)})
    return candidates


def get_strategy_config(indicator_name: str, selection_mode: str = FEATURE_SELECTION_MODE) -> Optional[Dict[str, Any]]:
    """Return strategy wiring for one indicator in the requested
    mode."""
    validated_mode = validate_selection_mode(selection_mode)
    metadata = get_indicator_research_metadata(indicator_name)
    if not metadata.supports_mode(validated_mode):
        return None
    return {"name": metadata.strategy_name, "params": deepcopy(metadata.strategy_params)}


def _validate_registry_coverage():
    """Ensure every registered indicator has explicit research
    metadata."""
    registered = set(IndicatorRegistry.list_all())
    configured = set(INDICATOR_RESEARCH_METADATA.keys())
    missing = registered - configured
    extra = configured - registered
    if missing or extra:
        raise RuntimeError(
            "Indicator research metadata is out of sync with IndicatorRegistry. "
            f"Missing: {sorted(missing)} Extra: {sorted(extra)}"
        )


_validate_registry_coverage()


# Compatibility view for older imports. Kept in the alpha layer, but now
# derived from explicit research metadata rather than a free-floating map.
INDICATOR_STRATEGY_MAP: Dict[str, Dict[str, Any]] = {
    name: {"name": metadata.strategy_name, "params": deepcopy(metadata.strategy_params)}
    for name, metadata in INDICATOR_RESEARCH_METADATA.items()
    if metadata.strategy_name is not None
}
