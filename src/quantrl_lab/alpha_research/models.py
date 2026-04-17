import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd

from .indicator_research import STRATEGY_SELECTION_MODE, validate_selection_mode


@dataclass
class AlphaJob:
    """Defines a single alpha research task."""

    data: pd.DataFrame  # The data to test on
    indicator_name: str  # e.g., "RSI" (for calculation)
    strategy_name: str  # e.g., "mean_reversion" (for logic)
    indicator_params: Dict[str, Any] = field(default_factory=dict)  # e.g. {"window": 14}
    strategy_params: Dict[str, Any] = field(default_factory=dict)  # e.g. {"oversold": 30}
    allow_short: bool = True
    evaluation_mode: str = STRATEGY_SELECTION_MODE

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.indicator_name or not self.indicator_name.strip():
            raise ValueError("indicator_name must be a non-empty string")
        if not self.strategy_name or not self.strategy_name.strip():
            raise ValueError("strategy_name must be a non-empty string")
        self.evaluation_mode = validate_selection_mode(self.evaluation_mode)


@dataclass
class AlphaResult:
    """Results of an alpha research job."""

    job: AlphaJob
    metrics: Dict[str, Any]
    equity_curve: Optional[pd.Series] = None
    signals: Optional[pd.Series] = None
    scores: Optional[pd.Series] = None
    status: str = "completed"
    error: Optional[str] = None
