import json
import uuid
from typing import Dict, List, Optional

import pandas as pd
from rich.console import Console

from quantrl_lab.alpha_research.converters import (
    metric_threshold_passes,
    results_to_pipeline_config,
    sort_alpha_results,
)
from quantrl_lab.alpha_research.indicator_research import (
    FEATURE_SELECTION_MODE,
)
from quantrl_lab.alpha_research.indicator_research import get_default_candidates as get_research_default_candidates
from quantrl_lab.alpha_research.indicator_research import get_strategy_config as get_research_strategy_config
from quantrl_lab.alpha_research.indicator_research import (
    validate_selection_mode,
)
from quantrl_lab.alpha_research.models import AlphaJob
from quantrl_lab.alpha_research.runner import AlphaRunner

console = Console()

FEATURE_MODE_METRICS = {
    "ic",
    "rank_ic",
    "ic_p_value",
    "rank_ic_p_value",
    "mutual_info",
    "feature_ic",
    "feature_rank_ic",
    "feature_ic_p_value",
    "feature_rank_ic_p_value",
    "feature_mutual_info",
    "score_mean",
    "score_std",
    "score_abs_mean",
}
STRATEGY_MODE_METRICS = FEATURE_MODE_METRICS | {
    "signal_ic",
    "signal_rank_ic",
    "signal_ic_p_value",
    "signal_rank_ic_p_value",
    "signal_mutual_info",
    "total_return",
    "annual_return",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "volatility",
    "win_rate",
    "win_loss_ratio",
    "profit_factor",
    "turnover",
}


class AlphaSelector:
    """Selects the best alpha factors (indicators) for a given
    dataset."""

    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        self.data = data
        self.verbose = verbose
        self.runner = AlphaRunner(verbose=False)  # Run quiet, we control output

    def suggest_indicators(
        self,
        candidates: Optional[List[Dict]] = None,
        metric: str = "ic",
        threshold: float = 0.0,
        top_k: int = 5,
        selection_mode: str = FEATURE_SELECTION_MODE,
    ) -> List[Dict]:
        """
        Test a set of candidate indicators and return the best
        performing ones.

        Args:
            candidates: List of indicator configs to test. If None, uses a default grid.
                        Format: [{"name": "RSI", "params": {"window": 14}}]
            metric: Metric to optimize. Defaults to "ic".
            threshold: Minimum acceptable value for higher-is-better metrics,
                or maximum acceptable value for p-value metrics.
            top_k: Number of top indicators to return.
            selection_mode: ``"feature"`` ranks indicators by continuous score
                predictive power. ``"strategy"`` ranks indicators by discrete
                trading-rule behavior and backtest metrics.

        Returns:
            List of indicator configurations ready for DataProcessor.
        """
        validated_mode = validate_selection_mode(selection_mode)
        self._validate_metric(metric, validated_mode)

        if candidates is None:
            candidates = self._get_default_candidates(validated_mode)

        if self.verbose:
            console.print(
                f"[cyan]Evaluating {len(candidates)} candidate indicators " f"in {validated_mode} mode...[/cyan]"
            )

        jobs = []
        for cand in candidates:
            strategy_config = self._get_strategy_config(cand, selection_mode=validated_mode)
            if not strategy_config:
                if self.verbose:
                    console.print(f"[yellow]Skipping {cand['name']}: No strategy mapping found.[/yellow]")
                continue

            job = AlphaJob(
                data=self.data,
                indicator_name=cand["name"],
                indicator_params=cand.get("params", {}),
                strategy_name=strategy_config["name"],
                strategy_params=strategy_config["params"],
                evaluation_mode=validated_mode,
                id=f"job_{cand['name']}_{uuid.uuid4().hex[:4]}",
                tags={"selection_mode": validated_mode},
            )
            jobs.append(job)

        results = self.runner.run_batch(jobs)

        # Filter by threshold before passing to converter.
        above_threshold = [r for r in results if metric_threshold_passes(r.metrics.get(metric), threshold, metric)]
        ranked_results = sort_alpha_results(above_threshold, metric=metric)

        if self.verbose:
            console.print(f"[green]Found {len(above_threshold)} indicators passing threshold {threshold}[/green]")
            for r in ranked_results[:top_k]:
                console.print(
                    f"  - {r.job.indicator_name} ({r.job.indicator_params}): {metric}={r.metrics.get(metric, 0):.4f}"
                )

        return results_to_pipeline_config(ranked_results, top_n=top_k, metric=metric)

    @classmethod
    def suggest_for_universe(
        cls,
        raw_data: Dict[str, pd.DataFrame],
        candidates: Optional[List[Dict]] = None,
        metric: str = "ic",
        threshold: float = 0.0,
        top_k: int = 5,
        verbose: bool = False,
        selection_mode: str = FEATURE_SELECTION_MODE,
    ) -> List[Dict]:
        """
        Select indicators across multiple symbols and return the
        deduplicated union of suggested indicators.

        Runs ``suggest_indicators`` on each symbol's raw OHLCV DataFrame
        and deduplicates results by JSON-serialised indicator spec.  The
        union approach ensures that any alpha-generating signal found for
        any individual symbol is included in the shared feature set.

        Args:
            raw_data: Mapping of ``{symbol: ohlcv_df}`` (raw, unprocessed).
            candidates: Candidate indicator configs to test. If None, uses the default grid.
            metric: Scoring metric (e.g. ``"ic"``).
            threshold: Minimum acceptable value for higher-is-better metrics,
                or maximum acceptable value for p-value metrics.
            top_k: Maximum indicators to pick *per symbol* before deduplication.
            verbose: Whether to print per-symbol progress.
            selection_mode: ``"feature"`` or ``"strategy"``.

        Returns:
            Deduplicated list of indicator spec dicts ready for ``DataProcessor``.
        """
        all_suggested: List[Dict] = []
        for symbol, df in raw_data.items():
            selector = cls(df, verbose=verbose)
            indicators = selector.suggest_indicators(
                candidates=candidates,
                metric=metric,
                threshold=threshold,
                top_k=top_k,
                selection_mode=selection_mode,
            )
            if verbose:
                console.print(f"  [cyan]{symbol}[/cyan]: {indicators}")
            all_suggested.extend(indicators)

        # Deduplicate while preserving insertion order
        seen: set = set()
        unique_indicators: List[Dict] = []
        for indicator in all_suggested:
            key = json.dumps(indicator, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique_indicators.append(indicator)

        console.print(
            f"[green]Alpha selection: {len(unique_indicators)} unique indicators "
            f"across {len(raw_data)} symbol(s).[/green]"
        )
        return unique_indicators

    def _get_default_candidates(self, selection_mode: str = FEATURE_SELECTION_MODE) -> List[Dict]:
        """Return a default grid covering all indicators that have a
        strategy mapping in the research metadata."""
        return get_research_default_candidates(selection_mode)

    def _get_strategy_config(self, indicator: Dict, selection_mode: str = FEATURE_SELECTION_MODE) -> Optional[Dict]:
        """
        Look up the strategy config for an indicator from research
        metadata.

        Returns None if no mapping exists for the mode.
        """
        name = indicator.get("name", "").upper()
        return get_research_strategy_config(name, selection_mode=selection_mode)

    @staticmethod
    def _validate_metric(metric: str, selection_mode: str) -> None:
        """Reject metrics that do not make sense for the requested
        mode."""
        allowed_metrics = FEATURE_MODE_METRICS if selection_mode == FEATURE_SELECTION_MODE else STRATEGY_MODE_METRICS
        if metric not in allowed_metrics:
            raise ValueError(
                f"Metric '{metric}' is not supported for selection_mode='{selection_mode}'. "
                f"Supported metrics: {sorted(allowed_metrics)}"
            )
