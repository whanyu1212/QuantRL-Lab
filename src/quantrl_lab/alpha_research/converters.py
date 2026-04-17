from typing import Any, Dict, List

import pandas as pd

from quantrl_lab.alpha_research.models import AlphaResult

LOWER_IS_BETTER_METRICS = {
    "ic_p_value",
    "rank_ic_p_value",
    "feature_ic_p_value",
    "feature_rank_ic_p_value",
    "signal_ic_p_value",
    "signal_rank_ic_p_value",
}


def metric_prefers_lower_values(metric: str) -> bool:
    """Return whether lower values are better for the given metric."""
    return metric in LOWER_IS_BETTER_METRICS


def metric_threshold_passes(metric_value: float, threshold: float, metric: str) -> bool:
    """Return whether a metric value passes the configured threshold."""
    if metric_value is None or pd.isna(metric_value):
        return False
    if metric_prefers_lower_values(metric):
        return metric_value <= threshold
    return metric_value >= threshold


def sort_alpha_results(results: List[AlphaResult], metric: str) -> List[AlphaResult]:
    """Sort alpha results by metric with the correct direction."""
    default_value = float("inf") if metric_prefers_lower_values(metric) else float("-inf")
    return sorted(
        results,
        key=lambda result: result.metrics.get(metric, default_value),
        reverse=not metric_prefers_lower_values(metric),
    )


def results_to_pipeline_config(
    results: List[AlphaResult], top_n: int = 5, metric: str = "ic", deduplicate: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert Alpha Research results into a DataPipeline configuration.

    Filters results to find the best performing indicators and formats them
    into the configuration structure expected by the DataPipeline's
    TechnicalIndicatorStep.

    Args:
        results (List[AlphaResult]): List of completed alpha research results.
        top_n (int): Number of top indicators to select. Defaults to 5.
        metric (str): Metric to use for ranking ("ic", "sharpe_ratio",
            "annual_return"). Defaults to "ic" (Information Coefficient).
        deduplicate (bool): When True, keep only the best-scoring result per
            indicator name. Useful when the same indicator was tested with
            multiple parameter sets (e.g., RSI window=14 and window=21) and
            you only want one entry per indicator type in the pipeline config.
            Defaults to False (all top_n results are included regardless of
            name).

    Returns:
        List[Dict[str, Any]]: A list of indicator configurations compatible
        with TechnicalIndicatorStep. Always dict format, e.g.
        ``[{"RSI": {"window": 14}}, {"SMA": {"window": 50}}, {"OBV": {}}]``.
    """
    if not results:
        return []

    # Filter for successful jobs only
    completed_jobs = [r for r in results if r.status == "completed" and r.metrics]

    if not completed_jobs:
        return []

    # Sort by metric using its correct optimization direction.
    sorted_results = sort_alpha_results(completed_jobs, metric=metric)

    if deduplicate:
        # Keep only the highest-scoring result per indicator name.
        # sorted_results is already in descending order, so the first
        # occurrence of each name is the best.
        seen: set = set()
        deduped = []
        for r in sorted_results:
            if r.job.indicator_name not in seen:
                seen.add(r.job.indicator_name)
                deduped.append(r)
        sorted_results = deduped

    # Select top N
    top_results = sorted_results[:top_n]

    pipeline_config = []
    for result in top_results:
        indicator_name = result.job.indicator_name
        params = result.job.indicator_params

        # Always use dict format so the exact research params are preserved
        # downstream by TechnicalIndicatorStep, even when params is empty.
        pipeline_config.append({indicator_name: params})

    return pipeline_config
