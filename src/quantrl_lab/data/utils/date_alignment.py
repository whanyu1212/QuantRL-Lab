"""Helpers for aligning sparse feature data to market data
timestamps."""

from typing import Iterable, List, Optional, Tuple

import pandas as pd

DATE_COLUMN_CANDIDATES = ("Timestamp", "Date", "date")


def to_naive_datetime(series: pd.Series) -> pd.Series:
    """Convert arbitrary date-like values to timezone-naive
    datetimes."""
    dt_series = pd.to_datetime(series)
    if getattr(dt_series.dt, "tz", None) is not None:
        dt_series = dt_series.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt_series


def merge_asof_features(
    base_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    *,
    feature_date_column: str,
    drop_columns: Iterable[str] = (),
    prefix: str = "",
    base_by_columns: Iterable[str] = (),
    feature_by_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Backward-align sparse feature data to the base market-data timeline.

    Returns the merged DataFrame plus the list of feature columns that
    were added.
    """
    if feature_df is None or feature_df.empty:
        return base_df.copy(), []

    base_working, base_date_column, restore_index = _prepare_base_frame(base_df)
    if feature_date_column not in feature_df.columns:
        return base_df.copy(), []

    base_group_columns = list(base_by_columns)
    feature_group_columns = list(feature_by_columns) if feature_by_columns is not None else list(base_group_columns)
    if len(base_group_columns) != len(feature_group_columns):
        raise ValueError("base_by_columns and feature_by_columns must have the same length")

    for column in base_group_columns:
        if column not in base_working.columns:
            return base_df.copy(), []
    for column in feature_group_columns:
        if column not in feature_df.columns:
            return base_df.copy(), []

    feature_working = feature_df.copy()
    feature_working["__merge_date__"] = to_naive_datetime(feature_working[feature_date_column])

    excluded = set(drop_columns)
    excluded.add(feature_date_column)
    excluded.update(feature_group_columns)
    feature_columns = [col for col in feature_working.columns if col not in excluded and col != "__merge_date__"]
    if not feature_columns:
        return base_df.copy(), []

    rename_map = {col: f"{prefix}{col}" for col in feature_columns}
    group_rename_map = {
        source: target for source, target in zip(feature_group_columns, base_group_columns) if source != target
    }
    payload = feature_working[["__merge_date__", *feature_group_columns, *feature_columns]].rename(
        columns={**rename_map, **group_rename_map}
    )
    sort_columns = [*base_group_columns, "__merge_date__"] if base_group_columns else ["__merge_date__"]
    payload = payload.sort_values(sort_columns).reset_index(drop=True)

    base_working["__merge_date__"] = to_naive_datetime(base_working[base_date_column])
    base_working["__row_order__"] = range(len(base_working))
    base_sorted = base_working.sort_values(sort_columns).reset_index(drop=True)

    merge_kwargs = {"on": "__merge_date__", "direction": "backward"}
    if base_group_columns:
        merge_kwargs["by"] = base_group_columns

    merged = pd.merge_asof(base_sorted, payload, **merge_kwargs)
    merged = merged.sort_values("__row_order__").drop(columns=["__merge_date__", "__row_order__"])

    if restore_index:
        merged = merged.set_index(base_date_column)
        merged.index.name = base_df.index.name

    return merged, list(rename_map.values())


def _prepare_base_frame(base_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, bool]:
    """Prepare the base DataFrame for date-based alignment."""
    if isinstance(base_df.index, pd.DatetimeIndex):
        base_working = base_df.reset_index()
        base_date_column = base_df.index.name or "index"
        if base_working.columns[0] != base_date_column:
            base_working = base_working.rename(columns={base_working.columns[0]: base_date_column})
        return base_working, base_date_column, True

    for column in DATE_COLUMN_CANDIDATES:
        if column in base_df.columns:
            return base_df.copy(), column, False

    raise ValueError(
        "DataFrame must contain a DatetimeIndex or one of the date columns: " f"{', '.join(DATE_COLUMN_CANDIDATES)}"
    )
