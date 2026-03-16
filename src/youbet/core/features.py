"""Base feature engineering — differentials, rolling windows, and normalization."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_differentials(
    df: pd.DataFrame,
    team_a_prefix: str,
    team_b_prefix: str,
    stat_columns: list[str],
) -> pd.DataFrame:
    """Compute stat differentials (team_a - team_b) for specified columns.

    Args:
        df: DataFrame with columns prefixed by team_a_prefix and team_b_prefix.
        team_a_prefix: Prefix for team A columns (e.g., "team_a_").
        team_b_prefix: Prefix for team B columns (e.g., "team_b_").
        stat_columns: Base stat names (without prefix) to compute differentials for.

    Returns:
        DataFrame with new columns named "diff_{stat}" for each stat.
    """
    result = df.copy()
    for stat in stat_columns:
        col_a = f"{team_a_prefix}{stat}"
        col_b = f"{team_b_prefix}{stat}"
        if col_a in df.columns and col_b in df.columns:
            result[f"diff_{stat}"] = df[col_a] - df[col_b]
        else:
            logger.warning("Missing columns for differential: %s, %s", col_a, col_b)
    return result


def rolling_stats(
    df: pd.DataFrame,
    group_col: str,
    stat_columns: list[str],
    windows: list[int],
    sort_col: str = "date",
) -> pd.DataFrame:
    """Compute rolling averages per group (team) for specified stats and windows.

    Args:
        df: DataFrame sorted by date with team stats per game.
        group_col: Column to group by (e.g., "team_id").
        stat_columns: Columns to compute rolling averages for.
        windows: List of window sizes (e.g., [5, 10]).
        sort_col: Column to sort by before computing rolling stats.

    Returns:
        DataFrame with new columns named "{stat}_last_{window}" for each combination.
    """
    result = df.sort_values(sort_col).copy()
    for stat in stat_columns:
        for window in windows:
            col_name = f"{stat}_last_{window}"
            result[col_name] = (
                result.groupby(group_col)[stat]
                .transform(lambda x: x.rolling(window, min_periods=1).mean().shift(1))
            )
    return result


def normalize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, dict]:
    """Normalize features using standard scaling or min-max.

    Args:
        df: DataFrame with features.
        feature_columns: Columns to normalize.
        method: "standard" (z-score) or "minmax".

    Returns:
        Tuple of (normalized DataFrame, params dict for inverse transform).
    """
    result = df.copy()
    params = {}
    for col in feature_columns:
        if col not in df.columns:
            continue
        if method == "standard":
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                std = 1.0
            result[col] = (df[col] - mean) / std
            params[col] = {"mean": mean, "std": std}
        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            if range_val == 0:
                range_val = 1.0
            result[col] = (df[col] - min_val) / range_val
            params[col] = {"min": min_val, "max": max_val}
    return result, params


def select_top_features(
    importances: dict[str, float],
    n: int = 20,
) -> list[str]:
    """Select top N features by importance score."""
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_features[:n]]
