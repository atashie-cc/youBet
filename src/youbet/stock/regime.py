"""Regime-split and subset helpers for Phase 2 robustness analysis.

Phase 2 tests whether Phase 1 point estimates are regime-stable by
recomputing them on date-masked subsets: pre-2013 vs post-2013, excluding
the GFC window, excluding the COVID window. Regime fragility (large
differences between subsets) is often greater than the underlying factor
signal, so these masks matter for interpretation.

All functions are PURE: they take a return series (or DataFrame) and a
mask specification, return the subset. No I/O, no global state. The
orchestrator in `phase2_robustness.py` composes them with evaluate_gate
to produce per-strategy × per-regime metric tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegimeMask:
    """A named subset of a time index.

    Attributes:
        name: Label for reporting (e.g., "pre_2013", "ex_gfc").
        description: Human-readable description.
        include_mask: Boolean Series (aligned to the series being subset)
            — True for dates to KEEP. Constructed via one of the helpers
            below.
    """

    name: str
    description: str
    include_mask: pd.Series


def _ts(date: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(date)


def pre_break_mask(
    index: pd.DatetimeIndex,
    break_date: str | pd.Timestamp,
) -> RegimeMask:
    """Keep dates strictly BEFORE break_date."""
    d = _ts(break_date)
    mask = pd.Series(index < d, index=index)
    return RegimeMask(
        name=f"pre_{d.year}",
        description=f"Dates strictly before {d.date()}",
        include_mask=mask,
    )


def post_break_mask(
    index: pd.DatetimeIndex,
    break_date: str | pd.Timestamp,
) -> RegimeMask:
    """Keep dates on or AFTER break_date."""
    d = _ts(break_date)
    mask = pd.Series(index >= d, index=index)
    return RegimeMask(
        name=f"post_{d.year}",
        description=f"Dates on or after {d.date()}",
        include_mask=mask,
    )


def exclude_window_mask(
    index: pd.DatetimeIndex,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    window_name: str,
) -> RegimeMask:
    """Keep all dates EXCEPT those in [start, end] (inclusive)."""
    s, e = _ts(start), _ts(end)
    if s > e:
        raise ValueError(f"Exclusion window start {s} > end {e}")
    mask = pd.Series(~((index >= s) & (index <= e)), index=index)
    return RegimeMask(
        name=f"ex_{window_name}",
        description=f"All dates except [{s.date()}..{e.date()}]",
        include_mask=mask,
    )


def full_mask(index: pd.DatetimeIndex) -> RegimeMask:
    """No-op mask: every date kept. Useful as baseline for comparison."""
    return RegimeMask(
        name="full",
        description="Full sample (no exclusions)",
        include_mask=pd.Series(True, index=index),
    )


def apply_mask(series: pd.Series, mask: RegimeMask) -> pd.Series:
    """Return the subset of `series` where mask.include_mask is True.

    The mask index must match `series.index`. Dates present in the mask
    but absent from series (or vice versa) are silently intersected;
    caller should pre-align if exact control is required.
    """
    if len(series) == 0:
        return series
    aligned = mask.include_mask.reindex(series.index, fill_value=False)
    return series[aligned]


def describe_subset(series: pd.Series, mask: RegimeMask) -> dict:
    """Report coverage of a mask on a series (for audit)."""
    sub = apply_mask(series, mask)
    n_full = int(len(series))
    n_sub = int(len(sub))
    if n_full == 0:
        return {
            "name": mask.name, "description": mask.description,
            "n_full": 0, "n_kept": 0, "fraction_kept": 0.0,
            "years_kept": 0.0,
            "first_date": None, "last_date": None,
        }
    return {
        "name": mask.name,
        "description": mask.description,
        "n_full": n_full,
        "n_kept": n_sub,
        "fraction_kept": n_sub / n_full,
        "years_kept": n_sub / 252.0,
        "first_date": str(sub.index.min().date()) if n_sub > 0 else None,
        "last_date": str(sub.index.max().date()) if n_sub > 0 else None,
    }


def standard_regime_set(
    index: pd.DatetimeIndex,
    pre_post_break: str | pd.Timestamp,
    exclude_windows: dict[str, tuple[str, str]],
) -> list[RegimeMask]:
    """Build the standard Phase 2 regime set from config.

    Returns masks: full, pre_<year>, post_<year>, ex_<each exclude_windows key>.
    """
    masks = [
        full_mask(index),
        pre_break_mask(index, pre_post_break),
        post_break_mask(index, pre_post_break),
    ]
    for key, (start, end) in exclude_windows.items():
        masks.append(exclude_window_mask(index, start, end, window_name=key))
    return masks


def apply_masks_to_pair(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    masks: Iterable[RegimeMask],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """Apply each mask to both series, preserving alignment.

    Returns {mask_name: (strat_subset, bench_subset)}.
    """
    common = strat_returns.index.intersection(bench_returns.index)
    s = strat_returns.loc[common]
    b = bench_returns.loc[common]
    out: dict[str, tuple[pd.Series, pd.Series]] = {}
    for m in masks:
        out[m.name] = (apply_mask(s, m), apply_mask(b, m))
    return out
