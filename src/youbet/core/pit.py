"""Point-in-time validation checks.

Structural guards against the most common leakage patterns found across
NBA, MLB, and MMA workflows:
1. Train/test temporal overlap
2. Transforms fitted on future data
3. Features containing future information

These checks are designed to fail fast and loudly rather than silently
producing optimistic results.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PITViolation(Exception):
    """Raised when a point-in-time violation is detected."""


def validate_temporal_split(
    train_dates: pd.Series,
    test_dates: pd.Series,
    label: str = "",
) -> None:
    """Assert no training date >= the earliest test date.

    This catches the most basic leakage: training on data from the same
    time period as the test set.
    """
    train_max = pd.Timestamp(train_dates.max())
    test_min = pd.Timestamp(test_dates.min())

    if train_max >= test_min:
        raise PITViolation(
            f"Temporal overlap{f' ({label})' if label else ''}: "
            f"latest train date ({train_max.date()}) >= "
            f"earliest test date ({test_min.date()}). "
            f"Training data must be strictly before test data."
        )

    logger.debug(
        "PIT check passed%s: train max=%s, test min=%s, gap=%d days",
        f" ({label})" if label else "",
        train_max.date(), test_min.date(),
        (test_min - train_max).days,
    )


def validate_no_overlap(
    train_ids: pd.Series,
    test_ids: pd.Series,
    label: str = "",
) -> None:
    """Assert no row IDs appear in both train and test sets."""
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise PITViolation(
            f"ID overlap{f' ({label})' if label else ''}: "
            f"{len(overlap)} rows appear in both train and test. "
            f"Examples: {list(overlap)[:5]}"
        )


def validate_feature_pit(
    df: pd.DataFrame,
    date_col: str,
    feature_cols: list[str],
    group_col: str | None = None,
) -> dict[str, float]:
    """Check that feature values change over time per entity.

    For each feature, computes the fraction of entities (fighters/teams)
    where the feature varies across fights. A feature that is constant
    across all fights for an entity may be a static snapshot from the
    future (e.g., career averages computed at scraping time).

    Returns dict of {feature: fraction_of_entities_with_variation}.
    Low fractions (< 0.5) are suspicious.
    """
    if group_col is None:
        logger.warning("No group_col provided — skipping feature PIT check")
        return {}

    results = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        # For each entity, check if the feature varies across their rows
        variation = df.groupby(group_col)[col].apply(
            lambda x: x.nunique() > 1 if len(x) > 1 else True
        )
        pct_varying = float(variation.mean())
        results[col] = pct_varying

        if pct_varying < 0.5:
            logger.warning(
                "PIT WARNING: feature '%s' varies for only %.0f%% of entities. "
                "May be a static future snapshot.",
                col, pct_varying * 100,
            )

    return results


def validate_calibration_split(
    train_dates: pd.Series,
    cal_dates: pd.Series,
    test_dates: pd.Series,
    label: str = "",
) -> None:
    """Assert calibration data is disjoint from training and before test.

    Catches the pattern found in MMA Codex R1: calibrating on data the
    model was already trained on.
    """
    # Cal must not overlap with train
    train_ids = set(train_dates.index)
    cal_ids = set(cal_dates.index)
    overlap = train_ids & cal_ids
    if overlap:
        raise PITViolation(
            f"Calibration/train overlap{f' ({label})' if label else ''}: "
            f"{len(overlap)} rows in both sets. "
            f"Calibration data must be held out from training."
        )

    # Cal must be before test
    cal_max = pd.Timestamp(cal_dates.max())
    test_min = pd.Timestamp(test_dates.min())
    if cal_max >= test_min:
        raise PITViolation(
            f"Calibration after test{f' ({label})' if label else ''}: "
            f"latest cal date ({cal_max.date()}) >= "
            f"earliest test date ({test_min.date()})."
        )


def audit_fold(
    fold_name: str,
    train_dates: pd.Series,
    cal_dates: pd.Series | None,
    test_dates: pd.Series,
    n_train: int,
    n_cal: int,
    n_test: int,
) -> dict:
    """Generate audit metadata for a walk-forward fold.

    Returns a dict suitable for logging or saving as experiment metadata.
    """
    audit = {
        "fold": fold_name,
        "n_train": n_train,
        "n_cal": n_cal,
        "n_test": n_test,
        "train_date_range": f"{train_dates.min()} to {train_dates.max()}",
        "test_date_range": f"{test_dates.min()} to {test_dates.max()}",
        "train_test_gap_days": (
            pd.Timestamp(test_dates.min()) - pd.Timestamp(train_dates.max())
        ).days,
    }
    if cal_dates is not None and len(cal_dates) > 0:
        audit["cal_date_range"] = f"{cal_dates.min()} to {cal_dates.max()}"

    return audit
