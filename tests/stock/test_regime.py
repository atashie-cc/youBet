"""Regime mask correctness (Phase 2 infrastructure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from youbet.stock.regime import (
    RegimeMask,
    apply_mask,
    apply_masks_to_pair,
    describe_subset,
    exclude_window_mask,
    full_mask,
    post_break_mask,
    pre_break_mask,
    standard_regime_set,
)


@pytest.fixture
def daily_series() -> pd.Series:
    idx = pd.bdate_range("2005-01-01", "2024-12-31")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def test_pre_break_strict_less(daily_series):
    mask = pre_break_mask(daily_series.index, "2013-01-01")
    sub = apply_mask(daily_series, mask)
    assert sub.index.max() < pd.Timestamp("2013-01-01")
    assert pd.Timestamp("2012-12-31") in sub.index or pd.Timestamp("2012-12-28") in sub.index


def test_post_break_inclusive(daily_series):
    mask = post_break_mask(daily_series.index, "2013-01-01")
    sub = apply_mask(daily_series, mask)
    assert sub.index.min() >= pd.Timestamp("2013-01-01")


def test_pre_post_partition_is_complete(daily_series):
    pre = apply_mask(daily_series, pre_break_mask(daily_series.index, "2013-01-01"))
    post = apply_mask(daily_series, post_break_mask(daily_series.index, "2013-01-01"))
    assert len(pre) + len(post) == len(daily_series)
    assert pre.index.intersection(post.index).empty


def test_exclude_window_inclusive_bounds(daily_series):
    mask = exclude_window_mask(
        daily_series.index, "2008-01-01", "2009-12-31", "gfc"
    )
    sub = apply_mask(daily_series, mask)
    # No dates within 2008-2009 should remain
    in_window = sub.index[
        (sub.index >= pd.Timestamp("2008-01-01"))
        & (sub.index <= pd.Timestamp("2009-12-31"))
    ]
    assert len(in_window) == 0
    # Dates just outside should remain
    assert pd.Timestamp("2007-12-31") in sub.index
    assert pd.Timestamp("2010-01-01") in sub.index


def test_exclude_window_bad_bounds_raises():
    idx = pd.bdate_range("2010-01-01", "2010-12-31")
    with pytest.raises(ValueError, match="start .* > end"):
        exclude_window_mask(idx, "2010-06-01", "2010-01-01", "bad")


def test_full_mask_keeps_everything(daily_series):
    mask = full_mask(daily_series.index)
    sub = apply_mask(daily_series, mask)
    assert len(sub) == len(daily_series)


def test_describe_subset_reports_coverage(daily_series):
    mask = pre_break_mask(daily_series.index, "2013-01-01")
    d = describe_subset(daily_series, mask)
    assert d["name"].startswith("pre_")
    assert d["n_full"] == len(daily_series)
    assert d["fraction_kept"] < 0.5  # ~8y of 20y
    assert 7.5 < d["years_kept"] < 8.5


def test_standard_regime_set_builds_expected_masks(daily_series):
    masks = standard_regime_set(
        daily_series.index,
        pre_post_break="2013-01-01",
        exclude_windows={
            "gfc": ("2008-01-01", "2009-12-31"),
            "covid": ("2020-02-01", "2021-06-30"),
        },
    )
    names = {m.name for m in masks}
    assert "full" in names
    assert "pre_2013" in names
    assert "post_2013" in names
    assert "ex_gfc" in names
    assert "ex_covid" in names


def test_apply_masks_to_pair_aligns_indices():
    idx = pd.bdate_range("2015-01-01", "2020-12-31")
    # Strategy and bench have slightly different indices
    strat = pd.Series(np.random.randn(len(idx)), index=idx)
    bench_idx = idx[10:]  # bench starts 10 days later
    bench = pd.Series(np.random.randn(len(bench_idx)), index=bench_idx)

    masks = [full_mask(idx), pre_break_mask(idx, "2018-01-01")]
    out = apply_masks_to_pair(strat, bench, masks)

    for name, (s, b) in out.items():
        # Must be aligned within each subset
        assert list(s.index) == list(b.index)
        # Never expanded beyond common intersection
        assert s.index.min() >= bench.index.min()


def test_mask_applied_to_returns_preserves_order(daily_series):
    mask = exclude_window_mask(daily_series.index, "2008-01-01", "2009-12-31", "gfc")
    sub = apply_mask(daily_series, mask)
    # Index must remain sorted
    assert sub.index.is_monotonic_increasing
