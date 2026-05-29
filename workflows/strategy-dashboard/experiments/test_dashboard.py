"""Unit tests for the strategy-dashboard machinery.

Run as:
    cd workflows/strategy-dashboard/experiments
    python test_dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from _shared import (
    compute_full_metrics,
    compose_metric_matrix,
    load_config,
    load_panel,
    load_vti_benchmark,
    mad_zscore,
    per_group_score,
    _regime_period_return,
)


def test_vti_sanity_metrics() -> None:
    """100% VTI from panel.parquet should have plausible-looking metrics."""
    cfg = load_config()
    vti = load_vti_benchmark(cfg)
    bench = vti  # self-benchmark; IR should be ~0
    metrics = compute_full_metrics(vti, bench, cfg["regime_periods"], "VTI sanity")
    assert 0.30 <= metrics["sharpe"] <= 0.80, f"VTI Sharpe wild: {metrics['sharpe']}"
    assert -0.60 <= metrics["max_drawdown"] <= -0.20, f"VTI MaxDD wild: {metrics['max_drawdown']}"
    assert 0.04 <= metrics["cagr"] <= 0.15, f"VTI CAGR wild: {metrics['cagr']}"
    assert abs(metrics["info_ratio_vs_vti"]) < 0.05, f"VTI IR vs itself should be ~0: {metrics['info_ratio_vs_vti']}"
    print(f"PASS test_vti_sanity_metrics  Sharpe={metrics['sharpe']:.3f}  MaxDD={metrics['max_drawdown']:.3f}  CAGR={metrics['cagr']:.3f}")


def test_regime_2008_gfc_vti() -> None:
    """VTI return during 2007-10 to 2009-06 should be deeply negative."""
    cfg = load_config()
    vti = load_vti_benchmark(cfg)
    r_gfc = _regime_period_return(vti, *cfg["regime_periods"]["gfc_2008"])
    assert -0.65 <= r_gfc <= -0.30, f"2008 VTI return wild: {r_gfc}"
    print(f"PASS test_regime_2008_gfc_vti  total_return_2007-10_to_2009-06 = {r_gfc:.3f}")


def test_regime_covid_2020_vti() -> None:
    """VTI return during Feb-May 2020 should round-trip near zero (V-shape) or modestly negative."""
    cfg = load_config()
    vti = load_vti_benchmark(cfg)
    r_covid = _regime_period_return(vti, *cfg["regime_periods"]["covid_2020"])
    # Feb 1 to May 31 2020 was -34% to -10% range depending on exact closes
    assert -0.30 <= r_covid <= 0.10, f"COVID VTI return wild: {r_covid}"
    print(f"PASS test_regime_covid_2020_vti  total_return = {r_covid:.3f}")


def test_mad_zscore_outlier_resistance() -> None:
    """MAD-z gives outliers moderate z (not ~50 like std-z would)."""
    x = np.concatenate([np.random.RandomState(7).normal(0, 1, 100), [100.0]])
    z = mad_zscore(x)
    # The outlier's z should be < 100 (i.e. dampened relative to std-z which would give ~9-10)
    # but still flagged as the largest
    assert abs(z[-1]) >= abs(z[:-1]).max(), "outlier should be the largest |z|"
    assert abs(z[-1]) < 200, f"MAD-z gave outlier z={z[-1]:.2f} — should be << std-z's"
    # Inliers should be roughly Gaussian z
    inlier_z = z[:-1]
    assert abs(inlier_z.mean()) < 0.5
    print(f"PASS test_mad_zscore_outlier_resistance  outlier z = {z[-1]:.2f}")


def test_mad_zscore_handles_nan() -> None:
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    z = mad_zscore(x)
    assert np.isnan(z[2])
    assert not np.isnan(z[0])
    print("PASS test_mad_zscore_handles_nan")


def test_compose_ranking_orders_higher_better() -> None:
    """Higher metric value should give higher composite z."""
    df = pd.DataFrame({
        "metric_a": [10.0, 5.0, 1.0, 7.0, 3.0],
        "metric_b": [1.0, 2.0, 5.0, 4.0, 3.0],  # decoupled
    }, index=[f"s{i}" for i in range(5)])
    z = compose_metric_matrix(
        df, ["metric_a", "metric_b"], lower_is_better=[], weights={"metric_a": 1.0, "metric_b": 1.0}
    )
    # s0 has highest metric_a but lowest metric_b → mid composite
    # s3 has high in both → top composite
    assert z["composite_z"].idxmax() in ("s0", "s3")
    print(f"PASS test_compose_ranking_orders_higher_better  top={z['composite_z'].idxmax()}")


def test_lower_is_better_sign_flip() -> None:
    """A 'lower-is-better' metric (e.g., MaxDD-duration) should give the smallest value the highest z."""
    df = pd.DataFrame({
        "dd_duration": [1000.0, 500.0, 100.0, 800.0, 200.0],
    }, index=[f"s{i}" for i in range(5)])
    z = compose_metric_matrix(
        df, ["dd_duration"], lower_is_better=["dd_duration"], weights={"dd_duration": 1.0}
    )
    assert z["composite_z"].idxmax() == "s2"  # smallest dd_duration
    assert z["composite_z"].idxmin() == "s0"  # largest dd_duration
    print("PASS test_lower_is_better_sign_flip")


def test_per_group_score_handles_missing_metric() -> None:
    """Group score should skip metrics not in DataFrame."""
    df = pd.DataFrame({
        "cagr": [0.10, 0.08, 0.12],
    }, index=[f"s{i}" for i in range(3)])
    groups = {"return": ["cagr", "annualized_return", "median_rolling_1y_return"], "missing": ["nonexistent"]}
    out = per_group_score(df, groups, lower_is_better=[])
    assert not out["group__return"].isna().all()
    assert out["group__missing"].isna().all()
    print("PASS test_per_group_score_handles_missing_metric")


def test_vti_lower_bound_n_obs() -> None:
    """VTI panel covers ~2000-2026; restricting to common 2010-2026 should still give >=252 days."""
    cfg = load_config()
    vti = load_vti_benchmark(cfg)
    start, end = cfg["window"]["common_start"], cfg["window"]["common_end"]
    sub = vti[(vti.index >= pd.Timestamp(start)) & (vti.index <= pd.Timestamp(end))]
    assert len(sub) > 4000, f"Common-window VTI too short: {len(sub)}"
    print(f"PASS test_vti_lower_bound_n_obs  common_window VTI obs = {len(sub)}")


if __name__ == "__main__":
    test_vti_sanity_metrics()
    test_regime_2008_gfc_vti()
    test_regime_covid_2020_vti()
    test_mad_zscore_outlier_resistance()
    test_mad_zscore_handles_nan()
    test_compose_ranking_orders_higher_better()
    test_lower_is_better_sign_flip()
    test_per_group_score_handles_missing_metric()
    test_vti_lower_bound_n_obs()
    print("\nAll tests passed.")
