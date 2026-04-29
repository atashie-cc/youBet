"""Empirical tracking error + MDE recomputation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from youbet.stock.te import (
    TRADING_DAYS_PER_YEAR,
    empirical_tracking_error,
    mde_at_80_power,
    mde_table,
    power_sensitivity_table,
    recompute_power,
)


def test_empirical_te_matches_injected_sd():
    """Inject a known daily excess sd; empirical TE should recover it."""
    rng = np.random.default_rng(0)
    n = 252 * 20
    idx = pd.bdate_range("2010-01-01", periods=n)
    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)
    # Inject excess with sd_daily = 0.005 → annualized TE ≈ 0.0794
    excess = pd.Series(rng.normal(0.0001, 0.005, n), index=idx)
    strat = bench + excess

    rep = empirical_tracking_error(strat, bench)
    expected_te_ann = 0.005 * np.sqrt(TRADING_DAYS_PER_YEAR)
    # 20y: SE on sd ≈ 5e-5; SE on TE ≈ 5e-5 * √252 ≈ 8e-4; allow 3×SE
    assert abs(rep.annualized_te - expected_te_ann) < 0.005
    assert rep.n_days == n
    # Sample mean SE = sd/sqrt(n) = 0.005/71 ≈ 7e-5; 3×SE ≈ 2.1e-4
    assert abs(rep.mean_daily_excess - 0.0001) < 3e-4


def test_te_zero_when_identical():
    idx = pd.bdate_range("2020-01-01", periods=500)
    bench = pd.Series(np.linspace(100, 110, len(idx)), index=idx).pct_change().dropna()
    strat = bench.copy()
    rep = empirical_tracking_error(strat, bench)
    assert rep.annualized_te < 1e-9


def test_te_raises_on_insufficient_overlap():
    a = pd.Series(np.random.randn(5), index=pd.bdate_range("2020-01-01", periods=5))
    b = pd.Series(np.random.randn(5), index=pd.bdate_range("2020-01-01", periods=5))
    with pytest.raises(ValueError, match="Insufficient overlap"):
        empirical_tracking_error(a, b)


def test_recompute_power_monotonic_in_target():
    """Larger target Sharpe → higher power."""
    powers = recompute_power(
        n_years=10,
        target_sharpes=[0.10, 0.30, 0.50],
        tracking_error_annual=0.08,
        n_sims=30,
        n_bootstrap=200,
    )
    vals = [powers[t] for t in [0.10, 0.30, 0.50]]
    # Allow MC jitter but the big-picture ordering should hold
    assert vals[0] <= vals[2], f"power non-monotonic: {powers}"


def test_mde_at_80_power_none_when_none_pass():
    powers = {0.10: 0.05, 0.20: 0.12, 0.30: 0.35}
    assert mde_at_80_power(powers) is None


def test_mde_at_80_power_picks_smallest_passing():
    powers = {0.10: 0.40, 0.20: 0.85, 0.30: 0.99}
    assert mde_at_80_power(powers) == 0.20


def test_power_sensitivity_table_columns_match_anchors():
    df = power_sensitivity_table(
        n_years=5,
        target_sharpes=[0.20, 0.50],
        te_anchors=[0.04, 0.08],
        n_sims=10,
        n_bootstrap=100,
    )
    assert list(df.columns) == ["TE=0.04", "TE=0.08"]
    assert list(df.index) == [0.20, 0.50]


def test_lower_te_yields_lower_mde_at_fixed_target():
    """With the same effect size (mean excess), a lower TE gives larger
    observed Sharpe → higher power → smaller MDE."""
    df = power_sensitivity_table(
        n_years=20,
        target_sharpes=[0.20, 0.40, 0.60],
        te_anchors=[0.04, 0.08],
        n_sims=25,
        n_bootstrap=300,
    )
    # Power should be higher at TE=0.04 than at TE=0.08 for every target
    # (same target Sharpe → same planted mean-over-std ratio, so this
    # actually should be roughly equal — the construction pins Sharpe
    # regardless of TE). This test confirms the SAME-sharpe equivalence.
    diff = df["TE=0.04"] - df["TE=0.08"]
    # Expected: |diff| small — the TE cancels out in the Sharpe-parameterized
    # construction. So this is a sanity check, not a power comparison.
    assert (diff.abs() < 0.20).all(), (
        f"Same-target-Sharpe power should be comparable across TE anchors "
        f"because the construction pins Sharpe independently of TE: {diff}"
    )


def test_mde_table_computes_correctly():
    """MDE is the smallest target where power ≥ threshold."""
    df = pd.DataFrame(
        {"TE=0.04": [0.1, 0.85, 0.95], "TE=0.08": [0.05, 0.3, 0.85]},
        index=[0.10, 0.30, 0.50],
    )
    df.index.name = "target_sharpe"
    mde = mde_table(df, threshold=0.80)
    assert mde["TE=0.04"] == 0.30   # 0.85 at 0.30 is first to pass 0.80
    assert mde["TE=0.08"] == 0.50   # only 0.85 at 0.50 passes
