"""Phase 15: Regime-Gated Timing Overlay — EXPLORATORY.

Tests whether a regime indicator can identify periods when SMA timing
is more vs less valuable. Primary diagnostic: state-conditional alpha.

Four indicators + random null:
  1. Factor breadth (count below SMA100, threshold >= 3)
  2. Cross-factor correlation (rolling 63d avg pairwise, > trailing median)
  3. Factor volatility (avg trailing 63d vol > 1.5x trailing 252d median)
  4. Market gate benchmark (Mkt-RF < SMA100) — backdoor market-timing control

State-conditional alpha: compute SMA timing alpha WITHIN each regime state.
The indicator is informative if alpha is significantly higher in "active" states.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import load_factors

from youbet.factor.simulator import (
    BuyAndHoldFactor,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)
from youbet.etf.risk import sharpe_ratio as compute_sharpe
from youbet.factor.data import fetch_international_factors

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SNAP_DIR = WORKFLOW_ROOT / "data" / "snapshots"
TRADING_DAYS = 252
FACTORS = ["HML", "SMB", "RMW", "CMA"]
SIM_CONFIG = SimulationConfig(train_months=36, test_months=12, step_months=12)


# ---------------------------------------------------------------------------
# Regime indicators
# ---------------------------------------------------------------------------

def factor_breadth_indicator(factors: pd.DataFrame, threshold: int = 3) -> pd.Series:
    """1 = broad distress (>=threshold factors below SMA100), 0 = normal."""
    n_below = pd.Series(0, index=factors.index, dtype=int)
    for f in FACTORS:
        if f not in factors.columns:
            continue
        cum = (1 + factors[f]).cumprod()
        sma = cum.rolling(100, min_periods=100).mean()
        below = (cum < sma).shift(1).fillna(False)
        n_below += below.astype(int)
    return (n_below >= threshold).astype(int)


def cross_factor_corr_indicator(
    factors: pd.DataFrame, lookback: int = 63, percentile: float = 0.50
) -> pd.Series:
    """1 = elevated cross-factor correlation, 0 = normal."""
    avail = [f for f in FACTORS if f in factors.columns]
    if len(avail) < 2:
        return pd.Series(0, index=factors.index)

    # Rolling pairwise correlation average
    rolling_corrs = []
    for i, f1 in enumerate(avail):
        for f2 in avail[i+1:]:
            rc = factors[f1].rolling(lookback, min_periods=lookback//2).corr(factors[f2])
            rolling_corrs.append(rc)
    avg_corr = pd.concat(rolling_corrs, axis=1).mean(axis=1)

    # Compare to trailing median
    trailing_median = avg_corr.rolling(252, min_periods=126).quantile(percentile)
    return (avg_corr > trailing_median).shift(1).fillna(False).astype(int)


def factor_vol_indicator(
    factors: pd.DataFrame, lookback: int = 63, multiplier: float = 1.5
) -> pd.Series:
    """1 = elevated factor vol, 0 = normal."""
    avail = [f for f in FACTORS if f in factors.columns]
    if not avail:
        return pd.Series(0, index=factors.index)

    vols = pd.DataFrame({f: factors[f].rolling(lookback, min_periods=lookback//2).std() for f in avail})
    avg_vol = vols.mean(axis=1)
    trailing_median = avg_vol.rolling(252, min_periods=126).median()
    return (avg_vol > multiplier * trailing_median).shift(1).fillna(False).astype(int)


def market_gate_indicator(factors: pd.DataFrame) -> pd.Series:
    """1 = market below SMA100, 0 = above. Backdoor market-timing proxy test."""
    if "Mkt-RF" not in factors.columns:
        return pd.Series(0, index=factors.index)
    mkt = factors["Mkt-RF"]
    if "RF" in factors.columns:
        mkt = mkt + factors["RF"]  # total market return
    cum = (1 + mkt).cumprod()
    sma = cum.rolling(100, min_periods=100).mean()
    return (cum < sma).shift(1).fillna(False).astype(int)


# ---------------------------------------------------------------------------
# State-conditional alpha
# ---------------------------------------------------------------------------

def state_conditional_alpha(
    factor_returns: pd.Series,
    rf: pd.Series,
    indicator: pd.Series,
    factor_name: str,
) -> dict:
    """Compute SMA timing alpha within gate-active vs gate-inactive states."""

    bh = simulate_factor_timing(factor_returns, rf, BuyAndHoldFactor(), SIM_CONFIG, factor_name)
    sma = simulate_factor_timing(factor_returns, rf, SMATrendFilter(100), SIM_CONFIG, factor_name)

    # Daily excess returns (timed - B&H)
    common = sma.overall_returns.index.intersection(bh.overall_returns.index).intersection(indicator.dropna().index)
    excess = sma.overall_returns[common] - bh.overall_returns[common]
    ind = indicator[common]

    active = ind == 1
    inactive = ind == 0

    n_active = active.sum()
    n_inactive = inactive.sum()

    alpha_active = float(excess[active].mean() * TRADING_DAYS) if n_active > 50 else None
    alpha_inactive = float(excess[inactive].mean() * TRADING_DAYS) if n_inactive > 50 else None

    return {
        "factor": factor_name,
        "pct_active": float(active.mean()),
        "n_active": int(n_active),
        "n_inactive": int(n_inactive),
        "alpha_active": alpha_active,
        "alpha_inactive": alpha_inactive,
        "alpha_diff": (alpha_active - alpha_inactive) if alpha_active is not None and alpha_inactive is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print("PHASE 15: REGIME-GATED TIMING OVERLAY — EXPLORATORY")
    print("=" * 110)

    factors = load_factors()
    rf = factors["RF"]

    indicators = {
        "breadth_3": ("Factor Breadth >= 3", lambda f: factor_breadth_indicator(f, 3)),
        "breadth_2": ("Factor Breadth >= 2", lambda f: factor_breadth_indicator(f, 2)),
        "corr_50": ("Cross-Factor Corr > p50", lambda f: cross_factor_corr_indicator(f, 63, 0.50)),
        "corr_40": ("Cross-Factor Corr > p40", lambda f: cross_factor_corr_indicator(f, 63, 0.40)),
        "corr_60": ("Cross-Factor Corr > p60", lambda f: cross_factor_corr_indicator(f, 63, 0.60)),
        "vol_150": ("Factor Vol > 1.5x median", lambda f: factor_vol_indicator(f, 63, 1.5)),
        "vol_125": ("Factor Vol > 1.25x median", lambda f: factor_vol_indicator(f, 63, 1.25)),
        "vol_175": ("Factor Vol > 1.75x median", lambda f: factor_vol_indicator(f, 63, 1.75)),
        "mkt_gate": ("Market < SMA100", lambda f: market_gate_indicator(f)),
    }

    primary_indicators = ["breadth_3", "corr_50", "vol_150", "mkt_gate"]
    robustness_indicators = ["breadth_2", "corr_40", "corr_60", "vol_125", "vol_175"]

    # =====================================================================
    # PRIMARY: State-Conditional Alpha (4 indicators × 4 factors)
    # =====================================================================
    print("\n" + "=" * 110)
    print("PRIMARY: STATE-CONDITIONAL ALPHA")
    print("=" * 110)
    print("\nQuestion: Is SMA timing alpha higher when the regime indicator is active?")
    print("Active = the indicator says 'conditions are favorable for timing'\n")

    for ind_key in primary_indicators:
        ind_name, ind_fn = indicators[ind_key]
        ind_series = ind_fn(factors.copy())

        print(f"\n--- {ind_name} (active {ind_series.mean():.0%} of time) ---")
        print(f"{'Factor':<8} {'Active%':>8} {'Alpha Active':>13} {'Alpha Inactive':>15} "
              f"{'Difference':>11} {'Informative?':>13}")
        print("-" * 75)

        for factor in FACTORS:
            result = state_conditional_alpha(factors[factor], rf, ind_series, factor)
            if result["alpha_diff"] is not None:
                informative = "YES" if result["alpha_diff"] > 0 else "NO"
                print(f"{factor:<8} {result['pct_active']:>7.0%} "
                      f"{result['alpha_active']:>+12.1%} {result['alpha_inactive']:>+14.1%} "
                      f"{result['alpha_diff']:>+10.1%} {informative:>13}")

    # =====================================================================
    # HORSE RACE: Breadth vs Market Gate
    # =====================================================================
    print("\n" + "=" * 110)
    print("HORSE RACE: BREADTH GATE vs MARKET GATE")
    print("=" * 110)
    print("\nDoes factor breadth add information beyond market trend?")

    breadth = factor_breadth_indicator(factors.copy(), 3)
    mkt = market_gate_indicator(factors.copy())

    # Overlap
    common = breadth.index.intersection(mkt.index)
    breadth_c = breadth[common]
    mkt_c = mkt[common]
    both_active = ((breadth_c == 1) & (mkt_c == 1)).mean()
    breadth_only = ((breadth_c == 1) & (mkt_c == 0)).mean()
    mkt_only = ((breadth_c == 0) & (mkt_c == 1)).mean()
    neither = ((breadth_c == 0) & (mkt_c == 0)).mean()
    corr = float(breadth_c.corr(mkt_c))

    print(f"\n  Correlation: {corr:.3f}")
    print(f"  Both active:   {both_active:.0%}")
    print(f"  Breadth only:  {breadth_only:.0%}")
    print(f"  Market only:   {mkt_only:.0%}")
    print(f"  Neither:       {neither:.0%}")

    # State-conditional alpha in 4 states
    print(f"\n  Annualized SMA timing alpha by state (HML):")
    bh = simulate_factor_timing(factors["HML"], rf, BuyAndHoldFactor(), SIM_CONFIG, "HML")
    sma = simulate_factor_timing(factors["HML"], rf, SMATrendFilter(100), SIM_CONFIG, "HML")
    excess = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
    common_ex = excess.index.intersection(common)
    bc = breadth[common_ex]
    mc = mkt[common_ex]
    ex = excess[common_ex]

    for state, mask in [("Both active", (bc==1)&(mc==1)), ("Breadth only", (bc==1)&(mc==0)),
                        ("Market only", (bc==0)&(mc==1)), ("Neither", (bc==0)&(mc==0))]:
        if mask.sum() > 50:
            alpha = float(ex[mask].mean() * TRADING_DAYS)
            print(f"    {state:<16}: {alpha:>+6.1%} ({mask.sum()} days, {mask.mean():.0%})")

    # =====================================================================
    # ROBUSTNESS: Threshold Sensitivity (primary indicators only)
    # =====================================================================
    print("\n" + "=" * 110)
    print("ROBUSTNESS: THRESHOLD SENSITIVITY")
    print("=" * 110)

    for ind_key in robustness_indicators:
        ind_name, ind_fn = indicators[ind_key]
        ind_series = ind_fn(factors.copy())

        # Just show HML and CMA as representative factors
        print(f"\n  {ind_name} (active {ind_series.mean():.0%}):", end="")
        for factor in ["HML", "CMA"]:
            result = state_conditional_alpha(factors[factor], rf, ind_series, factor)
            if result["alpha_diff"] is not None:
                print(f"  {factor} diff={result['alpha_diff']:>+5.1%}", end="")
        print()

    # =====================================================================
    # RANDOM NULL
    # =====================================================================
    print("\n" + "=" * 110)
    print("RANDOM NULL (200 simulations, matched activation fraction)")
    print("=" * 110)

    # Use breadth_3 activation fraction for matching
    target_frac = float(breadth.mean())
    null_diffs = {f: [] for f in FACTORS}

    for sim in range(200):
        rng = np.random.default_rng(42 + sim)
        random_gate = pd.Series(
            (rng.random(len(factors)) < target_frac).astype(int),
            index=factors.index,
        )
        for factor in FACTORS:
            result = state_conditional_alpha(factors[factor], rf, random_gate, factor)
            if result["alpha_diff"] is not None:
                null_diffs[factor].append(result["alpha_diff"])

    print(f"\nRandom gate null (activation frac = {target_frac:.0%}):")
    print(f"{'Factor':<8} {'Null Mean':>10} {'Null Std':>9} {'Breadth Diff':>13} {'p-value':>8}")
    print("-" * 55)

    for factor in FACTORS:
        breadth_result = state_conditional_alpha(factors[factor], rf, breadth, factor)
        if breadth_result["alpha_diff"] is not None and null_diffs[factor]:
            null_arr = np.array(null_diffs[factor])
            obs = breadth_result["alpha_diff"]
            p_val = float((null_arr >= obs).sum() + 1) / (len(null_arr) + 1)
            print(f"{factor:<8} {null_arr.mean():>+9.1%} {null_arr.std():>8.1%} "
                  f"{obs:>+12.1%} {p_val:>7.3f}")

    # =====================================================================
    # INTERNATIONAL OOS (best indicator: breadth_3)
    # =====================================================================
    print("\n" + "=" * 110)
    print("INTERNATIONAL OOS: BREADTH GATE ON DEV EX-US")
    print("=" * 110)

    try:
        intl = fetch_international_factors("developed_ex_us", snapshot_dir=SNAP_DIR)
        rf_intl = intl.get("RF", pd.Series(0.0, index=intl.index))
        intl_breadth = factor_breadth_indicator(intl.copy(), 3)

        print(f"\n  Breadth >= 3 active: {intl_breadth.mean():.0%}")
        print(f"  {'Factor':<8} {'Alpha Active':>13} {'Alpha Inactive':>15} {'Diff':>8}")
        print(f"  " + "-" * 50)

        for factor in FACTORS:
            if factor in intl.columns:
                result = state_conditional_alpha(intl[factor], rf_intl, intl_breadth, factor)
                if result["alpha_diff"] is not None:
                    print(f"  {factor:<8} {result['alpha_active']:>+12.1%} "
                          f"{result['alpha_inactive']:>+14.1%} {result['alpha_diff']:>+7.1%}")
    except Exception as e:
        print(f"  FAILED: {e}")

    # =====================================================================
    # VERDICT
    # =====================================================================
    print(f"\n" + "=" * 110)
    print("PHASE 15 VERDICT")
    print("=" * 110)


if __name__ == "__main__":
    main()
