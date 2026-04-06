"""Regime-stratified analysis: how do strategies perform in different market conditions?

Splits the test period into regimes based on VTI returns and VIX levels.
Reports per-regime excess returns for top strategies.

Usage:
    python experiments/regime_analysis.py
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
sys.path.insert(0, str(WORKFLOW_ROOT))

from youbet.etf.backtester import Backtester, BacktestConfig
from youbet.etf.benchmark import BuyAndHold
from youbet.etf.costs import CostModel
from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.macro.fetchers import fetch_all_tier1
from youbet.utils.io import load_config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

STRATEGIES_DIR = WORKFLOW_ROOT / "strategies"


def classify_regimes(
    vti_returns: pd.Series,
    vix_series: pd.Series | None = None,
) -> pd.DataFrame:
    """Classify each trading day into market regimes.

    Returns DataFrame with boolean columns for each regime.
    """
    regimes = pd.DataFrame(index=vti_returns.index)

    # Trailing 12-month VTI return
    trailing_12m = vti_returns.rolling(252).sum()

    regimes["bull"] = trailing_12m > 0.10
    regimes["bear"] = trailing_12m < -0.10
    regimes["sideways"] = ~regimes["bull"] & ~regimes["bear"]

    # VTI drawdown regimes
    cum_wealth = (1 + vti_returns).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    regimes["crisis"] = drawdown < -0.20  # VTI in >20% drawdown

    # VIX-based regimes
    if vix_series is not None:
        vix_aligned = vix_series.reindex(vti_returns.index, method="ffill")
        regimes["high_vol"] = vix_aligned > 25
        regimes["low_vol"] = vix_aligned < 15
    else:
        regimes["high_vol"] = False
        regimes["low_vol"] = False

    return regimes


def regime_performance(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    regimes: pd.DataFrame,
) -> dict:
    """Compute annualized excess return for each regime."""
    results = {}
    for regime_name in regimes.columns:
        mask = regimes[regime_name]
        if mask.sum() < 20:  # Need at least 20 days
            results[regime_name] = {"n_days": int(mask.sum()), "excess_ann": np.nan}
            continue

        strat_regime = strat_returns[mask]
        bench_regime = bench_returns[mask]
        excess = strat_regime - bench_regime

        ann_excess = float(excess.mean() * 252)
        ann_vol = float(excess.std() * np.sqrt(252))
        info_ratio = ann_excess / ann_vol if ann_vol > 1e-10 else 0.0

        results[regime_name] = {
            "n_days": int(mask.sum()),
            "strat_ann": float(strat_regime.mean() * 252),
            "bench_ann": float(bench_regime.mean() * 252),
            "excess_ann": ann_excess,
            "info_ratio": info_ratio,
        }
    return results


def main():
    print("=" * 90)
    print("REGIME-STRATIFIED ANALYSIS")
    print("=" * 90)
    print()

    universe = load_universe()
    all_tickers = universe["ticker"].tolist()
    prices = fetch_prices(all_tickers, start="2003-01-01")
    tbill = fetch_tbill_rates(start="2003-01-01", allow_fallback=True)
    macro_features = fetch_all_tier1(start="2003-01-01")

    cost_model = CostModel.from_universe(universe)
    config = BacktestConfig(
        train_months=36, test_months=12, step_months=12,
        rebalance_frequency="monthly", initial_capital=100_000,
    )
    benchmark = BuyAndHold({"VTI": 1.0})
    bt = Backtester(config=config, prices=prices, cost_model=cost_model,
                    tbill_rates=tbill, universe=universe)

    # Load top strategies
    from strategies.trend_following.scripts.run import TrendFollowing
    from strategies.risk_parity.scripts.run import RiskParity
    from strategies.asset_class_rotation.scripts.run import AssetClassRotation
    from strategies.full_universe_momentum.scripts.run import FullUniverseMomentum
    from strategies.vol_targeting.scripts.run import VolTargeting
    from strategies.combined_blend.scripts.run import CombinedBlend

    strategies = {}

    tf_cfg = load_config(STRATEGIES_DIR / "trend_following" / "config.yaml")
    strategies["trend_following"] = TrendFollowing.from_config(tf_cfg)

    rp_cfg = load_config(STRATEGIES_DIR / "risk_parity" / "config.yaml")
    strategies["risk_parity"] = RiskParity.from_config(rp_cfg)

    acr_cfg = load_config(STRATEGIES_DIR / "asset_class_rotation" / "config.yaml")
    acr = AssetClassRotation.from_config(acr_cfg)
    acr.set_features(macro_features)
    strategies["asset_class_rotation"] = acr

    vol_cfg = load_config(STRATEGIES_DIR / "vol_targeting" / "config.yaml")
    strategies["vol_targeting"] = VolTargeting.from_config(vol_cfg)

    fum_cfg = load_config(STRATEGIES_DIR / "full_universe_momentum" / "config.yaml")
    strategies["full_universe_momentum"] = FullUniverseMomentum.from_config(fum_cfg, universe)

    # Blend
    tf2 = TrendFollowing.from_config(tf_cfg)
    rp2 = RiskParity.from_config(rp_cfg)
    strategies["blend_trend_rp"] = CombinedBlend(
        members=[tf2, rp2], blend_weights=[0.5, 0.5], variant_name="trend_rp",
    )

    # Run all strategies
    results = {}
    for name, strat in strategies.items():
        print(f"  Running {name}...")
        results[name] = bt.run(strat, benchmark)

    # Get VIX for regime classification
    vix_vals = macro_features.get("vix")
    vix_series = vix_vals.values if vix_vals is not None else None

    # Use benchmark returns to define regimes
    bench_rets = list(results.values())[0].benchmark_returns
    regimes = classify_regimes(bench_rets, vix_series)

    # Count regime days
    print()
    print("Regime distribution:")
    for col in regimes.columns:
        n = regimes[col].sum()
        pct = n / len(regimes) * 100
        print(f"  {col:<12}: {n:>5} days ({pct:.1f}%)")
    print()

    # --- Per-regime performance ---
    regime_names = ["bull", "bear", "sideways", "crisis", "high_vol", "low_vol"]

    print("=" * 100)
    print("ANNUALIZED EXCESS RETURN BY REGIME (strategy - VTI)")
    print("=" * 100)
    print()
    header = f"{'Strategy':<28}"
    for r in regime_names:
        header += f" {r:>10}"
    print(header)
    print("-" * (28 + 11 * len(regime_names)))

    all_regime_results = {}
    for name, result in results.items():
        rp = regime_performance(result.overall_returns, result.benchmark_returns, regimes)
        all_regime_results[name] = rp
        row = f"{name:<28}"
        for r in regime_names:
            val = rp[r]["excess_ann"]
            if np.isnan(val):
                row += f" {'n/a':>10}"
            else:
                row += f" {val:>+10.1%}"
        print(row)
    print()

    # --- Crisis deep-dive ---
    print("=" * 90)
    print("CRISIS DEEP-DIVE: Performance during VTI drawdowns > 20%")
    print("=" * 90)
    print()
    crisis_mask = regimes["crisis"]
    if crisis_mask.sum() > 0:
        print(f"Crisis days: {crisis_mask.sum()} ({crisis_mask.sum()/len(regimes)*100:.1f}% of test period)")
        print()
        print(f"{'Strategy':<28} {'Crisis Return':>14} {'VTI Return':>12} {'Excess':>10}")
        print("-" * 68)
        for name, result in results.items():
            strat_crisis = result.overall_returns[crisis_mask].sum()
            bench_crisis = result.benchmark_returns[crisis_mask].sum()
            excess = strat_crisis - bench_crisis
            print(f"{name:<28} {strat_crisis:>+14.1%} {bench_crisis:>+12.1%} {excess:>+10.1%}")
    else:
        print("No crisis periods detected in the test window.")

    return all_regime_results


if __name__ == "__main__":
    main()
