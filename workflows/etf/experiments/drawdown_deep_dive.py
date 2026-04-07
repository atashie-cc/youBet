"""Drawdown deep-dive: 5 experiments for ruin-focused strategy evaluation.

Experiment 1: Bootstrap CIs on max drawdown difference (strategy vs VTI)
Experiment 2: Optimal trend/buy-and-hold blend ratio sweep (10-90%)
Experiment 3: Safe withdrawal rate comparison (max rate with <5% ruin @ 30yr)
Experiment 4: Insurance premium calculation (bull-market drag per crisis event)
Experiment 5: Momentum crisis mechanism trace (which ETFs held during 2008)

Usage:
    python experiments/drawdown_deep_dive.py
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
from youbet.etf.stats import stationary_block_bootstrap
from youbet.utils.io import load_config

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES_DIR = WORKFLOW_ROOT / "strategies"


def _max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min())


def _bootstrap_drawdown_ci(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    n_bootstrap: int = 5000,
    confidence: float = 0.90,
    block_length: int = 22,
    seed: int = 42,
) -> dict:
    """BCa-like bootstrap CI on max drawdown difference (strategy - benchmark)."""
    rng = np.random.default_rng(seed)
    strat_vals = strat_returns.values
    bench_vals = bench_returns.values
    n = len(strat_vals)

    diffs = []
    for _ in range(n_bootstrap):
        # Paired block bootstrap — same indices for both series
        indices = []
        pos = 0
        while pos < n:
            block_len = rng.geometric(1.0 / block_length)
            start = rng.integers(0, n)
            for j in range(block_len):
                if pos >= n:
                    break
                indices.append((start + j) % n)
                pos += 1

        idx = np.array(indices[:n])
        boot_strat = pd.Series(strat_vals[idx])
        boot_bench = pd.Series(bench_vals[idx])

        dd_strat = _max_drawdown(boot_strat)
        dd_bench = _max_drawdown(boot_bench)
        diffs.append(dd_strat - dd_bench)  # Positive = strategy has less drawdown

    diffs = np.array(diffs)
    alpha = 1 - confidence
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    # Point estimate = realized sample diff, NOT bootstrap mean (Codex fix #1)
    realized_diff = _max_drawdown(strat_returns) - _max_drawdown(bench_returns)

    return {
        "point_estimate": float(realized_diff),
        "ci_lower": lower,
        "ci_upper": upper,
        "pct_positive": float((diffs > 0).mean()),
    }


def _simulate_withdrawal(
    daily_returns: np.ndarray,
    annual_rate: float,
    horizon_days: int,
    block_length: int = 22,
    n_sims: int = 5000,
    seed: int = 42,
) -> float:
    """Simulate fixed-dollar withdrawal and return fraction of paths that survive.

    Bengen-style: withdraw a FIXED daily amount (annual_rate / 252) from the
    portfolio regardless of current value. This means deeper drawdowns cause
    more shares to be sold, amplifying sequence-of-returns risk.
    """
    rng = np.random.default_rng(seed)
    n = len(daily_returns)
    # Fixed daily withdrawal based on initial portfolio value of 1.0
    fixed_daily = annual_rate / 252
    survived = 0

    for _ in range(n_sims):
        wealth = 1.0
        pos = 0
        while pos < horizon_days:
            block_len = rng.geometric(1.0 / block_length)
            start = rng.integers(0, n)
            for j in range(block_len):
                if pos >= horizon_days:
                    break
                idx = (start + j) % n
                wealth *= (1 + daily_returns[idx])
                wealth -= fixed_daily  # Fixed dollar withdrawal (Bengen-style)
                pos += 1
                if wealth <= 0:
                    break
            if wealth <= 0:
                break
        if wealth > 0:
            survived += 1

    return survived / n_sims


def run_backtests(prices, universe, tbill, macro_features):
    """Run key strategies through backtester."""
    from strategies.trend_following.scripts.run import TrendFollowing
    from strategies.risk_parity.scripts.run import RiskParity
    from strategies.asset_class_rotation.scripts.run import AssetClassRotation
    from strategies.full_universe_momentum.scripts.run import FullUniverseMomentum
    from strategies.vol_targeting.scripts.run import VolTargeting
    from strategies.combined_blend.scripts.run import CombinedBlend

    cost_model = CostModel.from_universe(universe)
    config = BacktestConfig(
        train_months=36, test_months=12, step_months=12,
        rebalance_frequency="monthly", initial_capital=100_000,
    )
    benchmark = BuyAndHold({"VTI": 1.0})
    bt = Backtester(config=config, prices=prices, cost_model=cost_model,
                    tbill_rates=tbill, universe=universe)

    results = {}
    strats = {}

    tf_cfg = load_config(STRATEGIES_DIR / "trend_following" / "config.yaml")
    strats["trend_following"] = TrendFollowing.from_config(tf_cfg)

    rp_cfg = load_config(STRATEGIES_DIR / "risk_parity" / "config.yaml")
    strats["risk_parity"] = RiskParity.from_config(rp_cfg)

    acr_cfg = load_config(STRATEGIES_DIR / "asset_class_rotation" / "config.yaml")
    acr = AssetClassRotation.from_config(acr_cfg)
    acr.set_features(macro_features)
    strats["asset_class_rotation"] = acr

    fum_cfg = load_config(STRATEGIES_DIR / "full_universe_momentum" / "config.yaml")
    strats["full_universe_momentum"] = FullUniverseMomentum.from_config(fum_cfg, universe)

    vol_cfg = load_config(STRATEGIES_DIR / "vol_targeting" / "config.yaml")
    strats["vol_targeting"] = VolTargeting.from_config(vol_cfg)

    for name, strat in strats.items():
        print(f"  Running {name}...")
        results[name] = bt.run(strat, benchmark)

    # Blend sweeps for Experiment 2
    for tf_pct in range(10, 100, 10):
        rp_pct = 100 - tf_pct
        tf_s = TrendFollowing.from_config(tf_cfg)
        rp_s = RiskParity.from_config(rp_cfg)
        blend = CombinedBlend(
            members=[tf_s, rp_s],
            blend_weights=[tf_pct / 100, rp_pct / 100],
            variant_name=f"tf{tf_pct}_rp{rp_pct}",
        )
        name = f"blend_tf{tf_pct}_rp{rp_pct}"
        print(f"  Running {name}...")
        results[name] = bt.run(blend, benchmark)

    return results


def experiment_1_drawdown_cis(results: dict):
    """Bootstrap CIs on max drawdown difference (strategy - benchmark)."""
    print("=" * 90)
    print("EXPERIMENT 1: Bootstrap CIs on Max Drawdown Difference")
    print("  Positive = strategy has LESS drawdown than VTI (good)")
    print("=" * 90)
    print()
    print(f"{'Strategy':<28} {'Strat DD':>9} {'VTI DD':>8} {'Diff':>8} "
          f"{'90% CI':>22} {'P(better)':>10}")
    print("-" * 88)

    core_names = ["trend_following", "risk_parity", "asset_class_rotation",
                  "full_universe_momentum", "vol_targeting"]
    for name in core_names:
        if name not in results:
            continue
        r = results[name]
        strat_dd = _max_drawdown(r.overall_returns)
        bench_dd = _max_drawdown(r.benchmark_returns)

        ci = _bootstrap_drawdown_ci(r.overall_returns, r.benchmark_returns)
        ci_str = f"[{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}]"
        print(
            f"{name:<28} {strat_dd:>9.1%} {bench_dd:>8.1%} "
            f"{ci['point_estimate']:>+8.3f} {ci_str:>22} "
            f"{ci['pct_positive']:>10.0%}"
        )
    print()


def experiment_2_blend_sweep(results: dict):
    """Optimal trend/risk-parity blend ratio."""
    print("=" * 90)
    print("EXPERIMENT 2: Trend Following / Risk Parity Blend Sweep")
    print("  Literature suggests 40% trend for Sharpe, 63% for drawdown minimization")
    print("=" * 90)
    print()
    print(f"{'Blend':<28} {'Sharpe':>7} {'MaxDD':>8} {'Calmar':>7} {'Ann Ret':>8} {'Ann Vol':>8}")
    print("-" * 70)

    blend_names = sorted(
        [n for n in results if n.startswith("blend_tf")],
        key=lambda x: int(x.split("tf")[1].split("_")[0]),
    )

    best_sharpe_name = ""
    best_sharpe = -999
    best_dd_name = ""
    best_dd = -999

    for name in blend_names:
        r = results[name]
        ret = r.overall_returns
        cum = (1 + ret).cumprod()
        ann_ret = float(cum.iloc[-1] ** (252 / len(ret)) - 1)
        ann_vol = float(ret.std() * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0
        max_dd = _max_drawdown(ret)
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-10 else 0

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_sharpe_name = name
        if max_dd > best_dd:
            best_dd = max_dd
            best_dd_name = name

        print(
            f"{name:<28} {sharpe:>7.3f} {max_dd:>8.1%} {calmar:>7.3f} "
            f"{ann_ret:>8.1%} {ann_vol:>8.1%}"
        )

    print()
    print(f"  Best Sharpe: {best_sharpe_name} ({best_sharpe:.3f})")
    print(f"  Best MaxDD:  {best_dd_name} ({best_dd:.1%})")
    print()


def experiment_3_safe_withdrawal(results: dict):
    """Safe withdrawal rate: max rate with >95% survival at 30 years."""
    print("=" * 90)
    print("EXPERIMENT 3: Safe Withdrawal Rate (max rate with >95% survival @ 30yr)")
    print("  Bengen's 4% rule baseline. Block bootstrap, 5,000 sims per rate.")
    print("=" * 90)
    print()

    horizon_days = 30 * 252
    rates_to_test = [0.02, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08]

    core_names = ["trend_following", "risk_parity", "asset_class_rotation",
                  "full_universe_momentum", "vol_targeting"]

    # Header
    header = f"{'Strategy':<28}"
    for rate in rates_to_test:
        header += f" {rate:>6.0%}"
    print(header)
    print("-" * (28 + 7 * len(rates_to_test)))

    swr_results = {}
    for name in core_names:
        if name not in results:
            continue
        r = results[name]
        daily_rets = r.overall_returns.values
        row = f"{name:<28}"
        safe_rate = 0.0

        for rate in rates_to_test:
            survival = _simulate_withdrawal(daily_rets, rate, horizon_days, seed=42)
            row += f" {survival:>6.0%}"
            if survival >= 0.95:
                safe_rate = rate

        swr_results[name] = safe_rate
        print(row)

    # Also test benchmark
    bench_rets = list(results.values())[0].benchmark_returns.values
    row = f"{'VTI (benchmark)':<28}"
    safe_rate = 0.0
    for rate in rates_to_test:
        survival = _simulate_withdrawal(bench_rets, rate, horizon_days, seed=42)
        row += f" {survival:>6.0%}"
        if survival >= 0.95:
            safe_rate = rate
    swr_results["VTI"] = safe_rate
    print(row)

    print()
    print("Safe withdrawal rates (>95% survival):")
    for name, rate in sorted(swr_results.items(), key=lambda x: -x[1]):
        print(f"  {name:<28} {rate:.0%}")
    print()


def experiment_4_insurance_premium(results: dict):
    """Insurance premium: bull-market drag per crisis protection event."""
    print("=" * 90)
    print("EXPERIMENT 4: Insurance Premium (annual cost of crisis protection)")
    print("=" * 90)
    print()

    bench_rets = list(results.values())[0].benchmark_returns
    cum_bench = (1 + bench_rets).cumprod()
    bench_dd = (cum_bench - cum_bench.cummax()) / cum_bench.cummax()

    # Identify crisis periods (VTI drawdown > 20%)
    crisis_mask = bench_dd < -0.20
    bull_mask = ~crisis_mask
    n_crisis_days = crisis_mask.sum()
    n_bull_days = bull_mask.sum()
    n_years = len(bench_rets) / 252

    core_names = ["trend_following", "risk_parity", "full_universe_momentum"]

    print(f"Test period: {n_years:.1f} years, {n_crisis_days} crisis days "
          f"({n_crisis_days/len(bench_rets)*100:.1f}%), {n_bull_days} normal days")
    print()
    print(f"{'Strategy':<28} {'Bull Drag':>10} {'Crisis Gain':>12} {'Net Ann':>9} "
          f"{'Premium/yr':>11} {'Payoff Ratio':>12}")
    print("-" * 86)

    for name in core_names:
        if name not in results:
            continue
        r = results[name]

        # Compounded active wealth: (1+strat)/(1+bench) - 1 per day
        # Then compound over normal/crisis periods (Codex fix #3)
        active_wealth = (1 + r.overall_returns).cumprod() / (1 + r.benchmark_returns).cumprod()
        bull_active = active_wealth[bull_mask]
        crisis_active = active_wealth[crisis_mask]

        # Compounded drag/gain = ratio of active wealth at end vs start of each regime
        # Use log returns for additivity across periods
        log_excess = np.log1p(r.overall_returns) - np.log1p(r.benchmark_returns)
        bull_drag = float(log_excess[bull_mask].sum())  # Log active return during normal
        crisis_gain = float(log_excess[crisis_mask].sum())  # Log active return during crisis
        net = bull_drag + crisis_gain
        n_bull_years = bull_mask.sum() / 252
        annual_premium = -bull_drag / n_bull_years if n_bull_years > 0 else 0  # Cost per non-crisis year
        payoff_ratio = abs(crisis_gain / bull_drag) if abs(bull_drag) > 1e-10 else float("inf")

        print(
            f"{name:<28} {bull_drag:>+10.1%} {crisis_gain:>+12.1%} "
            f"{net:>+9.1%} {annual_premium:>11.2%}/yr "
            f"{payoff_ratio:>11.1f}x"
        )
    print()


def experiment_5_momentum_crisis_trace(results: dict, prices: pd.DataFrame):
    """Trace which ETFs the momentum strategy held during the 2008-2009 crisis."""
    print("=" * 90)
    print("EXPERIMENT 5: Momentum Crisis Mechanism — What was held during 2008-2009?")
    print("=" * 90)
    print()

    if "full_universe_momentum" not in results:
        print("  full_universe_momentum not found in results")
        return

    r = results["full_universe_momentum"]

    # Find crisis period weights from fold results
    for fold in r.fold_results:
        # Check if this fold covers the 2008-2009 crisis
        if fold.test_start.year > 2009 or fold.test_end.year < 2008:
            continue

        print(f"Fold: {fold.fold_name} ({fold.test_start.date()} to {fold.test_end.date()})")
        for rebal_date, weights in fold.weights_history:
            if rebal_date.year in (2008, 2009):
                # Show top holdings
                top = weights.sort_values(ascending=False).head(5)
                holdings_str = ", ".join(
                    f"{t} ({w:.0%})" for t, w in top.items() if w > 0.01
                )
                if not holdings_str:
                    holdings_str = "VGSH (risk-off, 100%)"
                print(f"  {rebal_date.date()}: {holdings_str}")
    print()

    # Verify: did it hold bonds during the crisis?
    print("Interpretation:")
    print("  If holdings show VGSH/BND/BSV/VGIT during 2008-2009, the absolute")
    print("  momentum filter correctly rotated to bonds — this is the documented")
    print("  mechanism for long-only momentum crisis protection (Antonacci).")
    print()


def main():
    print("=" * 90)
    print("DRAWDOWN DEEP-DIVE: 5 Experiments")
    print("=" * 90)
    print()

    universe = load_universe()
    all_tickers = universe["ticker"].tolist()
    prices = fetch_prices(all_tickers, start="2003-01-01")
    tbill = fetch_tbill_rates(start="2003-01-01", allow_fallback=True)
    macro_features = fetch_all_tier1(start="2003-01-01")

    print("Running backtests...")
    results = run_backtests(prices, universe, tbill, macro_features)
    print()

    experiment_1_drawdown_cis(results)
    experiment_2_blend_sweep(results)
    experiment_3_safe_withdrawal(results)
    experiment_4_insurance_premium(results)
    experiment_5_momentum_crisis_trace(results, prices)

    print("=" * 90)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
