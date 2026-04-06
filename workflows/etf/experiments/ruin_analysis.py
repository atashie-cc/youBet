"""Ruin probability analysis: bootstrap-simulated portfolio paths.

For each strategy, simulates 10,000 paths of 10/20/30-year holding periods
using block bootstrap to preserve autocorrelation. Reports:
  - P(portfolio < 50% of initial)
  - P(portfolio < 25% of initial) — severe ruin
  - Expected minimum portfolio value
  - Drawdown metrics with bootstrap CIs

Also produces a drawdown comparison table and Pareto frontier analysis.

Usage:
    python experiments/ruin_analysis.py
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
from youbet.etf.risk import compute_risk_metrics, risk_of_ruin
from youbet.etf.stats import stationary_block_bootstrap
from youbet.utils.io import load_config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

STRATEGIES_DIR = WORKFLOW_ROOT / "strategies"
N_BOOTSTRAP = 5_000
HORIZONS_YEARS = [10, 20, 30]
RUIN_THRESHOLDS = [0.50, 0.25]  # P(portfolio < X * initial)


def simulate_ruin_paths(
    daily_returns: pd.Series,
    n_sims: int,
    horizon_days: int,
    expected_block_length: int = 22,
    seed: int = 42,
) -> np.ndarray:
    """Simulate portfolio paths via block bootstrap.

    Returns array of shape (n_sims, horizon_days) with cumulative wealth ratios.
    """
    n_available = len(daily_returns)
    rng = np.random.default_rng(seed)
    values = daily_returns.values

    wealth_paths = np.ones((n_sims, horizon_days))

    for sim in range(n_sims):
        # Generate a bootstrapped return sequence of length horizon_days
        boot_returns = []
        pos = 0
        while pos < horizon_days:
            # Random block length from geometric distribution
            block_len = rng.geometric(1.0 / expected_block_length)
            start = rng.integers(0, n_available - 1)
            for j in range(block_len):
                if pos >= horizon_days:
                    break
                idx = (start + j) % n_available
                boot_returns.append(values[idx])
                pos += 1

        boot_returns = np.array(boot_returns[:horizon_days])
        wealth_paths[sim] = np.cumprod(1 + boot_returns)

    return wealth_paths


def analyze_strategy(
    name: str,
    daily_returns: pd.Series,
    bench_returns: pd.Series,
) -> dict:
    """Compute drawdown metrics and ruin probabilities for one strategy."""
    # Realized metrics from actual backtest
    cum_wealth = (1 + daily_returns).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    max_dd = float(drawdown.min())

    ann_return = float((cum_wealth.iloc[-1]) ** (252 / len(daily_returns)) - 1)
    ann_vol = float(daily_returns.std() * np.sqrt(252))
    sharpe = ann_return / ann_vol if ann_vol > 1e-10 else 0.0
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
    cvar_95 = float(daily_returns.sort_values().iloc[:max(1, int(len(daily_returns) * 0.05))].mean())
    sortino_denom = float(daily_returns[daily_returns < 0].std() * np.sqrt(252))
    sortino = ann_return / sortino_denom if sortino_denom > 1e-10 else 0.0

    # Drawdown duration
    is_dd = drawdown < -0.001
    if is_dd.any():
        groups = (~is_dd).cumsum()
        dd_groups = groups[is_dd]
        durations = []
        for _, group in dd_groups.groupby(dd_groups):
            durations.append((group.index[-1] - group.index[0]).days)
        max_dd_duration = max(durations) if durations else 0
    else:
        max_dd_duration = 0

    # Ruin probability via bootstrap simulation
    ruin_probs = {}
    for horizon_yr in HORIZONS_YEARS:
        horizon_days = horizon_yr * 252
        paths = simulate_ruin_paths(
            daily_returns, N_BOOTSTRAP, horizon_days, seed=42
        )
        min_wealth = paths.min(axis=1)  # Minimum wealth along each path
        final_wealth = paths[:, -1]

        for threshold in RUIN_THRESHOLDS:
            key = f"P(min<{threshold:.0%})_{horizon_yr}yr"
            ruin_probs[key] = float((min_wealth < threshold).mean())

        ruin_probs[f"median_final_{horizon_yr}yr"] = float(np.median(final_wealth))
        ruin_probs[f"p5_final_{horizon_yr}yr"] = float(np.percentile(final_wealth, 5))

    # Excess over benchmark
    bench_wealth = (1 + bench_returns).cumprod()
    bench_max_dd = float(((bench_wealth - bench_wealth.cummax()) / bench_wealth.cummax()).min())

    return {
        "strategy": name,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "max_dd_duration_days": max_dd_duration,
        "calmar": calmar,
        "cvar_95": cvar_95,
        "bench_max_dd": bench_max_dd,
        "dd_reduction": 1.0 - (max_dd / bench_max_dd) if bench_max_dd < 0 else 0.0,
        **ruin_probs,
    }


def load_all_strategy_results(prices, universe, tbill, macro_features):
    """Run all strategies through backtester, return dict of BacktestResults."""
    from strategies.vol_targeting.scripts.run import VolTargeting
    from strategies.momentum_rotation.scripts.run import MomentumRotation
    from strategies.macro_risk_composite.scripts.run import MacroRiskComposite
    from strategies.trend_following.scripts.run import TrendFollowing
    from strategies.dual_momentum.scripts.run import DualMomentum
    from strategies.sentiment_extremes.scripts.run import SentimentExtremes
    from strategies.vol_risk_premium.scripts.run import VolRiskPremium
    from strategies.asset_class_rotation.scripts.run import AssetClassRotation
    from strategies.sector_rotation.scripts.run import SectorRotation
    from strategies.factor_timing.scripts.run import FactorTiming
    from strategies.risk_parity.scripts.run import RiskParity
    from strategies.full_universe_momentum.scripts.run import FullUniverseMomentum
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

    # Subset of strategies — focus on top performers + VTI-timing strategies
    strategies = []

    vol_cfg = load_config(STRATEGIES_DIR / "vol_targeting" / "config.yaml")
    strategies.append(("vol_targeting", VolTargeting.from_config(vol_cfg)))

    tf_cfg = load_config(STRATEGIES_DIR / "trend_following" / "config.yaml")
    strategies.append(("trend_following", TrendFollowing.from_config(tf_cfg)))

    dm_cfg = load_config(STRATEGIES_DIR / "dual_momentum" / "config.yaml")
    strategies.append(("dual_momentum", DualMomentum.from_config(dm_cfg)))

    rp_cfg = load_config(STRATEGIES_DIR / "risk_parity" / "config.yaml")
    strategies.append(("risk_parity", RiskParity.from_config(rp_cfg)))

    acr_cfg = load_config(STRATEGIES_DIR / "asset_class_rotation" / "config.yaml")
    acr = AssetClassRotation.from_config(acr_cfg)
    acr.set_features(macro_features)
    strategies.append(("asset_class_rotation", acr))

    fum_cfg = load_config(STRATEGIES_DIR / "full_universe_momentum" / "config.yaml")
    strategies.append(("full_universe_momentum", FullUniverseMomentum.from_config(fum_cfg, universe)))

    sr_cfg = load_config(STRATEGIES_DIR / "sector_rotation" / "config.yaml")
    strategies.append(("sector_rotation", SectorRotation.from_config(sr_cfg)))

    se_cfg = load_config(STRATEGIES_DIR / "sentiment_extremes" / "config.yaml")
    se = SentimentExtremes.from_config(se_cfg)
    se.set_features(macro_features)
    strategies.append(("sentiment_extremes", se))

    # Blend strategies
    tf_strat = TrendFollowing.from_config(tf_cfg)
    rp_strat = RiskParity.from_config(rp_cfg)
    acr_strat2 = AssetClassRotation.from_config(acr_cfg)
    acr_strat2.set_features(macro_features)

    blend_top3 = CombinedBlend(
        members=[tf_strat, rp_strat, acr_strat2],
        blend_weights=[0.333, 0.333, 0.334],
        variant_name="top3_equal",
    )
    strategies.append(("blend_top3_equal", blend_top3))

    tf_strat2 = TrendFollowing.from_config(tf_cfg)
    rp_strat2 = RiskParity.from_config(rp_cfg)
    blend_tf_rp = CombinedBlend(
        members=[tf_strat2, rp_strat2],
        blend_weights=[0.5, 0.5],
        variant_name="trend_riskparity",
    )
    strategies.append(("blend_trend_riskparity", blend_tf_rp))

    for name, strat in strategies:
        print(f"  Running {name}...")
        results[name] = bt.run(strat, benchmark)

    return results


def main():
    print("=" * 90)
    print("RUIN PROBABILITY ANALYSIS")
    print("=" * 90)
    print()

    universe = load_universe()
    all_tickers = universe["ticker"].tolist()
    prices = fetch_prices(all_tickers, start="2003-01-01")
    tbill = fetch_tbill_rates(start="2003-01-01", allow_fallback=True)
    macro_features = fetch_all_tier1(start="2003-01-01")

    print("Running strategies...")
    results = load_all_strategy_results(prices, universe, tbill, macro_features)
    print()

    # Analyze each strategy
    analyses = []
    for name, result in results.items():
        a = analyze_strategy(name, result.overall_returns, result.benchmark_returns)
        analyses.append(a)

    # Sort by max drawdown (least negative = best)
    analyses.sort(key=lambda x: x["max_dd"], reverse=True)

    # --- Drawdown comparison table ---
    print("=" * 90)
    print("DRAWDOWN COMPARISON (sorted by max drawdown, best first)")
    print("=" * 90)
    print()
    print(f"{'Strategy':<28} {'Sharpe':>7} {'MaxDD':>8} {'DD Dur':>8} {'Calmar':>7} "
          f"{'CVaR95':>8} {'DD Red':>7}")
    print("-" * 80)
    for a in analyses:
        print(
            f"{a['strategy']:<28} {a['sharpe']:>7.3f} {a['max_dd']:>8.1%} "
            f"{a['max_dd_duration_days']:>7}d {a['calmar']:>7.3f} "
            f"{a['cvar_95']:>8.4f} {a['dd_reduction']:>7.0%}"
        )
    print()

    # --- Ruin probability table ---
    print("=" * 90)
    print("RUIN PROBABILITY (block bootstrap, 5,000 simulations)")
    print("=" * 90)
    print()
    for horizon in HORIZONS_YEARS:
        print(f"--- {horizon}-Year Horizon ---")
        print(f"{'Strategy':<28} {'P(min<50%)':>11} {'P(min<25%)':>11} "
              f"{'Median Final':>13} {'5th Pctl':>10}")
        print("-" * 78)
        for a in analyses:
            p50 = a[f"P(min<50%)_{horizon}yr"]
            p25 = a[f"P(min<25%)_{horizon}yr"]
            med = a[f"median_final_{horizon}yr"]
            p5 = a[f"p5_final_{horizon}yr"]
            print(
                f"{a['strategy']:<28} {p50:>11.1%} {p25:>11.1%} "
                f"{med:>12.2f}x {p5:>9.2f}x"
            )
        print()

    # --- Pareto frontier ---
    print("=" * 90)
    print("PARETO FRONTIER: Sharpe vs Max Drawdown")
    print("=" * 90)
    print()
    print("Pareto-optimal = not dominated on BOTH Sharpe AND MaxDD by any other strategy")
    print()

    pareto = []
    for a in analyses:
        dominated = False
        for b in analyses:
            if b["strategy"] == a["strategy"]:
                continue
            # b dominates a if b has higher Sharpe AND less negative MaxDD
            if b["sharpe"] >= a["sharpe"] and b["max_dd"] >= a["max_dd"]:
                if b["sharpe"] > a["sharpe"] or b["max_dd"] > a["max_dd"]:
                    dominated = True
                    break
        if not dominated:
            pareto.append(a)

    print(f"{'Strategy':<28} {'Sharpe':>7} {'MaxDD':>8} {'Ann Ret':>8} {'Pareto':>8}")
    print("-" * 65)
    for a in analyses:
        is_pareto = any(p["strategy"] == a["strategy"] for p in pareto)
        print(
            f"{a['strategy']:<28} {a['sharpe']:>7.3f} {a['max_dd']:>8.1%} "
            f"{a['ann_return']:>8.1%} {'  YES' if is_pareto else '':>8}"
        )
    print()
    print(f"Pareto-optimal strategies: {[p['strategy'] for p in pareto]}")

    return analyses


if __name__ == "__main__":
    main()
