"""Phase 0: Market Efficiency Test — strict gate + CI diagnostic.

Tests Phase 0 candidates (vol targeting, momentum rotation) against VTI
buy-and-hold with pre-committed default parameters.

## Dual-track reporting

**STRICT GATE (authoritative)**: strategy PROCEEDS iff all three hold:
  - excess_sharpe_point > 0.20  (economically meaningful, per CLAUDE.md #1)
  - holm_adjusted_p < 0.05       (statistically significant)
  - ci_lower > 0                 (positive with uncertainty accounted for)

**CI DIAGNOSTIC (interpretation only)**: 90% bootstrap CI on Sharpe diff.
  Explains why the gate passed/failed. Does NOT override the strict gate.

  Diagnostic tiers:
    STRONG_EDGE:           CI lower > 0.10
    WEAK_EDGE:             CI lower > 0
    INCONCLUSIVE_POSITIVE: CI spans zero, upper > 0.10
    INCONCLUSIVE:          CI spans zero, small magnitude
    NEGATIVE:              CI upper < 0

Usage:
    python experiments/efficiency_test.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT))  # for strategy imports

import pandas as pd

from youbet.etf.backtester import Backtester, BacktestConfig
from youbet.etf.benchmark import BuyAndHold
from youbet.etf.costs import CostModel
from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.macro.fetchers import fetch_all_tier1
from youbet.etf.stats import (
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)
from youbet.utils.io import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

STRATEGIES_DIR = WORKFLOW_ROOT / "strategies"

# STRICT GATE thresholds — LOCKED per CLAUDE.md #18
STRICT_MIN_EXCESS_SHARPE = 0.20
STRICT_MAX_PVALUE = 0.05


def evaluate_strict_gate(
    excess_sharpe_point: float,
    holm_adjusted_p: float,
    ci_lower: float,
) -> tuple[bool, list[str]]:
    """Apply the pre-committed strict gate.

    Returns (passed, reasons) where reasons lists why each criterion
    passed or failed, for transparent reporting.
    """
    passed_magnitude = excess_sharpe_point > STRICT_MIN_EXCESS_SHARPE
    passed_pvalue = holm_adjusted_p < STRICT_MAX_PVALUE
    passed_ci = ci_lower > 0

    reasons = []
    reasons.append(
        f"ExSharpe > {STRICT_MIN_EXCESS_SHARPE}: "
        f"{'PASS' if passed_magnitude else 'FAIL'} "
        f"(point={excess_sharpe_point:+.3f})"
    )
    reasons.append(
        f"Holm p < {STRICT_MAX_PVALUE}: "
        f"{'PASS' if passed_pvalue else 'FAIL'} "
        f"(p={holm_adjusted_p:.4f})"
    )
    reasons.append(
        f"CI lower > 0: "
        f"{'PASS' if passed_ci else 'FAIL'} "
        f"(lower={ci_lower:+.3f})"
    )

    return (passed_magnitude and passed_pvalue and passed_ci), reasons


def format_verdict(verdict: str) -> str:
    """Color-free verdict formatter for terminal output."""
    tags = {
        "STRONG_EDGE": "[STRONG EDGE]",
        "WEAK_EDGE": "[WEAK EDGE]",
        "INCONCLUSIVE_POSITIVE": "[INCONCLUSIVE-POS]",
        "INCONCLUSIVE": "[INCONCLUSIVE]",
        "NEGATIVE": "[NEGATIVE]",
    }
    return tags.get(verdict, verdict)


def main():
    print("=" * 70)
    print("PHASE 0: MARKET EFFICIENCY TEST")
    print("Does ANY systematic strategy beat buy-and-hold VTI?")
    print("=" * 70)
    print()

    # Load universe
    universe = load_universe()
    cost_model = CostModel.from_universe(universe)

    all_tickers = universe["ticker"].tolist()
    for required in ["VTI", "VGSH"]:
        if required not in all_tickers:
            all_tickers.append(required)

    print("Fetching price data...")
    prices = fetch_prices(all_tickers, start="2003-01-01")
    print(f"  Loaded {len(prices)} days, {len(prices.columns)} tickers")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print()

    print("Fetching T-bill rates...")
    # allow_fallback=True for now; authoritative runs should set FRED_API_KEY
    # or cache T-bill data to data/reference/tbill_3m_cache.csv
    tbill = fetch_tbill_rates(start="2003-01-01", allow_fallback=True)
    print(f"  Loaded {len(tbill)} days of T-bill rates")
    print()

    config = BacktestConfig(
        train_months=36,
        test_months=12,
        step_months=12,
        rebalance_frequency="monthly",
        initial_capital=100_000,
    )

    benchmark = BuyAndHold({"VTI": 1.0})

    bt = Backtester(
        config=config,
        prices=prices,
        cost_model=cost_model,
        tbill_rates=tbill,
        universe=universe,
    )

    # Run strategies
    print("=" * 70)
    print("Running strategies...")
    print("=" * 70)
    print()

    from strategies.vol_targeting.scripts.run import VolTargeting
    from strategies.momentum_rotation.scripts.run import MomentumRotation
    from strategies.macro_risk_composite.scripts.run import MacroRiskComposite
    from strategies.ml_signals.scripts.run import (
        MLLogistic, MLRidge, MLXGBoost, MLLightGBM, MLEnsemble,
    )
    from strategies.trend_following.scripts.run import TrendFollowing
    from strategies.dual_momentum.scripts.run import DualMomentum
    from strategies.sentiment_extremes.scripts.run import SentimentExtremes
    from strategies.vol_risk_premium.scripts.run import VolRiskPremium
    from strategies.asset_class_rotation.scripts.run import AssetClassRotation
    from strategies.sector_rotation.scripts.run import SectorRotation
    from strategies.factor_timing.scripts.run import FactorTiming
    from strategies.risk_parity.scripts.run import RiskParity
    from strategies.hierarchical_ml.scripts.run import HierarchicalML
    from strategies.full_universe_momentum.scripts.run import FullUniverseMomentum

    # ---- Fetch shared data ----
    print("Fetching Tier 1 macro features...")
    macro_features = fetch_all_tier1(start="2003-01-01")
    print(f"  Loaded {len(macro_features)} macro features: {list(macro_features.keys())}")
    print()

    results_by_name = {}

    def run_strategy(name, strategy):
        print(f"--- {name} ---")
        result = bt.run(strategy, benchmark)
        print(result.summary())
        print()
        results_by_name[name] = result

    # ---- Phase 0 strategies (rule-based, existing) ----
    vol_config = load_config(STRATEGIES_DIR / "vol_targeting" / "config.yaml")
    run_strategy("vol_targeting", VolTargeting.from_config(vol_config))

    mom_config = load_config(STRATEGIES_DIR / "momentum_rotation" / "config.yaml")
    mom_strategy = MomentumRotation.from_config(mom_config, universe)
    run_strategy("momentum_rotation", mom_strategy)

    macro_config = load_config(STRATEGIES_DIR / "macro_risk_composite" / "config.yaml")
    macro_strategy = MacroRiskComposite.from_config(macro_config)
    macro_strategy.set_features(macro_features)
    run_strategy("macro_risk_composite", macro_strategy)

    # ---- ML strategies (Phase A) ----
    ml_config = load_config(STRATEGIES_DIR / "ml_signals" / "config.yaml")

    for MLClass in [MLLogistic, MLRidge, MLXGBoost, MLLightGBM, MLEnsemble]:
        strat = MLClass.from_config(ml_config)
        strat.set_features(macro_features, tbill)
        run_strategy(strat.name, strat)

    # ---- Catalog strategies (Phase B) ----
    tf_config = load_config(STRATEGIES_DIR / "trend_following" / "config.yaml")
    run_strategy("trend_following", TrendFollowing.from_config(tf_config))

    dm_config = load_config(STRATEGIES_DIR / "dual_momentum" / "config.yaml")
    run_strategy("dual_momentum", DualMomentum.from_config(dm_config))

    se_config = load_config(STRATEGIES_DIR / "sentiment_extremes" / "config.yaml")
    se_strat = SentimentExtremes.from_config(se_config)
    se_strat.set_features(macro_features)
    run_strategy("sentiment_extremes", se_strat)

    vrp_config = load_config(STRATEGIES_DIR / "vol_risk_premium" / "config.yaml")
    vrp_strat = VolRiskPremium.from_config(vrp_config)
    vrp_strat.set_features(macro_features)
    run_strategy("vol_risk_premium", vrp_strat)

    # ---- Multi-asset strategies (52-ETF universe) ----
    acr_config = load_config(STRATEGIES_DIR / "asset_class_rotation" / "config.yaml")
    acr_strat = AssetClassRotation.from_config(acr_config)
    acr_strat.set_features(macro_features)
    run_strategy("asset_class_rotation", acr_strat)

    sr_config = load_config(STRATEGIES_DIR / "sector_rotation" / "config.yaml")
    run_strategy("sector_rotation", SectorRotation.from_config(sr_config))

    ft_config = load_config(STRATEGIES_DIR / "factor_timing" / "config.yaml")
    ft_strat = FactorTiming.from_config(ft_config)
    ft_strat.set_features(macro_features)
    run_strategy("factor_timing", ft_strat)

    rp_config = load_config(STRATEGIES_DIR / "risk_parity" / "config.yaml")
    run_strategy("risk_parity", RiskParity.from_config(rp_config))

    fum_config = load_config(STRATEGIES_DIR / "full_universe_momentum" / "config.yaml")
    run_strategy("full_universe_momentum", FullUniverseMomentum.from_config(fum_config, universe))

    hml_config = load_config(STRATEGIES_DIR / "hierarchical_ml" / "config.yaml")
    hml_strat = HierarchicalML.from_config(hml_config)
    hml_strat.set_features(macro_features, tbill)
    run_strategy("hierarchical_ml", hml_strat)

    # Compute CI + hypothesis tests for all strategies (used by both tracks)
    ci_results = {}
    ht_results = {}
    for name, result in results_by_name.items():
        ci_results[name] = excess_sharpe_ci(
            result.overall_returns,
            result.benchmark_returns,
            n_bootstrap=10_000,
            confidence=0.90,
            expected_block_length=22,
        )
        ht_results[name] = block_bootstrap_test(
            result.overall_returns,
            result.benchmark_returns,
            n_bootstrap=10_000,
            expected_block_length=22,
        )

    raw_ps = {name: r["p_value"] for name, r in ht_results.items()}
    corrected = holm_bonferroni(raw_ps)

    # ========================================================================
    # STRICT GATE (AUTHORITATIVE) — per CLAUDE.md #1 and #18
    # ========================================================================
    print("=" * 72)
    print("STRICT GATE (AUTHORITATIVE) — pre-committed thresholds")
    print("=" * 72)
    print(f"Criteria: ExSharpe > {STRICT_MIN_EXCESS_SHARPE} "
          f"AND Holm p < {STRICT_MAX_PVALUE} AND CI lower > 0")
    print()
    print(f"{'Strategy':<26} {'ExSharpe':>9} {'Holm p':>9} {'CI Lower':>10}  GATE")
    print("-" * 69)

    gate_results = {}
    any_passed = False
    for name in results_by_name:
        ci = ci_results[name]
        adj_p = corrected[name]["adjusted_p"]
        passed, reasons = evaluate_strict_gate(
            excess_sharpe_point=ci["point_estimate"],
            holm_adjusted_p=adj_p,
            ci_lower=ci["ci_lower"],
        )
        gate_results[name] = {"passed": passed, "reasons": reasons}
        if passed:
            any_passed = True
        print(
            f"{name:<26} "
            f"{ci['point_estimate']:>+9.3f} "
            f"{adj_p:>9.4f} "
            f"{ci['ci_lower']:>+10.3f}  "
            f"{'PASS' if passed else 'FAIL'}"
        )
    print()
    if any_passed:
        print("RESULT: At least one strategy PASSES the strict gate.")
        print("        AUTHORIZED to PROCEED to Phase 1 for passing strategies.")
    else:
        print("RESULT: No strategy passes the strict gate.")
        print("        Phase 0 does NOT authorize progression.")
    print()

    # ========================================================================
    # CI DIAGNOSTIC (INTERPRETATION ONLY) — does not override strict gate
    # ========================================================================
    print("=" * 72)
    print("CI DIAGNOSTIC (interpretation only — does not override strict gate)")
    print("=" * 72)
    print()
    print(f"{'Strategy':<26} {'Strat SR':>9} {'Bench SR':>9} {'Diff':>8} "
          f"{'90% CI':>20}  Diagnostic")
    print("-" * 102)
    for name in results_by_name:
        ci = ci_results[name]
        ci_str = f"[{ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}]"
        print(
            f"{name:<26} {ci['strategy_sharpe']:>9.3f} "
            f"{ci['benchmark_sharpe']:>9.3f} "
            f"{ci['point_estimate']:>+8.3f} {ci_str:>20}  "
            f"{format_verdict(ci['diagnostic_verdict'])}"
        )
    print()
    print("Supplementary: Sharpe of excess returns (active return signal)")
    print(f"{'Strategy':<26} {'Point':>8} {'90% CI':>22}")
    print("-" * 59)
    for name, ci in ci_results.items():
        ci_str = f"[{ci['excess_sharpe_lower']:+.3f}, {ci['excess_sharpe_upper']:+.3f}]"
        print(
            f"{name:<26} {ci['excess_sharpe_point']:>+8.3f} {ci_str:>22}"
        )
    print()

    # ========================================================================
    # Per-strategy breakdown: gate decision + why
    # ========================================================================
    print("=" * 72)
    print("PER-STRATEGY BREAKDOWN")
    print("=" * 72)
    for name in results_by_name:
        ci = ci_results[name]
        gate = gate_results[name]
        print()
        print(f"  {name}:")
        print(f"    Strict gate:        {'PASS' if gate['passed'] else 'FAIL'}")
        for r in gate["reasons"]:
            print(f"      - {r}")
        print(f"    CI diagnostic:      "
              f"{format_verdict(ci['diagnostic_verdict'])}")
        print(f"    Interpretation:     "
              f"ExSharpe {ci['point_estimate']:+.3f} "
              f"[90% CI: {ci['ci_lower']:+.3f}, {ci['ci_upper']:+.3f}]")

    return ci_results, ht_results, corrected, gate_results


if __name__ == "__main__":
    main()
