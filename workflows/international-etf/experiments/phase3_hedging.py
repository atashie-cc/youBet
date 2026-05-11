"""Phase 3 — Currency hedging EXPLORATORY.

Per plan v1.3 (Round 1 codex review H1 fix): synthetic hedged-EAFE pre-2014
formula was wrong (ignored rate-differential carry), so this phase uses
ONLY real HEFA returns from inception 2014-01-31 forward. The 12-year
sample fails Phase 4 P3's mechanical equal-thirds requirement, so Phase 3
is EXPLORATORY: not in the workflow's Holm denominator, no PROCEED can
be claimed.

Three strategies (all 60/40 VTI/ex-US, annual rebalance, 2014-2026):
  C1a: 60% VTI / 40% VEA   (always-unhedged baseline)
  C1b: 60% VTI / 40% HEFA  (always-hedged baseline)
  C3 : 60% VTI / 40% [HEFA when DXY 12m > 0, else VEA]  (dynamic hedge)

The Phase 3 question is: does hedging the ex-US sleeve via HEFA, or
toggling it dynamically, change the verdict relative to always-VEA?

Note on costs: HEFA expense ratio (35bps) is in the universe CSV; the
backtester applies it as daily drag. An additional ~15 bps/yr hedge-
roll friction (per plan §L3) is NOT in the expense ratio. To approximate
it conservatively, this script reports results both at the universe-only
expense ratio AND with a +15bps friction drag added to HEFA.

Outputs:
  - artifacts/phase3_results.json
  - research/phase3_results.md
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(WORKFLOW_DIR / "experiments"))

from youbet.etf.backtester import Backtester, BacktestConfig            # noqa: E402
from youbet.etf.benchmark import BuyAndHold                              # noqa: E402

from phase0_efficiency import (                                          # noqa: E402
    build_cost_model,
    compute_inferential_stats,
    load_gate,
    load_prices,
    load_tbill_daily,
    load_workflow_config,
)
from strategies.dynamic_hedge import DynamicHedgeStrategy                # noqa: E402
from strategies.regime_signals import dxy_12m_positive_signal           # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


COST_LEVELS_BPS = [3.0, 5.0, 10.0]
HEDGE_FRICTION_VARIANTS_BPS = [0, 15]   # additional yr drag on HEFA (per plan L3)
PHASE3_START = "2014-01-31"             # HEFA inception
PHASE3_END = "2026-04-30"


def build_universe_with_hedge_friction(
    universe: pd.DataFrame, hedge_friction_bps_annual: float
) -> pd.DataFrame:
    """Bump HEFA expense ratio by hedge_friction_bps_annual."""
    u = universe.copy()
    if hedge_friction_bps_annual <= 0:
        return u
    extra = hedge_friction_bps_annual / 10_000.0
    mask = u["ticker"] == "HEFA"
    u.loc[mask, "expense_ratio"] = u.loc[mask, "expense_ratio"] + extra
    return u


def run_strategy(
    name: str,
    strategy_fn,
    tickers: list[str],
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
    intl_cost_bps: float,
    shared_index: pd.DatetimeIndex | None = None,
) -> dict:
    """Generic Phase 3 runner. strategy_fn() returns a BaseStrategy.

    shared_index forces all 3 strategies to use a single common date index
    (Round 2 review H3 fix). Without it, dropna(how="any") on different
    ticker subsets gave each strategy a slightly different start date.
    """
    cost_model = build_cost_model(universe, intl_cost_bps)

    sub = prices[tickers].dropna(how="any")
    if shared_index is not None:
        sub = sub.loc[sub.index.intersection(shared_index)]
    sub = sub.loc[sub.index >= PHASE3_START]
    sub = sub.loc[sub.index <= PHASE3_END]
    if sub.empty:
        raise RuntimeError(f"Empty price slice for {tickers} {PHASE3_START}..{PHASE3_END}")

    bt = Backtester(
        config=bt_config,
        prices=sub,
        cost_model=cost_model,
        tbill_rates=tbill,
        universe=universe,
    )
    strategy = strategy_fn()
    benchmark = BuyAndHold({"VTI": 1.0})
    result = bt.run(strategy, benchmark)

    return {
        "name": name,
        "intl_cost_bps": intl_cost_bps,
        "n_folds": len(result.fold_results),
        "test_start": result.overall_returns.index[0].date().isoformat(),
        "test_end": result.overall_returns.index[-1].date().isoformat(),
        "n_test_days": len(result.overall_returns),
        "strat_sharpe": result.overall_metrics.sharpe_ratio,
        "bench_sharpe": result.benchmark_metrics.sharpe_ratio,
        "strat_vol_annual": result.overall_metrics.annualized_volatility,
        "bench_vol_annual": result.benchmark_metrics.annualized_volatility,
        "excess_sharpe": result.excess_sharpe,
        "total_turnover": result.total_turnover,
        "total_cost_drag": result.total_cost_drag,
        "_strategy_returns": result.overall_returns,
        "_benchmark_returns": result.benchmark_returns,
    }


def make_strategies():
    """Return (name, factory, tickers) tuples for Phase 3."""
    dxy_pos = dxy_12m_positive_signal()
    return [
        ("C1a_VEA40", lambda: BuyAndHold({"VTI": 0.6, "VEA": 0.4}), ["VTI", "VEA"]),
        ("C1b_HEFA40", lambda: BuyAndHold({"VTI": 0.6, "HEFA": 0.4}), ["VTI", "HEFA"]),
        (
            "C3_dynhedge40",
            lambda: DynamicHedgeStrategy(
                signal=dxy_pos,
                hedged_ticker="HEFA",
                unhedged_ticker="VEA",
                ex_us_weight=0.40,
                name="C3_dynhedge40",
            ),
            ["VTI", "HEFA", "VEA"],
        ),
    ]


def main() -> None:
    print("=" * 70)
    print("PHASE 3: CURRENCY HEDGING EXPLORATORY (2014-2026, NOT IN HOLM DENOMINATOR)")
    print("60/40 VTI/ex-US: VEA vs HEFA vs Dynamic Hedge")
    print("=" * 70)

    cfg = load_workflow_config()
    gate = load_gate(cfg)
    print(f"Gate (LOCKED v1.3, exploratory only): ExSharpe > {gate.excess_sharpe_min}")
    print(f"Workflow tier: {cfg['gate'].get('workflow_tier')}")
    print(f"Sample window: {PHASE3_START} to {PHASE3_END}")
    print(f"Hedge friction variants tested: {HEDGE_FRICTION_VARIANTS_BPS} bps/yr extra HEFA drag")
    print()

    universe_base = pd.read_csv(WORKFLOW_DIR / "data" / "reference" / "international_universe.csv")
    universe_base["inception_date"] = pd.to_datetime(universe_base["inception_date"])

    prices = load_prices()
    tbill = load_tbill_daily(prices.index)

    # Round 2 review H3 fix: compute SHARED date index across ALL Phase 3 tickers
    # so C1a (VTI+VEA), C1b (VTI+HEFA), C3 (VTI+HEFA+VEA) all use identical windows.
    shared_index = prices[["VTI", "VEA", "HEFA"]].dropna(how="any").index
    print(f"Shared sample window (post-dropna): {shared_index.min().date()} to {shared_index.max().date()}, n={len(shared_index)}")
    print()

    bt_config = BacktestConfig(
        train_months=cfg["walk_forward"]["train_months"],
        test_months=cfg["walk_forward"]["test_months"],
        step_months=cfg["walk_forward"]["step_months"],
        rebalance_frequency="annual",
        initial_capital=cfg["walk_forward"]["initial_capital"],
    )

    strategies = make_strategies()
    print(f"Strategies: {[s[0] for s in strategies]}\n")

    runs = {}
    for friction_bps in HEDGE_FRICTION_VARIANTS_BPS:
        universe = build_universe_with_hedge_friction(universe_base, friction_bps)
        for name, factory, tickers in strategies:
            for cost_bps in COST_LEVELS_BPS:
                key = f"{name}_intl{int(cost_bps)}bps_friction{friction_bps:02d}"
                print(f"--- {key} ---")
                try:
                    run = run_strategy(
                        name, factory, tickers, prices, universe, tbill, bt_config, cost_bps,
                        shared_index=shared_index,
                    )
                    inf = compute_inferential_stats(run, n_bootstrap=10_000)
                    runs[key] = {**run, "hedge_friction_bps": friction_bps, **inf}
                    print(f"  sharpe_diff={inf['sharpe_diff_point']:+.3f}  "
                          f"raw_p={inf['raw_p_upper']:.4f}  "
                          f"ci=[{inf['sharpe_diff_ci_lower']:+.3f}, {inf['sharpe_diff_ci_upper']:+.3f}]  "
                          f"vol={run['strat_vol_annual']:.3f}  "
                          f"turnover={run['total_turnover']:.3f}")
                except Exception as e:
                    logger.exception("Failed %s", key)
                    runs[key] = {
                        "error": str(e),
                        "name": name,
                        "intl_cost_bps": cost_bps,
                        "hedge_friction_bps": friction_bps,
                    }

    # Print headline comparison: at strict cost (10 bps) and zero friction
    print()
    print("=" * 70)
    print("PHASE 3 HEADLINE — strict cost 10bps, friction variants both reported")
    print("=" * 70)
    print(f"{'Strategy':<20} {'Friction':>10} {'SharpeDiff':>12} {'CI_low':>10} {'CI_high':>10} {'Vol':>8} {'Turnover':>10}")
    for friction_bps in HEDGE_FRICTION_VARIANTS_BPS:
        for name, _, _ in strategies:
            key = f"{name}_intl10bps_friction{friction_bps:02d}"
            run = runs.get(key, {})
            if "sharpe_diff_point" in run:
                print(f"{name:<20} {friction_bps:>9}bp "
                      f"{run['sharpe_diff_point']:>+12.3f} "
                      f"{run['sharpe_diff_ci_lower']:>+10.3f} "
                      f"{run['sharpe_diff_ci_upper']:>+10.3f} "
                      f"{run['strat_vol_annual']:>8.3f} "
                      f"{run['total_turnover']:>10.3f}")
            else:
                print(f"{name:<20} {friction_bps:>9}bp ERROR")

    # Persist
    artifacts_dir = WORKFLOW_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    runs_serializable = {}
    for key, run in runs.items():
        runs_serializable[key] = {
            k: v for k, v in run.items() if not k.startswith("_")
        }

    out_path = artifacts_dir / "phase3_results.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "run_date": date.today().isoformat(),
                "gate": {
                    "excess_sharpe_min": gate.excess_sharpe_min,
                    "holm_p_max": gate.holm_p_max,
                    "ci_lower_min": gate.ci_lower_min,
                },
                "workflow_tier": cfg["gate"].get("workflow_tier"),
                "phase3_status": "EXPLORATORY ONLY — NOT IN HOLM DENOMINATOR",
                "phase3_strategies": [s[0] for s in strategies],
                "cost_levels_bps": COST_LEVELS_BPS,
                "hedge_friction_variants_bps": HEDGE_FRICTION_VARIANTS_BPS,
                "sample_window": [PHASE3_START, PHASE3_END],
                "runs": runs_serializable,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", out_path)

    # Markdown summary
    md = ["# Phase 3 — Currency Hedging Exploratory", ""]
    md.append(f"- Run date: {date.today()}")
    md.append("- **Status: EXPLORATORY ONLY — NOT IN HOLM DENOMINATOR**")
    md.append(f"- Sample: {PHASE3_START} to {PHASE3_END} ({(pd.to_datetime(PHASE3_END) - pd.to_datetime(PHASE3_START)).days/365.25:.1f} years; below P3 mechanical-thirds threshold)")
    md.append(f"- Workflow tier: **{cfg['gate'].get('workflow_tier')}**")
    md.append(f"- Gate (informational only): ExSharpe > {gate.excess_sharpe_min}, Holm p < {gate.holm_p_max}, CI_lower > 0")
    md.append("")
    md.append("## Headline comparison @ strict 10 bps cost")
    md.append("")
    md.append("| Strategy | HEFA friction | SharpeDiff | 90% CI | Vol | Turnover |")
    md.append("|---|---|---|---|---|---|")
    for friction_bps in HEDGE_FRICTION_VARIANTS_BPS:
        for name, _, _ in strategies:
            key = f"{name}_intl10bps_friction{friction_bps:02d}"
            run = runs.get(key, {})
            if "sharpe_diff_point" in run:
                md.append(
                    f"| {name} | +{friction_bps} bps | "
                    f"{run['sharpe_diff_point']:+.3f} | "
                    f"[{run['sharpe_diff_ci_lower']:+.3f}, {run['sharpe_diff_ci_upper']:+.3f}] | "
                    f"{run['strat_vol_annual']:.3f} | "
                    f"{run['total_turnover']:.3f} |"
                )
    md.append("")
    md.append("## Hedged-vs-unhedged delta (informational)")
    md.append("")
    md.append("| Friction | C1b HEFA - C1a VEA | C3 dyn - C1a VEA |")
    md.append("|---|---|---|")
    for friction_bps in HEDGE_FRICTION_VARIANTS_BPS:
        a = runs.get(f"C1a_VEA40_intl10bps_friction{friction_bps:02d}", {}).get("sharpe_diff_point")
        b = runs.get(f"C1b_HEFA40_intl10bps_friction{friction_bps:02d}", {}).get("sharpe_diff_point")
        c = runs.get(f"C3_dynhedge40_intl10bps_friction{friction_bps:02d}", {}).get("sharpe_diff_point")
        if a is not None and b is not None and c is not None:
            md.append(f"| +{friction_bps} bps | {b - a:+.3f} | {c - a:+.3f} |")
    md.append("")
    md.append("See `artifacts/phase3_results.json` for full per-cost breakdown.")

    md_path = WORKFLOW_DIR / "research" / "phase3_results.md"
    md_path.write_text("\n".join(md))
    logger.info("Wrote %s", md_path)
    print(f"\nResults: {out_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
