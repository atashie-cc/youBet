"""E2 — Conditionally-leveraged factor SMA paper portfolio.

Apply 1.5x leverage during the on-state of CMA/SMB/HML/RMW SMA100 factor
timing; 0 exposure during the off-state. Financing on the 0.5x leverage
increment = T-bill + 50 bps borrow spread, charged only during on-periods.

Benchmarks:
  - Unlevered factor SMA100 per factor (exposure in {0, 1})
  - CMA SMA100 locked benchmark: ExSharpe 0.687 from factor-timing Phase 6

Paper portfolio. Long-short Ken French factors. Not directly investable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import pandas as pd

from youbet.factor.data import load_french_snapshot
from youbet.factor.simulator import (
    ConditionallyLeveragedSMA,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)

from _common import (
    bootstrap_excess_sharpe,
    check_elevation,
    compute_metrics,
    format_report,
    load_workflow_config,
    save_result,
    subperiod_consistency,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

SNAPSHOT_DIR = REPO_ROOT / "workflows" / "factor-timing" / "data" / "snapshots"


def main():
    cfg = load_workflow_config()
    experiment = "e2_leveraged_factor_sma"
    factors_to_test = ["CMA", "SMB", "HML", "RMW"]
    on_leverage = 1.5
    borrow_spread_bps = cfg["costs"]["financing"]["borrow_spread_bps"]
    sma_window = 100

    print(f"[{experiment}] Loading Ken French factors...")
    ff_full = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    print(f"  Loaded {len(ff_full)} days, {ff_full.index[0].date()} to {ff_full.index[-1].date()}")

    # Enforce the workflow's locked start_date. Per Codex review, not slicing made
    # E2 score on ~53yr of history while other experiments run on the 2003+ window,
    # which inflated the headline and broke cross-experiment comparability.
    # The 120-month factor train window means the first test day is start_date + 10yr,
    # so we slice `train_months` before start_date to preserve walk-forward structure.
    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from = start_date - pd.DateOffset(years=int(train_years))
    ff = ff_full.loc[ff_full.index >= slice_from]
    print(f"  Sliced to {ff.index[0].date()} to {ff.index[-1].date()} "
          f"({len(ff)} days) for locked start_date {start_date.date()} "
          f"with {int(train_years)}yr warmup")

    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=ff.index)

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    comparisons = {}
    per_factor_elevation = {}

    for factor in factors_to_test:
        print(f"\n[{experiment}] Factor: {factor}")
        if factor not in ff.columns:
            print(f"  Missing in snapshot, skipping")
            continue

        # Leveraged variant
        lev_strategy = ConditionallyLeveragedSMA(
            window=sma_window, on_leverage=on_leverage, off_exposure=0.0
        )
        lev_result = simulate_factor_timing(
            factor_returns=ff[factor],
            rf_returns=rf,
            strategy=lev_strategy,
            config=sim_cfg,
            factor_name=factor,
            borrow_spread_bps=borrow_spread_bps,
        )

        # Unlevered SMA100 variant (same cash optionality, no leverage)
        unlev_strategy = SMATrendFilter(window=sma_window)
        unlev_result = simulate_factor_timing(
            factor_returns=ff[factor],
            rf_returns=rf,
            strategy=unlev_strategy,
            config=sim_cfg,
            factor_name=factor,
            borrow_spread_bps=0.0,
        )

        lev_ret = lev_result.overall_returns
        unlev_ret = unlev_result.overall_returns
        bench_ret = lev_result.benchmark_returns  # buy-and-hold factor

        lev_m = compute_metrics(lev_ret, f"{factor}_sma100_lev1.5")
        unlev_m = compute_metrics(unlev_ret, f"{factor}_sma100_unlev")
        bench_m = compute_metrics(bench_ret, f"{factor}_buyhold")

        print(f"  lev1.5   Sharpe {lev_m['sharpe']:+.3f}  CAGR {lev_m['cagr']:+.2%}  MaxDD {lev_m['max_dd']:+.1%}")
        print(f"  unlev    Sharpe {unlev_m['sharpe']:+.3f}  CAGR {unlev_m['cagr']:+.2%}  MaxDD {unlev_m['max_dd']:+.1%}")
        print(f"  buyhold  Sharpe {bench_m['sharpe']:+.3f}  CAGR {bench_m['cagr']:+.2%}  MaxDD {bench_m['max_dd']:+.1%}")

        # CI: leveraged vs unlevered SMA (isolate leverage contribution)
        ci_lev_vs_unlev = bootstrap_excess_sharpe(
            lev_ret, unlev_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )
        # CI: leveraged vs buy-and-hold factor (total vs benchmark)
        ci_lev_vs_bench = bootstrap_excess_sharpe(
            lev_ret, bench_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )
        # CI: unlevered SMA vs buy-and-hold (replication of known result)
        ci_unlev_vs_bench = bootstrap_excess_sharpe(
            unlev_ret, bench_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )

        sub = subperiod_consistency(lev_ret, bench_ret, cfg["subperiods"])

        elevation_pass, elevation_reasons = check_elevation(
            excess_sharpe_point=ci_lev_vs_bench["excess_sharpe_point"],
            ci_lower=ci_lev_vs_bench["excess_sharpe_lower"],
            subperiod_same_sign=sub["same_sign_positive_excess_sharpe"],
            sharpe_diff_point=ci_lev_vs_bench["point_estimate"],
            threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
        )

        comparisons[f"{factor}_lev_vs_buyhold"] = {
            "strategy_metrics": lev_m,
            "benchmark_metrics": bench_m,
            "excess_sharpe_ci": ci_lev_vs_bench,
            "subperiods": sub,
        }
        comparisons[f"{factor}_lev_vs_unlev_sma"] = {
            "strategy_metrics": lev_m,
            "benchmark_metrics": unlev_m,
            "excess_sharpe_ci": ci_lev_vs_unlev,
        }
        comparisons[f"{factor}_unlev_vs_buyhold_replication"] = {
            "strategy_metrics": unlev_m,
            "benchmark_metrics": bench_m,
            "excess_sharpe_ci": ci_unlev_vs_bench,
        }

        per_factor_elevation[factor] = {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
        }

    out = {
        "experiment": experiment,
        "description": f"Conditionally-leveraged ({on_leverage}x) factor SMA{sma_window} paper portfolio with {borrow_spread_bps}bps borrow spread",
        "parameters": {
            "factors": factors_to_test,
            "sma_window": sma_window,
            "on_leverage": on_leverage,
            "borrow_spread_bps": borrow_spread_bps,
            "train_months": sim_cfg.train_months,
            "test_months": sim_cfg.test_months,
        },
        "comparisons": comparisons,
        "per_factor_elevation": per_factor_elevation,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["factor_paper"],
        "notes": [
            "Paper portfolio (Ken French factors, long-short self-financing)",
            "Financing only on the leverage increment (exposure - 1) during on-periods",
            "No implementation costs (borrow availability, factor ETF tracking error not modeled)",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))

    # Summary table
    print("\nSummary (leveraged vs buy-and-hold factor):")
    print(f"{'Factor':<8} {'Lev Sharpe':>12} {'Bench Sharpe':>14} {'ExSharpe':>10} {'CI Lower':>10} {'CI Upper':>10}  Elevated?")
    print("-" * 80)
    for factor in factors_to_test:
        key = f"{factor}_lev_vs_buyhold"
        if key not in comparisons:
            continue
        cmp = comparisons[key]
        ci = cmp["excess_sharpe_ci"]
        ls = cmp["strategy_metrics"]["sharpe"]
        bs = cmp["benchmark_metrics"]["sharpe"]
        elev = per_factor_elevation[factor]["passed"]
        print(
            f"{factor:<8} {ls:>+12.3f} {bs:>+14.3f} "
            f"{ci['excess_sharpe_point']:>+10.3f} "
            f"{ci['excess_sharpe_lower']:>+10.3f} "
            f"{ci['excess_sharpe_upper']:>+10.3f}  "
            f"{'YES' if elev else 'no'}"
        )
    print(f"\nSaved: {path}")

    return out


if __name__ == "__main__":
    main()
