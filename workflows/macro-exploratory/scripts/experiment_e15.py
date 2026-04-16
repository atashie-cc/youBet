"""E15 — E4 weekly-cadence robustness test.

Identical to E4 (12-sleeve pooled regional SMA100, equal weight, annual
rebalance) except each sleeve's SMA signal is evaluated at weekly boundaries
via CheckedFactorStrategy(inner=SMATrendFilter(100), check_period="W").

Tests whether E4's +0.635 Sharpe-difference depends on daily paper-data
smoothness. Factor-timing Phase 6 found weekly captures 85-90% of daily signal
quality; E4's pooled construction should be similar if the mechanism is honest.

Pass criterion: gate v2 on the primary comparison AND less than 15% Sharpe-diff
collapse relative to daily E4, consistent with the Phase 6 benchmark.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import pandas as pd

from youbet.factor.simulator import (
    CheckedFactorStrategy,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
    simulate_pooled_regional,
)

from experiment_e4 import (
    FACTOR_NAMES,
    INTL_REGIONS,
    SNAPSHOT_DIR,
    _load_all_regions,
    _slice_to_common_window,
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


def main():
    cfg = load_workflow_config()
    experiment = "e15_e4_weekly_cadence"
    sma_window = 100
    check_period = "W"

    print(f"[{experiment}] Loading all regional factors (identical to E4)...")
    regional_factors, regional_rf = _load_all_regions()

    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from = start_date - pd.DateOffset(years=int(train_years))

    regional_factors, regional_rf, common_start, common_end = _slice_to_common_window(
        regional_factors, regional_rf, slice_from,
    )
    print(f"  Common window: {common_start.date()} to {common_end.date()}")

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    # --- Weekly pooled simulation ---
    print(f"\n[{experiment}] Running pooled simulation (weekly cadence)...")
    pool_result = simulate_pooled_regional(
        regional_factors=regional_factors,
        regional_rf=regional_rf,
        strategy_factory=lambda: CheckedFactorStrategy(
            inner=SMATrendFilter(window=sma_window),
            check_period=check_period,
        ),
        factor_names=FACTOR_NAMES,
        config=sim_cfg,
        borrow_spread_bps=0.0,
        rebalance_freq="A",
    )

    pool_returns = pool_result["pool_returns"]
    pool_benchmark = pool_result["pool_benchmark"]

    assert not pool_returns.isna().any(), "pool_returns has NaN"
    assert not pool_benchmark.isna().any(), "pool_benchmark has NaN"

    pool_m = compute_metrics(pool_returns, f"pool_weekly_sma{sma_window}")
    bench_m = compute_metrics(pool_benchmark, "pool_benchmark_buyhold")

    print(f"  pool wk  Sharpe {pool_m['sharpe']:+.3f}  CAGR {pool_m['cagr']:+.2%}  MaxDD {pool_m['max_dd']:+.1%}")
    print(f"  bench    Sharpe {bench_m['sharpe']:+.3f}  CAGR {bench_m['cagr']:+.2%}  MaxDD {bench_m['max_dd']:+.1%}")

    # --- Signal cadence validation (persisted per Codex R1) ---
    sample_sleeve_label = list(pool_result["sleeve_results"].keys())[0]
    sample_sleeve = pool_result["sleeve_results"][sample_sleeve_label]
    sample_exp = pd.concat([fr.exposure for fr in sample_sleeve.fold_results])
    changes = sample_exp.diff().abs() > 1e-8
    change_days = sample_exp.index[changes]
    signal_validation = {"sampled_sleeve": sample_sleeve_label}
    if len(change_days) > 0:
        weekdays = change_days.dayofweek
        non_monday = int((weekdays != 0).sum())
        n_changes = len(change_days)
        pct_monday = 1.0 - float(non_monday) / n_changes
        signal_validation.update({
            "total_changes": n_changes,
            "monday_changes": n_changes - non_monday,
            "non_monday_changes": non_monday,
            "pct_monday": pct_monday,
        })
        print(
            f"  Signal change validation: {n_changes} changes, "
            f"{pct_monday:.0%} on Monday/first-of-week"
        )
        if non_monday > 5:
            print(
                f"  WARNING: {non_monday} signal changes on non-Monday -- "
                f"these may be first-of-week adjustments after holidays"
            )

    # --- Primary: pool weekly vs pool benchmark ---
    ci_primary = bootstrap_excess_sharpe(
        pool_returns, pool_benchmark,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    sub_primary = subperiod_consistency(pool_returns, pool_benchmark, cfg["subperiods"])

    print(
        f"  ExSharpe {ci_primary['excess_sharpe_point']:+.3f} "
        f"[{ci_primary['excess_sharpe_lower']:+.3f}, {ci_primary['excess_sharpe_upper']:+.3f}]  "
        f"Sharpe-diff {ci_primary['point_estimate']:+.3f}"
    )

    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci_primary["excess_sharpe_point"],
        ci_lower=ci_primary["excess_sharpe_lower"],
        subperiod_same_sign=sub_primary["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_primary["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # --- Collapse diagnostic vs E4 daily ---
    e4_sharpe_diff = None
    collapse_pct = None
    e4_json_path = WORKFLOW_ROOT / "results" / "e4_pooled_regional.json"
    if e4_json_path.exists():
        with open(e4_json_path) as f:
            e4_saved = json.load(f)
        e4_sharpe_diff = e4_saved["comparisons"]["pool_vs_pool_benchmark"]["excess_sharpe_ci"]["point_estimate"]
        weekly_sharpe_diff = ci_primary["point_estimate"]
        if e4_sharpe_diff > 0:
            collapse_pct = 1.0 - weekly_sharpe_diff / e4_sharpe_diff
        print(
            f"\n  Sharpe-diff collapse: E4 daily {e4_sharpe_diff:+.3f} -> "
            f"E15 weekly {weekly_sharpe_diff:+.3f}"
        )
        if collapse_pct is not None:
            print(f"  Collapse: {collapse_pct:+.1%} (threshold: <15% from Phase 6)")
            if collapse_pct < 0.15:
                print("  PASS — cadence-robust per Phase 6 benchmark")
            elif collapse_pct < 0.30:
                print("  MARGINAL — more collapse than expected")
            else:
                print("  FAIL — material cadence dependence detected")

    # --- Secondary: pool weekly vs US CMA SMA weekly ---
    print(f"\n[{experiment}] Running US-only CMA SMA100 weekly reference...")
    us_cma_result = simulate_factor_timing(
        factor_returns=regional_factors["us"]["CMA"],
        rf_returns=regional_rf["us"],
        strategy=CheckedFactorStrategy(
            inner=SMATrendFilter(window=sma_window),
            check_period=check_period,
        ),
        config=sim_cfg,
        factor_name="us_cma_sma100_weekly_ref",
        borrow_spread_bps=0.0,
    )
    us_cma_returns = us_cma_result.overall_returns

    common_idx = pool_returns.index.intersection(us_cma_returns.index)
    pool_for_uscma = pool_returns.loc[common_idx]
    uscma_aligned = us_cma_returns.loc[common_idx]

    ci_uscma = bootstrap_excess_sharpe(
        pool_for_uscma, uscma_aligned,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # Per-sleeve summaries
    sleeve_summary = {}
    for label, res in pool_result["sleeve_results"].items():
        sm = compute_metrics(res.overall_returns, label)
        bm = compute_metrics(res.benchmark_returns, f"{label}_bench")
        sleeve_summary[label] = {
            "sharpe": sm["sharpe"],
            "cagr": sm["cagr"],
            "max_dd": sm["max_dd"],
            "bench_sharpe": bm["sharpe"],
        }

    out = {
        "experiment": experiment,
        "description": (
            f"E4 weekly-cadence robustness test. Identical 12-sleeve pooled "
            f"SMA{sma_window} construction with CheckedFactorStrategy(check_period='W')."
        ),
        "parameters": {
            "regions": list(regional_factors.keys()),
            "factors": FACTOR_NAMES,
            "sma_window": sma_window,
            "check_period": check_period,
            "rebalance_freq": "A",
            "n_sleeves": pool_result["n_sleeves"],
            "train_months": sim_cfg.train_months,
            "test_months": sim_cfg.test_months,
            "common_start": str(common_start.date()),
            "common_end": str(common_end.date()),
        },
        "comparisons": {
            "pool_weekly_vs_pool_benchmark": {
                "strategy_metrics": pool_m,
                "benchmark_metrics": bench_m,
                "excess_sharpe_ci": ci_primary,
                "subperiods": sub_primary,
            },
            "pool_weekly_vs_us_cma_sma_weekly": {
                "strategy_metrics": compute_metrics(pool_for_uscma, "pool_weekly_uscma_idx"),
                "benchmark_metrics": compute_metrics(uscma_aligned, "us_cma_sma_weekly"),
                "excess_sharpe_ci": ci_uscma,
                "note": "pool reindexed to US CMA weekly dates",
            },
        },
        "cadence_collapse_diagnostic": {
            "e4_daily_sharpe_diff": e4_sharpe_diff,
            "e15_weekly_sharpe_diff": ci_primary["point_estimate"],
            "collapse_pct": collapse_pct,
            "threshold": 0.15,
            "source": "Factor-timing Phase 6: weekly captures 85-90% of daily alpha",
        },
        "signal_validation": signal_validation,
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "primary_comparison": "pool_weekly_vs_pool_benchmark",
            "version": 2,
        },
        "elevation_version": 2,
        "sleeve_summary": sleeve_summary,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["e4_paper_frozen"],
        "notes": [
            "Paper portfolio (Ken French factors, long-short self-financing)",
            "Weekly signal evaluation via CheckedFactorStrategy wrapper",
            "Tests cadence sensitivity — daily vs weekly decision frequency",
            "Phase 6 benchmark: weekly captures 85-90% of daily single-factor alpha",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))
    print(f"\nElevation: {'PASS' if elevation_pass else 'FAIL'}")
    for r in elevation_reasons:
        print(f"    {r}")
    print(f"\nSaved: {path}")

    return out


if __name__ == "__main__":
    main()
