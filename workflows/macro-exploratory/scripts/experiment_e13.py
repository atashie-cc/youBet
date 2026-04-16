"""E13 — E4 frozen quasi-holdout replication.

Reruns E4's exact construction (12-sleeve pooled regional SMA100, equal weight,
annual rebalance) but evaluates gate-v2 metrics ONLY on the post-2016 holdout
slice. The pre-2016 data is used for discovery (already seen in E4).

IMPORTANT CAVEAT: this is a quasi-holdout, not a true holdout. E4 already saw
the full 2003-2026 window, so the workflow owner has observed the post-2016
data. What this test CAN do: identify if E4's positive result depends
structurally on the early-sample regime (2003-2012, which contains the GFC and
is the strongest sub-period in E4 at ExS +0.714). What it CANNOT do: replace a
true prospective holdout.
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
    experiment = "e13_e4_holdout"
    sma_window = 100

    holdout_start = pd.Timestamp(cfg["holdout_e13"]["start"])
    holdout_end = pd.Timestamp(cfg["holdout_e13"]["end"])

    # --- Identical E4 construction ---
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

    print(f"\n[{experiment}] Running pooled simulation (identical to E4)...")
    pool_result = simulate_pooled_regional(
        regional_factors=regional_factors,
        regional_rf=regional_rf,
        strategy_factory=lambda: SMATrendFilter(window=sma_window),
        factor_names=FACTOR_NAMES,
        config=sim_cfg,
        borrow_spread_bps=0.0,
        rebalance_freq="A",
    )

    pool_returns = pool_result["pool_returns"]
    pool_benchmark = pool_result["pool_benchmark"]

    assert not pool_returns.isna().any(), "pool_returns has NaN"
    assert not pool_benchmark.isna().any(), "pool_benchmark has NaN"

    # --- Identity check vs E4's saved result ---
    print(f"\n[{experiment}] Identity check vs E4 saved result...")
    e4_json_path = WORKFLOW_ROOT / "results" / "e4_pooled_regional.json"
    if e4_json_path.exists():
        with open(e4_json_path) as f:
            e4_saved = json.load(f)
        e4_sharpe = e4_saved["comparisons"]["pool_vs_pool_benchmark"]["strategy_metrics"]["sharpe"]
        e4_cagr = e4_saved["comparisons"]["pool_vs_pool_benchmark"]["strategy_metrics"]["cagr"]

        full_m = compute_metrics(pool_returns, "full_pool")
        sharpe_match = abs(full_m["sharpe"] - e4_sharpe) < 0.0001
        cagr_match = abs(full_m["cagr"] - e4_cagr) < 0.0001
        print(
            f"  Full-sample Sharpe: E13={full_m['sharpe']:.6f} vs E4={e4_sharpe:.6f} "
            f"{'MATCH' if sharpe_match else 'DRIFT'}"
        )
        print(
            f"  Full-sample CAGR:   E13={full_m['cagr']:.6f} vs E4={e4_cagr:.6f} "
            f"{'MATCH' if cagr_match else 'DRIFT'}"
        )
        if not (sharpe_match and cagr_match):
            print(
                "  WARNING: E13 construction has drifted from E4's saved result. "
                "Check for data vintage changes (refetched snapshots) before "
                "trusting the holdout evaluation."
            )
    else:
        print("  E4 result JSON not found — skipping identity check")

    # --- Holdout slice ---
    print(f"\n[{experiment}] Evaluating holdout {holdout_start.date()} to {holdout_end.date()}...")

    holdout_mask = (pool_returns.index >= holdout_start) & (pool_returns.index <= holdout_end)
    pool_holdout = pool_returns.loc[holdout_mask]
    bench_holdout = pool_benchmark.loc[holdout_mask]

    print(f"  Holdout days: {len(pool_holdout)}")
    if len(pool_holdout) < 100:
        print("  WARNING: very short holdout window, results will be noisy")

    pool_h_m = compute_metrics(pool_holdout, "pool_holdout")
    bench_h_m = compute_metrics(bench_holdout, "pool_benchmark_holdout")

    print(f"  pool     Sharpe {pool_h_m['sharpe']:+.3f}  CAGR {pool_h_m['cagr']:+.2%}  MaxDD {pool_h_m['max_dd']:+.1%}")
    print(f"  bench    Sharpe {bench_h_m['sharpe']:+.3f}  CAGR {bench_h_m['cagr']:+.2%}  MaxDD {bench_h_m['max_dd']:+.1%}")

    # Bootstrap CI on holdout
    ci_holdout = bootstrap_excess_sharpe(
        pool_holdout, bench_holdout,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    print(
        f"  ExSharpe {ci_holdout['excess_sharpe_point']:+.3f} "
        f"[{ci_holdout['excess_sharpe_lower']:+.3f}, {ci_holdout['excess_sharpe_upper']:+.3f}]  "
        f"Sharpe-diff {ci_holdout['point_estimate']:+.3f}"
    )

    # Sub-period consistency on holdout sub-windows
    sub_holdout = subperiod_consistency(
        pool_holdout, bench_holdout, cfg["subperiods_e13"]
    )

    # Elevation check
    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci_holdout["excess_sharpe_point"],
        ci_lower=ci_holdout["excess_sharpe_lower"],
        subperiod_same_sign=sub_holdout["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_holdout["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # --- Secondary: pool vs US CMA SMA on holdout ---
    print(f"\n[{experiment}] Running US-only CMA SMA100 reference (for secondary comparison)...")
    us_cma_result = simulate_factor_timing(
        factor_returns=regional_factors["us"]["CMA"],
        rf_returns=regional_rf["us"],
        strategy=SMATrendFilter(window=sma_window),
        config=sim_cfg,
        factor_name="us_cma_sma100_ref",
        borrow_spread_bps=0.0,
    )
    us_cma_returns = us_cma_result.overall_returns

    common_idx = pool_holdout.index.intersection(us_cma_returns.index)
    pool_for_uscma = pool_holdout.loc[common_idx]
    uscma_aligned = us_cma_returns.loc[common_idx]

    ci_uscma = bootstrap_excess_sharpe(
        pool_for_uscma, uscma_aligned,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # --- Full-sample metrics (for comparison with E4, not for the gate) ---
    full_m = compute_metrics(pool_returns, "pool_full_sample")
    full_bench_m = compute_metrics(pool_benchmark, "pool_bench_full_sample")

    out = {
        "experiment": experiment,
        "description": (
            f"E4 frozen quasi-holdout replication. Identical 12-sleeve pooled "
            f"SMA{sma_window} construction, evaluated on holdout "
            f"{holdout_start.date()} to {holdout_end.date()} only."
        ),
        "parameters": {
            "regions": list(regional_factors.keys()),
            "factors": FACTOR_NAMES,
            "sma_window": sma_window,
            "rebalance_freq": "A",
            "holdout_start": str(holdout_start.date()),
            "holdout_end": str(holdout_end.date()),
            "n_sleeves": pool_result["n_sleeves"],
            "train_months": sim_cfg.train_months,
            "test_months": sim_cfg.test_months,
            "common_start": str(common_start.date()),
            "common_end": str(common_end.date()),
        },
        "comparisons": {
            "pool_vs_pool_benchmark_holdout": {
                "strategy_metrics": pool_h_m,
                "benchmark_metrics": bench_h_m,
                "excess_sharpe_ci": ci_holdout,
                "subperiods": sub_holdout,
            },
            "pool_vs_us_cma_sma_holdout": {
                "strategy_metrics": compute_metrics(pool_for_uscma, "pool_holdout_uscma_idx"),
                "benchmark_metrics": compute_metrics(uscma_aligned, "us_cma_sma_holdout"),
                "excess_sharpe_ci": ci_uscma,
                "note": "pool reindexed to US CMA dates, holdout window only",
            },
        },
        "identity_check": {
            "full_sample_sharpe": full_m["sharpe"],
            "full_sample_cagr": full_m["cagr"],
            "note": "Compare to E4 saved result for drift detection",
        },
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "primary_comparison": "pool_vs_pool_benchmark_holdout",
            "version": 2,
        },
        "elevation_version": 2,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["e4_paper_frozen"],
        "notes": [
            "Quasi-holdout, NOT true holdout — E4 already saw full 2003-2026 sample",
            "Paper portfolio (Ken French factors, long-short self-financing)",
            "Tests whether E4's +0.635 Sharpe-diff depends on early-sample regime (2003-2012)",
            "Gate evaluated on holdout sub-periods only, not on original E4 sub-periods",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))
    print(f"\nElevation (holdout): {'PASS' if elevation_pass else 'FAIL'}")
    for r in elevation_reasons:
        print(f"    {r}")
    print(f"\nSaved: {path}")

    return out


if __name__ == "__main__":
    main()
