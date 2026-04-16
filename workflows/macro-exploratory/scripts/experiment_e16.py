"""E16 — Pooled UMD momentum timing across regions.

Combines E4's pooling mechanism with E12's finding that BSC/momentum is
the right factor for vol management. Two arms:
  A) UMD-only pool: {US, Dev ex-US, Europe, Japan} x {UMD} = 4 sleeves
  B) Expanded pool: {US, Dev ex-US, Europe, Japan} x {CMA, HML, RMW, UMD} = 16 sleeves

Tests whether momentum adds diversification value to the pooled construction.
UMD has momentum crashes (2001, 2009) correlated across regions, so the
4-sleeve pool may have less diversification than E4's cross-factor pool.

Paper portfolio (Ken French factors). Not directly investable.
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

import numpy as np
import pandas as pd

from youbet.factor.simulator import (
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
    simulate_pooled_regional,
)

from experiment_e4 import (
    FACTOR_NAMES as E4_FACTOR_NAMES,
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

UMD_FACTOR_NAMES = ["UMD"]
EXPANDED_FACTOR_NAMES = ["CMA", "HML", "RMW", "UMD"]


def _signal_correlation(pool_result: dict) -> dict:
    """Cross-region correlation of on/off signals across sleeves."""
    exposures = {}
    for label, res in pool_result["sleeve_results"].items():
        exp = pd.concat([fr.exposure for fr in res.fold_results])
        exposures[label] = (exp > 0.5).astype(float)
    if len(exposures) < 2:
        return {"n_sleeves": len(exposures), "note": "too few sleeves"}
    exp_df = pd.DataFrame(exposures)
    corr = exp_df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().dropna()
    return {
        "mean_pairwise_corr": float(pairs.mean()),
        "max_pairwise_corr": float(pairs.max()),
        "min_pairwise_corr": float(pairs.min()),
        "n_pairs": int(len(pairs)),
    }


def _run_arm(
    label: str,
    regional_factors: dict,
    regional_rf: dict,
    factor_names: list[str],
    sim_cfg: SimulationConfig,
    cfg: dict,
    sma_window: int = 100,
) -> dict:
    """Run one arm of E16 and return full comparison data."""
    print(f"\n[e16] Running {label} ({len(regional_factors)} regions x {len(factor_names)} factors)...")

    pool_result = simulate_pooled_regional(
        regional_factors=regional_factors,
        regional_rf=regional_rf,
        strategy_factory=lambda: SMATrendFilter(window=sma_window),
        factor_names=factor_names,
        config=sim_cfg,
        borrow_spread_bps=0.0,
        rebalance_freq="A",
    )

    pool_returns = pool_result["pool_returns"]
    pool_benchmark = pool_result["pool_benchmark"]
    n_sleeves = pool_result["n_sleeves"]

    assert not pool_returns.isna().any(), f"{label}: pool_returns has NaN"
    assert not pool_benchmark.isna().any(), f"{label}: pool_benchmark has NaN"

    pool_m = compute_metrics(pool_returns, f"{label}_pool")
    bench_m = compute_metrics(pool_benchmark, f"{label}_benchmark")

    print(f"  pool   Sharpe {pool_m['sharpe']:+.3f}  CAGR {pool_m['cagr']:+.2%}  MaxDD {pool_m['max_dd']:+.1%}")
    print(f"  bench  Sharpe {bench_m['sharpe']:+.3f}  CAGR {bench_m['cagr']:+.2%}  MaxDD {bench_m['max_dd']:+.1%}")

    ci = bootstrap_excess_sharpe(
        pool_returns, pool_benchmark,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    sub = subperiod_consistency(pool_returns, pool_benchmark, cfg["subperiods"])

    print(
        f"  ExSharpe {ci['excess_sharpe_point']:+.3f} "
        f"[{ci['excess_sharpe_lower']:+.3f}, {ci['excess_sharpe_upper']:+.3f}]  "
        f"Sharpe-diff {ci['point_estimate']:+.3f}"
    )

    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci["excess_sharpe_point"],
        ci_lower=ci["excess_sharpe_lower"],
        subperiod_same_sign=sub["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    signal_corr = _signal_correlation(pool_result)
    print(f"  Signal correlation: mean {signal_corr.get('mean_pairwise_corr', 'N/A'):.3f}")

    sleeve_summary = {}
    for slabel, res in pool_result["sleeve_results"].items():
        sm = compute_metrics(res.overall_returns, slabel)
        bm = compute_metrics(res.benchmark_returns, f"{slabel}_bench")
        sleeve_summary[slabel] = {
            "sharpe": sm["sharpe"],
            "cagr": sm["cagr"],
            "max_dd": sm["max_dd"],
            "bench_sharpe": bm["sharpe"],
        }

    return {
        "pool_returns": pool_returns,
        "pool_benchmark": pool_benchmark,
        "comparison": {
            "strategy_metrics": pool_m,
            "benchmark_metrics": bench_m,
            "excess_sharpe_ci": ci,
            "subperiods": sub,
        },
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "version": 2,
        },
        "signal_correlation": signal_corr,
        "sleeve_summary": sleeve_summary,
        "n_sleeves": n_sleeves,
    }


def main():
    cfg = load_workflow_config()
    experiment = "e16_pooled_umd"
    sma_window = 100

    print(f"[{experiment}] Loading all regional factors...")
    regional_factors, regional_rf = _load_all_regions()

    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from = start_date - pd.DateOffset(years=int(train_years))

    regional_factors, regional_rf, common_start, common_end = _slice_to_common_window(
        regional_factors, regional_rf, slice_from,
    )
    print(f"  Common window: {common_start.date()} to {common_end.date()}")

    for region, df in regional_factors.items():
        has_umd = "UMD" in df.columns
        print(f"  {region:18s} {len(df)} days  UMD: {'YES' if has_umd else 'MISSING'}")
        assert has_umd, f"{region} missing UMD column"

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    # --- Arm A: UMD-only pool (4 sleeves) ---
    arm_a = _run_arm(
        "arm_a_umd_only", regional_factors, regional_rf,
        UMD_FACTOR_NAMES, sim_cfg, cfg, sma_window,
    )

    # --- Arm B: Expanded 16-sleeve pool ---
    arm_b = _run_arm(
        "arm_b_expanded_16", regional_factors, regional_rf,
        EXPANDED_FACTOR_NAMES, sim_cfg, cfg, sma_window,
    )

    # --- US-only UMD SMA reference (for diversification isolation) ---
    print(f"\n[{experiment}] Running US-only UMD SMA100 reference...")
    us_umd_result = simulate_factor_timing(
        factor_returns=regional_factors["us"]["UMD"],
        rf_returns=regional_rf["us"],
        strategy=SMATrendFilter(window=sma_window),
        config=sim_cfg,
        factor_name="us_umd_sma100_ref",
        borrow_spread_bps=0.0,
    )
    us_umd_returns = us_umd_result.overall_returns

    # Arm A vs US-only UMD SMA (international diversification value)
    common_idx = arm_a["pool_returns"].index.intersection(us_umd_returns.index)
    ci_pool_vs_us = bootstrap_excess_sharpe(
        arm_a["pool_returns"].loc[common_idx],
        us_umd_returns.loc[common_idx],
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # Arm B expanded vs E4 original (does adding UMD improve E4?)
    e4_json_path = WORKFLOW_ROOT / "results" / "e4_pooled_regional.json"
    e4_sharpe_diff = None
    if e4_json_path.exists():
        with open(e4_json_path) as f:
            e4_saved = json.load(f)
        e4_sharpe_diff = e4_saved["comparisons"]["pool_vs_pool_benchmark"]["excess_sharpe_ci"]["point_estimate"]

    out = {
        "experiment": experiment,
        "description": (
            f"Pooled UMD momentum timing. Arm A: 4 sleeves (4 regions x UMD only). "
            f"Arm B: 16 sleeves (4 regions x CMA/HML/RMW/UMD). SMA{sma_window}, "
            f"equal weight, annual rebalance."
        ),
        "parameters": {
            "regions": list(regional_factors.keys()),
            "arm_a_factors": UMD_FACTOR_NAMES,
            "arm_b_factors": EXPANDED_FACTOR_NAMES,
            "sma_window": sma_window,
            "rebalance_freq": "A",
            "train_months": sim_cfg.train_months,
            "test_months": sim_cfg.test_months,
            "common_start": str(common_start.date()),
            "common_end": str(common_end.date()),
        },
        "comparisons": {
            "umd_pool_vs_umd_buyhold": arm_a["comparison"],
            "expanded_pool_vs_expanded_buyhold": arm_b["comparison"],
            "umd_pool_vs_us_umd_sma": {
                "strategy_metrics": compute_metrics(arm_a["pool_returns"].loc[common_idx], "umd_pool_on_us_idx"),
                "benchmark_metrics": compute_metrics(us_umd_returns.loc[common_idx], "us_umd_sma100"),
                "excess_sharpe_ci": ci_pool_vs_us,
                "note": "Does international UMD pooling beat US-only UMD timing?",
            },
        },
        "arm_a": {
            "elevation": arm_a["elevation"],
            "signal_correlation": arm_a["signal_correlation"],
            "sleeve_summary": arm_a["sleeve_summary"],
            "n_sleeves": arm_a["n_sleeves"],
        },
        "arm_b": {
            "elevation": arm_b["elevation"],
            "signal_correlation": arm_b["signal_correlation"],
            "sleeve_summary": arm_b["sleeve_summary"],
            "n_sleeves": arm_b["n_sleeves"],
            "e4_sharpe_diff_reference": e4_sharpe_diff,
            "expanded_sharpe_diff": arm_b["comparison"]["excess_sharpe_ci"]["point_estimate"],
        },
        "elevation_version": 2,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["factor_paper"],
        "notes": [
            "Paper portfolio (Ken French factors, long-short self-financing)",
            "UMD has correlated momentum crashes across regions — less diversification than CMA/HML/RMW",
            "Arm B tests whether adding UMD to E4's CMA/HML/RMW improves the pool",
            "Signal correlation diagnostic flags whether UMD sleeves fire together (diversification loss)",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))

    print(f"\nArm A (UMD-only): {'PASS' if arm_a['elevation']['passed'] else 'FAIL'}")
    for r in arm_a["elevation"]["reasons"]:
        print(f"    {r}")
    print(f"\nArm B (expanded): {'PASS' if arm_b['elevation']['passed'] else 'FAIL'}")
    for r in arm_b["elevation"]["reasons"]:
        print(f"    {r}")

    if e4_sharpe_diff is not None:
        exp_diff = arm_b["comparison"]["excess_sharpe_ci"]["point_estimate"]
        print(f"\nE4 Sharpe-diff: {e4_sharpe_diff:+.3f} -> Expanded: {exp_diff:+.3f}")

    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
