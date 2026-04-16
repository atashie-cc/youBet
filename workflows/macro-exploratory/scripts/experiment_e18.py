"""E18 — True OOS: monthly frequency + Asia-Pac absorption.

Two independent OOS dimensions for E4's pooled factor-vs-cash mechanism:

Arm A: Run E4's exact construction on MONTHLY Ken French factor returns.
  Monthly data is a different series (not resampled daily). SMA window = 5
  months (pre-committed, ~100 trading days / 20 days per month).
  Tests: does the mechanism depend on daily noise/SMA dynamics?

Arm B: Add Asia-Pacific ex-Japan as a 5th region to E4's daily construction.
  Phase 7 found Asia-Pac CMA/HML/RMW timing FAILS individually (ExS -0.212,
  whipsaw). Tests: can pooling diversification absorb a known-bad region?
  Pass criterion: degradation < 20% of E4's Sharpe-diff.
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

from youbet.factor.data import (
    fetch_french_factors,
    fetch_international_factors,
    load_french_snapshot,
)
from youbet.factor.simulator import (
    SMATrendFilter,
    SimulationConfig,
    simulate_pooled_regional,
)

from experiment_e4 import (
    FACTOR_NAMES,
    SNAPSHOT_DIR,
)

from experiment_e12 import compute_metrics_monthly

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

E4_INTL_REGIONS = ["developed_ex_us", "europe", "japan"]
ARM_B_INTL_REGIONS = ["developed_ex_us", "europe", "japan", "asia_pacific_ex_japan"]
MONTHLY_SMA_WINDOW = 5


def _load_monthly_regions() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Load monthly factor data for US + international regions."""
    regional_factors: dict[str, pd.DataFrame] = {}
    regional_rf: dict[str, pd.Series] = {}

    # US monthly
    try:
        us_monthly = load_french_snapshot(SNAPSHOT_DIR, frequency="monthly")
    except FileNotFoundError:
        us_monthly = fetch_french_factors(frequency="monthly", snapshot_dir=SNAPSHOT_DIR)
    regional_factors["us"] = us_monthly
    regional_rf["us"] = us_monthly["RF"] if "RF" in us_monthly.columns else pd.Series(0.0, index=us_monthly.index)

    # International monthly
    for region in E4_INTL_REGIONS:
        df = fetch_international_factors(region, snapshot_dir=SNAPSHOT_DIR, frequency="monthly")
        regional_factors[region] = df
        regional_rf[region] = df["RF"] if "RF" in df.columns else pd.Series(0.0, index=df.index)

    return regional_factors, regional_rf


def _load_daily_regions_with_asiapac() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Load daily factor data for 5 regions (E4's 4 + Asia-Pac)."""
    regional_factors: dict[str, pd.DataFrame] = {}
    regional_rf: dict[str, pd.Series] = {}

    us_df = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    regional_factors["us"] = us_df
    regional_rf["us"] = us_df["RF"] if "RF" in us_df.columns else pd.Series(0.0, index=us_df.index)

    for region in ARM_B_INTL_REGIONS:
        df = fetch_international_factors(region, snapshot_dir=SNAPSHOT_DIR)
        regional_factors[region] = df
        regional_rf[region] = df["RF"] if "RF" in df.columns else pd.Series(0.0, index=df.index)

    return regional_factors, regional_rf


def _slice_to_common(
    regional_factors: dict[str, pd.DataFrame],
    regional_rf: dict[str, pd.Series],
    slice_from: pd.Timestamp,
) -> tuple[dict, dict, pd.Timestamp, pd.Timestamp]:
    starts = [df.index[0] for df in regional_factors.values()]
    ends = [df.index[-1] for df in regional_factors.values()]
    common_start = max(slice_from, max(starts))
    common_end = min(ends)
    sliced_f, sliced_r = {}, {}
    for region, df in regional_factors.items():
        mask = (df.index >= common_start) & (df.index <= common_end)
        sliced_f[region] = df.loc[mask]
        sliced_r[region] = regional_rf[region].loc[mask]
    return sliced_f, sliced_r, common_start, common_end


def main():
    cfg = load_workflow_config()
    experiment = "e18_oos_monthly_asiapac"

    # ===== ARM A: MONTHLY 12-SLEEVE POOL =====
    print(f"[{experiment}] ARM A: Monthly 12-sleeve pool")
    print(f"  Loading monthly factor data for 4 regions...")
    monthly_factors, monthly_rf = _load_monthly_regions()
    for region, df in monthly_factors.items():
        print(f"  {region:18s} {df.index[0].date()} to {df.index[-1].date()} ({len(df)} months)")

    # Slice to common window with warmup
    monthly_train = int(cfg["pit_protocol"]["e18_monthly_oos"]["train_months"]["value"])
    slice_from_monthly = pd.Timestamp("1990-01-01")
    monthly_factors, monthly_rf, m_start, m_end = _slice_to_common(
        monthly_factors, monthly_rf, slice_from_monthly,
    )
    print(f"  Common monthly window: {m_start.date()} to {m_end.date()}")

    monthly_sim_cfg = SimulationConfig(
        train_months=monthly_train,
        test_months=12,
        step_months=12,
        min_test_obs=6,
    )

    print(f"  Running pooled simulation (SMA{MONTHLY_SMA_WINDOW} on monthly)...")
    monthly_pool = simulate_pooled_regional(
        regional_factors=monthly_factors,
        regional_rf=monthly_rf,
        strategy_factory=lambda: SMATrendFilter(window=MONTHLY_SMA_WINDOW),
        factor_names=FACTOR_NAMES,
        config=monthly_sim_cfg,
        borrow_spread_bps=0.0,
        rebalance_freq="A",
    )

    m_pool_ret = monthly_pool["pool_returns"]
    m_pool_bench = monthly_pool["pool_benchmark"]
    assert not m_pool_ret.isna().any(), "monthly pool has NaN"

    m_pool_m = compute_metrics_monthly(m_pool_ret, "monthly_pool_12sleeve")
    m_bench_m = compute_metrics_monthly(m_pool_bench, "monthly_pool_benchmark")

    print(f"  pool   Sharpe {m_pool_m['sharpe']:+.3f}  CAGR {m_pool_m['cagr']:+.2%}")
    print(f"  bench  Sharpe {m_bench_m['sharpe']:+.3f}  CAGR {m_bench_m['cagr']:+.2%}")

    ci_monthly = bootstrap_excess_sharpe(
        m_pool_ret, m_pool_bench,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=3,
        periods_per_year=12,
    )
    sub_monthly = subperiod_consistency(
        m_pool_ret, m_pool_bench, cfg["subperiods_e18_monthly"],
        periods_per_year=12, min_obs=6,
    )

    print(
        f"  ExSharpe {ci_monthly['excess_sharpe_point']:+.3f} "
        f"[{ci_monthly['excess_sharpe_lower']:+.3f}, {ci_monthly['excess_sharpe_upper']:+.3f}]"
    )

    elev_a_pass, elev_a_reasons = check_elevation(
        excess_sharpe_point=ci_monthly["excess_sharpe_point"],
        ci_lower=ci_monthly["excess_sharpe_lower"],
        subperiod_same_sign=sub_monthly["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_monthly["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # Per-sleeve monthly summaries
    monthly_sleeve_summary = {}
    for label, res in monthly_pool["sleeve_results"].items():
        sm = compute_metrics_monthly(res.overall_returns, label)
        bm = compute_metrics_monthly(res.benchmark_returns, f"{label}_bench")
        monthly_sleeve_summary[label] = {
            "sharpe": sm["sharpe"], "cagr": sm["cagr"],
            "bench_sharpe": bm["sharpe"],
        }

    # ===== ARM B: 15-SLEEVE POOL WITH ASIA-PAC (DAILY) =====
    print(f"\n[{experiment}] ARM B: 15-sleeve pool with Asia-Pac (daily)")
    print(f"  Loading daily factor data for 5 regions...")
    daily_factors, daily_rf = _load_daily_regions_with_asiapac()

    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from_daily = start_date - pd.DateOffset(years=int(train_years))

    daily_factors, daily_rf, d_start, d_end = _slice_to_common(
        daily_factors, daily_rf, slice_from_daily,
    )
    print(f"  Common daily window: {d_start.date()} to {d_end.date()}")
    for region, df in daily_factors.items():
        print(f"  {region:25s} {len(df)} days")

    daily_sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    print(f"  Running 15-sleeve pooled simulation (SMA100, daily)...")
    daily_pool_15 = simulate_pooled_regional(
        regional_factors=daily_factors,
        regional_rf=daily_rf,
        strategy_factory=lambda: SMATrendFilter(window=100),
        factor_names=FACTOR_NAMES,
        config=daily_sim_cfg,
        borrow_spread_bps=0.0,
        rebalance_freq="A",
    )

    d_pool_ret = daily_pool_15["pool_returns"]
    d_pool_bench = daily_pool_15["pool_benchmark"]
    assert not d_pool_ret.isna().any(), "daily 15-sleeve pool has NaN"

    d_pool_m = compute_metrics(d_pool_ret, "daily_pool_15sleeve")
    d_bench_m = compute_metrics(d_pool_bench, "daily_pool_15sleeve_benchmark")

    print(f"  pool15 Sharpe {d_pool_m['sharpe']:+.3f}  CAGR {d_pool_m['cagr']:+.2%}  MaxDD {d_pool_m['max_dd']:+.1%}")
    print(f"  bench  Sharpe {d_bench_m['sharpe']:+.3f}  CAGR {d_bench_m['cagr']:+.2%}  MaxDD {d_bench_m['max_dd']:+.1%}")

    ci_daily_15 = bootstrap_excess_sharpe(
        d_pool_ret, d_pool_bench,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )
    sub_daily_15 = subperiod_consistency(d_pool_ret, d_pool_bench, cfg["subperiods"])

    print(
        f"  ExSharpe {ci_daily_15['excess_sharpe_point']:+.3f} "
        f"[{ci_daily_15['excess_sharpe_lower']:+.3f}, {ci_daily_15['excess_sharpe_upper']:+.3f}]"
    )

    elev_b_pass, elev_b_reasons = check_elevation(
        excess_sharpe_point=ci_daily_15["excess_sharpe_point"],
        ci_lower=ci_daily_15["excess_sharpe_lower"],
        subperiod_same_sign=sub_daily_15["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_daily_15["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # Degradation vs E4
    e4_json_path = WORKFLOW_ROOT / "results" / "e4_pooled_regional.json"
    degradation_pct = None
    e4_sharpe_diff = None
    if e4_json_path.exists():
        with open(e4_json_path) as f:
            e4_saved = json.load(f)
        e4_sharpe_diff = e4_saved["comparisons"]["pool_vs_pool_benchmark"]["excess_sharpe_ci"]["point_estimate"]
        pool15_sharpe_diff = ci_daily_15["point_estimate"]
        if e4_sharpe_diff > 0:
            degradation_pct = 1.0 - pool15_sharpe_diff / e4_sharpe_diff
        print(f"\n  E4 (12-sleeve) Sharpe-diff: {e4_sharpe_diff:+.3f}")
        print(f"  Pool15 Sharpe-diff:         {pool15_sharpe_diff:+.3f}")
        if degradation_pct is not None:
            print(f"  Degradation: {degradation_pct:+.1%} (threshold: <20%)")
            if degradation_pct < 0.20:
                print("  PASS -- pooling absorbs Asia-Pac")
            else:
                print("  FAIL -- Asia-Pac drags pool beyond proportional weight")

    # Per-sleeve daily summaries (highlight Asia-Pac)
    daily_sleeve_summary = {}
    for label, res in daily_pool_15["sleeve_results"].items():
        sm = compute_metrics(res.overall_returns, label)
        bm = compute_metrics(res.benchmark_returns, f"{label}_bench")
        daily_sleeve_summary[label] = {
            "sharpe": sm["sharpe"], "cagr": sm["cagr"], "max_dd": sm["max_dd"],
            "bench_sharpe": bm["sharpe"],
            "is_asia_pac": "asia_pacific" in label,
        }

    out = {
        "experiment": experiment,
        "description": (
            f"True OOS. Arm A: monthly 12-sleeve pool (SMA{MONTHLY_SMA_WINDOW}, "
            f"{monthly_train}mo train). Arm B: daily 15-sleeve pool with Asia-Pac absorption."
        ),
        "arm_a_monthly": {
            "parameters": {
                "regions": list(monthly_factors.keys()),
                "factors": FACTOR_NAMES,
                "sma_window": MONTHLY_SMA_WINDOW,
                "train_months": monthly_train,
                "frequency": "monthly",
                "common_start": str(m_start.date()),
                "common_end": str(m_end.date()),
                "n_sleeves": monthly_pool["n_sleeves"],
            },
            "comparison": {
                "strategy_metrics": m_pool_m,
                "benchmark_metrics": m_bench_m,
                "excess_sharpe_ci": ci_monthly,
                "subperiods": sub_monthly,
            },
            "elevation": {
                "passed": elev_a_pass,
                "reasons": elev_a_reasons,
                "version": 2,
            },
            "sleeve_summary": monthly_sleeve_summary,
        },
        "arm_b_asiapac": {
            "parameters": {
                "regions": list(daily_factors.keys()),
                "factors": FACTOR_NAMES,
                "sma_window": 100,
                "n_sleeves": daily_pool_15["n_sleeves"],
                "common_start": str(d_start.date()),
                "common_end": str(d_end.date()),
            },
            "comparison": {
                "strategy_metrics": d_pool_m,
                "benchmark_metrics": d_bench_m,
                "excess_sharpe_ci": ci_daily_15,
                "subperiods": sub_daily_15,
            },
            "elevation": {
                "passed": elev_b_pass,
                "reasons": elev_b_reasons,
                "version": 2,
            },
            "degradation_diagnostic": {
                "e4_12sleeve_sharpe_diff": e4_sharpe_diff,
                "pool15_sharpe_diff": ci_daily_15["point_estimate"],
                "degradation_pct": degradation_pct,
                "threshold": 0.20,
            },
            "sleeve_summary": daily_sleeve_summary,
        },
        "elevation_version": 2,
        "notes": [
            "Arm A: monthly data is a different Ken French series, not resampled daily",
            "Arm A: SMA5 on monthly ~ SMA100 on daily (pre-committed)",
            "Arm B: Asia-Pac failed individually in Phase 7 (whipsaw, ExS -0.212)",
            "Arm B: tests whether pooling can absorb a known-bad region",
            "Paper portfolio in both arms",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))

    print(f"\nArm A (monthly): {'PASS' if elev_a_pass else 'FAIL'}")
    for r in elev_a_reasons:
        print(f"    {r}")
    print(f"\nArm B (Asia-Pac): {'PASS' if elev_b_pass else 'FAIL'}")
    for r in elev_b_reasons:
        print(f"    {r}")

    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
