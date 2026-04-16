"""E4 — Pooled regional 12-sleeve factor SMA timing (paper portfolio).

Equal-weight composite of {US, Dev ex-US, Europe, Japan} x {CMA, HML, RMW}
running unlevered SMA100 timing per sleeve. Annual rebalance to equal weights.

Tests whether international diversification adds value on top of the known
factor-timing Phase 6 result (single-factor US CMA SMA100 ExSharpe +0.687).
Mechanism is distinct from E2: no leverage, pure diversification across
regions and factors.

Benchmarks:
  1. Pool vs pool-benchmark (equal-weight buy-and-hold of the same 12 factors)
  2. Pool vs US-only CMA SMA100 (isolates the international diversification benefit)
  3. Pool vs VTI (sanity check against workflow's locked equity benchmark)

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

from youbet.factor.data import fetch_international_factors, load_french_snapshot
from youbet.factor.simulator import (
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
    simulate_pooled_regional,
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
INTL_REGIONS = ["developed_ex_us", "europe", "japan"]
FACTOR_NAMES = ["CMA", "HML", "RMW"]


def _load_all_regions() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    regional_factors: dict[str, pd.DataFrame] = {}
    regional_rf: dict[str, pd.Series] = {}

    # US — loaded via the main daily snapshot (frequency="daily")
    us_df = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    regional_factors["us"] = us_df
    regional_rf["us"] = us_df["RF"] if "RF" in us_df.columns else pd.Series(0.0, index=us_df.index)

    # International regions — one parquet per region
    for region in INTL_REGIONS:
        df = fetch_international_factors(region, snapshot_dir=SNAPSHOT_DIR)
        regional_factors[region] = df
        regional_rf[region] = df["RF"] if "RF" in df.columns else pd.Series(0.0, index=df.index)

    return regional_factors, regional_rf


def _slice_to_common_window(
    regional_factors: dict[str, pd.DataFrame],
    regional_rf: dict[str, pd.Series],
    slice_from: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], pd.Timestamp, pd.Timestamp]:
    """Slice all regions to [max(slice_from, latest_region_start), min(region_ends)].

    This guarantees every sleeve runs on the same date range, so fold schedules
    and pool indices align exactly — no NaN days in the equal-weight composite.
    """
    starts = [df.index[0] for df in regional_factors.values()]
    ends = [df.index[-1] for df in regional_factors.values()]
    common_start = max(slice_from, max(starts))
    common_end = min(ends)
    if common_start >= common_end:
        raise ValueError(
            f"No common window across regions: "
            f"common_start={common_start}, common_end={common_end}"
        )

    sliced_factors: dict[str, pd.DataFrame] = {}
    sliced_rf: dict[str, pd.Series] = {}
    for region, df in regional_factors.items():
        mask = (df.index >= common_start) & (df.index <= common_end)
        sliced_factors[region] = df.loc[mask]
        sliced_rf[region] = regional_rf[region].loc[mask]

    return sliced_factors, sliced_rf, common_start, common_end


def main():
    cfg = load_workflow_config()
    experiment = "e4_pooled_regional"
    sma_window = 100

    print(f"[{experiment}] Loading all regional factors...")
    regional_factors, regional_rf = _load_all_regions()
    for region, df in regional_factors.items():
        print(f"  {region:18s} {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")

    # Enforce workflow's locked start_date (2003-01-01), with factor_train_months
    # of warmup upstream of it. All regions sliced to the same window so fold
    # schedules and pool indices align.
    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from = start_date - pd.DateOffset(years=int(train_years))

    regional_factors, regional_rf, common_start, common_end = _slice_to_common_window(
        regional_factors, regional_rf, slice_from,
    )
    print(
        f"\n[{experiment}] Common window: {common_start.date()} to {common_end.date()}"
    )
    for region, df in regional_factors.items():
        missing = [f for f in FACTOR_NAMES if f not in df.columns]
        status = "OK" if not missing else f"MISSING {missing}"
        print(f"  {region:18s} {len(df)} days  factors: {status}")

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    print(
        f"\n[{experiment}] Running pooled simulation "
        f"({len(regional_factors)} regions x {len(FACTOR_NAMES)} factors = "
        f"{len(regional_factors) * len(FACTOR_NAMES)} sleeves)..."
    )
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
    sleeve_results = pool_result["sleeve_results"]
    n_sleeves = pool_result["n_sleeves"]

    # Per Codex R1: fail loudly on any NaN rather than silently dropping them.
    # A defensive dropna() here would mask alignment bugs in simulate_pooled_regional
    # (e.g., per-sleeve fold schedules not lining up, RF reindex gaps).
    assert not pool_returns.isna().any(), (
        f"pool_returns has {int(pool_returns.isna().sum())} NaN rows — "
        f"sleeve alignment bug in simulate_pooled_regional"
    )
    assert not pool_benchmark.isna().any(), (
        f"pool_benchmark has {int(pool_benchmark.isna().sum())} NaN rows"
    )

    # US-only CMA SMA100 as a second comparison — isolates the international diversification benefit
    print(f"\n[{experiment}] Running US-only CMA SMA100 reference...")
    us_cma_result = simulate_factor_timing(
        factor_returns=regional_factors["us"]["CMA"],
        rf_returns=regional_rf["us"],
        strategy=SMATrendFilter(window=sma_window),
        config=sim_cfg,
        factor_name="us_cma_sma100_ref",
        borrow_spread_bps=0.0,
    )
    us_cma_returns = us_cma_result.overall_returns

    # Per Codex R2: reindex pool to us_cma_returns for the secondary comparison so
    # the reported metrics and bootstrap CI are computed on the exact same sample.
    # Otherwise pool runs on 12-sleeve union index while us_cma runs on US-only
    # fold schedule, giving a sample mismatch the bootstrap silently intersects.
    common_idx = pool_returns.index.intersection(us_cma_returns.index)
    pool_returns_for_uscma = pool_returns.loc[common_idx]
    us_cma_returns_aligned = us_cma_returns.loc[common_idx]

    pool_m = compute_metrics(pool_returns, f"pool_{n_sleeves}sleeves_sma{sma_window}")
    bench_m = compute_metrics(pool_benchmark, f"pool_benchmark_buyhold")
    pool_m_vs_uscma = compute_metrics(
        pool_returns_for_uscma, f"pool_{n_sleeves}sleeves_sma{sma_window}_on_uscma_idx"
    )
    us_cma_m = compute_metrics(us_cma_returns_aligned, f"us_cma_sma{sma_window}")

    print(f"\n  pool     Sharpe {pool_m['sharpe']:+.3f}  CAGR {pool_m['cagr']:+.2%}  MaxDD {pool_m['max_dd']:+.1%}")
    print(f"  poolbh   Sharpe {bench_m['sharpe']:+.3f}  CAGR {bench_m['cagr']:+.2%}  MaxDD {bench_m['max_dd']:+.1%}")
    print(f"  us_cma   Sharpe {us_cma_m['sharpe']:+.3f}  CAGR {us_cma_m['cagr']:+.2%}  MaxDD {us_cma_m['max_dd']:+.1%}")

    # CI 1: pool vs pool-benchmark — does SMA100 timing beat buy-hold at pooled level
    ci_pool_vs_poolbh = bootstrap_excess_sharpe(
        pool_returns, pool_benchmark,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # CI 2: pool vs US-only CMA SMA — does international diversification add
    # Use aligned series so the CI and the reported metrics describe the same sample.
    ci_pool_vs_uscma = bootstrap_excess_sharpe(
        pool_returns_for_uscma, us_cma_returns_aligned,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    # Sub-period consistency (vs pool-benchmark — the primary gate metric)
    sub_vs_poolbh = subperiod_consistency(pool_returns, pool_benchmark, cfg["subperiods"])
    sub_vs_uscma = subperiod_consistency(
        pool_returns_for_uscma, us_cma_returns_aligned, cfg["subperiods"]
    )

    # Elevation check (vs pool-benchmark is the primary question)
    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci_pool_vs_poolbh["excess_sharpe_point"],
        ci_lower=ci_pool_vs_poolbh["excess_sharpe_lower"],
        subperiod_same_sign=sub_vs_poolbh["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci_pool_vs_poolbh["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # Per-sleeve summaries
    sleeve_summary = {}
    for label, res in sleeve_results.items():
        sleeve_summary[label] = {
            "sharpe": compute_metrics(res.overall_returns, label)["sharpe"],
            "cagr": compute_metrics(res.overall_returns, label)["cagr"],
            "max_dd": compute_metrics(res.overall_returns, label)["max_dd"],
            "bench_sharpe": compute_metrics(res.benchmark_returns, f"{label}_bench")["sharpe"],
        }

    out = {
        "experiment": experiment,
        "description": (
            f"Pooled regional 12-sleeve ({len(regional_factors)}x{len(FACTOR_NAMES)}) "
            f"SMA{sma_window} equal-weight with annual rebalance"
        ),
        "parameters": {
            "regions": list(regional_factors.keys()),
            "factors": FACTOR_NAMES,
            "sma_window": sma_window,
            "rebalance_freq": "A",
            "n_sleeves": n_sleeves,
            "train_months": sim_cfg.train_months,
            "test_months": sim_cfg.test_months,
            "common_start": str(common_start.date()),
            "common_end": str(common_end.date()),
            "locked_start_date": str(start_date.date()),
        },
        "comparisons": {
            "pool_vs_pool_benchmark": {
                "strategy_metrics": pool_m,
                "benchmark_metrics": bench_m,
                "excess_sharpe_ci": ci_pool_vs_poolbh,
                "subperiods": sub_vs_poolbh,
            },
            "pool_vs_us_cma_sma": {
                "strategy_metrics": pool_m_vs_uscma,   # pool reindexed to us_cma dates
                "benchmark_metrics": us_cma_m,
                "excess_sharpe_ci": ci_pool_vs_uscma,
                "subperiods": sub_vs_uscma,
                "note": "pool reindexed to us_cma index for sample-matched comparison (Codex R2 fix)",
            },
        },
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "primary_comparison": "pool_vs_pool_benchmark",
            "version": 2,
        },
        "elevation_version": 2,
        "sleeve_summary": sleeve_summary,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["factor_paper"],
        "notes": [
            "Paper portfolio (Ken French factors, long-short self-financing)",
            "Equal weight across 12 sleeves with annual rebalance (drift within year)",
            "Primary elevation gate is vs pool_benchmark (does pooled timing beat pooled buy-hold)",
            "Secondary question (vs us_cma_sma) isolates the international+factor diversification lift",
            "Unlevered — no financing costs applied",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))

    # Per-sleeve summary table
    print("\nPer-sleeve Sharpes (SMA100 timed vs buy-hold):")
    print(f"{'Sleeve':<35} {'Timed':>10} {'BuyHold':>10} {'Delta':>10}")
    print("-" * 70)
    for label in sorted(sleeve_summary.keys()):
        s = sleeve_summary[label]
        delta = s["sharpe"] - s["bench_sharpe"]
        print(f"{label:<35} {s['sharpe']:>+10.3f} {s['bench_sharpe']:>+10.3f} {delta:>+10.3f}")

    print(f"\nSaved: {path}")
    return out


if __name__ == "__main__":
    main()
