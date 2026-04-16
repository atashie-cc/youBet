"""E12 — Canonical monthly Barroso & Santa-Clara (2015) UMD replication.

Tests vol-managed momentum on the canonical BSC construction: monthly UMD from
Ken French Library, trailing 6-month daily realized variance, monthly rebalance,
target-vol 12%, max leverage 2.0. All parameters literature-anchored (BSC 2015
Table I/III) and locked in config.yaml.pit_protocol.e12_bsc_canonical_umd.

E9 (daily BSC-style adaptation on CMA/HML/RMW/SMB) failed — but changed three
things at once: factor, cadence, and sample. E12 isolates whether the failure
was the adaptation or the mechanism itself.

Primary comparison on BSC paper window 1927-07 to 2011-12.
Secondary (diagnostic only) on extended window 1927-07 to 2026-02.

Paper portfolio (Ken French long-short construct). Not directly investable.
"""

from __future__ import annotations

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
    _download_french_dataset,
    fetch_french_factors,
    load_french_snapshot,
)
from youbet.factor.simulator import (
    SimulationConfig,
    VolTargetingMonthly,
    simulate_factor_timing,
)

from _common import (
    bootstrap_excess_sharpe,
    check_elevation,
    format_report,
    load_workflow_config,
    save_result,
    subperiod_consistency,
)


PERIODS_PER_YEAR = 12  # monthly data


def compute_metrics_monthly(returns: pd.Series, name: str = "") -> dict:
    """Metrics for MONTHLY return series (annualize with √12 and 12 obs/yr)."""
    if len(returns) == 0:
        return {"name": name, "error": "empty return series"}
    ann_vol = float(returns.std() * np.sqrt(PERIODS_PER_YEAR))
    mean_ret = float(returns.mean())
    sharpe = mean_ret / max(returns.std(), 1e-10) * np.sqrt(PERIODS_PER_YEAR)

    cum = (1 + returns).cumprod()
    final_wealth = float(cum.iloc[-1]) if len(cum) else 1.0
    n_years = len(returns) / PERIODS_PER_YEAR
    cagr = final_wealth ** (1.0 / max(n_years, 0.01)) - 1.0

    # Max drawdown
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    max_dd_dur = 0
    is_dd = dd < -0.001
    if is_dd.any():
        groups = (~is_dd).cumsum()
        dd_groups = groups[is_dd]
        for _, group in dd_groups.groupby(dd_groups):
            dur_months = len(group)
            max_dd_dur = max(max_dd_dur, dur_months)

    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    return {
        "name": name,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_dd_duration_months": max_dd_dur,
        "calmar": calmar,
        "final_wealth": final_wealth,
        "n_months": int(len(returns)),
        "frequency": "monthly",
    }

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

SNAPSHOT_DIR = REPO_ROOT / "workflows" / "factor-timing" / "data" / "snapshots"

TARGET_VOL = 0.12
LOOKBACK_DAYS = 126
MAX_LEVERAGE = 2.0

BSC_WINDOW_START = pd.Timestamp("1927-07-01")
BSC_WINDOW_END = pd.Timestamp("2011-12-31")
EXTENDED_END = pd.Timestamp("2026-02-28")


def _exposure_distribution(fold_results, max_lev: float) -> dict:
    """Summarize realized exposure distribution (per E9's Codex R2 diagnostic)."""
    exposures = pd.concat([fr.exposure for fr in fold_results]).dropna()
    if len(exposures) == 0:
        return {"n_obs": 0}
    arr = exposures.values
    return {
        "n_obs": int(len(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "pct_at_max_cap": float(np.mean(arr >= max_lev - 1e-6)),
        "pct_at_1x_or_below": float(np.mean(arr <= 1.0 + 1e-6)),
        "pct_below_1x": float(np.mean(arr < 1.0 - 1e-6)),
    }


def _run_comparison(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    strat_name: str,
    bench_name: str,
    cfg: dict,
    subperiods_key: str = "subperiods_e12",
) -> dict:
    """Compute metrics, bootstrap CI, and sub-period consistency for one comparison."""
    strat_m = compute_metrics_monthly(strat_returns, strat_name)
    bench_m = compute_metrics_monthly(bench_returns, bench_name)

    ci = bootstrap_excess_sharpe(
        strat_returns, bench_returns,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=3,  # ~quarterly blocks for monthly data (vs 22d for daily)
        periods_per_year=12,
    )

    windows = cfg.get(subperiods_key, cfg["subperiods_e12"])
    sub = subperiod_consistency(
        strat_returns, bench_returns, windows,
        periods_per_year=12, min_obs=6,
    )

    return {
        "strategy_metrics": strat_m,
        "benchmark_metrics": bench_m,
        "excess_sharpe_ci": ci,
        "subperiods": sub,
    }


def main():
    cfg = load_workflow_config()
    experiment = "e12_bsc_canonical_umd"
    borrow_spread_bps = cfg["costs"]["financing"]["borrow_spread_bps"]

    # --- Data ---
    # UMD monthly goes back to 1927 in the Ken French Library, but
    # fetch_french_factors inner-joins with FF5 (which starts 1963).
    # For E12 we only need UMD + RF, so fetch them separately to get
    # the full 1927+ history.
    print(f"[{experiment}] Fetching monthly UMD (standalone, for 1927+ history)...")
    mom_monthly = _download_french_dataset("momentum_monthly")
    mom_col = [c for c in mom_monthly.columns if "mom" in c.lower() or "umd" in c.lower()]
    if mom_col:
        mom_monthly = mom_monthly.rename(columns={mom_col[0]: "UMD"})
    elif len(mom_monthly.columns) == 1:
        mom_monthly.columns = ["UMD"]
    print(f"  UMD monthly: {mom_monthly.index[0].date()} to {mom_monthly.index[-1].date()} ({len(mom_monthly)} months)")

    # RF from the 3-factor monthly file (goes back to 1926-07, same as UMD)
    # The 5-factor file only starts 1963, which would leave RF=0 pre-1963
    # and bias leveraged months (Codex R1).
    print(f"[{experiment}] Fetching monthly 3-factor data (for RF series back to 1926)...")
    import requests, zipfile, io
    from youbet.factor.data import _parse_french_csv
    _FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    resp = requests.get(_FF3_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        content = zf.read(csv_names[0]).decode("utf-8", errors="replace")
    ff3_monthly = _parse_french_csv(content, ["Mkt-RF", "SMB", "HML", "RF"])
    rf_series = ff3_monthly["RF"] if "RF" in ff3_monthly.columns else pd.Series(0.0, index=ff3_monthly.index)
    print(f"  RF from 3-factor: {rf_series.index[0].date()} to {rf_series.index[-1].date()} ({len(rf_series)} months)")

    # Combine: UMD from momentum, RF from 3-factor — both go back to 1926/1927
    ff_monthly = mom_monthly[["UMD"]].join(rf_series, how="left")
    ff_monthly["RF"] = ff_monthly["RF"].fillna(0.0)
    print(
        f"  Combined monthly: {ff_monthly.index[0].date()} to {ff_monthly.index[-1].date()} "
        f"({len(ff_monthly)} months, RF available from {rf_series.index[0].date()})"
    )

    # Also save as a snapshot for future use
    snap_path = SNAPSHOT_DIR / f"french_umd_monthly_{pd.Timestamp.now().strftime('%Y-%m-%d')}.parquet"
    ff_monthly.to_parquet(snap_path)
    print(f"  Saved: {snap_path}")

    # Daily factors for variance estimation
    print(f"[{experiment}] Loading daily Ken French factors (for variance estimation)...")
    ff_daily = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    print(f"  Loaded daily: {len(ff_daily)} days, {ff_daily.index[0].date()} to {ff_daily.index[-1].date()}")

    assert "UMD" in ff_monthly.columns, f"UMD not in monthly columns: {ff_monthly.columns.tolist()}"
    assert "UMD" in ff_daily.columns, f"UMD not in daily columns: {ff_daily.columns.tolist()}"

    rf_monthly = ff_monthly["RF"]

    # --- Strategy ---
    strategy = VolTargetingMonthly(
        target_vol=TARGET_VOL,
        lookback_days=LOOKBACK_DAYS,
        max_leverage=MAX_LEVERAGE,
    )
    strategy.set_daily_returns(ff_daily["UMD"])

    # --- Primary: BSC canonical window (1927-2011) ---
    print(f"\n[{experiment}] Running BSC canonical window {BSC_WINDOW_START.date()} to {BSC_WINDOW_END.date()}...")

    monthly_canonical = ff_monthly.loc[
        (ff_monthly.index >= BSC_WINDOW_START) & (ff_monthly.index <= BSC_WINDOW_END)
    ]
    rf_canonical = rf_monthly.reindex(monthly_canonical.index).fillna(0.0)

    # Walk-forward config: 120-month train (10yr), 12-month test, 12-month step.
    # min_test_obs=6 because a 12-month test window has only ~12 monthly obs
    # (daily default of 21 would filter out all folds).
    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
        min_test_obs=6,
    )

    canonical_result = simulate_factor_timing(
        factor_returns=monthly_canonical["UMD"],
        rf_returns=rf_canonical,
        strategy=strategy,
        config=sim_cfg,
        factor_name="UMD",
        borrow_spread_bps=borrow_spread_bps,
    )

    canonical_strat = canonical_result.overall_returns
    canonical_bench = canonical_result.benchmark_returns

    print(
        f"  Canonical: {len(canonical_strat)} test months, "
        f"{canonical_result.n_folds} folds"
    )

    canonical_cmp = _run_comparison(
        canonical_strat, canonical_bench,
        f"UMD_bsc_monthly_tv{int(TARGET_VOL*100)}_lev{MAX_LEVERAGE:.1f}",
        "UMD_buyhold",
        cfg,
        subperiods_key="subperiods_e12",
    )

    strat_m = canonical_cmp["strategy_metrics"]
    bench_m = canonical_cmp["benchmark_metrics"]
    ci = canonical_cmp["excess_sharpe_ci"]
    print(
        f"  BSC:     Sharpe {strat_m['sharpe']:+.3f}  CAGR {strat_m['cagr']:+.2%}  MaxDD {strat_m['max_dd']:+.1%}"
    )
    print(
        f"  UMD B&H: Sharpe {bench_m['sharpe']:+.3f}  CAGR {bench_m['cagr']:+.2%}  MaxDD {bench_m['max_dd']:+.1%}"
    )
    print(
        f"  ExSharpe {ci['excess_sharpe_point']:+.3f} "
        f"[{ci['excess_sharpe_lower']:+.3f}, {ci['excess_sharpe_upper']:+.3f}]  "
        f"Sharpe-diff {ci['point_estimate']:+.3f}"
    )

    # Exposure diagnostic — split by variance regime (Codex R1: mixed estimator)
    exp_dist = _exposure_distribution(canonical_result.fold_results, MAX_LEVERAGE)

    # Regime-split: pre-1963 (monthly variance fallback) vs post-1963 (daily variance)
    DAILY_DATA_START = pd.Timestamp("1963-07-01")
    pre_1963_folds = [fr for fr in canonical_result.fold_results
                      if fr.strategy_returns.index[-1] < DAILY_DATA_START]
    post_1963_folds = [fr for fr in canonical_result.fold_results
                       if fr.strategy_returns.index[0] >= DAILY_DATA_START]
    exp_pre = _exposure_distribution(pre_1963_folds, MAX_LEVERAGE) if pre_1963_folds else {}
    exp_post = _exposure_distribution(post_1963_folds, MAX_LEVERAGE) if post_1963_folds else {}

    print(
        f"  Exposure (all): median {exp_dist['median']:.2f}  "
        f"[{exp_dist['p05']:.2f}, {exp_dist['p95']:.2f}]  "
        f"@cap {exp_dist['pct_at_max_cap']:.0%}  "
        f"@<=1x {exp_dist['pct_at_1x_or_below']:.0%}"
    )
    if exp_pre:
        print(
            f"  Exposure (pre-1963, monthly vol): median {exp_pre['median']:.2f}  "
            f"[{exp_pre['p05']:.2f}, {exp_pre['p95']:.2f}]  "
            f"@cap {exp_pre['pct_at_max_cap']:.0%}  n={exp_pre['n_obs']}"
        )
    if exp_post:
        print(
            f"  Exposure (post-1963, daily vol):  median {exp_post['median']:.2f}  "
            f"[{exp_post['p05']:.2f}, {exp_post['p95']:.2f}]  "
            f"@cap {exp_post['pct_at_max_cap']:.0%}  n={exp_post['n_obs']}"
        )

    # Elevation check on canonical window
    sub = canonical_cmp["subperiods"]
    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci["excess_sharpe_point"],
        ci_lower=ci["excess_sharpe_lower"],
        subperiod_same_sign=sub["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    # --- Secondary: extended window (1927-2026) ---
    print(f"\n[{experiment}] Running extended window to {EXTENDED_END.date()} (diagnostic only)...")

    monthly_extended = ff_monthly.loc[ff_monthly.index >= BSC_WINDOW_START]
    rf_extended = rf_monthly.reindex(monthly_extended.index).fillna(0.0)

    strategy_ext = VolTargetingMonthly(
        target_vol=TARGET_VOL,
        lookback_days=LOOKBACK_DAYS,
        max_leverage=MAX_LEVERAGE,
    )
    strategy_ext.set_daily_returns(ff_daily["UMD"])

    extended_result = simulate_factor_timing(
        factor_returns=monthly_extended["UMD"],
        rf_returns=rf_extended,
        strategy=strategy_ext,
        config=sim_cfg,
        factor_name="UMD",
        borrow_spread_bps=borrow_spread_bps,
    )

    extended_strat = extended_result.overall_returns
    extended_bench = extended_result.benchmark_returns

    extended_cmp = _run_comparison(
        extended_strat, extended_bench,
        f"UMD_bsc_monthly_extended",
        "UMD_buyhold_extended",
        cfg,
        subperiods_key="subperiods_e12",
    )

    ext_m = extended_cmp["strategy_metrics"]
    ext_bench_m = extended_cmp["benchmark_metrics"]
    ext_ci = extended_cmp["excess_sharpe_ci"]
    print(
        f"  BSC ext: Sharpe {ext_m['sharpe']:+.3f}  CAGR {ext_m['cagr']:+.2%}  MaxDD {ext_m['max_dd']:+.1%}"
    )
    print(
        f"  ExSharpe {ext_ci['excess_sharpe_point']:+.3f} "
        f"[{ext_ci['excess_sharpe_lower']:+.3f}, {ext_ci['excess_sharpe_upper']:+.3f}]"
    )

    exp_dist_ext = _exposure_distribution(extended_result.fold_results, MAX_LEVERAGE)

    # --- Assemble result ---
    comparisons = {
        "bsc_vs_umd_buyhold": canonical_cmp,
        "bsc_vs_umd_buyhold_extended": extended_cmp,
    }

    out = {
        "experiment": experiment,
        "description": (
            f"Canonical monthly Barroso & Santa-Clara (2015) vol-managed UMD. "
            f"Parameters: target_vol={TARGET_VOL}, lookback={LOOKBACK_DAYS}d daily "
            f"variance, monthly rebalance, max_leverage={MAX_LEVERAGE}. "
            f"Primary window: 1927-07 to 2011-12 (BSC paper). "
            f"Secondary: 1927-07 to 2026-02 (diagnostic only, NOT true OOS)."
        ),
        "parameters": {
            "factor": "UMD",
            "target_vol": TARGET_VOL,
            "lookback_days": LOOKBACK_DAYS,
            "max_leverage": MAX_LEVERAGE,
            "borrow_spread_bps": borrow_spread_bps,
            "train_months": sim_cfg.train_months,
            "test_months": sim_cfg.test_months,
            "step_months": sim_cfg.step_months,
            "canonical_start": str(BSC_WINDOW_START.date()),
            "canonical_end": str(BSC_WINDOW_END.date()),
            "extended_end": str(EXTENDED_END.date()),
        },
        "comparisons": comparisons,
        "elevation": {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "primary_comparison": "bsc_vs_umd_buyhold",
            "version": 2,
        },
        "exposure_diagnostics": {
            "canonical": exp_dist,
            "canonical_pre_1963_monthly_vol": exp_pre,
            "canonical_post_1963_daily_vol": exp_post,
            "extended": exp_dist_ext,
            "variance_regime_note": (
                "Pre-1963: variance from trailing 6-month monthly returns (x sqrt(12)). "
                "Post-1963: variance from trailing 126-day daily returns (x sqrt(252)). "
                "Regime break at 1963-07 may shift exposure distribution (Codex R1)."
            ),
        },
        "elevation_version": 2,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["umd_bsc_canonical"],
        "literature": "Barroso & Santa-Clara (2015) Momentum Has Its Moments, JFE",
        "notes": [
            "Paper portfolio (Ken French UMD, hypothetical long-short momentum)",
            "Monthly rebalance, daily-variance input — canonical BSC construction",
            "All parameters literature-anchored, locked in config.yaml.pit_protocol.e12_bsc_canonical_umd",
            "Extended window 1927-2026 is NOT true OOS (momentum extensively reanalyzed post-2011)",
            "Financing charged on leverage increment at T-bill + borrow_spread",
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
