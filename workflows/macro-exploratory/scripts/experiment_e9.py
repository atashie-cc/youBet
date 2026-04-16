"""E9 — Daily BSC-style vol-managed factor timing (paper).

NOTE (2026-04-14 Codex R2): this is an *adaptation* of Barroso & Santa-Clara
(2015), not a canonical replication. Canonical BSC is a *monthly*-rebalanced
risk-managed *momentum* strategy that estimates variance from trailing 6-month
daily returns and sizes UMD for the next month. E9 instead runs daily
inverse-vol scaling on CMA/HML/RMW/SMB with BSC-style parameters (target 12%
annualized, 126-day lookback, cap at 2.0x). A true canonical replication
would be a separate experiment (monthly UMD on the longer US history).

Apply constant-vol targeting to each of CMA/HML/RMW/SMB: scale exposure so
realized vol targets 12% annualized, using a 126-day (6-month) trailing vol
estimate and a hard cap at 2.0x leverage.

Mechanism is distinct from E2:
  - E2: binary conditional leverage (1.5x on-state, 0 off-state, SMA gate)
  - E9: continuous inverse-vol scaling, no trend filter, exposure in [0, 2.0]

Comparisons per factor:
  1. vol_target_lev2.0 vs buy-and-hold factor — primary elevation question
  2. vol_target_lev1.0 vs buy-and-hold factor — capped replication (pure
     vol-scaling, no leverage amplification)
  3. vol_target_lev2.0 vs vol_target_lev1.0 — isolates the leverage contribution

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

import numpy as np
import pandas as pd

from youbet.factor.data import load_french_snapshot
from youbet.factor.simulator import (
    SimulationConfig,
    VolTargeting,
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

# BSC-style literature defaults — locked pre-run
TARGET_VOL = 0.12          # 12% annualized (BSC literature default)
LOOKBACK_DAYS = 126        # ~6 months trailing realized vol
MAX_LEVERAGE_FULL = 2.0    # BSC-style practical cap
MAX_LEVERAGE_CAPPED = 1.0  # Mechanism comparison: no leverage amplification


def _exposure_distribution(fold_results) -> dict:
    """Summarize the realized exposure distribution across all fold test periods.

    Per Codex R2-2: the main remaining uncertainty is whether vol-scaling was
    actually modulating exposure finely or mostly pinning it at the cap / 1.0.
    This tells us whether E9 was truly testing vol management or just capped
    leverage.
    """
    exposures = pd.concat([fr.exposure for fr in fold_results])
    exposures = exposures.dropna()
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
        "pct_at_max_cap": float(np.mean(arr >= MAX_LEVERAGE_FULL - 1e-6)),
        "pct_at_1x_or_below": float(np.mean(arr <= 1.0 + 1e-6)),
        "pct_below_1x": float(np.mean(arr < 1.0 - 1e-6)),
        "pct_at_zero": float(np.mean(arr <= 1e-6)),
    }


def main():
    cfg = load_workflow_config()
    experiment = "e9_vol_managed_factor"
    factors_to_test = ["CMA", "HML", "RMW", "SMB"]
    borrow_spread_bps = cfg["costs"]["financing"]["borrow_spread_bps"]

    print(f"[{experiment}] Loading Ken French factors...")
    ff_full = load_french_snapshot(SNAPSHOT_DIR, frequency="daily")
    print(f"  Loaded {len(ff_full)} days, {ff_full.index[0].date()} to {ff_full.index[-1].date()}")

    # Same locked slice as E2/E4: start_date - factor_train_months warmup
    start_date = pd.Timestamp(cfg["backtest"]["start_date"])
    train_years = cfg["backtest"]["factor_train_months"] / 12
    slice_from = start_date - pd.DateOffset(years=int(train_years))
    ff = ff_full.loc[ff_full.index >= slice_from]
    print(
        f"  Sliced to {ff.index[0].date()} to {ff.index[-1].date()} "
        f"({len(ff)} days) for locked start_date {start_date.date()}"
    )

    rf = ff["RF"] if "RF" in ff.columns else pd.Series(0.0, index=ff.index)

    sim_cfg = SimulationConfig(
        train_months=cfg["backtest"]["factor_train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
    )

    comparisons = {}
    per_factor_elevation = {}
    exposure_diagnostics = {}

    for factor in factors_to_test:
        print(f"\n[{experiment}] Factor: {factor}")
        if factor not in ff.columns:
            print(f"  Missing in snapshot, skipping")
            continue

        # Full BSC construction (max_leverage=2.0)
        full_strategy = VolTargeting(
            target_vol=TARGET_VOL,
            lookback_days=LOOKBACK_DAYS,
            max_leverage=MAX_LEVERAGE_FULL,
        )
        full_result = simulate_factor_timing(
            factor_returns=ff[factor],
            rf_returns=rf,
            strategy=full_strategy,
            config=sim_cfg,
            factor_name=factor,
            borrow_spread_bps=borrow_spread_bps,
        )

        # Capped construction (max_leverage=1.0) — pure vol-scaling, no leverage
        capped_strategy = VolTargeting(
            target_vol=TARGET_VOL,
            lookback_days=LOOKBACK_DAYS,
            max_leverage=MAX_LEVERAGE_CAPPED,
        )
        capped_result = simulate_factor_timing(
            factor_returns=ff[factor],
            rf_returns=rf,
            strategy=capped_strategy,
            config=sim_cfg,
            factor_name=factor,
            borrow_spread_bps=0.0,   # no leverage, no borrow
        )

        full_ret = full_result.overall_returns
        capped_ret = capped_result.overall_returns
        bench_ret = full_result.benchmark_returns  # buy-and-hold factor

        full_m = compute_metrics(full_ret, f"{factor}_voltarget_lev{MAX_LEVERAGE_FULL:.1f}")
        capped_m = compute_metrics(capped_ret, f"{factor}_voltarget_lev{MAX_LEVERAGE_CAPPED:.1f}")
        bench_m = compute_metrics(bench_ret, f"{factor}_buyhold")

        print(f"  lev2.0   Sharpe {full_m['sharpe']:+.3f}  CAGR {full_m['cagr']:+.2%}  MaxDD {full_m['max_dd']:+.1%}")
        print(f"  lev1.0   Sharpe {capped_m['sharpe']:+.3f}  CAGR {capped_m['cagr']:+.2%}  MaxDD {capped_m['max_dd']:+.1%}")
        print(f"  buyhold  Sharpe {bench_m['sharpe']:+.3f}  CAGR {bench_m['cagr']:+.2%}  MaxDD {bench_m['max_dd']:+.1%}")

        # Exposure distribution diagnostics (Codex R2-2)
        full_exp = _exposure_distribution(full_result.fold_results)
        capped_exp = _exposure_distribution(capped_result.fold_results)
        exposure_diagnostics[factor] = {
            "full_lev2.0": full_exp,
            "capped_lev1.0": capped_exp,
        }
        print(
            f"  full exposure: median {full_exp['median']:.2f}  "
            f"[{full_exp['p05']:.2f}, {full_exp['p95']:.2f}]  "
            f"@cap {full_exp['pct_at_max_cap']:.0%}  "
            f"@<=1x {full_exp['pct_at_1x_or_below']:.0%}"
        )
        print(
            f"  capped exposure: median {capped_exp['median']:.2f}  "
            f"[{capped_exp['p05']:.2f}, {capped_exp['p95']:.2f}]  "
            f"@1x {capped_exp['pct_at_1x_or_below']:.0%}"
        )

        # Primary CI: full (lev2.0) vs buyhold
        ci_full_vs_bench = bootstrap_excess_sharpe(
            full_ret, bench_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )
        # Replication CI: capped (lev1.0) vs buyhold
        ci_capped_vs_bench = bootstrap_excess_sharpe(
            capped_ret, bench_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )
        # Mechanism isolation CI: full vs capped
        ci_full_vs_capped = bootstrap_excess_sharpe(
            full_ret, capped_ret,
            n_bootstrap=cfg["bootstrap"]["n_replicates"],
            confidence=cfg["bootstrap"]["confidence"],
            block_length=cfg["bootstrap"]["block_length"],
        )

        sub_full = subperiod_consistency(full_ret, bench_ret, cfg["subperiods"])
        sub_capped = subperiod_consistency(capped_ret, bench_ret, cfg["subperiods"])

        elevation_pass, elevation_reasons = check_elevation(
            excess_sharpe_point=ci_full_vs_bench["excess_sharpe_point"],
            ci_lower=ci_full_vs_bench["excess_sharpe_lower"],
            subperiod_same_sign=sub_full["same_sign_positive_excess_sharpe"],
            sharpe_diff_point=ci_full_vs_bench["point_estimate"],
            threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
        )

        comparisons[f"{factor}_full_vs_buyhold"] = {
            "strategy_metrics": full_m,
            "benchmark_metrics": bench_m,
            "excess_sharpe_ci": ci_full_vs_bench,
            "subperiods": sub_full,
        }
        comparisons[f"{factor}_capped_vs_buyhold"] = {
            "strategy_metrics": capped_m,
            "benchmark_metrics": bench_m,
            "excess_sharpe_ci": ci_capped_vs_bench,
            "subperiods": sub_capped,
        }
        comparisons[f"{factor}_full_vs_capped"] = {
            "strategy_metrics": full_m,
            "benchmark_metrics": capped_m,
            "excess_sharpe_ci": ci_full_vs_capped,
        }

        per_factor_elevation[factor] = {
            "passed": elevation_pass,
            "reasons": elevation_reasons,
            "version": 2,
        }

    out = {
        "experiment": experiment,
        "description": (
            f"Daily BSC-style inverse-vol scaling "
            f"(target_vol={TARGET_VOL}, lookback={LOOKBACK_DAYS}d, "
            f"max_leverage={MAX_LEVERAGE_FULL}) on CMA/HML/RMW/SMB. "
            f"Adaptation of Barroso & Santa-Clara (2015) from monthly UMD to "
            f"daily rebalanced factor-style portfolios."
        ),
        "parameters": {
            "factors": factors_to_test,
            "target_vol": TARGET_VOL,
            "lookback_days": LOOKBACK_DAYS,
            "max_leverage_full": MAX_LEVERAGE_FULL,
            "max_leverage_capped": MAX_LEVERAGE_CAPPED,
            "borrow_spread_bps": borrow_spread_bps,
            "train_months": sim_cfg.train_months,
            "test_months": sim_cfg.test_months,
        },
        "comparisons": comparisons,
        "per_factor_elevation": per_factor_elevation,
        "exposure_diagnostics": exposure_diagnostics,
        "elevation_version": 2,
        "locked_benchmark_ref": cfg["benchmarks"]["primary"]["factor_paper"],
        "literature": "Barroso & Santa-Clara (2015) Momentum Has Its Moments",
        "notes": [
            "Paper portfolio (Ken French factors, long-short self-financing)",
            "Continuous inverse-vol scaling with hard cap at max_leverage",
            "Financing on leverage increment (exposure-1) at T-bill + borrow_spread",
            "Literature defaults locked pre-run: target_vol=0.12, lookback=126, max_leverage=2.0",
        ],
    }

    path = save_result(experiment, out)
    print("\n" + format_report(experiment, out))

    # Summary table
    print(
        f"\nSummary (vol_target_lev{MAX_LEVERAGE_FULL:.1f} vs buy-and-hold factor):"
    )
    print(f"{'Factor':<8} {'Full Sharpe':>12} {'Bench Sharpe':>14} {'ExSharpe':>10} {'CI Lower':>10} {'CI Upper':>10}  Elevated?")
    print("-" * 84)
    for factor in factors_to_test:
        key = f"{factor}_full_vs_buyhold"
        if key not in comparisons:
            continue
        cmp = comparisons[key]
        ci = cmp["excess_sharpe_ci"]
        fs = cmp["strategy_metrics"]["sharpe"]
        bs = cmp["benchmark_metrics"]["sharpe"]
        elev = per_factor_elevation[factor]["passed"]
        print(
            f"{factor:<8} {fs:>+12.3f} {bs:>+14.3f} "
            f"{ci['excess_sharpe_point']:>+10.3f} "
            f"{ci['excess_sharpe_lower']:>+10.3f} "
            f"{ci['excess_sharpe_upper']:>+10.3f}  "
            f"{'YES' if elev else 'no'}"
        )
    print(f"\nSaved: {path}")

    return out


if __name__ == "__main__":
    main()
