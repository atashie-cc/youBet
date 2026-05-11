"""Phase 5 — Post-final-report addenda.

Four targeted experiments suggested as pre-wrap-up additions:

  #1 Drawdown analysis on Phase 1 sweep.
     Bootstrap CI on MaxDD-difference vs VTI for each weight. Answers: does
     adding ex-US reduce MaxDD enough to be a defensible "ruin-avoidance"
     trade-off, despite Sharpe degradation?

  #2 Mean-shifted placebo on Phase 1 at the 40% VXUS point.
     Shift VXUS daily USD returns so mean matches VTI's. Re-run 60/40 VTI/VXUS.
     If Sharpe-diff stays negative, Phase 1's "ex-US tilt hurts" finding has a
     structural (vol / correlation-timing) component. If Sharpe-diff goes
     to ~zero, the finding was driven by VXUS's lower mean in 2014-2026
     and would reverse under a mean-equalizing regime.

  #3 Correlated-nulls power analysis (Round 2 review M1).
     The Phase -1 power analysis used 18 INDEPENDENT nulls. The realistic
     workflow has clusters of highly-correlated null tests (M1×3 DXY-weight
     variants are ~99% correlated). Holm assumes independence; this script
     re-runs the FWER measurement under clustered null geometry. If FWER
     blows out > 0.10, switch to Romano-Wolf simultaneous CIs.

  #4 Phase 1 vol-min location CI (Round 2 review M4).
     Bootstrap 1000 samples of the daily return panel, recompute the
     vol curve at each weight {0,10,...,60%}, locate argmin, report
     fraction of bootstraps with argmin in the Vanguard 35-55% band.
     Tests whether "vol-min at 60%" is meaningfully different from
     vol-min at 40% or 50%.

Outputs:
  - artifacts/phase5_addenda.json
  - research/phase5_addenda.md
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(WORKFLOW_DIR / "experiments"))

from youbet.etf.backtester import Backtester, BacktestConfig            # noqa: E402
from youbet.etf.benchmark import BuyAndHold                              # noqa: E402
from youbet.etf.stats import holm_bonferroni                             # noqa: E402

from phase0_efficiency import (                                          # noqa: E402
    build_cost_model,
    compute_inferential_stats,
    load_gate,
    load_prices,
    load_tbill_daily,
    load_workflow_config,
)
from phase4_robustness_c1b import (                                      # noqa: E402
    _backtest_static_mix,
    mean_shift_returns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


GATING_COST_BPS = 10.0
PHASE1_START = "2011-01-26"  # VXUS inception
PHASE1_END = "2026-04-30"
PHASE1_WEIGHTS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 1.00]


# ============================================================================
# #1 — Drawdown analysis on Phase 1 sweep
# ============================================================================


def compute_max_drawdown(returns: pd.Series) -> float:
    """MaxDD as a positive fraction (e.g., 0.35 means -35% drawdown)."""
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (running_max - cum) / running_max
    return float(dd.max())


def drawdown_analysis(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
    n_bootstrap: int = 1_000,
) -> dict:
    """For each weight, compute realized MaxDD and bootstrap CI on diff vs VTI."""
    sub = prices[["VTI", "VXUS"]].dropna(how="any")
    sub = sub.loc[(sub.index >= PHASE1_START) & (sub.index <= PHASE1_END)]

    results = []
    bench_dd = None
    bench_returns = None

    for w in PHASE1_WEIGHTS:
        run = _backtest_static_mix(
            weight_ex_us=w,
            ex_us_ticker="VXUS",
            prices_in=sub[["VTI", "VXUS"]] if w > 0 else sub[["VTI"]],
            universe=universe,
            tbill=tbill,
            bt_config=bt_config,
            intl_cost_bps=GATING_COST_BPS,
        )
        strat_rets = run["_strategy_returns"]
        bench_rets = run["_benchmark_returns"]
        if bench_returns is None:
            bench_returns = bench_rets
            bench_dd = compute_max_drawdown(bench_rets)

        strat_dd = compute_max_drawdown(strat_rets)
        dd_diff = strat_dd - bench_dd

        # Bootstrap CI on MaxDD diff: stationary block bootstrap on PAIRED returns
        rng = np.random.default_rng(42 + int(w * 100))
        n = len(strat_rets)
        block_len = 22
        p_jump = 1.0 / block_len
        boot_diffs = np.empty(n_bootstrap)
        strat_arr = strat_rets.values
        bench_arr = bench_rets.values
        for b in range(n_bootstrap):
            indices = np.empty(n, dtype=np.int64)
            indices[0] = rng.integers(0, n)
            for i in range(1, n):
                if rng.random() < p_jump:
                    indices[i] = rng.integers(0, n)
                else:
                    indices[i] = (indices[i - 1] + 1) % n
            boot_s = pd.Series(strat_arr[indices])
            boot_b = pd.Series(bench_arr[indices])
            boot_diffs[b] = compute_max_drawdown(boot_s) - compute_max_drawdown(boot_b)
        ci_lo = float(np.percentile(boot_diffs, 5))
        ci_hi = float(np.percentile(boot_diffs, 95))
        results.append({
            "weight_vxus": w,
            "strat_maxdd": strat_dd,
            "bench_maxdd": bench_dd,
            "dd_diff": dd_diff,
            "dd_diff_ci_lower": ci_lo,
            "dd_diff_ci_upper": ci_hi,
            "ci_excludes_zero_lower": ci_hi < 0,   # strategy strictly LESS drawdown
        })
        logger.info(
            "  w=%2d%%: strat_DD=%.3f bench_DD=%.3f diff=%.3f CI=[%.3f, %.3f] %s",
            int(w * 100), strat_dd, bench_dd, dd_diff, ci_lo, ci_hi,
            "BETTER" if ci_hi < 0 else "INCONCLUSIVE" if ci_lo < 0 < ci_hi else "WORSE"
        )
    return {"bench_maxdd": bench_dd, "by_weight": results}


# ============================================================================
# #2 — Mean-shifted placebo on Phase 1 at 40% VXUS
# ============================================================================


def phase1_mean_shifted_placebo(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
) -> dict:
    """Shift VXUS daily USD returns to VTI's mean; re-run 40% VXUS strategy."""
    sub = prices[["VTI", "VXUS"]].dropna(how="any")
    sub = sub.loc[(sub.index >= PHASE1_START) & (sub.index <= PHASE1_END)]

    rets = sub.pct_change().dropna(how="any")
    delta_per_day = float(rets["VTI"].mean() - rets["VXUS"].mean())

    # Reconstruct VXUS prices from shifted returns
    base_price = sub["VXUS"].iloc[0]
    rets_shifted = rets.copy()
    rets_shifted["VXUS"] = rets["VXUS"] + delta_per_day

    vxus_recon = sub["VXUS"].copy()
    vxus_recon.iloc[1:] = base_price * (1 + rets_shifted["VXUS"]).cumprod().values
    prices_placebo = sub.copy()
    prices_placebo["VXUS"] = vxus_recon

    # Run 60/40 VTI/VXUS on placebo
    run = _backtest_static_mix(
        weight_ex_us=0.40,
        ex_us_ticker="VXUS",
        prices_in=prices_placebo,
        universe=universe,
        tbill=tbill,
        bt_config=bt_config,
        intl_cost_bps=GATING_COST_BPS,
    )
    inf = compute_inferential_stats(run, n_bootstrap=10_000)

    return {
        "subject": "Phase 1 at 40% VXUS — mean-shifted placebo",
        "delta_applied_per_day": delta_per_day,
        "delta_applied_annualized": delta_per_day * 252,
        "raw_vxus_mean_daily": float(rets["VXUS"].mean()),
        "raw_vti_mean_daily": float(rets["VTI"].mean()),
        **{k: v for k, v in run.items() if not k.startswith("_")},
        **inf,
    }


# ============================================================================
# #3 — Correlated-nulls power analysis
# ============================================================================


def correlated_nulls_power(
    n_simulations: int = 100,
    n_days: int = 6224,
    block_length: int = 22,
    n_bootstrap: int = 600,
) -> dict:
    """Re-run power analysis under clustered-null geometry.

    Cluster structure: 18 nulls organized as
      - 3 nulls perfectly correlated (M1-style: same DXY signal at 3 weights)
      - 3 nulls perfectly correlated (M4-style)
      - 2 nulls highly correlated (M3 + composite stand-in)
      - 10 nulls independent

    True alternative effect = 0 (everyone is null) — we measure FWER.
    """
    rng = np.random.default_rng(42)
    benchmark_mu = 0.0004
    benchmark_sigma = 0.01
    correlation = 0.70
    ann = np.sqrt(252)

    cluster_structure = [3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 14 unique base draws → 18 nulls
    assert sum(cluster_structure) == 18

    fwer_passes = 0
    sim_pass_counts = []

    for sim in range(n_simulations):
        rng_sim = np.random.default_rng(42 + sim * 1000)
        benchmark = rng_sim.normal(benchmark_mu, benchmark_sigma, size=n_days)
        bench_z = (benchmark - benchmark_mu) / benchmark_sigma

        candidates = []
        for cluster_size in cluster_structure:
            # Generate one underlying draw, replicate for each member of the cluster
            z_unit = rng_sim.standard_normal(n_days)
            eps_unit = bench_z * correlation + z_unit * np.sqrt(1.0 - correlation**2)
            base_strat = benchmark_mu + eps_unit * benchmark_sigma
            for _ in range(cluster_size):
                candidates.append(base_strat)
        assert len(candidates) == 18

        # Shared block-bootstrap indices for paired CI
        p_jump = 1.0 / block_length
        jump_draws = rng_sim.random((n_bootstrap, n_days), dtype=np.float32)
        jump_targets = rng_sim.integers(0, n_days, size=(n_bootstrap, n_days), dtype=np.int32)
        start_indices = rng_sim.integers(0, n_days, size=n_bootstrap, dtype=np.int32)
        indices = np.empty((n_bootstrap, n_days), dtype=np.int32)
        indices[:, 0] = start_indices
        for i in range(1, n_days):
            do_jump = jump_draws[:, i] < p_jump
            indices[:, i] = np.where(do_jump, jump_targets[:, i], (indices[:, i - 1] + 1) % n_days)

        boot_bench = benchmark[indices]
        sb = boot_bench.mean(axis=1) / np.maximum(boot_bench.std(axis=1), 1e-10) * ann

        p_values = {}
        ci_lowers = {}
        points = {}
        for idx, strat in enumerate(candidates):
            boot_strat = strat[indices]
            ss = boot_strat.mean(axis=1) / np.maximum(boot_strat.std(axis=1), 1e-10) * ann
            boot_diff = ss - sb
            ci_lo = float(np.percentile(boot_diff, 5))
            obs_strat = float(strat.mean() / max(strat.std(), 1e-10) * ann)
            obs_bench = float(benchmark.mean() / max(benchmark.std(), 1e-10) * ann)
            obs_diff = obs_strat - obs_bench
            excess = strat - benchmark
            excess_centered = excess - excess.mean()
            boot_excess = excess_centered[indices]
            null_es = boot_excess.mean(axis=1) / np.maximum(boot_excess.std(axis=1), 1e-10) * ann
            obs_es = float(excess.mean() / max(excess.std(), 1e-10) * ann)
            count_ge = int(np.sum(null_es >= obs_es))
            raw_p = (1 + count_ge) / (n_bootstrap + 1)
            p_values[f"c{idx:02d}"] = raw_p
            ci_lowers[f"c{idx:02d}"] = ci_lo
            points[f"c{idx:02d}"] = obs_diff

        holm = holm_bonferroni(p_values)
        # Gate (descriptive workflow uses 0.40; for FWER measurement under null
        # we use the OG plan's 0.20 to be conservative — at 0 effect, this
        # should not pass any null)
        passes = 0
        for name in p_values:
            if (points[name] > 0.20
                and holm[name]["adjusted_p"] < 0.05
                and ci_lowers[name] > 0):
                passes += 1
        sim_pass_counts.append(passes)
        if passes > 0:
            fwer_passes += 1

    return {
        "n_simulations": n_simulations,
        "n_days": n_days,
        "n_bootstrap_per_sim": n_bootstrap,
        "cluster_structure": cluster_structure,
        "n_candidates": 18,
        "true_effect_target": 0.0,
        "fwer_any_pass_rate": fwer_passes / n_simulations,
        "mean_pass_count_per_sim": float(np.mean(sim_pass_counts)),
        "max_pass_count_per_sim": int(np.max(sim_pass_counts)),
    }


# ============================================================================
# #4 — Phase 1 vol-min location CI
# ============================================================================


def vol_min_location_ci(
    prices: pd.DataFrame,
    n_bootstrap: int = 1_000,
) -> dict:
    """Bootstrap the daily return panel, locate vol-min argmin per replicate."""
    sub = prices[["VTI", "VXUS"]].dropna(how="any")
    sub = sub.loc[(sub.index >= PHASE1_START) & (sub.index <= PHASE1_END)]
    rets = sub.pct_change().dropna(how="any")

    grid_weights = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 1.00]
    rng = np.random.default_rng(42)
    n = len(rets)
    block_len = 22
    p_jump = 1.0 / block_len

    argmin_weights = []
    for b in range(n_bootstrap):
        indices = np.empty(n, dtype=np.int64)
        indices[0] = rng.integers(0, n)
        for i in range(1, n):
            if rng.random() < p_jump:
                indices[i] = rng.integers(0, n)
            else:
                indices[i] = (indices[i - 1] + 1) % n
        boot_rets = rets.iloc[indices].reset_index(drop=True)
        vti = boot_rets["VTI"].values
        vxus = boot_rets["VXUS"].values

        vols = []
        for w in grid_weights:
            port = (1 - w) * vti + w * vxus
            vols.append(float(port.std() * np.sqrt(252)))
        argmin_idx = int(np.argmin(vols))
        argmin_weights.append(grid_weights[argmin_idx])

    arr = np.array(argmin_weights)
    fraction_in_35_55 = float(np.mean((arr >= 0.35) & (arr <= 0.55)))
    counts = pd.Series(arr).value_counts().sort_index().to_dict()
    return {
        "n_bootstrap": n_bootstrap,
        "grid_weights": grid_weights,
        "argmin_distribution": {str(k): int(v) for k, v in counts.items()},
        "fraction_in_vanguard_band_35_55": fraction_in_35_55,
        "argmin_mean": float(arr.mean()),
        "argmin_median": float(np.median(arr)),
    }


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    print("=" * 70)
    print("PHASE 5 — POST-FINAL-REPORT ADDENDA")
    print("=" * 70)

    cfg = load_workflow_config()
    universe = pd.read_csv(WORKFLOW_DIR / "data" / "reference" / "international_universe.csv")
    universe["inception_date"] = pd.to_datetime(universe["inception_date"])
    prices = load_prices()
    tbill = load_tbill_daily(prices.index)

    bt_config = BacktestConfig(
        train_months=cfg["walk_forward"]["train_months"],
        test_months=cfg["walk_forward"]["test_months"],
        step_months=cfg["walk_forward"]["step_months"],
        rebalance_frequency="annual",
        initial_capital=cfg["walk_forward"]["initial_capital"],
    )

    print("\n--- #1: Drawdown analysis on Phase 1 sweep ---\n")
    drawdown = drawdown_analysis(prices, universe, tbill, bt_config, n_bootstrap=1_000)

    print("\n--- #2: Mean-shifted placebo on Phase 1 (40% VXUS) ---\n")
    placebo = phase1_mean_shifted_placebo(prices, universe, tbill, bt_config)
    print(f"  Delta applied: {placebo['delta_applied_per_day']:+.6f}/day "
          f"({placebo['delta_applied_annualized']:+.4f}/yr)")
    print(f"  Raw VXUS mean: {placebo['raw_vxus_mean_daily']:+.6f} | "
          f"raw VTI mean: {placebo['raw_vti_mean_daily']:+.6f}")
    print(f"  Placebo Sharpe-diff: {placebo['sharpe_diff_point']:+.3f} "
          f"(CI [{placebo['sharpe_diff_ci_lower']:+.3f}, {placebo['sharpe_diff_ci_upper']:+.3f}])")
    print(f"  Placebo log-excess daily: {placebo['log_excess_mean_daily']:+.6f} "
          f"(CI [{placebo['log_excess_ci_lower']:+.6f}, {placebo['log_excess_ci_upper']:+.6f}])")

    print("\n--- #3: Correlated-nulls power analysis ---\n")
    corr_power = correlated_nulls_power(n_simulations=80, n_days=6224, n_bootstrap=600)
    print(f"  Cluster structure: {corr_power['cluster_structure']}")
    print(f"  FWER (any null pass at gate 0.20): {corr_power['fwer_any_pass_rate']:.3f}")
    print(f"  Mean pass count per sim: {corr_power['mean_pass_count_per_sim']:.2f}")

    print("\n--- #4: Phase 1 vol-min location CI ---\n")
    volmin = vol_min_location_ci(prices, n_bootstrap=1_000)
    print(f"  Argmin distribution: {volmin['argmin_distribution']}")
    print(f"  Fraction in Vanguard 35-55% band: {volmin['fraction_in_vanguard_band_35_55']:.3f}")
    print(f"  Mean argmin: {volmin['argmin_mean']:.3f}")

    # Persist
    artifacts_dir = WORKFLOW_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "phase5_addenda.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "run_date": date.today().isoformat(),
                "drawdown_analysis": drawdown,
                "phase1_mean_shifted_placebo": placebo,
                "correlated_nulls_power": corr_power,
                "vol_min_location_ci": volmin,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", out_path)

    # Markdown
    md = ["# Phase 5 — Post-final-report Addenda", ""]
    md.append(f"- Run date: {date.today()}")
    md.append("- Four targeted validation experiments suggested as pre-wrap-up additions.")
    md.append("")

    md.append("## #1 Drawdown analysis on Phase 1 sweep")
    md.append("")
    md.append(f"- Benchmark (100% VTI) MaxDD: **{drawdown['bench_maxdd']:.3f}**")
    md.append("")
    md.append("| Weight VXUS | Strat MaxDD | Diff vs VTI | 90% CI | Verdict |")
    md.append("|---|---|---|---|---|")
    for r in drawdown["by_weight"]:
        if r["weight_vxus"] == 0:
            continue
        if r["dd_diff_ci_upper"] < 0:
            verdict = "STRICTLY LESS DD (significant)"
        elif r["dd_diff_ci_lower"] > 0:
            verdict = "STRICTLY MORE DD (significant)"
        else:
            verdict = "Inconclusive"
        md.append(
            f"| {int(r['weight_vxus'] * 100)}% | {r['strat_maxdd']:.3f} | "
            f"{r['dd_diff']:+.3f} | [{r['dd_diff_ci_lower']:+.3f}, {r['dd_diff_ci_upper']:+.3f}] | "
            f"{verdict} |"
        )
    md.append("")

    md.append("## #2 Mean-shifted placebo on Phase 1 (40% VXUS)")
    md.append("")
    md.append(f"- Delta applied: {placebo['delta_applied_per_day']:+.6f}/day "
              f"({placebo['delta_applied_annualized']:+.4f}/yr)")
    md.append(f"- Raw VXUS mean: {placebo['raw_vxus_mean_daily']:+.6f}/day; "
              f"raw VTI mean: {placebo['raw_vti_mean_daily']:+.6f}/day")
    md.append(f"- Original Phase 1 40% VXUS Sharpe-diff: -0.094")
    md.append(f"- **Placebo Sharpe-diff (VXUS mean-shifted to VTI's): "
              f"{placebo['sharpe_diff_point']:+.3f}** "
              f"(CI [{placebo['sharpe_diff_ci_lower']:+.3f}, "
              f"{placebo['sharpe_diff_ci_upper']:+.3f}])")
    md.append(f"- Placebo log-excess daily mean: {placebo['log_excess_mean_daily']:+.6f} "
              f"(CI [{placebo['log_excess_ci_lower']:+.6f}, "
              f"{placebo['log_excess_ci_upper']:+.6f}])")
    md.append("")

    md.append("## #3 Correlated-nulls power analysis (Round 2 review M1)")
    md.append("")
    md.append(f"- Cluster structure: {corr_power['cluster_structure']}")
    md.append(f"- FWER (any null pass at the 0.20-gate baseline): "
              f"**{corr_power['fwer_any_pass_rate']:.3f}** "
              f"(nominal Holm target: 0.05)")
    md.append(f"- Mean pass count per sim under H_0: {corr_power['mean_pass_count_per_sim']:.2f}")
    md.append("")

    md.append("## #4 Phase 1 vol-min location CI (Round 2 review M4)")
    md.append("")
    md.append(f"- Bootstrap N: {volmin['n_bootstrap']}")
    md.append(f"- Argmin distribution (weight VXUS):")
    for w, c in volmin["argmin_distribution"].items():
        md.append(f"  - {float(w)*100:.0f}%: {c} ({c/volmin['n_bootstrap']*100:.1f}%)")
    md.append(f"- **Fraction in Vanguard 35-55% band: "
              f"{volmin['fraction_in_vanguard_band_35_55']:.3f}**")
    md.append(f"- Mean argmin weight: {volmin['argmin_mean']:.3f}")
    md.append("")

    md_path = WORKFLOW_DIR / "research" / "phase5_addenda.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    logger.info("Wrote %s", md_path)
    print(f"\nResults: {out_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
