"""CI Calibration Simulation Study (Codex review Rec 1).

Monte Carlo validation of youbet.etf.stats.excess_sharpe_ci under
realistic data-generating processes. Answers: are our 90% CIs actually
90% coverage? What's the false positive rate? How sensitive to block
length and sample size?

Decision thresholds (committed BEFORE running):
  Coverage in [0.85, 0.95]           -> CALIBRATED, trust CIs
  Coverage in [0.80, 0.85] or [0.95, 0.98] -> WARN, use with caution
  Coverage outside [0.80, 0.98]      -> BROKEN, do not use for decisions

Data-generating process:
  - AR(1) return structure (phi captures autocorrelation)
  - Stochastic-vol (GARCH-lite): vol_t = beta*vol_{t-1} + (1-beta)*|shock_{t-1}|
  - Controlled strat-bench correlation (rho) mimicking real strategies
    (which are highly correlated with their benchmark)

Usage:
    python experiments/ci_calibration.py
    python experiments/ci_calibration.py --quick   (smaller sweep)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]  # workflows/etf/
REPO_ROOT = WORKFLOW_ROOT.parents[1]                 # youBet/
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.stats import excess_sharpe_ci
import pandas as pd


# --- Data-generating process -------------------------------------------------


def simulate_paired_returns(
    n_days: int,
    strat_annual_sharpe: float,
    bench_annual_sharpe: float,
    annual_vol: float = 0.16,
    phi: float = 0.05,
    vol_persistence: float = 0.90,
    rho: float = 0.85,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired (strategy, benchmark) daily return series.

    Returns have AR(1) autocorrelation, GARCH-like vol clustering,
    and controlled strat-bench correlation rho.

    Args:
        n_days: Number of daily observations.
        strat_annual_sharpe: True annualized Sharpe of strategy.
        bench_annual_sharpe: True annualized Sharpe of benchmark.
        annual_vol: Target annualized volatility for both series.
        phi: AR(1) coefficient (0 = iid, 0.1 = weak persistence).
        vol_persistence: Vol clustering parameter (0 = no clustering).
        rho: Correlation between strat and bench shocks.
        seed: Random seed.

    Returns:
        (strat_returns, bench_returns) as numpy arrays.
    """
    rng = np.random.default_rng(seed)

    daily_vol = annual_vol / np.sqrt(252)
    # Correct calibration: daily_mean = annual_sharpe * annual_vol / 252
    # (NOT sharpe * daily_vol, which gives sqrt(252)x too-large signal)
    strat_mean = strat_annual_sharpe * annual_vol / 252
    bench_mean = bench_annual_sharpe * annual_vol / 252

    # Correlated bivariate normal shocks
    # Covariance structure: unit variance, corr rho
    L = np.array([[1.0, 0.0], [rho, np.sqrt(1 - rho**2)]])
    z = rng.standard_normal((n_days, 2))
    shocks = z @ L.T  # shape (n_days, 2), columns = strat_shock, bench_shock

    # Stochastic volatility (shared between strat and bench)
    vol_scale = np.empty(n_days)
    vol_scale[0] = 1.0
    for t in range(1, n_days):
        vol_scale[t] = (
            vol_persistence * vol_scale[t - 1]
            + (1 - vol_persistence) * (1 + 0.5 * abs(shocks[t - 1, 0]))
        )
    # Normalize vol_scale so realized variance matches daily_vol^2
    # (otherwise realized vol > intended vol, shrinking realized Sharpe)
    vol_scale = vol_scale / np.sqrt(np.mean(vol_scale**2))

    # Apply vol scaling
    strat_innov = shocks[:, 0] * daily_vol * vol_scale
    bench_innov = shocks[:, 1] * daily_vol * vol_scale

    # AR(1) structure
    strat_ret = np.empty(n_days)
    bench_ret = np.empty(n_days)
    strat_ret[0] = strat_mean + strat_innov[0]
    bench_ret[0] = bench_mean + bench_innov[0]
    for t in range(1, n_days):
        strat_ret[t] = strat_mean + phi * (strat_ret[t - 1] - strat_mean) + strat_innov[t]
        bench_ret[t] = bench_mean + phi * (bench_ret[t - 1] - bench_mean) + bench_innov[t]

    return strat_ret, bench_ret


# --- Calibration evaluation --------------------------------------------------


def evaluate_cell(
    true_diff: float,
    n_days: int,
    block_length: int,
    n_replicates: int,
    n_bootstrap: int,
    rho: float = 0.85,
    phi: float = 0.05,
    annual_vol: float = 0.16,
    bench_sharpe: float = 0.50,
    confidence: float = 0.90,
) -> dict:
    """Estimate CI calibration for one parameter cell.

    For n_replicates simulated datasets, run excess_sharpe_ci and
    record whether the CI covers true_diff and which verdict tier
    the diagnostic assigned.
    """
    strat_sharpe = bench_sharpe + true_diff
    covered = 0
    excludes_zero = 0
    verdict_counts = {
        "STRONG_EDGE": 0, "WEAK_EDGE": 0,
        "INCONCLUSIVE_POSITIVE": 0, "INCONCLUSIVE": 0, "NEGATIVE": 0,
    }
    ci_widths = []
    point_estimates = []

    for rep in range(n_replicates):
        strat, bench = simulate_paired_returns(
            n_days=n_days,
            strat_annual_sharpe=strat_sharpe,
            bench_annual_sharpe=bench_sharpe,
            annual_vol=annual_vol,
            phi=phi,
            rho=rho,
            seed=rep,
        )
        dates = pd.bdate_range("2000-01-01", periods=n_days)
        strat_s = pd.Series(strat, index=dates)
        bench_s = pd.Series(bench, index=dates)

        ci = excess_sharpe_ci(
            strat_s, bench_s,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            expected_block_length=block_length,
            seed=rep * 97 + 3,
        )

        if ci["ci_lower"] <= true_diff <= ci["ci_upper"]:
            covered += 1
        if ci["ci_lower"] > 0 or ci["ci_upper"] < 0:
            excludes_zero += 1
        verdict_counts[ci["diagnostic_verdict"]] += 1
        ci_widths.append(ci["ci_upper"] - ci["ci_lower"])
        point_estimates.append(ci["point_estimate"])

    return {
        "true_diff": true_diff,
        "n_days": n_days,
        "block_length": block_length,
        "n_replicates": n_replicates,
        "coverage": covered / n_replicates,
        "excludes_zero_rate": excludes_zero / n_replicates,
        "median_ci_width": float(np.median(ci_widths)),
        "median_point_estimate": float(np.median(point_estimates)),
        "point_estimate_std": float(np.std(point_estimates)),
        "verdict_counts": verdict_counts,
    }


def classify_coverage(coverage: float, nominal: float = 0.90) -> str:
    """Apply pre-committed decision thresholds."""
    deviation = abs(coverage - nominal)
    if deviation <= 0.05:
        return "CALIBRATED"
    elif deviation <= 0.10:
        return "WARN"
    else:
        return "BROKEN"


# --- Main sweep --------------------------------------------------------------


def run_sweep(quick: bool = False) -> list[dict]:
    """Run the full calibration sweep.

    Sweeps over (true_diff, n_days, block_length) combinations.
    """
    if quick:
        true_diffs = [0.0, 0.10, 0.30]
        n_days_list = [2520, 5040]
        block_lengths = [22]
        n_replicates = 100
        n_bootstrap = 500
    else:
        true_diffs = [0.0, 0.05, 0.10, 0.20, 0.30]
        n_days_list = [1260, 2520, 5040]
        block_lengths = [5, 22, 63]
        n_replicates = 300
        n_bootstrap = 1000

    results = []
    total_cells = len(true_diffs) * len(n_days_list) * len(block_lengths)
    cell_idx = 0
    for true_diff in true_diffs:
        for n_days in n_days_list:
            for block in block_lengths:
                cell_idx += 1
                print(
                    f"[{cell_idx}/{total_cells}] "
                    f"true_diff={true_diff:+.2f} n_days={n_days} "
                    f"block={block}...",
                    flush=True,
                )
                cell = evaluate_cell(
                    true_diff=true_diff,
                    n_days=n_days,
                    block_length=block,
                    n_replicates=n_replicates,
                    n_bootstrap=n_bootstrap,
                )
                results.append(cell)
                verdict = classify_coverage(cell["coverage"])
                print(
                    f"   coverage={cell['coverage']:.3f} "
                    f"excl_zero={cell['excludes_zero_rate']:.3f} "
                    f"width={cell['median_ci_width']:.3f}  [{verdict}]",
                    flush=True,
                )
    return results


def print_report(results: list[dict], nominal_confidence: float = 0.90):
    """Print a formatted calibration report."""
    print()
    print("=" * 78)
    print("CI CALIBRATION RESULTS")
    print("=" * 78)
    print()
    print(f"Target coverage: {nominal_confidence:.2f}")
    print(f"Thresholds: CALIBRATED |c-{nominal_confidence:.2f}|<=0.05, "
          f"WARN <=0.10, BROKEN otherwise")
    print()
    print(f"{'true_diff':>9} {'n_days':>7} {'block':>6} {'coverage':>9} "
          f"{'excl_0':>8} {'width':>7} {'status':>10}")
    print("-" * 78)
    for r in results:
        status = classify_coverage(r["coverage"], nominal_confidence)
        print(
            f"{r['true_diff']:>+9.2f} "
            f"{r['n_days']:>7} "
            f"{r['block_length']:>6} "
            f"{r['coverage']:>9.3f} "
            f"{r['excludes_zero_rate']:>8.3f} "
            f"{r['median_ci_width']:>7.3f} "
            f"{status:>10}"
        )

    # Summary
    print()
    print("SUMMARY")
    print("-" * 78)
    n_broken = sum(
        1 for r in results
        if classify_coverage(r["coverage"], nominal_confidence) == "BROKEN"
    )
    n_warn = sum(
        1 for r in results
        if classify_coverage(r["coverage"], nominal_confidence) == "WARN"
    )
    n_calibrated = len(results) - n_broken - n_warn
    print(f"CALIBRATED cells: {n_calibrated}/{len(results)}")
    print(f"WARN cells:       {n_warn}/{len(results)}")
    print(f"BROKEN cells:     {n_broken}/{len(results)}")
    print()

    # FPR check: true_diff=0 cells should have excludes_zero_rate ~ 1 - confidence
    fpr_cells = [r for r in results if r["true_diff"] == 0.0]
    if fpr_cells:
        target_fpr = 1 - nominal_confidence
        print("FALSE POSITIVE RATE (true_diff = 0):")
        print(f"  Target: ~{target_fpr:.2f}")
        for r in fpr_cells:
            ok = abs(r["excludes_zero_rate"] - target_fpr) <= 0.05
            flag = "OK" if ok else "HIGH" if r["excludes_zero_rate"] > target_fpr + 0.05 else "LOW"
            print(
                f"  n_days={r['n_days']}, block={r['block_length']}: "
                f"FPR={r['excludes_zero_rate']:.3f}  [{flag}]"
            )
    print()

    # Overall verdict
    if n_broken > 0:
        print("VERDICT: BROKEN. Some cells have coverage outside [0.80, 0.98].")
        print("         CI interpretations are unreliable. Fix before using.")
    elif n_warn > len(results) // 2:
        print("VERDICT: MOSTLY WARN. Coverage is acceptable but not tight.")
        print("         Use CIs with caution, consider adjusting block length.")
    elif n_calibrated == len(results):
        print("VERDICT: CALIBRATED. All cells in [0.85, 0.95] coverage.")
        print("         CI interpretations are trustworthy.")
    else:
        print("VERDICT: ACCEPTABLE. Most cells calibrated, some warn.")
        print("         CI interpretations are trustworthy with minor caveats.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Small sweep for quick verification")
    args = parser.parse_args()

    print("CI CALIBRATION SIMULATION STUDY")
    print("Validating excess_sharpe_ci under realistic return DGP")
    print("(AR(1) + stochastic vol + correlated paired series)")
    print()

    results = run_sweep(quick=args.quick)
    print_report(results)

    return results


if __name__ == "__main__":
    main()
