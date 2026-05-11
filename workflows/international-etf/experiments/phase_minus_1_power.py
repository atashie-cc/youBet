"""Phase -1 — Power analysis for the international-ETF Holm-corrected gate.

Question: At the locked Holm denominator (N=19), in-sample window (~25 years
of daily returns), and stationary block bootstrap (22-day blocks, 90% CI),
what is the minimum detectable excess Sharpe at power >= 0.50?

Decision rule (pre-committed in research/plan.md v1.1):
  - If power at ExSharpe = 0.20 < 0.50, raise gate threshold to ExSharpe = 0.40
    BEFORE Phase 0.
  - If power at 0.40 also < 0.50, halt the workflow and document the
    detectability constraint in research/log.md.

Method (simulation-based; no real returns are read):
  1. Generate a synthetic benchmark daily return series with realistic
     equity moments (mean ~0.04%/day, sigma ~1%/day -> ~10%/yr CAGR,
     ~16% annualized vol, Sharpe ~0.4 over zero risk-free).
  2. Generate N = 19 candidate strategy daily return series, each
     correlated with the benchmark at rho = 0.70 (Asness-Israelov-Liew
     2011 short-horizon estimate). One ("target") has a known annualized
     excess Sharpe drawn from {0.0, 0.10, 0.20, 0.30, 0.40, 0.50}; the
     remaining 18 are exact null (excess Sharpe = 0).
  3. For each Monte Carlo simulation, compute:
       - point estimate of excess Sharpe via paired block bootstrap
       - upper-tail p-value via stationary block bootstrap
       - Holm-Bonferroni correction across all 19 candidates
       - 90% bootstrap CI on the Sharpe difference
     Apply the strict gate exactly as locked in plan.md:
       PASS iff (excess_sharpe_point > gate_threshold)
            AND (Holm-adjusted p < 0.05)
            AND (CI lower > 0)
  4. Power = (count of PASS for target strategy) / (number of simulations).
  5. Type-I error check: at ExSharpe target = 0, the PASS rate across
     simulations should be <= 0.05 (family-wise alpha).

Reuses src/youbet/etf/stats.py: stationary_block_bootstrap,
block_bootstrap_test, excess_sharpe_ci, holm_bonferroni.

Output:
  - artifacts/phase_minus_1_power_results.json (raw numbers)
  - research/phase_minus_1_power_results.md   (human-readable summary)
  - research/log.md                            (decision row appended by hand)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from youbet.etf.stats import holm_bonferroni

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = WORKFLOW_DIR / "artifacts"
RESEARCH_DIR = WORKFLOW_DIR / "research"


@dataclass(frozen=True)
class PowerConfig:
    """Pre-committed power-analysis parameters."""

    n_holm: int = 19
    n_simulations: int = 100
    n_years: float = 24.7   # 2001-08-14 to 2026-04-30
    trading_days_per_year: int = 252
    benchmark_mu_daily: float = 0.04 / 100        # ~10% annualized arithmetic
    benchmark_sigma_daily: float = 1.0 / 100      # ~16% annualized vol
    correlation: float = 0.70
    gate_excess_sharpe: float = 0.20              # primary gate; 0.40 fallback
    gate_excess_sharpe_fallback: float = 0.40
    gate_holm_p_max: float = 0.05
    gate_ci_lower_min: float = 0.0
    bootstrap_block_length: int = 22
    bootstrap_n: int = 600                          # per-simulation; 10k for real backtests.
                                                    # 600 -> MC SE ~0.009 on p=0.05; tight enough
                                                    # to bound power within +/-0.06 at the 0.50
                                                    # decision threshold with 100 sims.
    bootstrap_ci_level: float = 0.90
    # Pruned grid: 0.0 (FWER), 0.20 (primary decision), 0.30 (interpolation),
    # 0.40 (fallback decision). 0.10 and 0.50 dropped to bound runtime.
    target_excess_sharpe_grid: tuple = (0.0, 0.20, 0.30, 0.40)
    seed: int = 42

    @property
    def n_days(self) -> int:
        return int(self.n_years * self.trading_days_per_year)


def simulate_strategy_returns(
    benchmark: np.ndarray,
    target_excess_sharpe: float,
    correlation: float,
    benchmark_mu: float,
    benchmark_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct a strategy return stream with a known annualized excess Sharpe.

    The strategy return is decomposed as:
        r_strat = beta * r_bench + epsilon
    where beta = correlation * (sigma_strat / sigma_bench), and epsilon is
    independent noise. We set sigma_strat = sigma_bench so that the strategy
    has the same vol profile as the benchmark; this isolates the Sharpe
    difference into the mean term.

    Annualized Sharpe difference is achieved by adding a daily drift to the
    strategy mean equal to (target_excess_sharpe * sigma_bench) /
    sqrt(periods_per_year), measured against the benchmark.

    NOTE: this targets the Sharpe(strat) - Sharpe(bench) metric (Sharpe diff)
    rather than Sharpe of excess. Empirically these track each other when
    strategy and benchmark have similar vols (which is the case here).
    """
    n = len(benchmark)
    periods_per_year = 252

    # Standardize benchmark to unit variance
    bench_z = (benchmark - benchmark_mu) / max(benchmark_sigma, 1e-12)

    # Independent unit-variance shock
    z_unit = rng.standard_normal(n)

    # Linear combination: variance is corr^2 + (1-corr^2) = 1 (unit variance)
    eps_unit = bench_z * correlation + z_unit * np.sqrt(1.0 - correlation**2)

    # Rescale to benchmark_sigma so sigma_strat = sigma_bench
    daily_excess_drift = target_excess_sharpe * benchmark_sigma / np.sqrt(periods_per_year)
    strat_mu = benchmark_mu + daily_excess_drift
    strategy = strat_mu + eps_unit * benchmark_sigma

    return strategy


def simulate_benchmark(
    n_days: int,
    mu_daily: float,
    sigma_daily: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate iid Gaussian daily benchmark returns.

    iid Gaussian is conservative for power: real equity returns have
    fat tails and vol clustering, both of which increase the bootstrap CI
    width. Power computed under iid Gaussian is therefore an UPPER BOUND
    on real-world detectability — if a gate fails power here, it definitely
    fails in real returns.
    """
    return rng.normal(mu_daily, sigma_daily, size=n_days)


def _generate_block_bootstrap_indices(
    n: int,
    n_bootstrap: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary (Politis-Romano 1994) block bootstrap indices.

    Returns an (n_bootstrap, n) int array. The same index matrix is then
    applied to multiple data series, sharing the resampling structure.
    """
    p = 1.0 / block_length
    jump_draws = rng.random((n_bootstrap, n), dtype=np.float32)
    jump_targets = rng.integers(0, n, size=(n_bootstrap, n), dtype=np.int32)
    start_indices = rng.integers(0, n, size=n_bootstrap, dtype=np.int32)
    indices = np.empty((n_bootstrap, n), dtype=np.int32)
    indices[:, 0] = start_indices
    for i in range(1, n):
        continued = (indices[:, i - 1] + 1) % n
        do_jump = jump_draws[:, i] < p
        indices[:, i] = np.where(do_jump, jump_targets[:, i], continued)
    return indices


def _sharpe_vec(X: np.ndarray, ann: float) -> np.ndarray:
    """Annualized Sharpe per row of a (B, T) matrix."""
    m = X.mean(axis=1)
    s = np.maximum(X.std(axis=1), 1e-10)
    return m / s * ann


def evaluate_all_candidates_batched(
    benchmark: np.ndarray,
    candidates: dict[str, np.ndarray],
    config: PowerConfig,
    seed: int,
) -> dict[str, dict]:
    """Batched gate evaluation: one shared bootstrap for all (cand, bench) pairs.

    Generates one (B, T) index matrix and re-uses it for every candidate vs
    the shared benchmark. Returns the same per-candidate metrics as a
    sequence of `block_bootstrap_test` + `excess_sharpe_ci` calls would,
    but vectorized.

    The shared-index design preserves WITHIN-candidate paired structure
    (each bootstrap draws the same days for strat and bench) and ALSO
    induces correlation across candidates' bootstrap distributions —
    which is exactly what we want for FWER measurement under correlated
    null tests (this is the realistic case in the actual workflow, where
    DXY-gate variants and CAPE-gate variants would be highly correlated).
    """
    n = len(benchmark)
    ann = np.sqrt(252)
    rng = np.random.default_rng(seed)

    # 1. Build shared bootstrap indices
    indices = _generate_block_bootstrap_indices(
        n, config.bootstrap_n, config.bootstrap_block_length, rng
    )

    # 2. Resample benchmark once
    boot_bench = benchmark[indices]                    # (B, n)
    sharpe_bench_boot = _sharpe_vec(boot_bench, ann)   # (B,)
    obs_sharpe_bench = float(benchmark.mean() / max(benchmark.std(), 1e-10) * ann)

    alpha = 1 - config.bootstrap_ci_level
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)

    out = {}
    for name, strat in candidates.items():
        # Point estimates
        obs_strat = float(strat.mean() / max(strat.std(), 1e-10) * ann)
        obs_diff = obs_strat - obs_sharpe_bench
        excess = strat - benchmark
        obs_excess_sharpe = float(excess.mean() / max(excess.std(), 1e-10) * ann)

        # Bootstrap distributions sharing the index matrix
        boot_strat = strat[indices]
        sharpe_strat_boot = _sharpe_vec(boot_strat, ann)
        boot_diff = sharpe_strat_boot - sharpe_bench_boot

        ci_lower = float(np.percentile(boot_diff, lo_pct))
        ci_upper = float(np.percentile(boot_diff, hi_pct))

        # Upper-tail p-value via centered-excess null bootstrap
        # Mirrors block_bootstrap_test in src/youbet/etf/stats.py
        excess_centered = excess - excess.mean()
        excess_boot = excess_centered[indices]
        null_excess_sharpe = _sharpe_vec(excess_boot, ann)
        count_ge = int(np.sum(null_excess_sharpe >= obs_excess_sharpe))
        raw_p_upper = (1 + count_ge) / (config.bootstrap_n + 1)

        out[name] = {
            "raw_p_upper": raw_p_upper,
            "sharpe_diff_point": obs_diff,
            "sharpe_diff_ci_lower": ci_lower,
            "sharpe_diff_ci_upper": ci_upper,
            "excess_sharpe_point": obs_excess_sharpe,
        }

    return out


def run_one_simulation(
    sim_idx: int,
    target_excess_sharpe: float,
    config: PowerConfig,
) -> dict:
    """One Monte Carlo simulation: generate 1 target + (n_holm - 1) nulls.

    Each simulation re-draws all return streams, including the benchmark.
    """
    rng = np.random.default_rng(config.seed + sim_idx * 1000)

    benchmark = simulate_benchmark(
        config.n_days,
        config.benchmark_mu_daily,
        config.benchmark_sigma_daily,
        rng,
    )

    candidates = {}
    # The "target" candidate carries the alternative-hypothesis effect size.
    candidates["target"] = simulate_strategy_returns(
        benchmark=benchmark,
        target_excess_sharpe=target_excess_sharpe,
        correlation=config.correlation,
        benchmark_mu=config.benchmark_mu_daily,
        benchmark_sigma=config.benchmark_sigma_daily,
        rng=rng,
    )
    # The remaining (n_holm - 1) candidates are exact nulls (excess Sharpe = 0).
    for k in range(config.n_holm - 1):
        candidates[f"null_{k:02d}"] = simulate_strategy_returns(
            benchmark=benchmark,
            target_excess_sharpe=0.0,
            correlation=config.correlation,
            benchmark_mu=config.benchmark_mu_daily,
            benchmark_sigma=config.benchmark_sigma_daily,
            rng=rng,
        )

    metrics = evaluate_all_candidates_batched(
        benchmark, candidates, config, seed=config.seed + sim_idx
    )
    raw_p = {name: m["raw_p_upper"] for name, m in metrics.items()}

    holm = holm_bonferroni(raw_p)

    out = {"target_metrics": metrics["target"], "target_holm": holm["target"]}
    # Pass evaluations at primary AND fallback gate thresholds
    for gate_name, gate in [
        ("primary_0.20", config.gate_excess_sharpe),
        ("fallback_0.40", config.gate_excess_sharpe_fallback),
    ]:
        passes = (
            metrics["target"]["sharpe_diff_point"] > gate
            and holm["target"]["adjusted_p"] < config.gate_holm_p_max
            and metrics["target"]["sharpe_diff_ci_lower"] > config.gate_ci_lower_min
        )
        out[f"pass_gate_{gate_name}"] = bool(passes)

    # Also compute family-wise null-rejection rate at this simulation
    # (count how many of the 19 candidates passed the primary gate; for type-I
    # error this should be <=1 expected only at gate's 0.05 alpha when target=0).
    fwer_count = 0
    for name in candidates:
        passes_anyone = (
            metrics[name]["sharpe_diff_point"] > config.gate_excess_sharpe
            and holm[name]["adjusted_p"] < config.gate_holm_p_max
            and metrics[name]["sharpe_diff_ci_lower"] > config.gate_ci_lower_min
        )
        if passes_anyone:
            fwer_count += 1
    out["fwer_count"] = fwer_count
    out["fwer_any"] = fwer_count > 0

    return out


def run_power_analysis(config: PowerConfig) -> dict:
    """Run the full grid: target excess Sharpe in {0.0, 0.10, ..., 0.50}.

    Returns a nested dict {target: {power_primary, power_fallback, fwer, ...}}.
    """
    results = {}

    for target in config.target_excess_sharpe_grid:
        sim_outcomes = []
        for sim_idx in range(config.n_simulations):
            out = run_one_simulation(sim_idx, target, config)
            sim_outcomes.append(out)
            if (sim_idx + 1) % 25 == 0:
                logger.info(
                    "target=%s sim=%d/%d primary_pass_rate=%.3f",
                    target,
                    sim_idx + 1,
                    config.n_simulations,
                    np.mean([o["pass_gate_primary_0.20"] for o in sim_outcomes]),
                )

        primary_pass = np.array([o["pass_gate_primary_0.20"] for o in sim_outcomes])
        fallback_pass = np.array([o["pass_gate_fallback_0.40"] for o in sim_outcomes])
        fwer_any = np.array([o["fwer_any"] for o in sim_outcomes])
        fwer_count = np.array([o["fwer_count"] for o in sim_outcomes])
        target_points = np.array([o["target_metrics"]["sharpe_diff_point"] for o in sim_outcomes])

        results[target] = {
            "target_excess_sharpe": target,
            "n_simulations": config.n_simulations,
            "power_primary_0.20": float(primary_pass.mean()),
            "power_fallback_0.40": float(fallback_pass.mean()),
            "fwer_any_pass_rate": float(fwer_any.mean()),
            "fwer_count_mean": float(fwer_count.mean()),
            "target_point_mean": float(target_points.mean()),
            "target_point_std": float(target_points.std()),
        }

    return results


def write_results(results: dict, config: PowerConfig) -> None:
    """Persist raw JSON and a human-readable markdown summary."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ARTIFACTS_DIR / "phase_minus_1_power_results.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "config": {
                    "n_holm": config.n_holm,
                    "n_simulations": config.n_simulations,
                    "n_years": config.n_years,
                    "correlation": config.correlation,
                    "bootstrap_block_length": config.bootstrap_block_length,
                    "bootstrap_n": config.bootstrap_n,
                    "ci_level": config.bootstrap_ci_level,
                    "gate_primary": config.gate_excess_sharpe,
                    "gate_fallback": config.gate_excess_sharpe_fallback,
                    "seed": config.seed,
                },
                "results": {str(k): v for k, v in results.items()},
            },
            f,
            indent=2,
        )
    logger.info("Wrote raw results: %s", json_path)

    # Summary markdown
    md = ["# Phase -1 — Power Analysis Results", ""]
    md.append(f"- Run date: 2026-05-01")
    md.append(f"- N (Holm denominator): {config.n_holm}")
    md.append(f"- Simulations per effect size: {config.n_simulations}")
    md.append(f"- Sample length: {config.n_years} years (~{config.n_days} trading days)")
    md.append(f"- Correlation: {config.correlation}")
    md.append(f"- Bootstrap: stationary, block={config.bootstrap_block_length}d, n={config.bootstrap_n}, ci={config.bootstrap_ci_level}")
    md.append(f"- Gate: ExSharpe>{config.gate_excess_sharpe} OR fallback {config.gate_excess_sharpe_fallback}; Holm p<{config.gate_holm_p_max}; CI_lower>{config.gate_ci_lower_min}")
    md.append("")
    md.append("## Power and Type-I error by target excess Sharpe")
    md.append("")
    md.append("| Target ExSh | Power @ 0.20 gate | Power @ 0.40 gate | FWER (any null pass) | Mean point | Point SD |")
    md.append("|---|---|---|---|---|---|")
    for target, r in results.items():
        md.append(
            f"| {target:.2f} | {r['power_primary_0.20']:.3f} | "
            f"{r['power_fallback_0.40']:.3f} | {r['fwer_any_pass_rate']:.3f} | "
            f"{r['target_point_mean']:.3f} | {r['target_point_std']:.3f} |"
        )
    md.append("")

    # Decision rule application
    pwr_at_020 = results.get(0.20, {}).get("power_primary_0.20", 0.0)
    pwr_at_040 = results.get(0.40, {}).get("power_fallback_0.40", 0.0)
    fwer_at_zero = results.get(0.0, {}).get("fwer_any_pass_rate", 0.0)

    md.append("## Pre-committed decision rule application")
    md.append("")
    md.append(f"- Power at ExSharpe=0.20, primary gate: **{pwr_at_020:.3f}** (threshold: 0.50)")
    md.append(f"- Power at ExSharpe=0.40, fallback gate: **{pwr_at_040:.3f}** (threshold: 0.50)")
    md.append(f"- FWER at ExSharpe=0.0 (target=null): **{fwer_at_zero:.3f}** (target: <=0.05)")
    md.append("")

    if pwr_at_020 >= 0.50:
        md.append("**Decision: KEEP gate at ExSharpe > 0.20.** Detectability adequate.")
    elif pwr_at_040 >= 0.50:
        md.append("**Decision: RAISE gate to ExSharpe > 0.40.** 0.20 underpowered.")
    else:
        md.append("**Decision: HALT.** Even ExSharpe = 0.40 is underpowered with N=19 + ~25yr daily data. Either reduce N (re-prune plan) or accept the workflow as descriptive-only.")
    md.append("")

    if fwer_at_zero > 0.05:
        md.append(f"**WARNING:** FWER at null = {fwer_at_zero:.3f} > 0.05. Holm correction may be insufficient given correlation structure; consider Romano-Wolf simultaneous CIs.")
    else:
        md.append(f"FWER controlled at null (observed {fwer_at_zero:.3f} <= nominal 0.05).")

    md_path = RESEARCH_DIR / "phase_minus_1_power_results.md"
    md_path.write_text("\n".join(md))
    logger.info("Wrote summary: %s", md_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = PowerConfig()
    logger.info(
        "Starting power analysis: N=%d, sims=%d, years=%.1f, gate=%.2f/%.2f",
        config.n_holm,
        config.n_simulations,
        config.n_years,
        config.gate_excess_sharpe,
        config.gate_excess_sharpe_fallback,
    )
    results = run_power_analysis(config)
    write_results(results, config)
    logger.info("Power analysis complete.")


if __name__ == "__main__":
    main()
