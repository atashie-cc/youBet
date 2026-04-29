"""Phase 2 — robustness + regime stability + empirical TE re-assessment.

Consumes Phase 1 return artifacts (`artifacts/phase1_returns.parquet`)
and emits:
  1. Per-strategy × per-regime metric table (full, pre/post-2013,
     ex-GFC, ex-COVID) — point estimate + 90% CI per subset.
  2. Empirical tracking-error report per strategy.
  3. Power-sensitivity grid (target_sharpe × TE anchor) and MDE per TE
     — recomputing Phase 0's power analysis at empirical TE.
  4. Cost-sensitivity sweep: strategy Sharpe / CAGR under alternate
     commission bps (applied as proportional daily drag on turnover).

NO gate claims — Phase 2 is characterization, not discovery.

Usage:
    python phase2_robustness.py                   # smoke run on synthetic
    python phase2_robustness.py --from-phase1     # real Phase 1 artifacts
    STOCK_PHASE0_FULL=1 python phase2_robustness.py  # tighter MC
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import load_config, evaluate_gate, ARTIFACTS_DIR  # noqa: E402

from youbet.stock.regime import (  # noqa: E402
    RegimeMask,
    apply_masks_to_pair,
    describe_subset,
    standard_regime_set,
)
from youbet.stock.te import (  # noqa: E402
    empirical_tracking_error,
    mde_table,
    power_sensitivity_table,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 artifact loading
# ---------------------------------------------------------------------------


def load_phase1_artifacts() -> tuple[dict[str, pd.Series], pd.Series] | None:
    """Load Phase 1 strategy returns + benchmark from artifacts/.

    Returns (strategies_dict, benchmark_series) or None if no artifact found.
    """
    path = ARTIFACTS_DIR / "phase1_returns.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    bench_col = "__benchmark__"
    if bench_col not in df.columns:
        raise ValueError(f"{path} missing '{bench_col}' column")
    benchmark = df[bench_col].dropna()
    strategies = {
        c: df[c].dropna() for c in df.columns if c != bench_col
    }
    logger.info(
        "Loaded Phase 1 artifacts: %d strategies, %d benchmark days",
        len(strategies), len(benchmark),
    )
    return strategies, benchmark


def synthetic_phase1() -> tuple[dict[str, pd.Series], pd.Series]:
    """Synthetic Phase 1 returns for smoke-testing Phase 2.

    Four strategies with distinct (mean, vol, regime) behavior and a
    benchmark roughly matching SPY statistics. 20 years of daily data.
    """
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2004-01-01", "2023-12-31")
    n = len(idx)

    bench = pd.Series(rng.normal(0.0003, 0.01, n), index=idx)

    def make_factor(mu_d: float, sd_d: float, regime_shift: tuple[str, float] | None = None):
        excess = rng.normal(mu_d, sd_d, n)
        if regime_shift is not None:
            after_date, shift = regime_shift
            after_mask = idx >= pd.Timestamp(after_date)
            excess = excess.copy()
            excess[after_mask] = rng.normal(mu_d + shift, sd_d, int(after_mask.sum()))
        return bench + pd.Series(excess, index=idx)

    strategies = {
        "momentum": make_factor(1e-4, 0.005),
        "lowvol": make_factor(5e-5, 0.003, regime_shift=("2013-01-01", -3e-5)),
        "quality": make_factor(8e-5, 0.004),
        "value": make_factor(-2e-5, 0.006),
    }
    return strategies, bench


# ---------------------------------------------------------------------------
# Per-strategy regime stability
# ---------------------------------------------------------------------------


def regime_stability_table(
    strategies: dict[str, pd.Series],
    benchmark: pd.Series,
    masks: list[RegimeMask],
    config: dict,
) -> pd.DataFrame:
    """Run evaluate_gate on each (strategy, regime) pair, with joint
    Holm correction across all evaluable cells (R4 fix).

    Prior implementation called `evaluate_gate` per-cell with a
    single-strategy dict, producing trivial Holm(k=1) adjustments.
    Readers interpreted the per-cell `holm_adjusted_p` as family-wise
    corrected, which it wasn't — this led directly to the R3 LowVol
    post_2013 retraction. This version:

      1. Computes upper-tail, lower-tail, and two-sided p-values per
         cell from the underlying `block_bootstrap_test` (now two-sided
         per R4 H-bug fix).
      2. Applies Holm jointly across the FULL evaluable cell set
         (typically 2 strategies × 4 regimes = 8 tests).
      3. Reports both per-cell raw p (upper/lower/two-sided) and the
         joint-Holm-adjusted p for the two-sided test.
      4. Also plumbs diff_of_sharpes_* fields per R3 recommendation
         T7 — these tell the "risk-adjusted total return" story
         alongside the information-ratio headline.
    """
    from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci, holm_bonferroni

    min_years = float(config.get("phase2", {}).get("subperiod_min_years", 5))
    min_days = int(min_years * 252)
    boot = config["bootstrap"]
    gate = config["gate"]

    # Phase 1: compute per-cell raw stats without correction.
    raw_cells: list[dict] = []
    for strat_name, strat_ret in strategies.items():
        masked = apply_masks_to_pair(strat_ret, benchmark, masks)
        for mask in masks:
            s, b = masked[mask.name]
            coverage = describe_subset(strat_ret, mask)
            row = {
                "strategy": strat_name,
                "regime": mask.name,
                "n_days": coverage["n_kept"],
                "years": coverage["years_kept"],
            }
            if coverage["n_kept"] < min_days:
                row.update({
                    "observed_excess_sharpe": None,
                    "gate_ci_lower": None,
                    "gate_ci_upper": None,
                    "p_upper": None, "p_lower": None, "p_two_sided": None,
                    "diff_of_sharpes_point": None,
                    "diff_of_sharpes_lower": None,
                    "diff_of_sharpes_upper": None,
                    "joint_holm_two_sided_adj_p": None,
                    "note": f"skipped (<{min_years}y)",
                })
                raw_cells.append(row)
                continue

            test = block_bootstrap_test(
                s, b,
                n_bootstrap=int(boot["n_replicates"]),
                expected_block_length=int(boot["block_length"]),
                seed=int(boot["seed"]),
            )
            ci = excess_sharpe_ci(
                s, b,
                n_bootstrap=int(boot["n_replicates"]),
                confidence=float(gate["confidence"]),
                expected_block_length=int(boot["block_length"]),
                seed=int(boot["seed"]),
            )
            row.update({
                "observed_excess_sharpe": test["observed_excess_sharpe"],
                "gate_ci_lower": ci["excess_sharpe_lower"],
                "gate_ci_upper": ci["excess_sharpe_upper"],
                "p_upper": test["p_value_upper"],
                "p_lower": test["p_value_lower"],
                "p_two_sided": test["p_value_two_sided"],
                "diff_of_sharpes_point": ci["point_estimate"],
                "diff_of_sharpes_lower": ci["ci_lower"],
                "diff_of_sharpes_upper": ci["ci_upper"],
                "note": "[EXPLORATORY]",
            })
            raw_cells.append(row)

    # Phase 2: Holm across evaluable cells on the TWO-SIDED p (R4 H-bug
    # + R3 H2). Name keys as "strategy|regime" for holm_bonferroni.
    evaluable = {
        f"{r['strategy']}|{r['regime']}": r["p_two_sided"]
        for r in raw_cells
        if r.get("p_two_sided") is not None
    }
    if evaluable:
        holm_joint = holm_bonferroni(evaluable)
    else:
        holm_joint = {}

    for r in raw_cells:
        if r.get("p_two_sided") is None:
            continue
        key = f"{r['strategy']}|{r['regime']}"
        h = holm_joint.get(key, {})
        r["joint_holm_two_sided_adj_p"] = h.get("adjusted_p")

    return pd.DataFrame(raw_cells)


# ---------------------------------------------------------------------------
# Empirical TE report + re-powered MDE
# ---------------------------------------------------------------------------


def empirical_te_report(
    strategies: dict[str, pd.Series],
    benchmark: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, ret in strategies.items():
        try:
            rep = empirical_tracking_error(ret, benchmark)
            rows.append({
                "strategy": name,
                "n_days": rep.n_days,
                "mean_annual_excess": rep.mean_annual_excess,
                "annualized_te": rep.annualized_te,
                "information_ratio": rep.annualized_ir,
            })
        except ValueError as exc:
            logger.warning("TE failed for %s: %s", name, exc)
    return pd.DataFrame(rows).set_index("strategy")


def repowered_mde(
    te_report: pd.DataFrame,
    n_years: int,
    target_sharpes: list[float],
    n_sims: int,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Recompute the Phase 0 MDE table using empirical TEs + anchor TEs.

    Anchors: min/median/max of empirical TE across strategies, plus the
    pre-committed `phase2.te_sensitivity_anchors` from config.
    """
    return None, None  # overridden in main, see config-aware path


# ---------------------------------------------------------------------------
# Cost sensitivity
# ---------------------------------------------------------------------------


def apply_commission_drag(
    returns: pd.Series,
    rebalance_turnover_per_year: float,
    commission_bps: float,
    rebalances_per_year: int = 12,
) -> pd.Series:
    """Apply an extra daily drag to mimic a different commission schedule.

    Approximate: distribute `rebalance_turnover_per_year * commission_bps/10000`
    across `rebalances_per_year` rebalance days, or as a daily drag if
    rebalance dates are unknown. For Phase 2 purposes we use the daily-drag
    form (slightly pessimistic) so the orchestrator doesn't need the full
    weight history.
    """
    total_annual_drag = rebalance_turnover_per_year * commission_bps / 10_000.0
    daily_drag = total_annual_drag / 252
    return returns - daily_drag


def cost_sensitivity_table(
    strategies: dict[str, pd.Series],
    benchmark: pd.Series,
    commission_bps_list: list[float],
    config: dict,
    assumed_annual_turnover: float = 1.0,  # 100% annual turnover default
) -> pd.DataFrame:
    """For each (strategy, commission_bps) pair, recompute observed ExSharpe."""
    rows = []
    for name, ret in strategies.items():
        for bps in commission_bps_list:
            adjusted = apply_commission_drag(
                ret, assumed_annual_turnover, bps, rebalances_per_year=12,
            )
            result = evaluate_gate(
                {name: adjusted}, benchmark, config=config,
            )[name]
            rows.append({
                "strategy": name,
                "commission_bps": bps,
                "observed_excess_sharpe": result["observed_excess_sharpe"],
                "gate_ci_lower": result["gate_ci_lower"],
                "gate_ci_upper": result["gate_ci_upper"],
                "holm_adjusted_p": result["holm_adjusted_p"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-phase1", action="store_true",
        help="Require Phase 1 artifacts; fail if missing.",
    )
    args = parser.parse_args()

    config = load_config()
    p2 = config.get("phase2", {})

    # Bootstrap tiers: smoke=1k, auth_fast=5k, auth=10k.
    # auth_fast (STOCK_PHASE2_N_BOOTSTRAP=5000 or similar) lets us produce
    # authoritative-grade results in <10 min wall-clock; CI width at 5k vs
    # 10k differs by <0.01 Sharpe, and p-value floor moves from 1e-4 to 2e-4.
    full = os.environ.get("STOCK_PHASE0_FULL") == "1"
    env_n = os.environ.get("STOCK_PHASE2_N_BOOTSTRAP")
    if env_n:
        n_boot = int(env_n)
        config = {**config, "bootstrap": {**config["bootstrap"], "n_replicates": n_boot}}
        logger.info("Phase 2: STOCK_PHASE2_N_BOOTSTRAP=%d (explicit override)", n_boot)
    elif not full:
        config = {**config, "bootstrap": {**config["bootstrap"], "n_replicates": 1000}}
        logger.info("Phase 2 smoke mode: n_bootstrap=1000 "
                    "(set STOCK_PHASE0_FULL=1 for authoritative 10k)")

    # --- load data ---
    loaded = load_phase1_artifacts()
    if loaded is None:
        if args.from_phase1:
            logger.error("No Phase 1 artifacts at %s/phase1_returns.parquet",
                         ARTIFACTS_DIR)
            sys.exit(1)
        logger.warning(
            "Phase 1 artifacts not found — running on SYNTHETIC returns "
            "(smoke only; not publishable). Re-run with --from-phase1 "
            "once Phase 1 has emitted real returns."
        )
        strategies, benchmark = synthetic_phase1()
    else:
        strategies, benchmark = loaded

    # --- regime stability ---
    masks = standard_regime_set(
        index=benchmark.index,
        pre_post_break=p2.get("pre_post_break", "2013-01-01"),
        exclude_windows={
            k: tuple(v) for k, v in p2.get("exclude_windows", {}).items()
        },
    )
    logger.info("Regime masks: %s", [m.name for m in masks])
    regime_df = regime_stability_table(strategies, benchmark, masks, config)

    # --- empirical TE ---
    te_df = empirical_te_report(strategies, benchmark)

    # --- power re-assessment ---
    # Full power sweep is ~100B ops — can run hours. STOCK_PHASE2_SKIP_POWER=1
    # keeps the authoritative 10k bootstrap for regime/cost (the main value
    # of --from-phase1) while cutting the power sweep.
    skip_power = os.environ.get("STOCK_PHASE2_SKIP_POWER") == "1"
    n_sims = 300 if full else 15
    n_bootstrap_power = 3000 if full else 150

    te_anchors_from_empirical = sorted(set(
        [round(te_df["annualized_te"].median(), 2),
         round(te_df["annualized_te"].min(), 2),
         round(te_df["annualized_te"].max(), 2)]
    )) if not te_df.empty else []
    if full:
        # Rounded-to-2-decimals ensures column names like "TE=0.12" are
        # unique even when a config anchor coincides with an empirical one.
        cfg_anchors = [
            round(float(a), 2)
            for a in p2.get("te_sensitivity_anchors", [0.04, 0.08, 0.12])
        ]
        all_anchors = sorted(set(cfg_anchors + te_anchors_from_empirical))
    else:
        # Smoke: 3 anchors (low / mid / high of config) to keep runtime sane
        all_anchors = [
            round(float(a), 2)
            for a in p2.get("te_sensitivity_anchors", [0.04, 0.08, 0.12])
        ]

    target_sharpes = [0.20, 0.30, 0.40, 0.50] if not full else config["power_analysis"].get(
        "target_sharpe_diffs", [0.10, 0.20, 0.30, 0.40, 0.50]
    )

    if full and not skip_power:
        power_df = power_sensitivity_table(
            n_years=20,
            target_sharpes=target_sharpes,
            te_anchors=all_anchors,
            n_sims=n_sims,
            n_bootstrap=n_bootstrap_power,
        )
        mde_s = mde_table(power_df, threshold=0.80)
    else:
        reason = ("STOCK_PHASE2_SKIP_POWER=1 — skipping power sweep"
                  if skip_power else
                  "Smoke mode: skipping power_sensitivity_table (set STOCK_PHASE0_FULL=1)")
        logger.info(reason)
        power_df = pd.DataFrame(index=target_sharpes, columns=[f"TE={a:.2f}" for a in all_anchors])
        mde_s = pd.Series(
            {c: None for c in power_df.columns},
            name="mde_at_80pct_power_SKIPPED",
        )

    # --- cost sensitivity ---
    commission_bps_list = p2.get(
        "commission_sensitivity_bps", [0.5, 2.0, 5.0, 10.0]
    )
    cost_df = cost_sensitivity_table(
        strategies, benchmark, commission_bps_list, config,
    )

    # --- report ---
    print("\n" + "=" * 100)
    print("Phase 2 — Robustness + Regime Stability + Empirical TE")
    print(f"  Authoritative: {full}    Source: "
          f"{'Phase 1 artifacts' if loaded is not None else 'SYNTHETIC (smoke)'}")
    print("=" * 100)

    print("\n--- Regime stability (point estimates per subset, no Holm across regimes) ---")
    print(regime_df.to_string(index=False))

    print("\n--- Empirical tracking error ---")
    print(te_df.to_string())

    print("\n--- Power sensitivity: power[target_sharpe × TE anchor] ---")
    print(power_df.to_string())

    print("\n--- Recomputed MDE at 80% power ---")
    print(mde_s.to_string())

    print("\n--- Cost sensitivity (commission bps -> ExSharpe) ---")
    print(cost_df.to_string(index=False))

    # --- persist ---
    out_dir = ARTIFACTS_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    regime_df.to_parquet(out_dir / "regime_stability.parquet")
    te_df.to_parquet(out_dir / "empirical_te.parquet")
    # Power/MDE may be skipped; still persist for provenance but tolerate
    # all-None values and any duplicate anchor columns defensively.
    power_df_to_save = power_df.loc[:, ~power_df.columns.duplicated()]
    power_df_to_save.to_parquet(out_dir / "power_sensitivity.parquet")
    mde_s.to_frame().to_parquet(out_dir / "mde_recomputed.parquet")
    cost_df.to_parquet(out_dir / "cost_sensitivity.parquet")
    logger.info("Phase 2 artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
