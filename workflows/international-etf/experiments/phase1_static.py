"""Phase 1 — Static VTI/VXUS allocation sweep.

Weights ∈ {0%, 10%, 20%, 30%, 40%, 50%, 60%, 100%} VXUS in equity, with
the remainder in VTI. Annual rebalance only (per plan v1.1 H2 fix).
The 0% weight = pure VTI = benchmark, included as sanity-check (not gating).

Pre-committed gate (LOCKED v1.2 after Phase -1 power analysis):
  PASS iff (excess_sharpe_point > 0.40)
        AND (Holm-adjusted p < 0.05)
        AND (CI lower > 0)
        AND (mean log-excess > 0 with CI lower > 0)
        AND (passes at cost = 10 bps)

Workflow tier: DESCRIPTIVE/EXPLORATORY. Power at gate = 0.30; no PROCEED
will be claimed even if gate is met. Results reported as point + CI for
informational use.

Hypotheses operationalized:
  S1 (Vanguard 2021 rev 2019, primary-source verified): the realized
     10yr-rolling vol minimum sits in 35-55% ex-US for a US investor.
  S2 (Asness 2011): rolling 10y worst-case for 60/40 dominates worst-case
     for 100% US AND 100% ex-US.

Inputs read:
  - data/snapshots/<latest>/prices.parquet
  - data/snapshots/<latest>/macro/tb3ms.csv
  - data/reference/international_universe.csv
  - config.yaml (gate, walk-forward, cost levels)

Outputs:
  - artifacts/phase1_results.json
  - research/phase1_results.md
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_DIR / "experiments"))

from youbet.etf.backtester import Backtester, BacktestConfig            # noqa: E402
from youbet.etf.benchmark import BuyAndHold                              # noqa: E402
from youbet.etf.costs import COST_SCHEDULE, CostModel                    # noqa: E402
from youbet.etf.stats import (                                           # noqa: E402
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)

# Reuse helpers from phase0 by direct import. Both scripts are siblings.
from phase0_efficiency import (                                          # noqa: E402
    GateConfig,
    build_cost_model,
    compute_inferential_stats,
    evaluate_strict_gate,
    load_gate,
    load_prices,
    load_tbill_daily,
    load_workflow_config,
    log_excess_stats,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Pre-committed weight grid; 0 and 100 are descriptive (not gating).
WEIGHTS_VXUS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 1.00]
GATING_WEIGHTS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 1.00]
COST_LEVELS_BPS = [3.0, 5.0, 10.0]
GATING_COST_BPS = 10.0


def run_static_strategy(
    weight_vxus: float,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
    intl_cost_bps: float,
) -> dict:
    """Annual-rebalanced static VTI/VXUS mix vs 100% VTI buy-and-hold."""
    cost_model = build_cost_model(universe, intl_cost_bps)
    weights = {"VTI": 1 - weight_vxus, "VXUS": weight_vxus}
    weights = {k: v for k, v in weights.items() if v > 0}

    sub = prices[["VTI", "VXUS"]].dropna(how="any")
    sub = sub.loc[sub.index >= "2001-08-14"]
    sub = sub.loc[sub.index <= "2026-04-30"]

    bt = Backtester(
        config=bt_config,
        prices=sub,
        cost_model=cost_model,
        tbill_rates=tbill,
        universe=universe,
    )
    strategy = BuyAndHold(weights)
    benchmark = BuyAndHold({"VTI": 1.0})
    result = bt.run(strategy, benchmark)

    return {
        "weight_vxus": weight_vxus,
        "intl_cost_bps": intl_cost_bps,
        "n_folds": len(result.fold_results),
        "test_start": result.overall_returns.index[0].date().isoformat(),
        "test_end": result.overall_returns.index[-1].date().isoformat(),
        "n_test_days": len(result.overall_returns),
        "strat_sharpe": result.overall_metrics.sharpe_ratio,
        "bench_sharpe": result.benchmark_metrics.sharpe_ratio,
        "strat_vol_annual": result.overall_metrics.annualized_volatility,
        "bench_vol_annual": result.benchmark_metrics.annualized_volatility,
        "excess_sharpe": result.excess_sharpe,
        "total_turnover": result.total_turnover,
        "total_cost_drag": result.total_cost_drag,
        "_strategy_returns": result.overall_returns,
        "_benchmark_returns": result.benchmark_returns,
    }


def find_vol_minimum(runs_at_strict_cost: dict) -> dict:
    """S1 hypothesis: vol-minimum location among gating weights.

    Reports the weight whose realized annual vol is minimum, and whether
    that location falls in [35%, 55%] (the Vanguard 2021 rev 2019
    forward-VCMM band, primary-source verified).
    """
    table = []
    for run in runs_at_strict_cost.values():
        if "error" in run:
            continue
        table.append((run["weight_vxus"], run["strat_vol_annual"]))
    table.sort()
    weights, vols = zip(*table)
    idx = int(np.argmin(vols))
    min_weight = weights[idx]
    min_vol = vols[idx]
    return {
        "min_weight_vxus": min_weight,
        "min_vol_annual": min_vol,
        "in_vanguard_band_35_55": 0.35 <= min_weight <= 0.55,
        "all_weights": list(weights),
        "all_vols": list(vols),
    }


def main() -> None:
    print("=" * 70)
    print("PHASE 1: STATIC VTI/VXUS ALLOCATION SWEEP")
    print("Weights: 0/10/20/30/40/50/60/100% VXUS, annual rebalance")
    print("=" * 70)

    cfg = load_workflow_config()
    gate = load_gate(cfg)
    print(f"Gate (LOCKED v1.2): ExSharpe > {gate.excess_sharpe_min}, "
          f"Holm p < {gate.holm_p_max}, CI_lower > {gate.ci_lower_min}")
    print(f"Workflow tier: {cfg['gate'].get('workflow_tier', 'unspecified')}")
    print(f"Power at locked gate: {cfg['gate'].get('power_at_locked_gate', 'unknown')}")
    print()

    universe = pd.read_csv(WORKFLOW_DIR / "data" / "reference" / "international_universe.csv")
    universe["inception_date"] = pd.to_datetime(universe["inception_date"])

    prices = load_prices()
    tbill = load_tbill_daily(prices.index)

    # Annual rebalance per plan v1.1 H2 fix
    bt_config = BacktestConfig(
        train_months=cfg["walk_forward"]["train_months"],
        test_months=cfg["walk_forward"]["test_months"],
        step_months=cfg["walk_forward"]["step_months"],
        rebalance_frequency="annual",
        initial_capital=cfg["walk_forward"]["initial_capital"],
    )

    runs = {}
    for weight in WEIGHTS_VXUS:
        for cost_bps in COST_LEVELS_BPS:
            key = f"vxus{int(weight * 100):03d}_intl{int(cost_bps)}bps"
            print(f"\n--- Running {key} ---")
            try:
                run = run_static_strategy(weight, prices, universe, tbill, bt_config, cost_bps)
                inf = compute_inferential_stats(run, n_bootstrap=10_000)
                runs[key] = {**run, **inf}
                print(f"  excess_sharpe={inf['excess_sharpe_point']:+.3f}  "
                      f"raw_p_upper={inf['raw_p_upper']:.4f}  "
                      f"ci=[{inf['sharpe_diff_ci_lower']:+.3f}, "
                      f"{inf['sharpe_diff_ci_upper']:+.3f}]  "
                      f"vol={run['strat_vol_annual']:.3f}")
            except Exception as e:
                logger.exception("Failed %s: %s", key, e)
                runs[key] = {"error": str(e), "weight_vxus": weight, "intl_cost_bps": cost_bps}

    # Holm correction per cost level over GATING weights only
    holm_by_cost = {}
    for cost_bps in COST_LEVELS_BPS:
        per_cost_p = {}
        for weight in GATING_WEIGHTS:
            key = f"vxus{int(weight * 100):03d}_intl{int(cost_bps)}bps"
            run = runs.get(key, {})
            if "raw_p_upper" in run:
                per_cost_p[key] = run["raw_p_upper"]
        holm_by_cost[cost_bps] = holm_bonferroni(per_cost_p)

    # Apply gate
    verdicts = {}
    for weight in GATING_WEIGHTS:
        per_cost_verdicts = {}
        for cost_bps in COST_LEVELS_BPS:
            key = f"vxus{int(weight * 100):03d}_intl{int(cost_bps)}bps"
            run = runs.get(key, {})
            if "error" in run or "raw_p_upper" not in run:
                per_cost_verdicts[cost_bps] = {"passed": False, "reasons": [run.get("error", "no result")]}
                continue
            holm_p = holm_by_cost[cost_bps].get(key, {}).get("adjusted_p", 1.0)
            passed, reasons = evaluate_strict_gate(
                excess_sharpe=run["sharpe_diff_point"],
                holm_p=holm_p,
                ci_lower=run["sharpe_diff_ci_lower"],
                log_excess_mean=run["log_excess_mean_daily"],
                log_excess_ci_lower=run["log_excess_ci_lower"],
                gate=gate,
            )
            per_cost_verdicts[cost_bps] = {
                "passed": passed,
                "reasons": reasons,
                "holm_adjusted_p": holm_p,
                "raw_p_upper": run["raw_p_upper"],
                "sharpe_diff_point": run["sharpe_diff_point"],
                "sharpe_diff_ci": [run["sharpe_diff_ci_lower"], run["sharpe_diff_ci_upper"]],
            }
        proceed = per_cost_verdicts.get(GATING_COST_BPS, {}).get("passed", False)
        verdicts[weight] = {
            "proceed": proceed,
            "by_cost": per_cost_verdicts,
        }

    # S1 vol-minimum analysis at strict cost
    runs_at_strict = {k: v for k, v in runs.items() if v.get("intl_cost_bps") == GATING_COST_BPS}
    s1 = find_vol_minimum(runs_at_strict)

    # Print summary
    print()
    print("=" * 70)
    print("PHASE 1 VERDICTS (strict cost = 10 bps, gating ExSharpe > 0.40)")
    print("=" * 70)
    print(f"S1 (Vanguard variance-min): vol-min at weight={s1['min_weight_vxus']*100:.0f}% VXUS, "
          f"vol={s1['min_vol_annual']:.3f}, "
          f"in 35-55% band: {s1['in_vanguard_band_35_55']}")
    print()
    print(f"{'Weight':>8} {'Verdict':>10} {'ExSharpe':>10} {'Holm p':>10} {'CI_lower':>10} {'CI_upper':>10}")
    for weight in GATING_WEIGHTS:
        v = verdicts[weight]
        c10 = v["by_cost"].get(GATING_COST_BPS, {})
        if "sharpe_diff_point" in c10:
            print(f"{int(weight*100):>7}% {('PROCEED' if v['proceed'] else 'FAIL'):>10} "
                  f"{c10['sharpe_diff_point']:>+10.3f} "
                  f"{c10['holm_adjusted_p']:>10.4f} "
                  f"{c10['sharpe_diff_ci'][0]:>+10.3f} "
                  f"{c10['sharpe_diff_ci'][1]:>+10.3f}")
        else:
            print(f"{int(weight*100):>7}% {'ERROR':>10}")

    # Persist
    artifacts_dir = WORKFLOW_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    runs_serializable = {}
    for key, run in runs.items():
        runs_serializable[key] = {
            k: v for k, v in run.items() if not k.startswith("_")
        }
    holm_serializable = {str(c): h for c, h in holm_by_cost.items()}

    out_path = artifacts_dir / "phase1_results.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "run_date": date.today().isoformat(),
                "gate": {
                    "excess_sharpe_min": gate.excess_sharpe_min,
                    "holm_p_max": gate.holm_p_max,
                    "ci_lower_min": gate.ci_lower_min,
                },
                "workflow_tier": cfg["gate"].get("workflow_tier"),
                "power_at_locked_gate": cfg["gate"].get("power_at_locked_gate"),
                "weights_grid": WEIGHTS_VXUS,
                "gating_weights": GATING_WEIGHTS,
                "cost_levels_bps": COST_LEVELS_BPS,
                "verdicts": {str(k): v for k, v in verdicts.items()},
                "runs": runs_serializable,
                "holm_by_cost": holm_serializable,
                "s1_vol_minimum": s1,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", out_path)

    # Markdown summary
    md = ["# Phase 1 — Static VTI/VXUS Allocation Sweep", ""]
    md.append(f"- Run date: {date.today()}")
    md.append(f"- Gate: ExSharpe > {gate.excess_sharpe_min}, Holm p < {gate.holm_p_max}, CI_lower > 0")
    md.append(f"- Workflow tier: **{cfg['gate'].get('workflow_tier', 'unspecified')}**")
    md.append(f"- Power at locked gate: {cfg['gate'].get('power_at_locked_gate')}")
    md.append("")
    md.append("## S1 — Vanguard variance-minimum hypothesis")
    md.append("")
    md.append(f"- Vol-minimum weight: **{s1['min_weight_vxus']*100:.0f}% VXUS** "
              f"(annual vol = {s1['min_vol_annual']:.3f})")
    md.append(f"- In Vanguard 35-55% band: **{s1['in_vanguard_band_35_55']}**")
    md.append("")
    md.append("| Weight VXUS | Annual vol |")
    md.append("|---|---|")
    for w, v in zip(s1["all_weights"], s1["all_vols"]):
        md.append(f"| {int(w*100)}% | {v:.3f} |")
    md.append("")
    md.append("## Strict-gate verdicts at 10 bps cost")
    md.append("")
    md.append("| Weight | Verdict | ExSharpe | Holm p | CI 90% | Log-excess CI |")
    md.append("|---|---|---|---|---|---|")
    for weight in GATING_WEIGHTS:
        v = verdicts[weight]
        c10 = v["by_cost"].get(GATING_COST_BPS, {})
        if "sharpe_diff_point" in c10:
            run_at_10 = runs.get(f"vxus{int(weight*100):03d}_intl10bps", {})
            log_lo = run_at_10.get("log_excess_ci_lower", float("nan"))
            log_hi = run_at_10.get("log_excess_ci_upper", float("nan"))
            md.append(
                f"| {int(weight*100)}% | {'PROCEED' if v['proceed'] else 'FAIL'} | "
                f"{c10['sharpe_diff_point']:+.3f} | "
                f"{c10['holm_adjusted_p']:.4f} | "
                f"[{c10['sharpe_diff_ci'][0]:+.3f}, {c10['sharpe_diff_ci'][1]:+.3f}] | "
                f"[{log_lo:+.4f}, {log_hi:+.4f}] |"
            )
        else:
            md.append(f"| {int(weight*100)}% | ERROR | — | — | — | — |")
    md.append("")
    md.append("See `artifacts/phase1_results.json` for full per-cost-level breakdown.")

    md_path = WORKFLOW_DIR / "research" / "phase1_results.md"
    md_path.write_text("\n".join(md))
    logger.info("Wrote %s", md_path)
    print(f"\nResults: {out_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
