"""Phase 2 — Regime-conditional gates.

7 gating tests (M5 dropped due to no intl CAPE data; see plan v1.3 ledger):
  M1a: DXY 12m < 0  → 20% VEA, else 100% VTI
  M1b: DXY 12m < 0  → 40% VEA, else 100% VTI
  M1c: DXY 12m < 0  → 60% VEA, else 100% VTI
  M3:  US-Bund yield-diff narrowing → 40% VEA, else 100% VTI
  M4a: DBC 12m > 0 → 20% VWO, else 100% VTI
  M4b: DBC 12m > 0 → 40% VWO, else 100% VTI
  M6:  ex-US 36m relative return < 0 → 40% VEA, else 100% VTI

Strict gate (LOCKED v1.3 after Phase -1 power analysis):
  PASS iff (excess_sharpe_point > 0.40)
        AND (Holm-adjusted p < 0.05)
        AND (CI lower > 0)
        AND (mean log-excess > 0 with CI lower > 0)
        AND (passes at cost = 10 bps)

Workflow tier: DESCRIPTIVE/EXPLORATORY (power 0.30 at locked gate).
No PROCEED claim is confirmatory; results are point estimates + CIs.

Inputs read:
  - data/snapshots/<latest>/prices.parquet
  - data/snapshots/<latest>/macro/{dxy,dgs10,bund10y,dbc,tb3ms}.csv
  - data/reference/international_universe.csv
  - config.yaml

Outputs:
  - artifacts/phase2_results.json
  - research/phase2_results.md
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

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
    evaluate_strict_gate,
    load_gate,
    load_prices,
    load_tbill_daily,
    load_workflow_config,
)
from strategies.regime_conditional import RegimeConditionalStrategy     # noqa: E402
from strategies.regime_signals import (                                 # noqa: E402
    dbc_12m_positive_signal,
    dxy_12m_negative_signal,
    ex_us_36m_negative_signal,
    us_bund_yield_narrowing_signal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


COST_LEVELS_BPS = [3.0, 5.0, 10.0]
GATING_COST_BPS = 10.0


def build_strategy_specs() -> list[dict]:
    """Pre-committed Phase 2 strategy list. Order frozen at workflow inception."""
    dxy_sig = dxy_12m_negative_signal()
    bund_sig = us_bund_yield_narrowing_signal()
    dbc_sig = dbc_12m_positive_signal()
    rev_sig = ex_us_36m_negative_signal("VEA", "VTI")

    return [
        {"id": "M1a_dxy_VEA20", "signal": dxy_sig, "ex_us": "VEA", "weight": 0.20, "tickers": ["VTI", "VEA"]},
        {"id": "M1b_dxy_VEA40", "signal": dxy_sig, "ex_us": "VEA", "weight": 0.40, "tickers": ["VTI", "VEA"]},
        {"id": "M1c_dxy_VEA60", "signal": dxy_sig, "ex_us": "VEA", "weight": 0.60, "tickers": ["VTI", "VEA"]},
        {"id": "M3_bund_VEA40", "signal": bund_sig, "ex_us": "VEA", "weight": 0.40, "tickers": ["VTI", "VEA"]},
        {"id": "M4a_dbc_VWO20", "signal": dbc_sig, "ex_us": "VWO", "weight": 0.20, "tickers": ["VTI", "VWO"]},
        {"id": "M4b_dbc_VWO40", "signal": dbc_sig, "ex_us": "VWO", "weight": 0.40, "tickers": ["VTI", "VWO"]},
        {"id": "M6_rev_VEA40", "signal": rev_sig, "ex_us": "VEA", "weight": 0.40, "tickers": ["VTI", "VEA"]},
    ]


def run_one_regime_strategy(
    spec: dict,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
    intl_cost_bps: float,
) -> dict:
    """Run one regime-conditional strategy vs 100% VTI buy-and-hold."""
    cost_model = build_cost_model(universe, intl_cost_bps)

    sub = prices[spec["tickers"]].dropna(how="any")
    sub = sub.loc[sub.index >= "2001-08-14"]
    sub = sub.loc[sub.index <= "2026-04-30"]

    bt = Backtester(
        config=bt_config,
        prices=sub,
        cost_model=cost_model,
        tbill_rates=tbill,
        universe=universe,
    )
    strategy = RegimeConditionalStrategy(
        signal=spec["signal"],
        ex_us_ticker=spec["ex_us"],
        weight_when_on=spec["weight"],
        benchmark_ticker="VTI",
        name=spec["id"],
    )
    benchmark = BuyAndHold({"VTI": 1.0})
    result = bt.run(strategy, benchmark)

    # Estimate fraction of test days the signal was "on"
    test_dates = result.overall_returns.index
    sig = spec["signal"]
    sig_on_count = 0
    for t in test_dates:
        from strategies.regime_signals import signal_at_date
        if signal_at_date(sig, t):
            sig_on_count += 1
    sig_on_frac = sig_on_count / len(test_dates) if len(test_dates) > 0 else 0.0

    return {
        "id": spec["id"],
        "signal_name": spec["signal"].name,
        "ex_us": spec["ex_us"],
        "weight_when_on": spec["weight"],
        "intl_cost_bps": intl_cost_bps,
        "n_folds": len(result.fold_results),
        "test_start": result.overall_returns.index[0].date().isoformat(),
        "test_end": result.overall_returns.index[-1].date().isoformat(),
        "n_test_days": len(result.overall_returns),
        "signal_on_fraction": sig_on_frac,
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


def main() -> None:
    print("=" * 70)
    print("PHASE 2: REGIME-CONDITIONAL GATES")
    print("7 strategies: M1×3 (DXY) + M3 (yield-diff) + M4×2 (DBC) + M6 (mean-rev)")
    print("=" * 70)

    cfg = load_workflow_config()
    gate = load_gate(cfg)
    print(f"Gate (LOCKED v1.3): ExSharpe > {gate.excess_sharpe_min}, "
          f"Holm p < {gate.holm_p_max}, CI_lower > {gate.ci_lower_min}")
    print(f"Workflow tier: {cfg['gate'].get('workflow_tier')}")
    print(f"Power at locked gate: {cfg['gate'].get('power_at_locked_gate')}")
    print(f"Holm denominator (workflow-wide): {cfg['bootstrap']['holm_denominator']}")
    print()

    universe = pd.read_csv(WORKFLOW_DIR / "data" / "reference" / "international_universe.csv")
    universe["inception_date"] = pd.to_datetime(universe["inception_date"])

    prices = load_prices()
    tbill = load_tbill_daily(prices.index)

    bt_config = BacktestConfig(
        train_months=cfg["walk_forward"]["train_months"],
        test_months=cfg["walk_forward"]["test_months"],
        step_months=cfg["walk_forward"]["step_months"],
        rebalance_frequency="monthly",
        initial_capital=cfg["walk_forward"]["initial_capital"],
    )

    specs = build_strategy_specs()
    print(f"Strategies: {[s['id'] for s in specs]}\n")

    runs = {}
    for spec in specs:
        for cost_bps in COST_LEVELS_BPS:
            key = f"{spec['id']}_intl{int(cost_bps)}bps"
            print(f"--- Running {key} ---")
            try:
                run = run_one_regime_strategy(
                    spec, prices, universe, tbill, bt_config, cost_bps
                )
                inf = compute_inferential_stats(run, n_bootstrap=10_000)
                runs[key] = {**run, **inf}
                print(f"  excess_sharpe={inf['excess_sharpe_point']:+.3f}  "
                      f"sharpe_diff={inf['sharpe_diff_point']:+.3f}  "
                      f"raw_p={inf['raw_p_upper']:.4f}  "
                      f"ci=[{inf['sharpe_diff_ci_lower']:+.3f}, "
                      f"{inf['sharpe_diff_ci_upper']:+.3f}]  "
                      f"sig_on={run['signal_on_fraction']:.2f}")
            except Exception as e:
                logger.exception("Failed %s", key)
                runs[key] = {"error": str(e), "id": spec["id"], "intl_cost_bps": cost_bps}

    # Holm correction within Phase 2 per cost level
    holm_by_cost = {}
    for cost_bps in COST_LEVELS_BPS:
        per_cost_p = {}
        for spec in specs:
            key = f"{spec['id']}_intl{int(cost_bps)}bps"
            run = runs.get(key, {})
            if "raw_p_upper" in run:
                per_cost_p[key] = run["raw_p_upper"]
        holm_by_cost[cost_bps] = holm_bonferroni(per_cost_p)

    # Verdicts
    verdicts = {}
    for spec in specs:
        per_cost_verdicts = {}
        for cost_bps in COST_LEVELS_BPS:
            key = f"{spec['id']}_intl{int(cost_bps)}bps"
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
        verdicts[spec["id"]] = {
            "proceed": proceed,
            "by_cost": per_cost_verdicts,
        }

    # Print summary
    print()
    print("=" * 70)
    print(f"PHASE 2 VERDICTS (strict cost = {GATING_COST_BPS} bps, gate ExSharpe > {gate.excess_sharpe_min})")
    print("=" * 70)
    print(f"{'Strategy':<22} {'Verdict':>8} {'SharpeDiff':>12} {'Holm p':>10} "
          f"{'CI_lower':>10} {'CI_upper':>10} {'sig_on':>8}")
    for spec in specs:
        v = verdicts[spec["id"]]
        c10 = v["by_cost"].get(GATING_COST_BPS, {})
        run10 = runs.get(f"{spec['id']}_intl{int(GATING_COST_BPS)}bps", {})
        if "sharpe_diff_point" in c10:
            print(f"{spec['id']:<22} {('PROCEED' if v['proceed'] else 'FAIL'):>8} "
                  f"{c10['sharpe_diff_point']:>+12.3f} "
                  f"{c10['holm_adjusted_p']:>10.4f} "
                  f"{c10['sharpe_diff_ci'][0]:>+10.3f} "
                  f"{c10['sharpe_diff_ci'][1]:>+10.3f} "
                  f"{run10.get('signal_on_fraction', float('nan')):>8.2f}")
        else:
            print(f"{spec['id']:<22} {'ERROR':>8}")

    # Persist
    artifacts_dir = WORKFLOW_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    runs_serializable = {}
    for key, run in runs.items():
        runs_serializable[key] = {
            k: v for k, v in run.items() if not k.startswith("_")
        }
    holm_serializable = {str(c): h for c, h in holm_by_cost.items()}

    out_path = artifacts_dir / "phase2_results.json"
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
                "holm_denominator_workflow": cfg["bootstrap"]["holm_denominator"],
                "phase2_strategies": [s["id"] for s in specs],
                "cost_levels_bps": COST_LEVELS_BPS,
                "verdicts": verdicts,
                "runs": runs_serializable,
                "holm_by_cost": holm_serializable,
                "m5_dropped_reason": "intl CAPE data unavailable (Siblis/MSCI/Barclays paywalled or non-bulk)",
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", out_path)

    # Markdown summary
    md = ["# Phase 2 — Regime-Conditional Gates Results", ""]
    md.append(f"- Run date: {date.today()}")
    md.append(f"- Gate: ExSharpe > {gate.excess_sharpe_min}, Holm p < {gate.holm_p_max}, CI_lower > 0, log-excess CI_lower > 0")
    md.append(f"- Workflow tier: **{cfg['gate'].get('workflow_tier')}** (power {cfg['gate'].get('power_at_locked_gate')} at gate)")
    md.append(f"- Holm denominator (workflow-wide): {cfg['bootstrap']['holm_denominator']}")
    md.append(f"- M5 dropped (no intl CAPE data); within-Phase-2 Holm: 7 strategies per cost level")
    md.append("")
    md.append("## Verdicts at strict 10 bps cost")
    md.append("")
    md.append("| Strategy | Verdict | SharpeDiff | Holm p | 90% CI | sig_on frac | Log-excess daily |")
    md.append("|---|---|---|---|---|---|---|")
    for spec in specs:
        v = verdicts[spec["id"]]
        c10 = v["by_cost"].get(GATING_COST_BPS, {})
        run10 = runs.get(f"{spec['id']}_intl{int(GATING_COST_BPS)}bps", {})
        if "sharpe_diff_point" in c10:
            log_mean = run10.get("log_excess_mean_daily", float("nan"))
            md.append(
                f"| {spec['id']} | "
                f"{'PROCEED' if v['proceed'] else 'FAIL'} | "
                f"{c10['sharpe_diff_point']:+.3f} | "
                f"{c10['holm_adjusted_p']:.4f} | "
                f"[{c10['sharpe_diff_ci'][0]:+.3f}, {c10['sharpe_diff_ci'][1]:+.3f}] | "
                f"{run10.get('signal_on_fraction', float('nan')):.2f} | "
                f"{log_mean:+.6f} |"
            )
        else:
            md.append(f"| {spec['id']} | ERROR | — | — | — | — | — |")
    md.append("")
    md.append("See `artifacts/phase2_results.json` for full per-cost breakdown.")

    md_path = WORKFLOW_DIR / "research" / "phase2_results.md"
    md_path.write_text("\n".join(md))
    logger.info("Wrote %s", md_path)
    print(f"\nResults: {out_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
