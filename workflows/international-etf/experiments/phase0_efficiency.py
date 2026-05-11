"""Phase 0 — Single-ETF efficiency baseline.

Question: Does any individual ex-US ETF (VXUS / VEA / VWO / EFA) at 100%
weight beat 100% VTI buy-and-hold on walk-forward Sharpe?

Pre-committed gate (LOCKED v1.1; ExSharpe threshold subject to Phase -1
power-analysis revision via config.yaml):
  PASS iff (excess_sharpe_point > gate_min)
        AND (Holm-adjusted p < 0.05)
        AND (CI lower > 0)
        AND (mean log-excess > 0 with CI lower > 0)
        AND (passes at cost = 10 bps for broad_intl_equity)

Holm denominator across ALL Phase 0-2 gating tests = 19 (per plan v1.1).
This script runs only the 4 Phase-0 tests; the Holm correction across
the larger family is applied in a separate aggregation step after Phase 2.

Inputs read:
  - data/snapshots/<latest>/prices.parquet            (from phase_minus_1_fetch_and_hash.py)
  - data/snapshots/<latest>/macro/tb3ms.csv          (cash rate)
  - data/reference/international_universe.csv         (PIT survivorship + costs)
  - config.yaml                                       (gate thresholds, walk-forward params)

Outputs:
  - artifacts/phase0_results.json
  - research/phase0_results.md
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.backtester import Backtester, BacktestConfig            # noqa: E402
from youbet.etf.benchmark import BuyAndHold                              # noqa: E402
from youbet.etf.costs import COST_SCHEDULE, CostModel                    # noqa: E402
from youbet.etf.stats import (                                           # noqa: E402
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


CANDIDATES = ["VXUS", "VEA", "VWO", "EFA"]
COST_LEVELS_BPS = [3.0, 5.0, 10.0]  # one-way; gate requires PASS at 10 bps


@dataclass(frozen=True)
class GateConfig:
    """Strict gate parameters loaded from workflow config.yaml."""

    excess_sharpe_min: float
    holm_p_max: float
    ci_lower_min: float
    log_excess_min: float
    log_excess_ci_lower_min: float


def load_workflow_config() -> dict:
    cfg_path = WORKFLOW_DIR / "config.yaml"
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def load_gate(cfg: dict) -> GateConfig:
    g = cfg["gate"]
    return GateConfig(
        excess_sharpe_min=float(g["excess_sharpe_min"]),
        holm_p_max=float(g["holm_p_max"]),
        ci_lower_min=float(g["ci_lower_min"]),
        log_excess_min=float(g["log_excess_min"]),
        log_excess_ci_lower_min=float(g["log_excess_ci_lower_min"]),
    )


def load_prices() -> pd.DataFrame:
    snap_root = WORKFLOW_DIR / "data" / "snapshots"
    snaps = sorted([d for d in snap_root.iterdir() if d.is_dir()], reverse=True)
    if not snaps:
        raise FileNotFoundError(f"No snapshot in {snap_root}")
    path = snaps[0] / "prices.parquet"
    logger.info("Loading prices from %s", path)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def load_tbill_daily(prices_index: pd.DatetimeIndex) -> pd.Series:
    """Load TB3MS from snapshot, convert annualized % to daily decimal."""
    snap_root = WORKFLOW_DIR / "data" / "snapshots"
    snaps = sorted([d for d in snap_root.iterdir() if d.is_dir()], reverse=True)
    path = snaps[0] / "macro" / "tb3ms.csv"
    if not path.exists():
        logger.warning("No TB3MS snapshot — falling back to flat 2%/yr")
        return pd.Series(0.02 / 252, index=prices_index)
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    series = df.iloc[:, 0] / 100.0   # FRED reports as percent
    series = series.reindex(prices_index, method="ffill").fillna(0.02)
    return series


def build_cost_model(universe: pd.DataFrame, intl_bps_one_way: float) -> CostModel:
    """CostModel with broad_intl_equity overridden to a target one-way bps."""

    class OverrideCostModel(CostModel):
        def trade_cost_bps(self, ticker: str) -> float:
            category = self.ticker_categories.get(ticker, "default")
            if category == "broad_intl_equity":
                return float(intl_bps_one_way)
            schedule = COST_SCHEDULE.get(category, COST_SCHEDULE["default"])
            return schedule["bid_ask_bps"] + schedule["slippage_bps"]

    return OverrideCostModel.from_universe(universe)


def evaluate_strict_gate(
    excess_sharpe: float,
    holm_p: float,
    ci_lower: float,
    log_excess_mean: float,
    log_excess_ci_lower: float,
    gate: GateConfig,
) -> tuple[bool, list[str]]:
    """Apply locked strict gate; return (passed, reasons)."""
    checks = [
        ("ExSharpe > %.2f" % gate.excess_sharpe_min,
         excess_sharpe > gate.excess_sharpe_min,
         f"point={excess_sharpe:+.3f}"),
        ("Holm p < %.2f" % gate.holm_p_max,
         holm_p < gate.holm_p_max,
         f"holm_p={holm_p:.4f}"),
        ("CI_lower > %.2f" % gate.ci_lower_min,
         ci_lower > gate.ci_lower_min,
         f"ci_lower={ci_lower:+.3f}"),
        ("LogExcess > %.2f" % gate.log_excess_min,
         log_excess_mean > gate.log_excess_min,
         f"log_mean={log_excess_mean:+.4f}"),
        ("LogExcess CI_lower > %.2f" % gate.log_excess_ci_lower_min,
         log_excess_ci_lower > gate.log_excess_ci_lower_min,
         f"log_ci_lower={log_excess_ci_lower:+.4f}"),
    ]
    reasons = [f"{name}: {'PASS' if ok else 'FAIL'} ({val})" for name, ok, val in checks]
    return all(ok for _, ok, _ in checks), reasons


def log_excess_stats(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10_000,
    block_length: int = 22,
    seed: int = 42,
) -> dict:
    """Mean and 90% CI on log-excess return.

    log_excess = log(1 + r_strat) - log(1 + r_bench), summed over the test
    window. Bootstrap CI uses the same stationary block as the Sharpe test.
    """
    import numpy as np

    common = strategy_returns.index.intersection(benchmark_returns.index)
    log_strat = np.log1p(strategy_returns[common].values)
    log_bench = np.log1p(benchmark_returns[common].values)
    excess = log_strat - log_bench
    point_total = float(excess.sum())
    point_mean_daily = float(excess.mean())

    rng = np.random.default_rng(seed)
    n = len(excess)
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
    boot_excess = excess[indices]
    boot_means = boot_excess.mean(axis=1)
    ci_lo = float(np.percentile(boot_means, 5))
    ci_hi = float(np.percentile(boot_means, 95))

    return {
        "log_excess_total": point_total,
        "log_excess_mean_daily": point_mean_daily,
        "log_excess_ci_lower": ci_lo,
        "log_excess_ci_upper": ci_hi,
    }


def run_one_strategy(
    ticker: str,
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
    intl_cost_bps: float,
) -> dict:
    """Run 100%-{ticker} buy-and-hold vs 100% VTI buy-and-hold."""
    cost_model = build_cost_model(universe, intl_cost_bps)

    # Trim to dates where both ticker and VTI are valid
    common_cols = ["VTI", ticker]
    sub = prices[common_cols].dropna(how="any")
    sub = sub.loc[sub.index >= "2001-08-14"]    # locked in-sample start
    sub = sub.loc[sub.index <= "2026-04-30"]

    bt = Backtester(
        config=bt_config,
        prices=sub,
        cost_model=cost_model,
        tbill_rates=tbill,
        universe=universe,
    )
    strategy = BuyAndHold({ticker: 1.0})
    benchmark = BuyAndHold({"VTI": 1.0})
    result = bt.run(strategy, benchmark)

    return {
        "ticker": ticker,
        "intl_cost_bps": intl_cost_bps,
        "n_folds": len(result.fold_results),
        "test_start": result.overall_returns.index[0].date().isoformat(),
        "test_end": result.overall_returns.index[-1].date().isoformat(),
        "n_test_days": len(result.overall_returns),
        "strat_sharpe": result.overall_metrics.sharpe_ratio,
        "bench_sharpe": result.benchmark_metrics.sharpe_ratio,
        "excess_sharpe": result.excess_sharpe,
        "total_turnover": result.total_turnover,
        "total_cost_drag": result.total_cost_drag,
        "_strategy_returns": result.overall_returns,    # private, dropped before serialization
        "_benchmark_returns": result.benchmark_returns,
    }


def compute_inferential_stats(run: dict, n_bootstrap: int = 10_000) -> dict:
    """Compute upper-tail p, paired Sharpe-diff CI, and log-excess stats."""
    strat = run["_strategy_returns"]
    bench = run["_benchmark_returns"]
    test = block_bootstrap_test(
        strat, bench, n_bootstrap=n_bootstrap, expected_block_length=22, seed=42
    )
    ci = excess_sharpe_ci(
        strat, bench, n_bootstrap=n_bootstrap, confidence=0.90,
        expected_block_length=22, seed=42,
    )
    log_stats = log_excess_stats(strat, bench, n_bootstrap=n_bootstrap)
    return {
        "raw_p_upper": test["p_value_upper"],
        "raw_p_lower": test["p_value_lower"],
        "raw_p_two_sided": test["p_value_two_sided"],
        "sharpe_diff_point": ci["point_estimate"],
        "sharpe_diff_ci_lower": ci["ci_lower"],
        "sharpe_diff_ci_upper": ci["ci_upper"],
        "excess_sharpe_point": ci["excess_sharpe_point"],
        "excess_sharpe_ci_lower": ci["excess_sharpe_lower"],
        "excess_sharpe_ci_upper": ci["excess_sharpe_upper"],
        **log_stats,
    }


def main() -> None:
    print("=" * 70)
    print("PHASE 0: SINGLE-ETF EFFICIENCY BASELINE")
    print("Does any 100% ex-US ETF beat 100% VTI buy-and-hold?")
    print("=" * 70)

    cfg = load_workflow_config()
    gate = load_gate(cfg)
    print(f"Gate (LOCKED): ExSharpe > {gate.excess_sharpe_min}, Holm p < {gate.holm_p_max}, "
          f"CI_lower > {gate.ci_lower_min}, log_excess > {gate.log_excess_min}, "
          f"log_ci_lower > {gate.log_excess_ci_lower_min}")
    print(f"Holm denominator (workflow-wide): {cfg['bootstrap']['holm_denominator']}")
    print(f"Cost levels (one-way bps): {COST_LEVELS_BPS}; PASS at {cfg['costs']['pass_at_strict_cost_bps']} bps required.")
    print()

    universe = pd.read_csv(WORKFLOW_DIR / "data" / "reference" / "international_universe.csv")
    universe["inception_date"] = pd.to_datetime(universe["inception_date"])
    print(f"Universe: {len(universe)} tickers")

    prices = load_prices()
    tbill = load_tbill_daily(prices.index)
    print(f"Prices: {prices.shape[0]} rows, {prices.shape[1]} columns, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")

    bt_config = BacktestConfig(
        train_months=cfg["walk_forward"]["train_months"],
        test_months=cfg["walk_forward"]["test_months"],
        step_months=cfg["walk_forward"]["step_months"],
        rebalance_frequency="monthly",
        initial_capital=cfg["walk_forward"]["initial_capital"],
    )

    # Run each candidate at each cost level
    runs = {}
    for ticker in CANDIDATES:
        for cost_bps in COST_LEVELS_BPS:
            key = f"{ticker}_intl{int(cost_bps)}bps"
            print(f"\n--- Running {key} ---")
            try:
                run = run_one_strategy(ticker, prices, universe, tbill, bt_config, cost_bps)
                inf = compute_inferential_stats(run, n_bootstrap=10_000)
                runs[key] = {**run, **inf}
                print(f"  excess_sharpe={inf['excess_sharpe_point']:+.3f}  "
                      f"raw_p_upper={inf['raw_p_upper']:.4f}  "
                      f"ci=[{inf['sharpe_diff_ci_lower']:+.3f}, {inf['sharpe_diff_ci_upper']:+.3f}]")
            except Exception as e:
                logger.exception("Failed to run %s: %s", key, e)
                runs[key] = {"error": str(e), "ticker": ticker, "intl_cost_bps": cost_bps}

    # Holm correction across the 4 candidates at each cost level separately
    # (the workflow-level Holm correction across all 19 tests is applied later
    # in a Phase 2 aggregation; this per-cost Holm is the within-Phase-0 view).
    holm_by_cost = {}
    for cost_bps in COST_LEVELS_BPS:
        per_cost_p = {
            t: runs[f"{t}_intl{int(cost_bps)}bps"]["raw_p_upper"]
            for t in CANDIDATES if "raw_p_upper" in runs[f"{t}_intl{int(cost_bps)}bps"]
        }
        holm_by_cost[cost_bps] = holm_bonferroni(per_cost_p)

    # Apply strict gate at each cost level; PROCEED requires pass at 10 bps
    verdicts = {}
    for ticker in CANDIDATES:
        per_cost_verdicts = {}
        for cost_bps in COST_LEVELS_BPS:
            key = f"{ticker}_intl{int(cost_bps)}bps"
            run = runs[key]
            if "error" in run:
                per_cost_verdicts[cost_bps] = {"passed": False, "reasons": [run["error"]]}
                continue
            holm_p = holm_by_cost[cost_bps][ticker]["adjusted_p"]
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
        # PROCEED iff pass at 10 bps (the strictest)
        proceed = per_cost_verdicts.get(10.0, {}).get("passed", False)
        verdicts[ticker] = {
            "proceed": proceed,
            "by_cost": per_cost_verdicts,
        }

    # Print summary
    print()
    print("=" * 70)
    print("PHASE 0 VERDICTS")
    print("=" * 70)
    for ticker, v in verdicts.items():
        status = "PROCEED" if v["proceed"] else "FAIL"
        print(f"\n[{status}] {ticker}")
        for cost_bps, c in v["by_cost"].items():
            tag = "PASS" if c["passed"] else "FAIL"
            print(f"  {cost_bps:.0f} bps: {tag}  "
                  f"holm_p={c.get('holm_adjusted_p', 'NA')} "
                  f"diff={c.get('sharpe_diff_point', 'NA')}")
            for r in c["reasons"]:
                print(f"     - {r}")

    # Persist
    artifacts_dir = WORKFLOW_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Strip non-serializable pandas objects
    runs_serializable = {}
    for key, run in runs.items():
        runs_serializable[key] = {
            k: v for k, v in run.items() if not k.startswith("_")
        }

    holm_serializable = {str(c): h for c, h in holm_by_cost.items()}

    out_path = artifacts_dir / "phase0_results.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "run_date": date.today().isoformat(),
                "gate": {
                    "excess_sharpe_min": gate.excess_sharpe_min,
                    "holm_p_max": gate.holm_p_max,
                    "ci_lower_min": gate.ci_lower_min,
                    "log_excess_min": gate.log_excess_min,
                    "log_excess_ci_lower_min": gate.log_excess_ci_lower_min,
                },
                "cost_levels_bps": COST_LEVELS_BPS,
                "pass_at_strict_cost_bps": cfg["costs"]["pass_at_strict_cost_bps"],
                "candidates": CANDIDATES,
                "verdicts": verdicts,
                "runs": runs_serializable,
                "holm_by_cost": holm_serializable,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", out_path)

    # Markdown summary
    md_path = WORKFLOW_DIR / "research" / "phase0_results.md"
    md = ["# Phase 0 — Single-ETF Efficiency Test Results", ""]
    md.append(f"- Run date: {date.today()}")
    md.append(f"- Gate: ExSharpe > {gate.excess_sharpe_min}, "
              f"Holm p < {gate.holm_p_max}, CI_lower > {gate.ci_lower_min}, "
              f"log_excess > {gate.log_excess_min}")
    md.append(f"- Holm denominator (workflow-wide): {cfg['bootstrap']['holm_denominator']}")
    md.append(f"- Within-phase Holm: 4 candidates per cost level, applied separately")
    md.append("")
    md.append("## Verdicts at strict 10 bps cost gate")
    md.append("")
    md.append("| Ticker | Verdict | ExSharpe | Holm p | CI 90% | Log-excess CI |")
    md.append("|---|---|---|---|---|---|")
    for ticker in CANDIDATES:
        v = verdicts[ticker]
        c10 = v["by_cost"].get(10.0, {})
        if "sharpe_diff_point" in c10:
            md.append(
                f"| {ticker} | {'PROCEED' if v['proceed'] else 'FAIL'} | "
                f"{c10['sharpe_diff_point']:+.3f} | "
                f"{c10.get('holm_adjusted_p', float('nan')):.4f} | "
                f"[{c10['sharpe_diff_ci'][0]:+.3f}, {c10['sharpe_diff_ci'][1]:+.3f}] | "
                f"see JSON |"
            )
        else:
            md.append(f"| {ticker} | ERROR | — | — | — | — |")
    md.append("")
    md.append(f"See `artifacts/phase0_results.json` for full per-cost-level breakdown.")

    md_path.write_text("\n".join(md))
    logger.info("Wrote %s", md_path)
    print(f"\nResults: {out_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    main()
