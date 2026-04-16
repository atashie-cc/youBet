"""E10 — Pure Antonacci GEM baseline (pipeline smoke test).

Runs the existing DualMomentum scaffold (workflows/etf/strategies/dual_momentum)
via the ETF backtester on the current data vintage. Validates the pipeline
before any other experiment is trusted.

No novel content — this is a clean replication for report isolation.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure repo imports are resolvable
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "workflows" / "etf"))
sys.path.insert(0, str(WORKFLOW_ROOT / "scripts"))

import pandas as pd

from youbet.etf.backtester import Backtester, BacktestConfig
from youbet.etf.benchmark import BuyAndHold
from youbet.etf.costs import CostModel
from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe

from strategies.dual_momentum.scripts.run import DualMomentum  # type: ignore

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


def main():
    cfg = load_workflow_config()
    experiment = "e10_gem_baseline"

    # Load data
    universe = load_universe()
    cost_model = CostModel.from_universe(universe)

    all_tickers = universe["ticker"].tolist()
    for req in ["VTI", "VXUS", "VEA", "VWO", "BND", "BSV", "BIV"]:
        if req not in all_tickers:
            all_tickers.append(req)

    print(f"[{experiment}] Fetching prices...")
    prices = fetch_prices(all_tickers, start=cfg["backtest"]["start_date"])
    print(f"  {len(prices)} days, {len(prices.columns)} tickers")

    tbill = fetch_tbill_rates(
        start=cfg["backtest"]["start_date"], allow_fallback=True
    )

    # Backtest config
    bt_cfg = BacktestConfig(
        train_months=cfg["backtest"]["train_months"],
        test_months=cfg["backtest"]["test_months"],
        step_months=cfg["backtest"]["step_months"],
        rebalance_frequency=cfg["backtest"]["rebalance_frequency"],
        initial_capital=cfg["backtest"]["initial_capital"],
    )

    bt = Backtester(
        config=bt_cfg,
        prices=prices,
        cost_model=cost_model,
        tbill_rates=tbill,
        universe=universe,
    )

    # Strategy + benchmark
    strategy = DualMomentum(
        us_equity="VTI", intl_equity="VXUS", bond="BND", lookback_months=12
    )
    benchmark = BuyAndHold({"VTI": 1.0})

    print(f"[{experiment}] Running backtest...")
    result = bt.run(strategy, benchmark)
    print(result.summary())
    print()

    # Metrics and CI
    strat_ret = result.overall_returns
    bench_ret = result.benchmark_returns

    strat_metrics = compute_metrics(strat_ret, "dual_momentum")
    bench_metrics = compute_metrics(bench_ret, "VTI")

    ci = bootstrap_excess_sharpe(
        strat_ret,
        bench_ret,
        n_bootstrap=cfg["bootstrap"]["n_replicates"],
        confidence=cfg["bootstrap"]["confidence"],
        block_length=cfg["bootstrap"]["block_length"],
    )

    sub = subperiod_consistency(
        strat_ret, bench_ret, cfg["subperiods"]
    )

    elevation_pass, elevation_reasons = check_elevation(
        excess_sharpe_point=ci["excess_sharpe_point"],
        ci_lower=ci["excess_sharpe_lower"],
        subperiod_same_sign=sub["same_sign_positive_excess_sharpe"],
        sharpe_diff_point=ci["point_estimate"],
        threshold_excess_sharpe=cfg["exploratory_gate"]["elevation_excess_sharpe"],
    )

    out = {
        "experiment": experiment,
        "description": "Pure Antonacci GEM replication (pipeline smoke test)",
        "strategy": strategy.params,
        "comparisons": {
            "VTI_buy_and_hold": {
                "strategy_metrics": strat_metrics,
                "benchmark_metrics": bench_metrics,
                "excess_sharpe_ci": ci,
            },
        },
        "subperiods": sub,
        "elevation": {"passed": elevation_pass, "reasons": elevation_reasons},
        "locked_benchmarks": cfg["benchmarks"]["primary"],
        "n_folds": len(result.fold_results),
        "period": {
            "start": str(strat_ret.index[0].date()),
            "end": str(strat_ret.index[-1].date()),
        },
    }

    path = save_result(experiment, out)
    print(format_report(experiment, out))
    print(f"\nSaved: {path}")

    return out


if __name__ == "__main__":
    main()
