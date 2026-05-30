"""Phase 1 — PIT enterprise-multiple (EV/EBITDA) valuation on real S&P 500 data.

Core (gate-tested): evebitda_yield — the enterprise-multiple construction the
stock-selection workflow did NOT run. Descriptive companion: earnings_yield_v2
(~95% clone of the prior value_EY; point estimate + CI only, NOT in the gate
family). Both top-decile, equal-weight, monthly, T+1, costs on, benchmark SPY.

Pre-registered expectation (Phase 0 MDE > +0.50): FAILS gate.

Usage:
    python workflows/individual-stocks-snp500/experiments/phase1_forward_valuation.py
    python ... phase1_forward_valuation.py --limit-tickers 60   # smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (  # noqa: E402
    clear_fundamentals_cache, evaluate_gate, get_tbill, load_config,
    load_facts_by_ticker, load_prices_with_benchmark, load_sp500_universe,
    make_backtest_config, make_cost_model, print_gate_table, save_phase_returns,
)
from youbet.stock.backtester import StockBacktester  # noqa: E402
from youbet.stock.fwd_valuation import EVEBITDAYield, EarningsYieldV2  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402
from youbet.stock.universe import Universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def run_phase1(limit_tickers: int | None = None, price_start: str = "2005-01-01"):
    config = load_config()
    clear_fundamentals_cache()
    universe = load_sp500_universe()
    if limit_tickers:
        keep = list(universe.active_as_of("2026-04-19"))[:limit_tickers]
        sub = universe.membership[universe.membership["ticker"].isin(keep)].copy()
        universe = Universe(membership=sub, delistings=universe.delistings,
                            index_name=f"S&P 500 (limit {limit_tickers})")
        logger.warning("LIMITED universe to %d tickers (smoke)", limit_tickers)

    t0 = time.monotonic()
    prices = load_prices_with_benchmark(universe, start=price_start, config=config)
    logger.info("Prices %s in %.1fs", prices.shape, time.monotonic() - t0)

    t0 = time.monotonic()
    facts = load_facts_by_ticker(universe)
    active_ever = set(universe.all_tickers_ever())
    facts = {t: f for t, f in facts.items()
             if t in active_ever and t in prices.columns}
    logger.info("Facts-by-ticker: %d in %.1fs", len(facts), time.monotonic() - t0)

    bt = StockBacktester(config=make_backtest_config(config), prices=prices,
                         universe=universe, cost_model=make_cost_model(config),
                         tbill_rates=get_tbill(prices), facts_by_ticker=facts)
    min_hold = 20 if limit_tickers is None else max(5, limit_tickers // 10)
    strategies = [EVEBITDAYield(use_ebit_fallback=True), EarningsYieldV2()]
    for s in strategies:
        s.min_holdings = min_hold
    benchmark = BuyAndHoldETF(config["benchmark"]["ticker"])

    returns, bench_ret = {}, None
    for strat in strategies:
        logger.info("=== %s ===", strat.name)
        t0 = time.monotonic()
        res = bt.run(strategy=strat, benchmark=benchmark)
        logger.info("%s: ExSharpe=%+.3f (strat %.3f / bench %.3f), folds=%d, %.1fs",
                    strat.name, res.excess_sharpe,
                    res.overall_metrics.sharpe_ratio,
                    res.benchmark_metrics.sharpe_ratio,
                    len(res.fold_results), time.monotonic() - t0)
        returns[strat.name] = res.overall_returns
        bench_ret = res.benchmark_returns if bench_ret is None else bench_ret
        # Incremental persist: each strategy's returns survive even if a
        # later strategy is killed by the 10-min tool ceiling.
        save_phase_returns(f"phase1_{strat.name}", {strat.name: res.overall_returns},
                           res.benchmark_returns)
    return returns, bench_ret


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-tickers", type=int, default=None)
    ap.add_argument("--price-start", default="2005-01-01")
    args = ap.parse_args()

    t0 = time.monotonic()
    returns, bench = run_phase1(args.limit_tickers, args.price_start)
    name = "phase1_smoke" if args.limit_tickers else "phase1"
    save_phase_returns(name, returns, bench)
    results = evaluate_gate(returns, bench)
    print_gate_table(results, f"Phase 1 — EV/EBITDA valuation "
                     f"({'smoke ' + str(args.limit_tickers) if args.limit_tickers else 'full S&P 500'}). "
                     f"earnings_yield_v2 is DESCRIPTIVE (not gate family).")
    logger.info("Phase 1 wall time %.1fs", time.monotonic() - t0)


if __name__ == "__main__":
    main()
