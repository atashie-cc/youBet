"""Phase 1 fast backtest from the precomputed signal panel.

Reads artifacts/signal_panel.parquet (built by precompute_signals.py) and runs
evebitda_yield (CORE) + earnings_yield_v2 (DESCRIPTIVE) through StockBacktester
via PrecomputedScoreStrategy (score = O(1) lookup), so T+1 / costs / T-bill /
delisting are handled by the proven backtester at full speed. Benchmark SPY.

Usage:
    python -u workflows/individual-stocks-snp500/experiments/phase1_run_from_panel.py
"""

from __future__ import annotations

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
    ARTIFACTS_DIR, evaluate_gate, get_tbill, load_config, load_prices_with_benchmark,
    load_sp500_universe, make_backtest_config, make_cost_model, print_gate_table,
    save_phase_returns,
)
from youbet.stock.backtester import StockBacktester  # noqa: E402
from youbet.stock.fwd_valuation import PrecomputedScoreStrategy  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    import glob
    t0 = time.monotonic()
    cfg = load_config()
    parts = []
    main = ARTIFACTS_DIR / "signal_panel.parquet"
    if main.exists():
        parts.append(pd.read_parquet(main))
    for p in sorted(glob.glob(str(ARTIFACTS_DIR / "signal_panel_shard*.parquet"))):
        parts.append(pd.read_parquet(p))
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.drop_duplicates(subset=["date", "ticker"], keep="first")
    lut_ev = panel.pivot_table(index="date", columns="ticker", values="evebitda_signal")
    lut_ey = panel.pivot_table(index="date", columns="ticker", values="earnings_yield")
    logger.info("Signal panel: %d rows, %d dates %s..%s, %d tickers",
                len(panel), lut_ev.shape[0], lut_ev.index.min().date(),
                lut_ev.index.max().date(), lut_ev.shape[1])

    uni = load_sp500_universe()
    px = load_prices_with_benchmark(uni, start="2005-01-01", config=cfg)
    bt = StockBacktester(config=make_backtest_config(cfg), prices=px, universe=uni,
                         cost_model=make_cost_model(cfg), tbill_rates=get_tbill(px))
    bench = BuyAndHoldETF(cfg["benchmark"]["ticker"])
    strategies = [
        PrecomputedScoreStrategy(lut_ev, "evebitda_yield", min_holdings=20),
        PrecomputedScoreStrategy(lut_ey, "earnings_yield_v2", min_holdings=20),
    ]

    returns, bench_ret = {}, None
    for s in strategies:
        res = bt.run(strategy=s, benchmark=bench)
        logger.info("%s: ExSharpe=%+.3f (strat %.3f / bench %.3f), folds=%d, turnover=%.2f",
                    s.name, res.excess_sharpe, res.overall_metrics.sharpe_ratio,
                    res.benchmark_metrics.sharpe_ratio, len(res.fold_results),
                    res.total_turnover)
        returns[s.name] = res.overall_returns
        bench_ret = res.benchmark_returns if bench_ret is None else bench_ret
        save_phase_returns(f"phase1_{s.name}", {s.name: res.overall_returns},
                           res.benchmark_returns)
    save_phase_returns("phase1", returns, bench_ret)
    results = evaluate_gate(returns, bench_ret)
    print_gate_table(results, "Phase 1 — EV/EBITDA valuation (full S&P 500, from panel). "
                     "earnings_yield_v2 is DESCRIPTIVE (not gate family).")
    logger.info("Phase 1 (from panel) wall time %.1fs", time.monotonic() - t0)


if __name__ == "__main__":
    main()
