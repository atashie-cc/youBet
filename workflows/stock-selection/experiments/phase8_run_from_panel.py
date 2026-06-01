"""Phase 8 fast backtest from the precomputed quality-composite panel.

Reads artifacts/phase8_panel.parquet (built by phase8_precompute.py) and runs
quality_composite_v1 through StockBacktester via PrecomputedScoreStrategy (O(1)
lookup), so T+1/costs/T-bill/delisting are handled at full speed. EXPLORATORY per
precommit/phase8_quality.json. MCAP-free -> no raw_prices needed. Benchmark SPY.

Usage: python -u workflows/stock-selection/experiments/phase8_run_from_panel.py
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
    ARTIFACTS_DIR, evaluate_gate, load_config, load_prices_with_benchmark,
    load_sp500_universe, make_backtest_config, make_cost_model,
)
from youbet.etf.data import fetch_tbill_rates  # noqa: E402
from youbet.stock.backtester import StockBacktester  # noqa: E402
from youbet.stock.fwd_valuation import PrecomputedScoreStrategy  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    t0 = time.monotonic()
    cfg = load_config()
    panel = pd.read_parquet(ARTIFACTS_DIR / "phase8_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.drop_duplicates(["date", "ticker"], keep="first")
    lut = panel.pivot_table(index="date", columns="ticker", values="score")
    logger.info("Panel: %d rows, %d dates %s..%s, %d tickers",
                len(panel), lut.shape[0], lut.index.min().date(),
                lut.index.max().date(), lut.shape[1])

    uni = load_sp500_universe()
    px = load_prices_with_benchmark(uni, start="2005-01-01", config=cfg)
    try:
        tbill = fetch_tbill_rates(start=px.index.min().strftime("%Y-%m-%d"),
                                  end=px.index.max().strftime("%Y-%m-%d"), allow_fallback=True)
    except Exception:
        tbill = pd.Series(0.04, index=px.index, name="tbill_3m")

    bt = StockBacktester(config=make_backtest_config(cfg), prices=px, universe=uni,
                         cost_model=make_cost_model(cfg), tbill_rates=tbill)
    strat = PrecomputedScoreStrategy(lut, "quality_composite_v1", min_holdings=20)
    bench = BuyAndHoldETF(cfg["benchmark"]["ticker"])

    res = bt.run(strategy=strat, benchmark=bench)
    logger.info("%s: ExSharpe=%+.3f strat=%.3f bench=%.3f folds=%d turn=%.1f in %.0fs",
                strat.name, res.excess_sharpe, res.overall_metrics.sharpe_ratio,
                res.benchmark_metrics.sharpe_ratio, len(res.fold_results),
                res.total_turnover, time.monotonic() - t0)

    out = ARTIFACTS_DIR / "phase8_returns.parquet"
    df = pd.DataFrame({strat.name: res.overall_returns})
    df["__benchmark__"] = res.benchmark_returns.reindex(df.index)
    df.to_parquet(out)
    logger.info("saved %s", out)

    g = evaluate_gate({strat.name: res.overall_returns}, res.benchmark_returns, config=cfg)
    r = g[strat.name]
    print("\n" + "=" * 92, flush=True)
    print("Phase 8 — quality_composite_v1 (mcap-free, from panel) — EXPLORATORY", flush=True)
    print("=" * 92, flush=True)
    print(f"  ExSharpe={r['observed_excess_sharpe']:+.3f}  raw_p={r['p_value']:.4f}  "
          f"hAdj={r['holm_adjusted_p']:.4f}  90% CI=[{r['gate_ci_lower']:+.3f}, {r['gate_ci_upper']:+.3f}]  "
          f"stratSharpe={r['strategy_sharpe']:.3f} benchSharpe={r['benchmark_sharpe']:.3f}  "
          f"GATE={'PASS' if r['passes_gate'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
