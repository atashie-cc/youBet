"""Phase 8 — mcap-free QualityComposite backtest.

Runs the pre-registered quality_composite_v1 (z-sum of ROE + GP/A + magic-rank,
all mcap-free) through the locked StockBacktester (60/24/12, monthly, top-decile,
T+1, costs, T-bill, SPY) + 10k block-bootstrap gate. EXPLORATORY per
precommit/phase8_quality.json. No raw_prices needed (strategy reads no price/mcap).

Usage: python -u workflows/stock-selection/experiments/phase8_quality.py
"""
from __future__ import annotations

import hashlib
import json
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
from youbet.stock.edgar import EdgarConfig, IndexedFacts, get_company_facts  # noqa: E402
from youbet.stock.fundamentals import TickerFundamentalsPanel, _clear_caches  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402
from youbet.stock.strategies.composites import QualityComposite  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EDGAR_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "edgar"
PRECOMMIT = WORKFLOW_ROOT / "precommit" / "phase8_quality.json"


def load_facts_by_ticker(universe):
    cfg = EdgarConfig(cache_dir=EDGAR_CACHE)
    t2c = {}
    for _, row in universe.membership.sort_values("start_date").iterrows():
        c = str(row["cik"]).strip() if pd.notna(row["cik"]) else ""
        if c and c != "nan":
            t2c[row["ticker"]] = c.zfill(10)
    panels = {}
    loaded = missing = 0
    for t, cik in t2c.items():
        p = EDGAR_CACHE / f"CIK{cik}.parquet"
        if not p.exists():
            missing += 1
            continue
        try:
            panels[t] = TickerFundamentalsPanel.build(t, IndexedFacts(get_company_facts(cik, cfg)))
            loaded += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("facts load failed %s: %s", t, exc)
            missing += 1
    logger.info("EDGAR panels: %d built, %d missing", loaded, missing)
    return panels


def main():
    sha = hashlib.sha256(PRECOMMIT.read_text(encoding="utf-8").encode()).hexdigest()[:16]
    pre = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    logger.info("Phase 8 precommit sha=%s framing=%s", sha, pre["framing"][:60])

    cfg = load_config()
    _clear_caches()
    uni = load_sp500_universe()
    bench_t = cfg["benchmark"]["ticker"]

    t0 = time.monotonic()
    prices = load_prices_with_benchmark(uni, start="2005-01-01", config=cfg)
    logger.info("Prices %s in %.1fs", prices.shape, time.monotonic() - t0)

    t0 = time.monotonic()
    facts = load_facts_by_ticker(uni)
    ae = set(uni.all_tickers_ever())
    facts = {t: f for t, f in facts.items() if t in ae and t in prices.columns}
    logger.info("Facts %d in %.1fs", len(facts), time.monotonic() - t0)

    try:
        tbill = fetch_tbill_rates(
            start=prices.index.min().strftime("%Y-%m-%d"),
            end=prices.index.max().strftime("%Y-%m-%d"), allow_fallback=True)
    except Exception:
        tbill = pd.Series(0.04, index=prices.index, name="tbill_3m")

    # first_test_start_min per precommit exclusions (2010-01-01)
    first_min = pd.Timestamp(pre["exclusions"]["first_fold_test_start_min"])
    bt = StockBacktester(
        config=make_backtest_config(cfg, first_test_start_min=first_min)
        if "first_test_start_min" in make_backtest_config.__code__.co_varnames
        else make_backtest_config(cfg),
        prices=prices, universe=uni, cost_model=make_cost_model(cfg),
        tbill_rates=tbill, facts_by_ticker=facts,
    )
    strat = QualityComposite(min_holdings=20)
    bench = BuyAndHoldETF(bench_t)

    ts = time.monotonic()
    res = bt.run(strategy=strat, benchmark=bench)
    logger.info("%s done in %.0fs: ExSharpe=%+.3f strat=%.3f bench=%.3f folds=%d turn=%.1f",
                strat.name, time.monotonic() - ts, res.excess_sharpe,
                res.overall_metrics.sharpe_ratio, res.benchmark_metrics.sharpe_ratio,
                len(res.fold_results), res.total_turnover)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "phase8_returns.parquet"
    df = pd.DataFrame({strat.name: res.overall_returns})
    df["__benchmark__"] = res.benchmark_returns.reindex(df.index)
    df.to_parquet(out)
    logger.info("saved %s", out)

    g = evaluate_gate({strat.name: res.overall_returns}, res.benchmark_returns, config=cfg)
    r = g[strat.name]
    print("\n" + "=" * 90, flush=True)
    print("Phase 8 — quality_composite_v1 (mcap-free) — EXPLORATORY", flush=True)
    print("=" * 90, flush=True)
    print(f"  ExSharpe={r['observed_excess_sharpe']:+.3f}  raw_p={r['p_value']:.4f}  "
          f"hAdj={r['holm_adjusted_p']:.4f}  90% CI=[{r['gate_ci_lower']:+.3f}, {r['gate_ci_upper']:+.3f}]  "
          f"stratSharpe={r['strategy_sharpe']:.3f} benchSharpe={r['benchmark_sharpe']:.3f}  "
          f"GATE={'PASS' if r['passes_gate'] else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
