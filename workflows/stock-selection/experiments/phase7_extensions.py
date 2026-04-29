"""Phase 7 — Codex R8 falsification tests (Step 1: Value + GP/A).

Runs `gross_profitability` (Novy-Marx GP/A standalone) and
`value_profitability` (AQR QMJ-lite z-sum of EY + GP/A) on the existing
S&P 500 universe. Both strategies use the same backtest infrastructure
as Phase 1 + Phase 3 (no ML, no OHLCV needed).

Joint Holm(N=11) is computed across all prior + new strategies on the
SAME cleaned bench/data. Per phase7 precommit framing, this is
EXPLORATORY (post-completion bonus rounds); claims require R6
contamination checks.

Usage:
    python phase7_extensions.py                    # full universe
    python phase7_extensions.py --limit-tickers 50 # smoke
"""

from __future__ import annotations

import argparse
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
    ARTIFACTS_DIR,
    evaluate_gate,
    load_canonical_benchmark,
    load_config,
    load_sp500_universe,
    make_backtest_config,
    make_cost_model,
    print_gate_table,
    save_phase_returns,
)

from youbet.etf.data import fetch_tbill_rates  # noqa: E402
from youbet.stock.backtester import StockBacktester  # noqa: E402
from youbet.stock.data import fetch_stock_prices  # noqa: E402
from youbet.stock.edgar import EdgarConfig, IndexedFacts, get_company_facts  # noqa: E402
from youbet.stock.fundamentals import TickerFundamentalsPanel, _clear_caches  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402
from youbet.stock.strategies.composites import ValueProfitability  # noqa: E402
from youbet.stock.strategies.rules import GrossProfitability  # noqa: E402
from youbet.stock.universe import Universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PRECOMMIT_PATH = WORKFLOW_ROOT / "precommit" / "phase7_extensions.json"
EDGAR_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "edgar"
PRICE_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "prices"


def verify_precommit() -> dict:
    content = PRECOMMIT_PATH.read_text(encoding="utf-8")
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]
    logger.info("Phase 7 precommit loaded: sha=%s", sha)
    return json.loads(content)


def load_facts_by_ticker(universe: Universe):
    cfg = EdgarConfig(cache_dir=EDGAR_CACHE)
    ticker_to_cik = {}
    for _, row in universe.membership.sort_values("start_date").iterrows():
        t = row["ticker"]
        c = str(row["cik"]).strip() if pd.notna(row["cik"]) else ""
        if c and c != "nan":
            ticker_to_cik[t] = c.zfill(10)

    panels: dict[str, TickerFundamentalsPanel] = {}
    loaded = missing = 0
    for t, cik in ticker_to_cik.items():
        path = EDGAR_CACHE / f"CIK{cik}.parquet"
        if not path.exists():
            missing += 1
            continue
        try:
            indexed = IndexedFacts(get_company_facts(cik, cfg))
            panels[t] = TickerFundamentalsPanel.build(t, indexed)
            loaded += 1
        except Exception as exc:
            logger.warning("Failed to load facts for %s (CIK %s): %s", t, cik, exc)
            missing += 1
    logger.info("EDGAR panels built: %d tickers (%d missing)", loaded, missing)
    return panels


def build_step1_strategies():
    return [
        GrossProfitability(min_holdings=20),
        ValueProfitability(min_holdings=20),
    ]


def run_phase7_step1(price_start: str = "2005-01-01"):
    verify_precommit()
    config = load_config()
    _clear_caches()
    universe = load_sp500_universe()

    bench_ticker = config["benchmark"]["ticker"]
    t0 = time.monotonic()
    prices = fetch_stock_prices(
        universe=universe, start=price_start,
        snapshot_dir=PRICE_CACHE, extra_tickers=[bench_ticker],
    )
    start_ts = pd.Timestamp(price_start)
    prices = prices.loc[prices.index >= start_ts]
    logger.info("Prices: %s in %.1fs", prices.shape, time.monotonic() - t0)

    t0 = time.monotonic()
    facts_by_ticker = load_facts_by_ticker(universe)
    active_ever = set(universe.all_tickers_ever())
    facts_by_ticker = {
        t: f for t, f in facts_by_ticker.items()
        if t in active_ever and t in prices.columns
    }
    logger.info("Facts-by-ticker: %d in %.1fs", len(facts_by_ticker), time.monotonic() - t0)

    try:
        tbill = fetch_tbill_rates(
            start=prices.index.min().strftime("%Y-%m-%d"),
            end=prices.index.max().strftime("%Y-%m-%d"),
            allow_fallback=True,
        )
    except Exception as exc:
        logger.warning("T-bill fetch failed (%s); using 4%% constant", exc)
        tbill = pd.Series(0.04, index=prices.index, name="tbill_3m")

    bt_cfg = make_backtest_config(config)
    cost = make_cost_model(config)
    bt = StockBacktester(
        config=bt_cfg, prices=prices, universe=universe,
        cost_model=cost, tbill_rates=tbill,
        facts_by_ticker=facts_by_ticker,
    )

    benchmark = BuyAndHoldETF(bench_ticker)
    strategies = build_step1_strategies()
    returns: dict[str, pd.Series] = {}
    bench_series: pd.Series | None = None

    for strat in strategies:
        logger.info("=== Running %s ===", strat.name)
        ts = time.monotonic()
        result = bt.run(strategy=strat, benchmark=benchmark)
        logger.info(
            "%s complete in %.1fs: strategy_sharpe=%.3f bench_sharpe=%.3f excess=%+.3f folds=%d",
            strat.name, time.monotonic() - ts,
            result.overall_metrics.sharpe_ratio,
            result.benchmark_metrics.sharpe_ratio,
            result.excess_sharpe, len(result.fold_results),
        )
        returns[strat.name] = result.overall_returns
        if bench_series is None:
            bench_series = result.benchmark_returns

    return returns, bench_series


def joint_holm_n11(
    p7_returns: dict[str, pd.Series],
    config: dict,
) -> pd.DataFrame:
    """Joint Holm across Phase 1 ∪ Phase 3 ∪ Phase 4 ∪ Phase 7-step1 = N=11.

    R9-HIGH-3: uses `load_canonical_benchmark` to pick the LONGEST saved
    __benchmark__ across artifacts and asserts overlap-consistency. Each
    strategy's full return series is used (no truncation to a runtime-
    specific bench window).
    """
    p1 = pd.read_parquet(ARTIFACTS_DIR / "phase1_returns.parquet")
    p3 = pd.read_parquet(ARTIFACTS_DIR / "phase3_returns.parquet")
    p4 = pd.read_parquet(ARTIFACTS_DIR / "phase4_returns.parquet")
    combined = {}
    for df in [p1, p3, p4]:
        for c in df.columns:
            if c != "__benchmark__":
                combined[c] = df[c].dropna()
    combined.update(p7_returns)
    logger.info("Joint Holm scope: %d strategies", len(combined))

    p1_set = set(p1.columns)
    p3_set = set(p3.columns)
    p4_set = set(p4.columns)

    canonical_bench = load_canonical_benchmark()
    results = evaluate_gate(combined, canonical_bench, config=config)
    rows = []
    for name, r in results.items():
        if name in p1_set:
            phase = "phase1"
        elif name in p3_set:
            phase = "phase3"
        elif name in p4_set:
            phase = "phase4"
        else:
            phase = "phase7"
        rows.append({
            "phase": phase,
            "strategy": name,
            "exSh": r["observed_excess_sharpe"],
            "ci_lo": r["gate_ci_lower"],
            "ci_hi": r["gate_ci_upper"],
            "raw_p_up": r["p_value"],
            "h_adj_up": r["holm_adjusted_p"],
            "passes": r["passes_gate"],
        })
    return pd.DataFrame(rows).sort_values("exSh", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-start", default="2005-01-01")
    args = parser.parse_args()

    t_start = time.monotonic()
    returns, bench = run_phase7_step1(price_start=args.price_start)
    save_phase_returns("phase7_step1", returns, bench)

    logger.info("Within-phase Holm(N=2)...")
    within = evaluate_gate(returns, bench)
    print_gate_table(within, "Phase 7 step 1 — WITHIN-PHASE Holm(N=2) — EXPLORATORY")

    logger.info("Joint Holm(N=11) Phase 1 + 3 + 4 + 7-step1 ...")
    joint = joint_holm_n11(returns, load_config())
    print("\n" + "=" * 130)
    print(f"Joint Holm Phase 1 + Phase 3 + Phase 4 + Phase 7 step 1 (N={len(joint)}) — AUTHORITATIVE")
    print("=" * 130)
    print(joint.to_string(index=False))
    print("\nGate: exSh > 0.20 AND h_adj_up < 0.05 AND ci_lo > 0  →  "
          f"{int(joint['passes'].sum())}/{len(joint)} pass")

    logger.info("Phase 7 step 1 wall: %.1fs", time.monotonic() - t_start)


if __name__ == "__main__":
    main()
