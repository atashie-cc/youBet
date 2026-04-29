"""Phase 1 — EXPLORATORY characterization of 4 factor strategies on real
S&P 500 data.

Per CLAUDE.md #1 and the 2026-04-19 research log reframing: Phase 1 is
point-estimate + CI reporting, NOT a confirmatory gate pass. The Phase 0
empirical power analysis showed MDE > +0.50 excess Sharpe at 20y daily —
literature factor premia (post-McLean-Pontiff halving) are in the
+0.05 to +0.20 range and will be below MDE by design. We still compute
the gate (Holm + CI + p-value) for audit, but the authoritative deliverable
is the point estimate + regime stability (phase2_robustness.py).

Strategies (pre-committed in `precommit/phase1_confirmatory.json`):
  - value_earnings_yield  (TTM NI / shares out / price)
  - momentum_252_21       (Jegadeesh-Titman 12-1)
  - quality_roe_ttm       (trailing-twelve-month ROE)
  - lowvol_252d           (inverse 1-year realized vol)

All top-decile, equal-weight, monthly rebalance. Benchmark = SPY.

Usage:
    python phase1_efficiency.py                    # full run on 646 tickers
    python phase1_efficiency.py --limit-tickers 50 # smoke run
    STOCK_PHASE0_FULL=1 python phase1_efficiency.py  # larger bootstrap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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
from youbet.stock.fundamentals import _clear_caches, TickerFundamentalsPanel  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402
from youbet.stock.strategies.rules import (  # noqa: E402
    LowVol252d,
    Momentum12m1,
    QualityROE,
    ValueScore,
)
from youbet.stock.universe import Universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PRECOMMIT_PATH = WORKFLOW_ROOT / "precommit" / "phase1_confirmatory.json"
EDGAR_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "edgar"
PRICE_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "prices"


def verify_precommit(path: Path = PRECOMMIT_PATH) -> dict:
    """Load + hash the precommit JSON (audit trail)."""
    content = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]
    logger.info("Precommit loaded: %s (sha=%s)", path.name, sha)
    return json.loads(content)


def load_facts_by_ticker(universe: Universe, limit: int | None = None) -> dict:
    """Load every cached EDGAR parquet, keyed by ticker.

    Tickers without a CIK or without a cached parquet are absent from the
    returned dict — the fundamentals strategies (Value/Quality) will
    simply score NaN for them and they'll drop out of the top-decile.
    """
    cfg = EdgarConfig(cache_dir=EDGAR_CACHE)

    # Build ticker -> cik from membership (dedupe to latest known cik per ticker)
    ticker_to_cik = {}
    for _, row in universe.membership.sort_values("start_date").iterrows():
        t = row["ticker"]
        c = str(row["cik"]).strip() if pd.notna(row["cik"]) else ""
        if c and c != "nan":
            ticker_to_cik[t] = c.zfill(10)

    facts_by_ticker: dict[str, TickerFundamentalsPanel] = {}
    tickers_iter = list(ticker_to_cik.items())
    if limit is not None:
        tickers_iter = tickers_iter[:limit]

    loaded = 0
    missing = 0
    for t, cik in tickers_iter:
        path = EDGAR_CACHE / f"CIK{cik}.parquet"
        if not path.exists():
            missing += 1
            continue
        try:
            # Build a per-ticker fundamentals panel: union-sort each alias
            # once, so every compute_fundamentals call becomes a filter +
            # quarterize on a small slice (~50ms cold vs ~550ms for the
            # legacy per-call union path).
            indexed = IndexedFacts(get_company_facts(cik, cfg))
            facts_by_ticker[t] = TickerFundamentalsPanel.build(t, indexed)
            loaded += 1
        except Exception as exc:
            logger.warning("Failed to load facts for %s (CIK %s): %s", t, cik, exc)
            missing += 1
    logger.info(
        "EDGAR panels built: %d tickers (%d missing / no cache)",
        loaded, missing,
    )
    return facts_by_ticker


def build_strategies(limit: int | None = None, price_only: bool = False):
    """Four Phase 1 strategies with pre-committed params.

    If `price_only=True`, skip Value and Quality (which need fundamentals)
    — useful for quickly validating the pipeline without the slow
    compute_fundamentals path.
    """
    min_holdings = 20 if limit is None else max(5, limit // 10)
    strategies = [
        Momentum12m1(lookback_days=252, skip_days=21, min_holdings=min_holdings),
        LowVol252d(lookback_days=252, min_holdings=min_holdings),
    ]
    if not price_only:
        strategies = [
            ValueScore(min_holdings=min_holdings),
            strategies[0],  # Momentum
            QualityROE(min_holdings=min_holdings),
            strategies[1],  # LowVol
        ]
    return strategies


def run_phase1(
    limit_tickers: int | None = None,
    price_start: str = "1998-01-01",
    price_only: bool = False,
) -> dict[str, pd.Series]:
    """Execute all Phase 1 strategies, return {name: returns_series} + benchmark."""
    precommit = verify_precommit()
    config = load_config()
    _clear_caches()  # flush any stale memoized fundamentals from prior runs
    universe = load_sp500_universe()
    logger.info("Universe: %d rows, %d unique tickers, %d delistings",
                len(universe.membership), universe.membership["ticker"].nunique(),
                len(universe.delistings))

    if limit_tickers:
        # Keep the most-recent `limit_tickers` members for smoke testing
        active_today = list(universe.active_as_of("2026-04-19"))[:limit_tickers]
        membership_sub = universe.membership[
            universe.membership["ticker"].isin(active_today)
        ].copy()
        universe = Universe(
            membership=membership_sub,
            delistings=universe.delistings,
            index_name=f"S&P 500 (limited to {limit_tickers})",
        )
        logger.warning("LIMITED universe to %d tickers for smoke run", limit_tickers)

    # --- prices ---
    bench_ticker = config["benchmark"]["ticker"]
    logger.info("Fetching/loading prices (from cache if available)...")
    t0 = time.monotonic()
    prices = fetch_stock_prices(
        universe=universe,
        start=price_start,
        snapshot_dir=PRICE_CACHE,
        extra_tickers=[bench_ticker],
    )
    # The shared snapshot cache was built with start='1990-01-01'; trim
    # to the requested price_start so we don't run pre-EDGAR folds that
    # burn compute on empty-weights warnings.
    start_ts = pd.Timestamp(price_start)
    pre_rows = len(prices)
    prices = prices.loc[prices.index >= start_ts]
    logger.info("Prices: %s in %.1fs (trimmed %d pre-%s rows)",
                prices.shape, time.monotonic() - t0,
                pre_rows - len(prices), price_start)

    # --- fundamentals ---
    logger.info("Loading EDGAR facts by ticker...")
    t0 = time.monotonic()
    facts_by_ticker = load_facts_by_ticker(universe, limit=None)
    # Restrict to tickers that are actually in prices AND in our universe
    active_ever = set(universe.all_tickers_ever())
    facts_by_ticker = {
        t: f for t, f in facts_by_ticker.items()
        if t in active_ever and t in prices.columns
    }
    logger.info("Facts-by-ticker: %d entries in %.1fs",
                len(facts_by_ticker), time.monotonic() - t0)

    # --- t-bill ---
    try:
        tbill = fetch_tbill_rates(
            start=prices.index.min().strftime("%Y-%m-%d"),
            end=prices.index.max().strftime("%Y-%m-%d"),
            allow_fallback=True,
        )
    except Exception as exc:
        logger.warning("T-bill fetch failed (%s); using 4%% constant", exc)
        tbill = pd.Series(0.04, index=prices.index, name="tbill_3m")

    # --- backtester setup ---
    bt_cfg = make_backtest_config(config)
    cost = make_cost_model(config)
    bt = StockBacktester(
        config=bt_cfg,
        prices=prices,
        universe=universe,
        cost_model=cost,
        tbill_rates=tbill,
        facts_by_ticker=facts_by_ticker,
    )

    strategies = build_strategies(limit=limit_tickers, price_only=price_only)
    benchmark = BuyAndHoldETF(bench_ticker)

    # --- run each strategy ---
    returns_by_strategy: dict[str, pd.Series] = {}
    benchmark_returns: pd.Series | None = None
    for strat in strategies:
        logger.info("=== Running %s ===", strat.name)
        t0 = time.monotonic()
        result = bt.run(strategy=strat, benchmark=benchmark)
        logger.info(
            "%s complete in %.1fs: strategy_sharpe=%.3f bench_sharpe=%.3f "
            "excess=%+.3f folds=%d",
            strat.name, time.monotonic() - t0,
            result.overall_metrics.sharpe_ratio,
            result.benchmark_metrics.sharpe_ratio,
            result.excess_sharpe,
            len(result.fold_results),
        )
        returns_by_strategy[strat.name] = result.overall_returns
        if benchmark_returns is None:
            benchmark_returns = result.benchmark_returns
        else:
            # Sanity: all strategies should produce the same benchmark
            diff = (benchmark_returns - result.benchmark_returns).abs().max()
            if diff > 1e-9:
                logger.warning(
                    "Benchmark mismatch between runs: max |diff|=%.6f", diff,
                )

    return returns_by_strategy, benchmark_returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-tickers", type=int, default=None,
                        help="Smoke-run: restrict universe to N most-recent members.")
    parser.add_argument("--price-only", action="store_true",
                        help="Skip fundamentals-backed strategies (Value, "
                             "Quality) to validate the pipeline end-to-end "
                             "quickly.")
    parser.add_argument("--price-start", default="2005-01-01",
                        help="XBRL mandate phased in 2009-2011; earliest "
                             "filings go back to ~2009. With 60-month train, "
                             "2005 start → first test fold 2010-01. Early "
                             "folds (2010-2012) have thin fundamentals "
                             "coverage; expect Value/Quality to hit "
                             "solid ground ~2012+.")
    args = parser.parse_args()

    t_start = time.monotonic()
    returns, bench = run_phase1(
        limit_tickers=args.limit_tickers,
        price_start=args.price_start,
        price_only=args.price_only,
    )

    # --- persist ---
    artifact_name = "phase1_smoke" if args.limit_tickers else "phase1"
    save_phase_returns(artifact_name, returns, bench)

    # --- gate evaluation (audit only) ---
    logger.info("Evaluating gate (exploratory framing — no PASS/FAIL claims)...")
    results = evaluate_gate(returns, bench)
    print_gate_table(
        results,
        f"Phase 1 — EXPLORATORY (universe={'subset ' + str(args.limit_tickers) if args.limit_tickers else 'full S&P 500'})",
    )

    logger.info("Phase 1 total wall time: %.1fs", time.monotonic() - t_start)


if __name__ == "__main__":
    main()
