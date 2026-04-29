"""Phase 3 — composite strategies (Piotroski F, Magic Formula, QV z-sum).

Run pattern mirrors `phase1_efficiency.py` — same universe load, same
panel construction, same backtester. The difference is the strategy
set: three fundamentals-backed composites instead of price-only + rules.

Per precommit/phase3_confirmatory.json: EXPLORATORY framing with joint
Holm(Phase1 ∪ Phase3) scoped reporting. Neither the phase gate nor the
joint Holm is expected to clear 0.05 given Phase 0's MDE > +0.5 finding;
we report point estimates + 90% CIs.

Usage:
    python phase3_composites.py                 # full universe (slow)
    python phase3_composites.py --limit-tickers 50  # smoke
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
from youbet.etf.stats import holm_bonferroni  # noqa: E402
from youbet.stock.backtester import StockBacktester  # noqa: E402
from youbet.stock.data import fetch_stock_prices  # noqa: E402
from youbet.stock.edgar import EdgarConfig, IndexedFacts, get_company_facts  # noqa: E402
from youbet.stock.fundamentals import TickerFundamentalsPanel, _clear_caches  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402
from youbet.stock.strategies.composites import (  # noqa: E402
    MagicFormula,
    PiotroskiF,
    QualityValue,
)
from youbet.stock.universe import Universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PRECOMMIT_PATH = WORKFLOW_ROOT / "precommit" / "phase3_confirmatory.json"
EDGAR_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "edgar"
PRICE_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "prices"


def verify_precommit() -> dict:
    content = PRECOMMIT_PATH.read_text(encoding="utf-8")
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]
    logger.info("Phase 3 precommit loaded: sha=%s", sha)
    return json.loads(content)


def load_facts_by_ticker(universe: Universe, limit: int | None = None):
    cfg = EdgarConfig(cache_dir=EDGAR_CACHE)
    ticker_to_cik = {}
    for _, row in universe.membership.sort_values("start_date").iterrows():
        t = row["ticker"]
        c = str(row["cik"]).strip() if pd.notna(row["cik"]) else ""
        if c and c != "nan":
            ticker_to_cik[t] = c.zfill(10)

    panels: dict[str, TickerFundamentalsPanel] = {}
    tickers_iter = list(ticker_to_cik.items())
    if limit is not None:
        tickers_iter = tickers_iter[:limit]
    loaded = missing = 0
    for t, cik in tickers_iter:
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
    logger.info(
        "EDGAR panels built: %d tickers (%d missing / no cache)",
        loaded, missing,
    )
    return panels


def build_strategies():
    return [
        PiotroskiF(min_f=7, min_holdings=20),
        MagicFormula(min_holdings=20),
        QualityValue(min_holdings=20),
    ]


def run_phase3(
    limit_tickers: int | None = None,
    price_start: str = "2005-01-01",
):
    verify_precommit()
    config = load_config()
    _clear_caches()
    universe = load_sp500_universe()

    if limit_tickers:
        active_today = list(universe.active_as_of("2026-04-20"))[:limit_tickers]
        membership_sub = universe.membership[
            universe.membership["ticker"].isin(active_today)
        ].copy()
        universe = Universe(
            membership=membership_sub,
            delistings=universe.delistings,
            index_name=f"S&P 500 (limited to {limit_tickers})",
        )
        logger.warning("LIMITED universe to %d tickers for smoke run", limit_tickers)

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
    facts_by_ticker = load_facts_by_ticker(universe, limit=None)
    active_ever = set(universe.all_tickers_ever())
    facts_by_ticker = {
        t: f for t, f in facts_by_ticker.items()
        if t in active_ever and t in prices.columns
    }
    logger.info("Facts-by-ticker: %d entries in %.1fs",
                len(facts_by_ticker), time.monotonic() - t0)

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
    strategies = build_strategies()
    returns: dict[str, pd.Series] = {}
    bench_series: pd.Series | None = None
    for strat in strategies:
        logger.info("=== Running %s ===", strat.name)
        ts = time.monotonic()
        result = bt.run(strategy=strat, benchmark=benchmark)
        logger.info(
            "%s complete in %.1fs: strategy_sharpe=%.3f excess=%+.3f folds=%d",
            strat.name, time.monotonic() - ts,
            result.overall_metrics.sharpe_ratio,
            result.excess_sharpe, len(result.fold_results),
        )
        returns[strat.name] = result.overall_returns
        if bench_series is None:
            bench_series = result.benchmark_returns

    return returns, bench_series


def joint_holm_phase1_plus_phase3(
    phase3_returns: dict[str, pd.Series],
    bench: pd.Series,
    phase1_returns_path: Path,
    config: dict,
) -> pd.DataFrame:
    """Report joint Holm(Phase1 ∪ Phase3) — per Phase 3 precommit framing."""
    phase1_df = pd.read_parquet(phase1_returns_path)
    phase1_returns = {
        c: phase1_df[c].dropna()
        for c in phase1_df.columns if c != "__benchmark__"
    }
    combined = {**phase1_returns, **phase3_returns}
    logger.info(
        "Joint Holm scope: %d strategies (Phase 1: %d, Phase 3: %d)",
        len(combined), len(phase1_returns), len(phase3_returns),
    )
    results = evaluate_gate(combined, bench, config=config)
    rows = []
    for name, r in results.items():
        is_p3 = name in phase3_returns
        rows.append({
            "phase": "phase3" if is_p3 else "phase1",
            "strategy": name,
            "observed_excess_sharpe": r["observed_excess_sharpe"],
            "gate_ci_lower": r["gate_ci_lower"],
            "gate_ci_upper": r["gate_ci_upper"],
            "holm_joint_adj_p": r["holm_adjusted_p"],
            "passes_joint_gate": r["passes_gate"],
        })
    return pd.DataFrame(rows).sort_values(
        "observed_excess_sharpe", ascending=False
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-tickers", type=int, default=None,
                        help="Smoke: restrict universe to N most-recent members.")
    parser.add_argument("--price-start", default="2005-01-01")
    args = parser.parse_args()

    t_start = time.monotonic()
    returns, bench = run_phase3(
        limit_tickers=args.limit_tickers,
        price_start=args.price_start,
    )

    artifact_name = "phase3_smoke" if args.limit_tickers else "phase3"
    save_phase_returns(artifact_name, returns, bench)

    # Within-phase Holm
    logger.info("Evaluating within-phase gate (EXPLORATORY)...")
    within_results = evaluate_gate(returns, bench)
    print_gate_table(
        within_results,
        f"Phase 3 — WITHIN-PHASE Holm(N={len(returns)}) — EXPLORATORY"
        f" (universe={'subset ' + str(args.limit_tickers) if args.limit_tickers else 'full S&P 500'})",
    )

    # Joint Holm with Phase 1 (if Phase 1 artifact available)
    phase1_path = ARTIFACTS_DIR / "phase1_returns.parquet"
    if phase1_path.exists():
        logger.info("Evaluating joint Holm(Phase1 ∪ Phase3)...")
        joint = joint_holm_phase1_plus_phase3(
            returns, bench, phase1_path, load_config(),
        )
        print("\n" + "=" * 110)
        print(f"Joint Holm across Phase 1 + Phase 3 (N={len(joint)}) — EXPLORATORY")
        print("=" * 110)
        print(joint.to_string(index=False))
    else:
        logger.warning("phase1_returns.parquet missing — skipping joint Holm")

    logger.info("Phase 3 total wall time: %.1fs", time.monotonic() - t_start)


if __name__ == "__main__":
    main()
