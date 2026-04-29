"""Phase 4b — R8 OHLCV+Amihud falsification test (step 2 of Phase 7).

Mirrors `phase4_ml_gkx.py` but uses the 20-feature set including 6
OHLCV-based volume/illiquidity features (turn_22d, dolvol_22d,
std_dolvol, ill_amihud, baspread_hl, zerotrade).

Per phase7 precommit step 2: tests whether ml_gkx_lightgbm's −0.218
moves toward zero when liquidity signals are added. R8 expected yield
8-12%; "ML failed because we omitted core liquidity/volume signals" is
the falsification target.

Joint Holm scope is N=11 if step 1 results are kept; this run REPLACES
the 14-feature ml_gkx_* with 20-feature ml_gkx_*_v20 in the family
(family stays at N=11 — replacement, not expansion).

Usage:
    python phase4b_ohlcv.py
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
from youbet.stock.fundamentals import build_pit_shares_series_from_panel  # noqa: E402
from youbet.stock.data import fetch_stock_ohlcv, fetch_stock_prices  # noqa: E402
from youbet.stock.edgar import EdgarConfig, IndexedFacts, get_company_facts  # noqa: E402
from youbet.stock.fundamentals import TickerFundamentalsPanel, _clear_caches  # noqa: E402
from youbet.stock.strategies.base import BuyAndHoldETF  # noqa: E402
from youbet.stock.strategies.ml_ranker import MLRanker  # noqa: E402

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


def load_facts_by_ticker(universe):
    cfg = EdgarConfig(cache_dir=EDGAR_CACHE)
    ticker_to_cik = {}
    for _, row in universe.membership.sort_values("start_date").iterrows():
        t = row["ticker"]
        c = str(row["cik"]).strip() if pd.notna(row["cik"]) else ""
        if c and c != "nan":
            ticker_to_cik[t] = c.zfill(10)
    panels = {}
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


def build_strategies():
    return [
        MLRanker(
            model_backend="elasticnet",
            model_params={"alpha": 1e-3, "l1_ratio": 0.5, "max_iter": 5000, "random_state": 42},
            validation_scheme="recent_dates_contiguous",
            validation_fraction_of_train=0.20,
            decile_breakpoint=0.10, min_holdings=20, max_holdings=100,
            feature_set="full",
        ),
        MLRanker(
            model_backend="lightgbm",
            model_params={
                "objective": "regression", "metric": "l2",
                "num_leaves": 31, "max_depth": 6, "learning_rate": 0.05,
                "n_estimators": 500, "min_child_samples": 50,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.1, "reg_lambda": 1.0,
                "random_state": 42, "verbose": -1,
            },
            early_stopping_rounds=20,
            validation_scheme="recent_dates_contiguous",
            validation_fraction_of_train=0.20,
            decile_breakpoint=0.10, min_holdings=20, max_holdings=100,
            feature_set="full",
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-start", default="2005-01-01")
    args = parser.parse_args()

    pre = verify_precommit()
    config = load_config()
    _clear_caches()
    universe = load_sp500_universe()

    bench_ticker = config["benchmark"]["ticker"]
    t0 = time.monotonic()
    prices = fetch_stock_prices(
        universe=universe, start=args.price_start,
        snapshot_dir=PRICE_CACHE, extra_tickers=[bench_ticker],
    )
    start_ts = pd.Timestamp(args.price_start)
    prices = prices.loc[prices.index >= start_ts]
    logger.info("Prices: %s in %.1fs", prices.shape, time.monotonic() - t0)

    t0 = time.monotonic()
    ohlcv = fetch_stock_ohlcv(
        universe=universe, start=args.price_start,
        snapshot_dir=PRICE_CACHE, extra_tickers=[bench_ticker],
    )
    for k, df in ohlcv.items():
        ohlcv[k] = df.loc[df.index >= start_ts]
    logger.info("OHLCV loaded: close shape %s in %.1fs", ohlcv["close"].shape, time.monotonic() - t0)

    t0 = time.monotonic()
    facts_by_ticker = load_facts_by_ticker(universe)
    active_ever = set(universe.all_tickers_ever())
    facts_by_ticker = {t: f for t, f in facts_by_ticker.items()
                       if t in active_ever and t in prices.columns}
    logger.info("Facts: %d in %.1fs", len(facts_by_ticker), time.monotonic() - t0)

    try:
        tbill = fetch_tbill_rates(
            start=prices.index.min().strftime("%Y-%m-%d"),
            end=prices.index.max().strftime("%Y-%m-%d"),
            allow_fallback=True,
        )
    except Exception as exc:
        logger.warning("T-bill fetch failed (%s); 4%% constant", exc)
        tbill = pd.Series(0.04, index=prices.index, name="tbill_3m")

    first_test_min = pre["walk_forward"]["first_test_start_min"]
    bt_cfg = make_backtest_config(config, first_test_start_min=first_test_min)
    cost = make_cost_model(config)

    # R9-HIGH-2: build PIT shares dict (filed_date-keyed) so turn_22d
    # actually has data. Prior runs passed nothing → turn_22d structurally
    # NaN → median-imputed to 0 → "20-feature" run was effectively
    # 19-feature with one constant-zero column.
    shares_outstanding_by_ticker = {
        t: build_pit_shares_series_from_panel(panel)
        for t, panel in facts_by_ticker.items()
    }
    n_with_shares = sum(1 for s in shares_outstanding_by_ticker.values() if len(s) > 0)
    logger.info(
        "PIT shares series built for %d/%d tickers (R9-HIGH-2 fix)",
        n_with_shares, len(shares_outstanding_by_ticker),
    )

    bt = StockBacktester(
        config=bt_cfg, prices=prices, universe=universe,
        cost_model=cost, tbill_rates=tbill,
        facts_by_ticker=facts_by_ticker,
        shares_outstanding_by_ticker=shares_outstanding_by_ticker,
        ohlcv=ohlcv,
    )

    benchmark = BuyAndHoldETF(bench_ticker)
    strategies = build_strategies()
    returns = {}
    bench_series = None
    for strat in strategies:
        strat_name = strat.name + "_v20"  # mark as 20-feature variant
        logger.info("=== Running %s (20-feature) ===", strat_name)
        ts = time.monotonic()
        result = bt.run(strategy=strat, benchmark=benchmark)
        logger.info(
            "%s complete in %.1fs: strategy_sharpe=%.3f bench=%.3f excess=%+.3f folds=%d",
            strat_name, time.monotonic() - ts,
            result.overall_metrics.sharpe_ratio,
            result.benchmark_metrics.sharpe_ratio,
            result.excess_sharpe, len(result.fold_results),
        )
        returns[strat_name] = result.overall_returns
        if bench_series is None:
            bench_series = result.benchmark_returns

    save_phase_returns("phase4b", returns, bench_series)

    logger.info("Within-phase Holm(N=2)...")
    within = evaluate_gate(returns, bench_series)
    print_gate_table(within, "Phase 4b — WITHIN-PHASE Holm(N=2) — EXPLORATORY")

    # Joint Holm: replace v14 ML in family with v20 ML; family stays N=11.
    # R9-HIGH-3: use canonical (longest) saved benchmark across artifacts
    # to avoid silent date-range truncation when a runtime bench is shorter
    # than some strategies' return history.
    p1 = pd.read_parquet(ARTIFACTS_DIR / "phase1_returns.parquet")
    p3 = pd.read_parquet(ARTIFACTS_DIR / "phase3_returns.parquet")
    p7 = pd.read_parquet(ARTIFACTS_DIR / "phase7_step1_returns.parquet")
    combined = {}
    for df in [p1, p3, p7]:
        for c in df.columns:
            if c != "__benchmark__":
                combined[c] = df[c].dropna()
    # Add v20 ML (note: v14 ml_gkx_* are EXCLUDED from this Joint Holm scope)
    combined.update(returns)
    logger.info("Joint Holm(N=%d) Phase 1 + 3 + 7-step1 + 4b-v20...", len(combined))
    canonical_bench = load_canonical_benchmark()
    results = evaluate_gate(combined, canonical_bench, config=config)
    p1_set, p3_set, p7_set = set(p1.columns), set(p3.columns), set(p7.columns)
    rows = []
    for name, r in results.items():
        if name in p1_set:
            phase = "P1"
        elif name in p3_set:
            phase = "P3"
        elif name in p7_set:
            phase = "P7-1"
        else:
            phase = "P4b"
        rows.append({
            "phase": phase, "strategy": name,
            "exSh": r["observed_excess_sharpe"],
            "ci_lo": r["gate_ci_lower"],
            "ci_hi": r["gate_ci_upper"],
            "h_adj_up": r["holm_adjusted_p"],
            "passes": r["passes_gate"],
        })
    rows.sort(key=lambda r: r["exSh"], reverse=True)
    print("\n" + "=" * 130)
    print(f"Joint Holm Phase 1 + 3 + 7-step1 + 4b-v20 (N={len(rows)}) — AUTHORITATIVE")
    print("=" * 130)
    print(f"{'rank':>4} {'phase':>5} {'strategy':<28} {'exSh':>8} {'CI lo':>8} {'CI hi':>8} {'hAdj':>7} {'pass':>5}")
    for i, r in enumerate(rows):
        print(f"{i+1:>4} {r['phase']:>5} {r['strategy']:<28} "
              f"{r['exSh']:>+8.3f} {r['ci_lo']:>+8.3f} {r['ci_hi']:>+8.3f} "
              f"{r['h_adj_up']:>7.4f} {str(r['passes']):>5}")
    print(f"Gate: {sum(r['passes'] for r in rows)}/{len(rows)} pass")


if __name__ == "__main__":
    main()
