"""Phase 4 — Gu-Kelly-Xiu-inspired ML ranking (14-feature MVP).

Runs 2 confirmatory strategies (ElasticNet + LightGBM) on the full S&P
500 universe. MLP is DEFERRED per Codex R6 and gated on
STOCK_PHASE4_ENABLE_MLP=1.

Per `precommit/phase4_confirmatory.json`:
- 14 features (10 price-based + 3 fundamentals + 1 sector-derived);
  6 volume-based features deferred to Phase 4b (need OHLCV pipeline).
- Target: 22d cross-sectionally demeaned forward log-return.
- Walk-forward 60/24/12, first test ≥ 2012-01-01.
- Coverage-invalid rebalances are EXCLUDED from the confirmatory return
  series (NOT parked in T-bill; the date is simply omitted).
- Holm scope: joint Phase 1 ∪ Phase 3 ∪ Phase 4 = N=9.

Contamination checks (run post hoc in a separate helper):
- Zero-return day dist, pre/post-AI split, decade split, COVID
  exclusion, stripped-window Holm, feature-importance stability, Mag7
  exposure, coverage heatmap, raw-return sensitivity.

Usage:
    python phase4_ml_gkx.py                    # full universe (slow)
    python phase4_ml_gkx.py --limit-tickers 50 # smoke

====================================================================
GKX 94-characteristic eligibility audit (committed here per R6-HIGH-Q1)
====================================================================

Full GKX (Gu-Kelly-Xiu 2020 RFS Internet Appendix Table B.1) list with
our status:

Price-based (✔ included in MVP):
  mom1m      ✔ mom_1m_1m
  mom6m      ✔ mom_6m_1m
  mom12m     ✔ mom_12m_1m
  mom36m     ✔ mom_36m_13m
  chmom      ✔ chmom
  maxret     ✔ maxret_22d
  retvol     ✔ retvol_22d
  idiovol    ✔ idiovol_126d
  beta       ✔ beta_252d
  betasq     ✔ betasq_252d

Volume-based (⏳ deferred to Phase 4b, OHLCV pipeline required):
  turn, dolvol, std_dolvol, ill, baspread, zerotrade

Fundamentals (✔ 3 included, ⏭ ~15 excluded):
  ep, sp, bm           ✔ MVP
  cfp, dy, rd_mve      ⏭ computable, coverage-artifact concern per R5
  agr, grltnoa, invest ⏭ computable, deferred
  gma, gm, roe, roa    ⏭ computable; overlap with Quality
  accruals, cashdebt   ⏭ computable, deferred

Derived (✔ 1 included):
  indmom     ✔ indmom (sector-mean mom_12m_1m)

Excluded — paid-data paywall:
  IBES-based (sue, chfeps, nanalyst, disp, fgr5yr, sfe, …): ✘ no IBES
  13F-based (ih, chih): ✘ requires 13F parse
  Short-interest (short, chshort): ✘ no FINRA pipe
  Credit rating (cr): ✘ paywall

Excluded — out of scope:
  mvel1 (market cap): ⏭ R5 Mag7 concern; ablation flag
  pricedelay: ⏭ high eng cost, low GKX R²; Phase 4b
  sic2: ⏭ we use GICS not SIC

Summary: 14 of ~42 computable; 20 of ~94 total. Selection was
literature-driven, not empirical-R² driven.
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
from youbet.stock.strategies.ml_ranker import MLRanker  # noqa: E402
from youbet.stock.universe import Universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PRECOMMIT_PATH = WORKFLOW_ROOT / "precommit" / "phase4_confirmatory.json"
EDGAR_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "edgar"
PRICE_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "prices"
AUDIT_DIR = WORKFLOW_ROOT / "artifacts"


def verify_precommit() -> dict:
    content = PRECOMMIT_PATH.read_text(encoding="utf-8")
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]
    logger.info("Phase 4 precommit loaded: sha=%s", sha)
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
        "EDGAR panels built: %d tickers (%d missing / no cache)", loaded, missing,
    )
    return panels


def build_strategies():
    strategies = [
        MLRanker(
            model_backend="elasticnet",
            model_params={"alpha": 1e-3, "l1_ratio": 0.5, "max_iter": 5000, "random_state": 42},
            validation_scheme="recent_dates_contiguous",
            validation_fraction_of_train=0.20,
            decile_breakpoint=0.10,
            min_holdings=20,
            max_holdings=100,
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
            decile_breakpoint=0.10,
            min_holdings=20,
            max_holdings=100,
        ),
    ]
    if os.environ.get("STOCK_PHASE4_ENABLE_MLP") == "1":
        raise NotImplementedError(
            "STOCK_PHASE4_ENABLE_MLP=1 set but MLP implementation is a stub. "
            "Run the MVP (ElasticNet + LightGBM) first per R6."
        )
    return strategies


def run_phase4(
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

    # R7-HIGH-1: enforce precommit first_test_start_min (2012-01-01).
    # Backtester will skip any fold whose test_start is before this floor.
    pre = verify_precommit()
    first_test_min = pre.get("walk_forward", {}).get("first_test_start_min")
    bt_cfg = make_backtest_config(config, first_test_start_min=first_test_min)
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
    audit_by_strategy: dict[str, dict] = {}

    for strat in strategies:
        logger.info("=== Running %s ===", strat.name)
        ts = time.monotonic()
        result = bt.run(strategy=strat, benchmark=benchmark)
        elapsed = time.monotonic() - ts
        logger.info(
            "%s complete in %.1fs: strategy_sharpe=%.3f excess=%+.3f folds=%d",
            strat.name, elapsed,
            result.overall_metrics.sharpe_ratio,
            result.excess_sharpe, len(result.fold_results),
        )

        # Coverage audit: which rebalances were invalid
        cov_rows = list(getattr(strat, "_coverage_audit", []))
        n_total = len(cov_rows)
        n_invalid = sum(1 for r in cov_rows if not r.get("is_valid", True))
        logger.info(
            "%s coverage audit: %d rebalances, %d invalid (%.1f%%)",
            strat.name, n_total, n_invalid,
            100 * n_invalid / max(n_total, 1),
        )

        # Coverage-invalid rebalances: the empty-weights path in
        # backtester already puts 100% cash → T-bill. Per R6-HIGH-Q4 we
        # must EXCLUDE those test dates from the confirmatory series.
        ret = result.overall_returns.copy()
        invalid_dates = [
            pd.Timestamp(r["decision_date"]) for r in cov_rows
            if not r.get("is_valid", True)
        ]
        if invalid_dates:
            # For each invalid rebal, drop the subsequent hold-window dates.
            # Simplification for MVP: drop the rebal_date and next 21 biz days.
            drop_mask = pd.Series(False, index=ret.index)
            for d in invalid_dates:
                drop_end = d + pd.Timedelta(days=31)
                drop_mask |= (ret.index >= d) & (ret.index < drop_end)
            n_dropped = int(drop_mask.sum())
            logger.info(
                "%s: excluding %d return-days from coverage-invalid rebals "
                "(per R6-HIGH-Q4 invalidation policy)",
                strat.name, n_dropped,
            )
            ret = ret[~drop_mask]

        returns[strat.name] = ret
        audit_by_strategy[strat.name] = {
            "elapsed_sec": elapsed,
            "n_folds": len(result.fold_results),
            "n_rebals_total": n_total,
            "n_rebals_invalid": n_invalid,
            "n_return_days_excluded": int(drop_mask.sum()) if invalid_dates else 0,
            "excess_sharpe_raw": result.excess_sharpe,
            "feature_importances_by_fold": getattr(strat, "_feature_importances_by_fold", []),
        }
        if bench_series is None:
            bench_series = result.benchmark_returns

    return returns, bench_series, audit_by_strategy


def joint_holm_full(
    phase4_returns: dict[str, pd.Series],
    bench: pd.Series,
    phase1_returns_path: Path,
    phase3_returns_path: Path,
    config: dict,
) -> pd.DataFrame:
    """Joint Holm across Phase 1 ∪ Phase 3 ∪ Phase 4.

    Authoritative PASS scope per CLAUDE.md #6 (extended 2026-04-22).
    Phase 4 MVP = 2 strategies → N=9.
    """
    p1_df = pd.read_parquet(phase1_returns_path)
    p3_df = pd.read_parquet(phase3_returns_path)
    combined = {
        c: p1_df[c].dropna()
        for c in p1_df.columns if c != "__benchmark__"
    }
    combined.update({
        c: p3_df[c].dropna()
        for c in p3_df.columns if c != "__benchmark__"
    })
    combined.update(phase4_returns)
    logger.info(
        "Joint Holm scope: %d strategies (Phase 1 + Phase 3 + Phase 4)",
        len(combined),
    )
    results = evaluate_gate(combined, bench, config=config)
    rows = []
    for name, r in results.items():
        if name in p1_df.columns:
            phase = "phase1"
        elif name in p3_df.columns:
            phase = "phase3"
        else:
            phase = "phase4"
        rows.append({
            "phase": phase,
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
                        help="Smoke: restrict universe to N recent members.")
    parser.add_argument("--price-start", default="2005-01-01")
    args = parser.parse_args()

    t_start = time.monotonic()
    returns, bench, audit = run_phase4(
        limit_tickers=args.limit_tickers,
        price_start=args.price_start,
    )

    artifact_name = "phase4_smoke" if args.limit_tickers else "phase4"
    save_phase_returns(artifact_name, returns, bench)

    # Save audit JSON alongside
    audit_path = AUDIT_DIR / f"{artifact_name}_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump({
            k: {kk: vv for kk, vv in v.items() if kk != "feature_importances_by_fold"}
            for k, v in audit.items()
        }, f, indent=2, default=str)
    logger.info("Audit saved: %s", audit_path)

    # Within-phase Holm
    logger.info("Evaluating within-phase Holm (EXPLORATORY)...")
    within_results = evaluate_gate(returns, bench)
    print_gate_table(
        within_results,
        f"Phase 4 — WITHIN-PHASE Holm(N={len(returns)}) — EXPLORATORY"
        f" (universe={'subset ' + str(args.limit_tickers) if args.limit_tickers else 'full S&P 500'})",
    )

    # Joint Holm across Phase 1 + Phase 3 + Phase 4 (authoritative)
    p1_path = ARTIFACTS_DIR / "phase1_returns.parquet"
    p3_path = ARTIFACTS_DIR / "phase3_returns.parquet"
    if p1_path.exists() and p3_path.exists():
        logger.info("Evaluating joint Holm(Phase1 ∪ Phase3 ∪ Phase4)...")
        joint = joint_holm_full(returns, bench, p1_path, p3_path, load_config())
        print("\n" + "=" * 120)
        print(f"Joint Holm Phase 1 + Phase 3 + Phase 4 (N={len(joint)}) — AUTHORITATIVE")
        print("=" * 120)
        print(joint.to_string(index=False))
    else:
        logger.warning("phase1 or phase3 returns missing — skipping joint Holm")

    logger.info("Phase 4 total wall time: %.1fs", time.monotonic() - t_start)


if __name__ == "__main__":
    main()
