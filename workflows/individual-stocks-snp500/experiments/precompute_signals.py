"""Resumable, time-budgeted precompute of the PIT EV/EBITDA + earnings-yield
signal panel (date x ticker), so the heavy compute_fundamentals work is done
ONCE and the backtest becomes a fast lookup.

Each call processes monthly rebal dates chronologically until `--budget-sec`,
appends to artifacts/signal_panel.parquet, and exits 0 if ALL dates are done
else 2 (more to do). Re-invoke until exit 0. Survives the 10-min tool ceiling.

Market cap is computed CORRECTLY as last_price(<date) x PIT shares (NOT the
price-proxy fallback), so EV = mcap + net_debt is right.

Usage:
    python -u workflows/individual-stocks-snp500/experiments/precompute_signals.py --budget-sec 520
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
    ARTIFACTS_DIR, clear_fundamentals_cache, load_config, load_facts_by_ticker,
    load_prices_with_benchmark, load_sp500_universe,
)
from youbet.stock.fundamentals import compute_fundamentals  # noqa: E402
from youbet.stock.fwd_valuation import (  # noqa: E402
    _finite_pos, earnings_yield_pit, ebit_yield_pit, ebitda_pit, ebitda_yield_pit,
    enterprise_value_pit, net_debt_pit,
)

# Plausibility backstop: |EBITDA/EV| or |E/P| above this is a data error
# (EV/EBITDA < ~0.7x or P/E < ~0.7x is economically impossible for a going
# concern) — drop rather than rank. With correct (unadjusted-price) mcap these
# are rare; the guard catches residual share/price-vintage misalignment.
_MAX_PLAUSIBLE_YIELD = 1.5

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
PANEL_PATH = ARTIFACTS_DIR / "signal_panel.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-sec", type=int, default=520)
    ap.add_argument("--start", default="2009-07-01",
                    help="First rebal date to compute (buffers the 2010 first test fold).")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1,
                    help="Run N parallel workers over disjoint date subsets; "
                         "each writes signal_panel_shard{id}.parquet, merged later.")
    args = ap.parse_args()

    cfg = load_config()
    clear_fundamentals_cache()
    uni = load_sp500_universe()
    px = load_prices_with_benchmark(uni, start="2005-01-01", config=cfg)
    # UNADJUSTED close for market cap (R1 fix): adjusted prices x as-reported
    # shares understates mcap by the split factor (~28x for AAPL) and corrupts
    # EV/EBITDA. mcap must use the as-traded price on the date.
    unadj_path = ARTIFACTS_DIR / "unadj_close.parquet"
    if not unadj_path.exists():
        raise FileNotFoundError(
            f"{unadj_path} missing — run the unadjusted-close fetch first.")
    unadj = pd.read_parquet(unadj_path)
    unadj.index = pd.to_datetime(unadj.index)
    facts = load_facts_by_ticker(uni)
    facts = {t: f for t, f in facts.items() if t in px.columns}

    idx = px.index
    months = pd.DatetimeIndex(idx.to_series().groupby(idx.to_period("M")).first().values)
    rebal = [d for d in months if d >= pd.Timestamp(args.start)]

    import glob
    cols = ["date", "ticker", "evebitda_signal", "earnings_yield",
            "mcap", "net_debt", "ebitda", "ebit", "ttm_ni", "shares"]
    # Global done-set = every date already in the main panel OR any shard file
    # (so parallel shards never redo a completed date).
    existing = [PANEL_PATH] + [Path(p) for p in
                glob.glob(str(ARTIFACTS_DIR / "signal_panel_shard*.parquet"))]
    done: set = set()
    for fp in existing:
        if fp.exists():
            done |= set(pd.to_datetime(pd.read_parquet(fp, columns=["date"])["date"]).unique())
    todo_all = [d for d in rebal if d not in done]
    if args.n_shards > 1:
        my_todo = [d for i, d in enumerate(todo_all) if i % args.n_shards == args.shard_id]
        out_path = ARTIFACTS_DIR / f"signal_panel_shard{args.shard_id}.parquet"
    else:
        my_todo = todo_all
        out_path = PANEL_PATH
    panel = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=cols)
    print(f"rebal total={len(rebal)} done={len(done)} todo_all={len(todo_all)} "
          f"my_todo={len(my_todo)} -> {out_path.name}", flush=True)

    t0 = time.monotonic()
    new_rows = []
    processed = 0
    for d in my_todo:
        if time.monotonic() - t0 > args.budget_sec:
            break
        active = uni.active_as_of(d)
        avail_u = unadj.loc[unadj.index < d]   # UNADJUSTED price for mcap
        if avail_u.empty:
            continue
        last_u = avail_u.ffill().iloc[-1]
        for tk in active:
            f0 = facts.get(tk)
            if f0 is None:
                continue
            try:
                f = compute_fundamentals(f0, d)
            except Exception:  # noqa: BLE001
                continue
            shares = _finite_pos(f.get("shares_outstanding"))
            lp = _finite_pos(last_u.get(tk))    # as-traded close < d
            mcap = (lp * shares) if (shares and shares > 0 and lp and lp > 0) else None
            nd = net_debt_pit(f)
            ebitda = ebitda_pit(f)
            ebit = _finite_pos(f.get("ttm_operating_income"))
            ttm_ni = _finite_pos(f.get("ttm_net_income"))
            ev_y = ebitda_yield_pit(f, mcap)
            if ev_y is None:
                ev_y = ebit_yield_pit(f, mcap)
            ey = earnings_yield_pit(f, mcap)
            # Plausibility backstop (residual share/price-vintage misalignment)
            if ev_y is not None and abs(ev_y) > _MAX_PLAUSIBLE_YIELD:
                ev_y = None
            if ey is not None and abs(ey) > _MAX_PLAUSIBLE_YIELD:
                ey = None
            if ev_y is not None or ey is not None:
                new_rows.append((d, tk, ev_y, ey, mcap, nd, ebitda, ebit, ttm_ni, shares))
        processed += 1
        if processed % 8 == 0:
            # Incremental flush: progress persists even if `timeout` kills
            # this process before the loop's budget break (load time + loop
            # can exceed the wall timeout under CPU contention).
            if new_rows:
                panel = pd.concat([panel, pd.DataFrame(new_rows, columns=cols)],
                                  ignore_index=True)
                new_rows = []
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                panel.to_parquet(out_path)
            print(f"  ...{processed}/{len(my_todo)} dates ({time.monotonic()-t0:.0f}s), "
                  f"panel rows={len(panel)}", flush=True)

    if new_rows:
        add = pd.DataFrame(new_rows, columns=cols)
        panel = pd.concat([panel, add], ignore_index=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)

    remaining = len(my_todo) - processed
    print(f"processed {processed} dates this call in {time.monotonic()-t0:.0f}s; "
          f"remaining={remaining}; panel rows={len(panel)}", flush=True)
    sys.exit(0 if remaining == 0 else 2)


if __name__ == "__main__":
    main()
