"""Resumable, time-budgeted precompute of the Phase-8 quality-composite signal
panel (date × ticker -> composite score), so the heavy compute_fundamentals work
is done ONCE and survives the 8 GB box's OOM/timeout kills.

Each call processes monthly rebal dates chronologically until --budget-sec, appends
to artifacts/phase8_panel.parquet, exits 0 if ALL dates done else 2. Re-invoke until 0.

The composite score is computed CROSS-SECTIONALLY per date (z-scores need the full
cross-section), exactly as QualityComposite.score does, but materialised to a LUT so
the backtest is an O(1) lookup (PrecomputedScoreStrategy). MCAP-FREE -> no raw_prices.

Usage: python -u workflows/stock-selection/experiments/phase8_precompute.py --budget-sec 520
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (  # noqa: E402
    ARTIFACTS_DIR, load_config, load_prices_with_benchmark, load_sp500_universe,
)
from youbet.stock.edgar import EdgarConfig, IndexedFacts, get_company_facts  # noqa: E402
from youbet.stock.fundamentals import (  # noqa: E402
    TickerFundamentalsPanel, _clear_caches, compute_fundamentals,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
EDGAR_CACHE = WORKFLOW_ROOT / "data" / "snapshots" / "edgar"
PANEL = ARTIFACTS_DIR / "phase8_panel.parquet"


def load_facts(universe):
    cfg = EdgarConfig(cache_dir=EDGAR_CACHE)
    t2c = {}
    for _, row in universe.membership.sort_values("start_date").iterrows():
        c = str(row["cik"]).strip() if pd.notna(row["cik"]) else ""
        if c and c != "nan":
            t2c[row["ticker"]] = c.zfill(10)
    out = {}
    for t, cik in t2c.items():
        p = EDGAR_CACHE / f"CIK{cik}.parquet"
        if not p.exists():
            continue
        try:
            out[t] = TickerFundamentalsPanel.build(t, IndexedFacts(get_company_facts(cik, cfg)))
        except Exception:  # noqa: BLE001
            pass
    return out


def _zscore(s: pd.Series) -> pd.Series:
    s = s.dropna()
    mu, sd = s.mean(), s.std()
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def quality_scores_at_date(active, facts, d) -> dict:
    """Replicate QualityComposite.score: z(ROE)+z(GP/A)+z(magic-rank), >=2 of 3."""
    roe_by, gpa_by, mey_by, mroic_by = {}, {}, {}, {}
    for tk in active:
        f0 = facts.get(tk)
        if f0 is None:
            continue
        try:
            f = compute_fundamentals(f0, d)
        except Exception:  # noqa: BLE001
            continue
        roe = f.get("roe_ttm")
        if roe is not None and np.isfinite(roe):
            roe_by[tk] = float(roe)
        gp, assets = f.get("ttm_gross_profit"), f.get("total_assets")
        if gp is not None and assets is not None and np.isfinite(gp) and np.isfinite(assets) and assets > 0:
            gpa_by[tk] = float(gp) / float(assets)
        op, cash = f.get("ttm_operating_income"), f.get("cash")
        if op is not None and assets is not None and np.isfinite(op) and np.isfinite(assets) and assets > 0:
            invested = float(assets) - (float(cash) if cash is not None and np.isfinite(cash) else 0.0)
            mey_by[tk] = float(op) / float(assets)
            if invested > 0:
                mroic_by[tk] = float(op) / invested
    if not roe_by and not gpa_by and not mey_by:
        return {}
    z_roe, z_gpa = _zscore(pd.Series(roe_by)), _zscore(pd.Series(gpa_by))
    z_magic = pd.Series(dtype=float)
    ey, roic = pd.Series(mey_by), pd.Series(mroic_by)
    common = ey.index.intersection(roic.index)
    if len(common) > 0:
        magic = -(ey.loc[common].rank(ascending=False, method="min")
                  + roic.loc[common].rank(ascending=False, method="min"))
        z_magic = _zscore(magic)
    out = {}
    for t in z_roe.index.union(z_gpa.index).union(z_magic.index):
        parts = [z_roe.get(t), z_gpa.get(t), z_magic.get(t)]
        parts = [p for p in parts if p is not None and np.isfinite(p)]
        if len(parts) >= 2:
            out[t] = float(sum(parts)) / len(parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-sec", type=int, default=520)
    ap.add_argument("--start", default="2009-07-01")
    a = ap.parse_args()

    cfg = load_config()
    _clear_caches()
    uni = load_sp500_universe()
    px = load_prices_with_benchmark(uni, start="2005-01-01", config=cfg)
    facts = load_facts(uni)
    facts = {t: f for t, f in facts.items() if t in px.columns}

    idx = px.index
    months = pd.DatetimeIndex(idx.to_series().groupby(idx.to_period("M")).first().values)
    rebal = [d for d in months if d >= pd.Timestamp(a.start)]

    done = set()
    if PANEL.exists():
        done = set(pd.to_datetime(pd.read_parquet(PANEL, columns=["date"])["date"]).unique())
    todo = [d for d in rebal if d not in done]
    panel = pd.read_parquet(PANEL) if PANEL.exists() else pd.DataFrame(columns=["date", "ticker", "score"])
    print(f"rebal={len(rebal)} done={len(done)} todo={len(todo)}", flush=True)

    t0 = time.monotonic()
    rows, proc = [], 0
    for d in todo:
        if time.monotonic() - t0 > a.budget_sec:
            break
        active = uni.active_as_of(d)
        sc = quality_scores_at_date(active, facts, d)
        for tk, v in sc.items():
            rows.append((d, tk, v))
        proc += 1
        if proc % 8 == 0:
            if rows:
                panel = pd.concat([panel, pd.DataFrame(rows, columns=["date", "ticker", "score"])],
                                  ignore_index=True)
                rows = []
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                panel.to_parquet(PANEL)
            print(f"  ...{proc}/{len(todo)} ({time.monotonic()-t0:.0f}s) rows={len(panel)}", flush=True)
    if rows:
        panel = pd.concat([panel, pd.DataFrame(rows, columns=["date", "ticker", "score"])],
                          ignore_index=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(PANEL)
    rem = len(todo) - proc
    print(f"processed {proc} dates in {time.monotonic()-t0:.0f}s; remaining={rem}; rows={len(panel)}", flush=True)
    sys.exit(0 if rem == 0 else 2)


if __name__ == "__main__":
    main()
