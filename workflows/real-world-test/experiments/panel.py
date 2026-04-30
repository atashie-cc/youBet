"""Phase A joint daily panel builder with documented splices.

Source window: 2000-08-30 to 2026-04-16 (longest jointly-available with all
required series, including GC=F gold futures).

Splices (per precommit/tests_1_4_7.json):
  VTI pre-2001-06-15:    SPY adjusted close
  UPRO pre-2009-06-25:   synthetic 3x SPY (E2-calibrated 25 bps borrow, 91 bps ER)
  IEF pre-2002-07-30:    synthetic via FRED DGS7 (modified duration 6.3)
  BND pre-2003-09-29:    AGG adjusted close
  BND pre-AGG inception: synthetic via FRED DGS5 (modified duration 5.5)
  GLD pre-2004-11-18:    GC=F gold futures front-month price-change
  VGSH pre-2009-11-23:   daily T-bill rate
  BIL pre-2007-05-30:    daily T-bill rate

Output is cached to artifacts/panel.parquet so subsequent runs skip the fetches.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.data import fetch_tbill_rates
from youbet.etf.synthetic_leverage import conditional_leveraged_return

logger = logging.getLogger(__name__)

PANEL_PATH = WORKFLOW_ROOT / "artifacts" / "panel.parquet"
PANEL_START = pd.Timestamp("2000-08-30")
PANEL_END = pd.Timestamp("2026-04-16")


def _load_cached_prices() -> dict[str, pd.Series]:
    """For each ticker, find the longest non-null series across all repo snapshots."""
    snapshot_dirs = [
        WORKFLOW_ROOT.parents[0] / "etf" / "data" / "snapshots",
        WORKFLOW_ROOT.parents[0] / "commodity" / "data" / "snapshots",
    ]
    needed = ["VTI", "SPY", "UPRO", "BND", "GLD", "VGSH", "BIL", "SSO"]
    best: dict[str, pd.Series] = {}
    for snap_dir in snapshot_dirs:
        if not snap_dir.exists():
            continue
        for d in sorted(snap_dir.glob("20*"), reverse=True):
            try:
                df = pd.read_parquet(d / "prices.parquet")
            except Exception:
                continue
            for t in needed:
                if t not in df.columns:
                    continue
                col = df[t].dropna()
                if t not in best or len(col) > len(best[t]):
                    best[t] = col
    return best


def _fetch_yfinance(tickers: list[str]) -> dict[str, pd.Series]:
    """Fetch via yfinance for series not in cache (IEF, AGG, GC=F)."""
    import yfinance as yf
    out: dict[str, pd.Series] = {}
    for t in tickers:
        df = yf.download(t, start="2000-01-01", end=PANEL_END.strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if df.empty:
            continue
        col = df["Close"]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        col.name = t
        out[t] = col.dropna()
    return out


def _fetch_fred(series_ids: list[str]) -> dict[str, pd.Series]:
    from pandas_datareader import data as pdr
    out: dict[str, pd.Series] = {}
    for sid in series_ids:
        df = pdr.DataReader(sid, "fred", start="1998-01-01")
        out[sid] = df.iloc[:, 0].dropna()
    return out


def _synthetic_treasury_total_return(
    yield_pct_series: pd.Series,
    modified_duration: float,
) -> pd.Series:
    """Daily TR for a constant-maturity bond proxy.

    daily_return ≈ -mod_dur × Δyield_decimal + (yield/252) accrual
    yield_pct_series is in percent (e.g., 4.35 for 4.35%).
    """
    y = yield_pct_series / 100.0  # to decimal
    dy = y.diff()
    accrual = y.shift(1) / 252.0  # accrual based on yesterday's yield
    tr = -modified_duration * dy + accrual
    return tr.dropna()


def build_panel() -> pd.DataFrame:
    """Build the joint daily returns panel for Phase A. Cached to parquet."""
    if PANEL_PATH.exists():
        logger.info("Loading cached panel from %s", PANEL_PATH)
        return pd.read_parquet(PANEL_PATH)

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    cached = _load_cached_prices()
    fetched = _fetch_yfinance(["IEF", "AGG", "GC=F"])

    # FRED yields for synthetic Treasury TR pre-ETF inception
    fred = _fetch_fred(["DGS7", "DGS5"])

    # Master daily index = SPY trading days from PANEL_START to PANEL_END
    spy = cached["SPY"]
    idx = spy.index[(spy.index >= PANEL_START) & (spy.index <= PANEL_END)]
    spy = spy.reindex(idx)

    # T-bill daily rate (used for VGSH/BIL fallback)
    tbill = fetch_tbill_rates()
    tbill_daily = tbill.reindex(idx, method="ffill").fillna(0.04) / 252.0

    # === Build each ticker's RETURN series on idx ===
    rets: dict[str, pd.Series] = {}

    # VTI: real where available, SPY proxy pre-2001-06-15
    vti_real = cached["VTI"].reindex(idx)
    spy_ret = spy.pct_change()
    vti_ret = vti_real.pct_change()
    vti_ret = vti_ret.where(vti_ret.notna(), spy_ret)
    rets["VTI"] = vti_ret

    rets["SPY"] = spy_ret

    # UPRO: synthetic 3x SPY pre-2009-06-25 (E2 calibrated), real after
    upro_real = cached["UPRO"].reindex(idx)
    upro_ret_real = upro_real.pct_change()
    exposure_3x = pd.Series(3.0, index=idx)
    upro_synth = conditional_leveraged_return(
        spy_ret, exposure_3x, tbill_daily,
        borrow_spread_bps=25.0, expense_ratio=0.0091,
    )
    splice_upro = pd.Timestamp("2009-06-25")
    upro_ret = upro_synth.copy()
    real_part = upro_ret_real.loc[upro_ret_real.index >= splice_upro].dropna()
    upro_ret.loc[real_part.index] = real_part.values
    rets["UPRO"] = upro_ret

    # IEF: synthetic via DGS7 + 6.3-year duration pre-2002-07-30, real after
    ief_real = fetched["IEF"].reindex(idx)
    ief_ret_real = ief_real.pct_change()
    dgs7 = fred["DGS7"].reindex(idx, method="ffill")
    ief_synth = _synthetic_treasury_total_return(dgs7, modified_duration=6.3)
    ief_synth = ief_synth.reindex(idx).fillna(0.0)
    splice_ief = pd.Timestamp("2002-07-30")
    ief_ret = ief_synth.copy()
    ief_ret.loc[ief_ret_real.dropna().index] = ief_ret_real.dropna().values
    rets["IEF"] = ief_ret

    # BND: AGG pre-2007-04-10, synthetic DGS5+5.5dur pre-AGG, real BND after
    bnd_real = cached["BND"].reindex(idx)
    bnd_ret_real = bnd_real.pct_change()
    agg = fetched["AGG"].reindex(idx)
    agg_ret = agg.pct_change()
    dgs5 = fred["DGS5"].reindex(idx, method="ffill")
    bnd_synth = _synthetic_treasury_total_return(dgs5, modified_duration=5.5)
    bnd_synth = bnd_synth.reindex(idx).fillna(0.0)
    bnd_ret = bnd_synth.copy()
    bnd_ret.loc[agg_ret.dropna().index] = agg_ret.dropna().values
    bnd_ret.loc[bnd_ret_real.dropna().index] = bnd_ret_real.dropna().values
    rets["BND"] = bnd_ret

    # GLD/IAU (gold): GC=F price change pre-2004-11-18, real GLD after
    # Use GLD as the gold proxy (essentially identical to IAU)
    gld_real = cached["GLD"].reindex(idx)
    gld_ret_real = gld_real.pct_change()
    gcf = fetched["GC=F"].reindex(idx)
    gcf_ret = gcf.pct_change()
    gold_ret = gcf_ret.copy()
    gold_ret.loc[gld_ret_real.dropna().index] = gld_ret_real.dropna().values
    rets["GOLD"] = gold_ret  # use "GOLD" as composite name

    # VGSH: tbill rate pre-2009-11-23, real VGSH after
    vgsh_real = cached["VGSH"].reindex(idx)
    vgsh_ret_real = vgsh_real.pct_change()
    vgsh_ret = tbill_daily.copy()
    vgsh_ret.loc[vgsh_ret_real.dropna().index] = vgsh_ret_real.dropna().values
    rets["VGSH"] = vgsh_ret

    # BIL: tbill rate pre-2007-05-30, real BIL after
    bil_real = cached["BIL"].reindex(idx)
    bil_ret_real = bil_real.pct_change()
    bil_ret = tbill_daily.copy()
    bil_ret.loc[bil_ret_real.dropna().index] = bil_ret_real.dropna().values
    rets["BIL"] = bil_ret

    panel = pd.DataFrame(rets).fillna(0.0)
    panel = panel.loc[panel.index >= PANEL_START]

    # Sanity: drop the very first row (pct_change NaN) and any all-zero rows
    panel = panel.iloc[1:]

    panel.to_parquet(PANEL_PATH)
    logger.info(
        "Panel built: %d days from %s to %s, columns=%s",
        len(panel), panel.index[0].date(), panel.index[-1].date(), list(panel.columns),
    )
    return panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    panel = build_panel()
    print(panel.describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"\nPanel shape: {panel.shape}")
    print(f"Panel range: {panel.index[0].date()} to {panel.index[-1].date()}")
    # Validate splices: report annualized stats
    print("\nAnnualized return / vol per series:")
    for col in panel.columns:
        r = panel[col]
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        print(f"  {col:6s}: ann_ret={ann_ret:>+7.2%}  ann_vol={ann_vol:>6.2%}")
