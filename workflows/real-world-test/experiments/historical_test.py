"""Historical backtest of v3 strategy across the longest available real sample.

Sample composition:
  - 1998-01-02 to 2009-06-24: synthetic 3x SPY (E2-calibrated, 25 bps borrow)
  - 2009-06-25 to 2026-04-16: real UPRO


VTI is buy-and-hold (with weekly SMA100 overlay) throughout (VTI inception 2001-05-24,
so for 1998-2001 we substitute SPY-as-VTI to avoid synthetic gap; this affects the
small head of the sample only).

IAU inception is 2005-01-21. For pre-IAU history we substitute the GLD-equivalent
gold spot return via $GOLDAMGBD228NLBM (LBMA Gold AM Fix from FRED, fetched separately
or proxied — we fall back to the GSCI gold sub-index where IAU is unavailable).
For simplicity here, we start the historical backtest at 2005-01-21 (IAU inception)
when running the full v3, and report a separate "no-gold" variant from 1998.

T-bill (BIL) inception 2007-05-25. For pre-2007, BIL is replaced by the daily
T-bill rate from FRED (fetched via fetch_tbill_rates).

Outputs:
  artifacts/historical_v3_full_returns.parquet
  artifacts/historical_summary.csv
  Console: comparison table + per-period (dot-com, GFC, COVID, 2022) breakdowns.
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
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from youbet.etf.data import fetch_prices, fetch_tbill_rates
from youbet.etf.synthetic_leverage import conditional_leveraged_return, sma_signal
from strategy import V3Config, metrics, run_v3

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = WORKFLOW_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def _build_synthetic_upro_pre_2009(
    spy_prices: pd.Series,
    spy_ret: pd.Series,
    tbill_daily: pd.Series,
) -> pd.Series:
    """Synthetic 3x SPY (always-on buy-hold leverage) pre-2009.

    The overlay logic in run_v3 will apply the SMA gate to this series. So this
    series should be the underlying leveraged-buy-hold returns — gating happens
    in the strategy.
    """
    exposure_3x = pd.Series(3.0, index=spy_ret.index)
    syn = conditional_leveraged_return(
        spy_ret,
        exposure_3x,
        tbill_daily,
        borrow_spread_bps=25.0,
        expense_ratio=0.0091,
    )
    return syn


def _splice(real: pd.Series, synthetic: pd.Series, splice_date: pd.Timestamp) -> pd.Series:
    """Use synthetic before splice_date, real on/after splice_date."""
    s = synthetic.copy()
    if real.empty:
        return s
    real_part = real.loc[real.index >= splice_date]
    s.loc[s.index >= splice_date] = real_part.reindex(s.loc[s.index >= splice_date].index).fillna(0.0)
    return s


def main() -> None:
    print("=" * 80, flush=True)
    print("v3 Historical Backtest — 1998-2026 (synthetic 3x pre-UPRO, real UPRO post)", flush=True)
    print("=" * 80, flush=True)

    tickers = ["VTI", "SPY", "UPRO", "BIL", "IAU"]
    snapshot_dirs = [
        WORKFLOW_ROOT.parents[0] / "etf" / "data" / "snapshots",
        WORKFLOW_ROOT.parents[0] / "commodity" / "data" / "snapshots",
    ]

    # For each ticker, pick the snapshot with the most non-null observations.
    best_per_ticker: dict[str, pd.Series] = {}
    for snap_dir in snapshot_dirs:
        if not snap_dir.exists():
            continue
        for date_dir in sorted(snap_dir.glob("20*"), reverse=True):
            try:
                df = pd.read_parquet(date_dir / "prices.parquet")
            except Exception:
                continue
            for t in tickers:
                if t not in df.columns:
                    continue
                col = df[t].dropna()
                if t not in best_per_ticker or len(col) > len(best_per_ticker[t]):
                    best_per_ticker[t] = col

    missing = [t for t in tickers if t not in best_per_ticker or len(best_per_ticker[t]) == 0]
    if missing:
        logger.warning("Tickers missing from cache, attempting yfinance fetch: %s", missing)
        extra = fetch_prices(missing, start="1998-01-01")
        for t in missing:
            if t in extra.columns:
                best_per_ticker[t] = extra[t].dropna()

    prices = pd.DataFrame(best_per_ticker)[tickers].copy()
    prices.index = pd.DatetimeIndex(prices.index)
    prices = prices.sort_index()

    tbill = fetch_tbill_rates()
    tbill_daily = tbill.reindex(prices.index, method="ffill").fillna(0.04) / 252.0

    print("\nData ranges:", flush=True)
    for t in tickers:
        s = prices[t].dropna()
        print(f"  {t:5s}: {s.index.min().date()} to {s.index.max().date()} ({len(s)} days)", flush=True)

    # --- Returns ---
    rets = prices.pct_change()
    spy_ret = rets["SPY"].dropna()
    vti_ret = rets["VTI"].dropna()

    # Synthetic UPRO pre-2009
    syn_upro = _build_synthetic_upro_pre_2009(prices["SPY"], spy_ret, tbill_daily)

    # Splice synthetic + real UPRO
    upro_ret = rets["UPRO"]
    splice_date = pd.Timestamp("2009-06-25")
    upro_full = syn_upro.copy()
    real_part = upro_ret.loc[upro_ret.index >= splice_date].dropna()
    upro_full.loc[real_part.index] = real_part.values

    # IAU: zero before 2005-01-21 (we'll start the with-gold variant at IAU inception)
    iau_ret = rets["IAU"].fillna(0.0)

    # BIL: use real returns when available, fall back to T-bill daily rate
    bil_ret = rets["BIL"].copy()
    bil_ret = bil_ret.where(bil_ret.notna(), tbill_daily)

    # VTI pre-inception (2001-05-24): use SPY as proxy for VTI sleeve
    vti_full = rets["VTI"].copy()
    pre_vti = vti_full.index[vti_full.isna()]
    vti_full.loc[pre_vti] = spy_ret.reindex(pre_vti).values
    vti_prices_full = prices["VTI"].copy()
    vti_prices_full = vti_prices_full.fillna(prices["SPY"])

    # --- Two variants ---
    # A) Full sample, no-gold (1998 → 2026); IAU sleeve held in BIL when IAU NaN
    # B) Post-IAU full v3 (2005-01-21 → 2026)

    config = V3Config()

    # Variant A: 1998-2026, no IAU
    print("\n--- Variant A: 1998-2026, no-gold (IAU sleeve held in BIL) ---", flush=True)
    config_no_gold = V3Config(w_vti=0.60, w_upro=0.30, w_iau=0.10)
    iau_proxy_no_gold = bil_ret  # 10% sleeve sits in BIL when IAU unavailable
    df_a = run_v3(
        vti_ret=vti_full,
        spy_prices=prices["SPY"],
        vti_prices=vti_prices_full,
        upro_ret=upro_full,
        iau_ret=iau_proxy_no_gold,
        bil_ret=bil_ret,
        config=config_no_gold,
    )

    # Variant B: 2005-2026, full v3 with real IAU
    start_b = pd.Timestamp("2005-01-21")
    mask_b_idx = vti_full.index[vti_full.index >= start_b]
    df_b = run_v3(
        vti_ret=vti_full.reindex(mask_b_idx),
        spy_prices=prices["SPY"].reindex(mask_b_idx),
        vti_prices=vti_prices_full.reindex(mask_b_idx),
        upro_ret=upro_full.reindex(mask_b_idx),
        iau_ret=iau_ret.reindex(mask_b_idx),
        bil_ret=bil_ret.reindex(mask_b_idx),
        config=config,
    )

    # Variant C: Real-only window 2009-06-25 → 2026 (pure real LETF)
    start_c = pd.Timestamp("2009-06-25")
    mask_c_idx = vti_full.index[vti_full.index >= start_c]
    df_c = run_v3(
        vti_ret=vti_full.reindex(mask_c_idx),
        spy_prices=prices["SPY"].reindex(mask_c_idx),
        vti_prices=vti_prices_full.reindex(mask_c_idx),
        upro_ret=upro_full.reindex(mask_c_idx),
        iau_ret=iau_ret.reindex(mask_c_idx),
        bil_ret=bil_ret.reindex(mask_c_idx),
        config=config,
    )

    # --- Benchmarks ---
    # VTI buy-and-hold (full sample)
    vti_bh_a = vti_full
    # 60/40 VTI/BND -- BND not in cache, use 60/40 VTI/BIL as a proxy
    vti60_bil40_a = 0.60 * vti_full + 0.40 * bil_ret

    # --- Metrics tables ---
    rows = []
    rows.append(metrics(df_a["port_return"], "v3 (1998-2026, no gold)"))
    rows.append(metrics(vti_bh_a, "VTI B&H (1998-2026)"))
    rows.append(metrics(vti60_bil40_a, "60/40 VTI/BIL (1998-2026)"))
    rows.append(metrics(df_b["port_return"], "v3 (2005-2026, with IAU)"))
    rows.append(metrics(vti_full.reindex(mask_b_idx), "VTI B&H (2005-2026)"))
    rows.append(metrics(df_c["port_return"], "v3 (2009-2026, real UPRO)"))
    rows.append(metrics(vti_full.reindex(mask_c_idx), "VTI B&H (2009-2026)"))

    summary = pd.DataFrame(rows)
    summary.to_csv(ARTIFACTS / "historical_summary.csv", index=False)

    print("\n=== HISTORICAL SUMMARY ===", flush=True)
    fmt = lambda x: f"{x:>+8.1%}" if isinstance(x, float) else str(x)
    print(
        f"{'Strategy':<35} {'Years':>6} {'CAGR':>8} {'Vol':>8} {'Sharpe':>7} "
        f"{'MaxDD':>9} {'Calmar':>7} {'Final$':>10}",
        flush=True,
    )
    print("-" * 100, flush=True)
    for r in rows:
        print(
            f"{r['name']:<35} {r['n_years']:>6.1f} {r['cagr']:>8.1%} {r['ann_vol']:>8.1%} "
            f"{r['sharpe']:>7.3f} {r['max_dd']:>9.1%} "
            f"{(r['calmar'] if r['calmar'] == r['calmar'] else 0):>7.2f} "
            f"{r['terminal_wealth']:>9.2f}x",
            flush=True,
        )

    # --- Crisis breakdown on Variant A (1998-2026 longest sample) ---
    print("\n=== CRISIS PERIOD BREAKDOWN (Variant A, 1998-2026) ===", flush=True)
    crises = [
        ("Dot-com bust", "2000-03-01", "2003-03-31"),
        ("GFC", "2007-10-01", "2009-06-30"),
        ("COVID crash", "2020-02-01", "2020-04-30"),
        ("2022 rate hike", "2022-01-01", "2022-12-31"),
        ("Full 2000-2013 (incl. lost decade)", "2000-01-01", "2013-01-01"),
    ]
    print(f"{'Crisis':<40} {'v3 ret':>10} {'v3 MDD':>10} {'VTI ret':>10} {'VTI MDD':>10}", flush=True)
    print("-" * 90, flush=True)
    for name, start, end in crises:
        v3_p = df_a["port_return"].loc[start:end]
        vti_p = vti_bh_a.loc[start:end]
        if len(v3_p) < 5:
            continue
        v3_cum = (1.0 + v3_p).cumprod()
        vti_cum = (1.0 + vti_p).cumprod()
        v3_dd = (v3_cum / v3_cum.cummax() - 1.0).min()
        vti_dd = (vti_cum / vti_cum.cummax() - 1.0).min()
        v3_total = v3_cum.iloc[-1] - 1.0
        vti_total = vti_cum.iloc[-1] - 1.0
        print(
            f"{name:<40} {v3_total:>+9.1%} {v3_dd:>+9.1%} {vti_total:>+9.1%} {vti_dd:>+9.1%}",
            flush=True,
        )

    # --- Save returns ---
    out = pd.DataFrame(
        {
            "v3_1998": df_a["port_return"],
            "vti_bh_1998": vti_bh_a,
            "v3_2005_with_gold": df_b["port_return"].reindex(df_a.index),
            "v3_2009_real_upro": df_c["port_return"].reindex(df_a.index),
        }
    )
    out.to_parquet(ARTIFACTS / "historical_v3_full_returns.parquet")
    print(f"\nSaved returns to {ARTIFACTS / 'historical_v3_full_returns.parquet'}", flush=True)

    # --- Overlay diagnostics ---
    print("\n=== OVERLAY DIAGNOSTICS (Variant A) ===", flush=True)
    n = len(df_a)
    print(f"  VTI overlay in-market: {df_a['vti_signal'].mean():.1%}", flush=True)
    print(f"  SPY overlay in-market: {df_a['spy_signal'].mean():.1%}", flush=True)
    print(f"  VTI overlay flips: {int((df_a['vti_signal'].diff().abs() > 0.5).sum())}", flush=True)
    print(f"  SPY overlay flips: {int((df_a['spy_signal'].diff().abs() > 0.5).sum())}", flush=True)


if __name__ == "__main__":
    main()
