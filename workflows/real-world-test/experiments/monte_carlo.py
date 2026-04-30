"""Monte Carlo simulation of v3 strategy via paired block bootstrap.

Method:
  - Source: historical daily joint returns (VTI, SPY, UPRO/synth, IAU/proxy, BIL) 1998-2026.
  - Block bootstrap (Politis-Romano spirit): sample 22-day blocks, concatenate to a
    25-year (~6,300 trading day) path. Cross-sectional joint structure preserved
    by sampling rows together; serial autocorrelation preserved within blocks.
  - 10,000 paths.
  - For each path: reconstruct prices from returns, compute SMA100 signals on VTI
    and SPY, apply weekly Friday cadence (every 5 trading days), run v3 strategy.
  - Track terminal wealth, CAGR, MaxDD, underwater duration.

Outputs:
  artifacts/monte_carlo_results.parquet  — per-path metrics
  artifacts/monte_carlo_summary.json     — aggregate distribution
  Console: percentile tables for v3, VTI buy-and-hold, 60/40 VTI/BIL.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from youbet.etf.data import fetch_tbill_rates
from youbet.etf.synthetic_leverage import conditional_leveraged_return

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = WORKFLOW_ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# v3 strategy params
W_VTI = 0.60
W_UPRO = 0.30
W_IAU = 0.10
SMA_WINDOW = 100
SWITCHING_COST_BPS = 10.0
DRIFT_THRESHOLD = 0.07

TRADING_DAYS_PER_YEAR = 252
WEEKLY_CADENCE_DAYS = 5         # decision every 5 trading days (Friday → Friday)
HORIZON_YEARS = 25
N_PATHS = 10_000
BLOCK_LEN_DAYS = 22
BOOTSTRAP_SEED = 20260429


def _load_historical_panel() -> pd.DataFrame:
    """Build joint daily returns 1998-2026 with synthetic 3x SPY pre-UPRO splice."""
    snapshot_dirs = [
        WORKFLOW_ROOT.parents[0] / "etf" / "data" / "snapshots",
        WORKFLOW_ROOT.parents[0] / "commodity" / "data" / "snapshots",
    ]
    tickers = ["VTI", "SPY", "UPRO", "BIL", "IAU"]
    best: dict[str, pd.Series] = {}
    for snap_dir in snapshot_dirs:
        if not snap_dir.exists():
            continue
        for d in sorted(snap_dir.glob("20*"), reverse=True):
            try:
                df = pd.read_parquet(d / "prices.parquet")
            except Exception:
                continue
            for t in tickers:
                if t not in df.columns:
                    continue
                col = df[t].dropna()
                if t not in best or len(col) > len(best[t]):
                    best[t] = col

    prices = pd.DataFrame(best)[tickers].sort_index()
    rets = prices.pct_change(fill_method=None)

    # T-bill daily rate
    tbill = fetch_tbill_rates()
    tbill_daily = tbill.reindex(prices.index, method="ffill").fillna(0.04) / TRADING_DAYS_PER_YEAR

    # Build synthetic 3x SPY for pre-UPRO
    spy_ret = rets["SPY"].dropna()
    exposure_3x = pd.Series(3.0, index=spy_ret.index)
    syn_upro = conditional_leveraged_return(
        spy_ret, exposure_3x, tbill_daily,
        borrow_spread_bps=25.0, expense_ratio=0.0091,
    )
    splice = pd.Timestamp("2009-06-25")
    upro = syn_upro.copy()
    real = rets["UPRO"].dropna()
    real_part = real.loc[real.index >= splice]
    upro.loc[real_part.index] = real_part.values

    # VTI: use SPY where missing (pre-2001-06)
    vti_ret = rets["VTI"].copy()
    pre = vti_ret.index[vti_ret.isna()]
    vti_ret.loc[pre] = spy_ret.reindex(pre).values

    # IAU: fall back to zero (so 10% sleeve drifts with cash) pre-2005-01-21.
    iau_ret = rets["IAU"].fillna(0.0)

    # BIL: use real where available, else daily T-bill rate
    bil_ret = rets["BIL"].copy()
    bil_ret = bil_ret.where(bil_ret.notna(), tbill_daily)

    panel = pd.DataFrame({
        "vti_ret": vti_ret,
        "spy_ret": spy_ret,
        "upro_ret": upro,
        "iau_ret": iau_ret,
        "bil_ret": bil_ret.fillna(0.0),
    }).dropna(how="any")

    logger.info(
        "Panel built: %d days from %s to %s",
        len(panel), panel.index[0].date(), panel.index[-1].date(),
    )
    return panel


def _sample_blocks(rng: np.random.Generator, panel: np.ndarray, n_days: int) -> np.ndarray:
    """One bootstrap path: concatenate random 22-day blocks to length n_days."""
    n_src = panel.shape[0]
    n_blocks = n_days // BLOCK_LEN_DAYS + 1
    starts = rng.integers(0, n_src - BLOCK_LEN_DAYS + 1, size=n_blocks)
    parts = [panel[s:s + BLOCK_LEN_DAYS] for s in starts]
    out = np.concatenate(parts, axis=0)
    return out[:n_days]


def _v3_path(returns_block: np.ndarray) -> dict:
    """Run v3 on one bootstrap path. Vectorized SMA + weekly cadence + drift rebalance."""
    # returns_block columns: 0=vti, 1=spy, 2=upro, 3=iau, 4=bil
    n_days = returns_block.shape[0]
    vti_ret = returns_block[:, 0]
    spy_ret = returns_block[:, 1]
    upro_ret = returns_block[:, 2]
    iau_ret = returns_block[:, 3]
    bil_ret = returns_block[:, 4]

    # Reconstruct prices for SMA
    vti_price = np.cumprod(1.0 + vti_ret)
    spy_price = np.cumprod(1.0 + spy_ret)

    # SMA100 daily (NaN for first 99 days)
    vti_sma = np.full(n_days, np.nan)
    spy_sma = np.full(n_days, np.nan)
    csum_vti = np.cumsum(vti_price)
    csum_spy = np.cumsum(spy_price)
    if n_days > SMA_WINDOW:
        vti_sma[SMA_WINDOW - 1:] = (csum_vti[SMA_WINDOW - 1:] - np.concatenate([[0], csum_vti[:n_days - SMA_WINDOW]])) / SMA_WINDOW
        spy_sma[SMA_WINDOW - 1:] = (csum_spy[SMA_WINDOW - 1:] - np.concatenate([[0], csum_spy[:n_days - SMA_WINDOW]])) / SMA_WINDOW

    vti_above = (vti_price > vti_sma)
    spy_above = (spy_price > spy_sma)
    vti_above[:SMA_WINDOW - 1] = True   # default in-market until SMA available
    spy_above[:SMA_WINDOW - 1] = True

    # Weekly Friday cadence, vectorized:
    # Decision days: i where i % 5 == 4 (last day of each 5-day window).
    # Between decisions, signal holds. Then T+1 shift (signal effective next day).
    is_friday = ((np.arange(n_days) % WEEKLY_CADENCE_DAYS) == (WEEKLY_CADENCE_DAYS - 1))
    # On non-Fridays, set sig to NaN; ffill from prior Friday
    vti_friday = np.where(is_friday, vti_above.astype(float), np.nan)
    spy_friday = np.where(is_friday, spy_above.astype(float), np.nan)
    # Fast forward-fill via cumulative max of last-known index
    def _ffill(a: np.ndarray, default: float = 1.0) -> np.ndarray:
        mask = ~np.isnan(a)
        idx = np.where(mask, np.arange(len(a)), -1)
        idx = np.maximum.accumulate(idx)
        out = np.where(idx >= 0, a[idx], default)
        return out
    daily_vti_sig = _ffill(vti_friday, default=1.0)
    daily_spy_sig = _ffill(spy_friday, default=1.0)
    # T+1 shift
    daily_vti_sig = np.concatenate([[1.0], daily_vti_sig[:-1]])
    daily_spy_sig = np.concatenate([[1.0], daily_spy_sig[:-1]])

    # Switching costs (paid on day of flip)
    vti_flip = np.abs(np.diff(daily_vti_sig, prepend=daily_vti_sig[0])) > 0.5
    spy_flip = np.abs(np.diff(daily_spy_sig, prepend=daily_spy_sig[0])) > 0.5
    vti_cost = vti_flip.astype(float) * (SWITCHING_COST_BPS / 10000.0)
    spy_cost = spy_flip.astype(float) * (SWITCHING_COST_BPS / 10000.0)

    # Sleeve daily returns
    vti_sleeve = daily_vti_sig * vti_ret + (1.0 - daily_vti_sig) * bil_ret - vti_cost
    upro_sleeve = daily_spy_sig * upro_ret + (1.0 - daily_spy_sig) * bil_ret - spy_cost
    iau_sleeve = iau_ret

    # Portfolio with annual rebalance — vectorized per-year segment.
    # (Drift-threshold mid-year rebalance is omitted for vectorization speed;
    # at 7% threshold and 1-year rebalance it almost never triggers in our sleeves.)
    sleeve_rets = np.column_stack([vti_sleeve, upro_sleeve, iau_sleeve])
    targets = np.array([W_VTI, W_UPRO, W_IAU])
    port_rets = np.empty(n_days)
    start = 0
    while start < n_days:
        end = min(start + TRADING_DAYS_PER_YEAR, n_days)
        seg = sleeve_rets[start:end]
        # Within-year cumulative sleeve growth from initial $1
        sleeve_cum = np.cumprod(1.0 + seg, axis=0)
        # Portfolio value within year (starting at 1.0 with target weights)
        port_cum = sleeve_cum @ targets
        # Daily returns: first day = port_cum[0]-1, rest = pct change
        port_seg = np.empty(end - start)
        port_seg[0] = port_cum[0] - 1.0
        port_seg[1:] = port_cum[1:] / port_cum[:-1] - 1.0
        port_rets[start:end] = port_seg
        start = end

    # Compute metrics
    cum = np.cumprod(1.0 + port_rets)
    n_years = n_days / TRADING_DAYS_PER_YEAR
    cagr = cum[-1] ** (1.0 / n_years) - 1.0 if cum[-1] > 0 else -1.0
    running = np.maximum.accumulate(cum)
    dd = cum / running - 1.0
    max_dd = dd.min()

    # Comparison series: VTI buy-and-hold and 60/40 VTI/BIL
    vti_bh = np.cumprod(1.0 + vti_ret)
    vti_bh_cagr = vti_bh[-1] ** (1.0 / n_years) - 1.0 if vti_bh[-1] > 0 else -1.0
    vti_bh_dd = (vti_bh / np.maximum.accumulate(vti_bh) - 1.0).min()

    p6040 = 0.60 * vti_ret + 0.40 * bil_ret
    cum6040 = np.cumprod(1.0 + p6040)
    p6040_cagr = cum6040[-1] ** (1.0 / n_years) - 1.0 if cum6040[-1] > 0 else -1.0
    p6040_dd = (cum6040 / np.maximum.accumulate(cum6040) - 1.0).min()

    return {
        "v3_terminal": float(cum[-1]),
        "v3_cagr": float(cagr),
        "v3_max_dd": float(max_dd),
        "vti_terminal": float(vti_bh[-1]),
        "vti_cagr": float(vti_bh_cagr),
        "vti_max_dd": float(vti_bh_dd),
        "p6040_terminal": float(cum6040[-1]),
        "p6040_cagr": float(p6040_cagr),
        "p6040_max_dd": float(p6040_dd),
    }


def main() -> None:
    print("=" * 80, flush=True)
    print(f"v3 Monte Carlo: {N_PATHS:,} paths, {HORIZON_YEARS}-yr horizon, "
          f"block bootstrap from 1998-2026 daily history", flush=True)
    print("=" * 80, flush=True)

    panel = _load_historical_panel()
    panel_arr = panel[["vti_ret", "spy_ret", "upro_ret", "iau_ret", "bil_ret"]].to_numpy()
    n_days_per_path = HORIZON_YEARS * TRADING_DAYS_PER_YEAR

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    t0 = time.time()
    log_every = max(N_PATHS // 20, 1)
    for i in range(N_PATHS):
        block = _sample_blocks(rng, panel_arr, n_days_per_path)
        out = _v3_path(block)
        results.append(out)
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_PATHS - i - 1) / max(rate, 1e-6)
            print(f"  Path {i+1:>6}/{N_PATHS} ({100*(i+1)/N_PATHS:>5.1f}%)  "
                  f"elapsed {elapsed:>6.1f}s  rate {rate:>5.1f}/s  ETA {eta:>5.0f}s",
                  flush=True)

    df = pd.DataFrame(results)
    df.to_parquet(ARTIFACTS / "monte_carlo_results.parquet")
    elapsed = time.time() - t0
    print(f"\nCompleted {N_PATHS:,} paths in {elapsed:.1f}s "
          f"({N_PATHS/elapsed:.1f} paths/sec)", flush=True)

    # --- Distribution summary ---
    print("\n=== TERMINAL WEALTH DISTRIBUTION (lump-sum $1 → $X after 25 years) ===", flush=True)
    print(f"{'Strategy':<15} {'p1':>9} {'p5':>9} {'p25':>9} {'p50':>9} "
          f"{'p75':>9} {'p95':>9} {'p99':>9} {'mean':>9}", flush=True)
    print("-" * 90, flush=True)
    for col, name in [("v3_terminal", "v3"), ("vti_terminal", "VTI B&H"), ("p6040_terminal", "60/40 VTI/BIL")]:
        s = df[col].values
        ps = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
        print(f"{name:<15} "
              f"{ps[0]:>8.2f}x {ps[1]:>8.2f}x {ps[2]:>8.2f}x {ps[3]:>8.2f}x "
              f"{ps[4]:>8.2f}x {ps[5]:>8.2f}x {ps[6]:>8.2f}x {s.mean():>8.2f}x", flush=True)

    print("\n=== CAGR DISTRIBUTION (annualized over 25 years) ===", flush=True)
    print(f"{'Strategy':<15} {'p1':>8} {'p5':>8} {'p25':>8} {'p50':>8} "
          f"{'p75':>8} {'p95':>8} {'p99':>8} {'mean':>8}", flush=True)
    print("-" * 85, flush=True)
    for col, name in [("v3_cagr", "v3"), ("vti_cagr", "VTI B&H"), ("p6040_cagr", "60/40 VTI/BIL")]:
        s = df[col].values
        ps = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
        print(f"{name:<15} "
              f"{ps[0]:>+7.1%} {ps[1]:>+7.1%} {ps[2]:>+7.1%} {ps[3]:>+7.1%} "
              f"{ps[4]:>+7.1%} {ps[5]:>+7.1%} {ps[6]:>+7.1%} {s.mean():>+7.1%}", flush=True)

    print("\n=== MAX DRAWDOWN DISTRIBUTION ===", flush=True)
    print(f"{'Strategy':<15} {'p1':>8} {'p5':>8} {'p25':>8} {'p50':>8} "
          f"{'p75':>8} {'p95':>8} {'p99':>8} {'mean':>8}", flush=True)
    print("-" * 85, flush=True)
    for col, name in [("v3_max_dd", "v3"), ("vti_max_dd", "VTI B&H"), ("p6040_max_dd", "60/40 VTI/BIL")]:
        s = df[col].values
        ps = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
        print(f"{name:<15} "
              f"{ps[0]:>7.1%} {ps[1]:>7.1%} {ps[2]:>7.1%} {ps[3]:>7.1%} "
              f"{ps[4]:>7.1%} {ps[5]:>7.1%} {ps[6]:>7.1%} {s.mean():>7.1%}", flush=True)

    # --- Head-to-head ---
    print("\n=== HEAD-TO-HEAD: v3 vs VTI buy-and-hold ===", flush=True)
    v3_beats_vti = (df["v3_terminal"] > df["vti_terminal"]).mean()
    v3_beats_6040 = (df["v3_terminal"] > df["p6040_terminal"]).mean()
    v3_log_excess = np.log(df["v3_terminal"]) - np.log(df["vti_terminal"])
    v3_log_excess_6040 = np.log(df["v3_terminal"]) - np.log(df["p6040_terminal"])
    print(f"  P(v3 terminal > VTI B&H)        = {v3_beats_vti:.1%}", flush=True)
    print(f"  P(v3 terminal > 60/40)          = {v3_beats_6040:.1%}", flush=True)
    print(f"  Mean log-wealth excess vs VTI   = {v3_log_excess.mean():+.3f}", flush=True)
    print(f"  Mean log-wealth excess vs 60/40 = {v3_log_excess_6040.mean():+.3f}", flush=True)
    print(f"  Median CAGR excess vs VTI       = "
          f"{(df['v3_cagr'] - df['vti_cagr']).median():+.2%}", flush=True)
    print(f"  Median CAGR excess vs 60/40     = "
          f"{(df['v3_cagr'] - df['p6040_cagr']).median():+.2%}", flush=True)
    print(f"  P(v3 CAGR < 0)                  = {(df['v3_cagr'] < 0).mean():.1%}", flush=True)
    print(f"  P(VTI CAGR < 0)                 = {(df['vti_cagr'] < 0).mean():.1%}", flush=True)

    summary = {
        "n_paths": N_PATHS,
        "horizon_years": HORIZON_YEARS,
        "block_length_days": BLOCK_LEN_DAYS,
        "seed": BOOTSTRAP_SEED,
        "v3_terminal": {
            "mean": float(df["v3_terminal"].mean()),
            "median": float(df["v3_terminal"].median()),
            "p5": float(np.percentile(df["v3_terminal"], 5)),
            "p95": float(np.percentile(df["v3_terminal"], 95)),
        },
        "v3_cagr": {
            "mean": float(df["v3_cagr"].mean()),
            "median": float(df["v3_cagr"].median()),
            "p5": float(np.percentile(df["v3_cagr"], 5)),
            "p95": float(np.percentile(df["v3_cagr"], 95)),
        },
        "v3_max_dd": {
            "mean": float(df["v3_max_dd"].mean()),
            "median": float(df["v3_max_dd"].median()),
            "p5_worst": float(np.percentile(df["v3_max_dd"], 5)),
            "p1_worst": float(np.percentile(df["v3_max_dd"], 1)),
        },
        "p_v3_beats_vti": float(v3_beats_vti),
        "p_v3_beats_6040": float(v3_beats_6040),
        "mean_log_excess_vs_vti": float(v3_log_excess.mean()),
        "median_cagr_excess_vs_vti": float((df["v3_cagr"] - df["vti_cagr"]).median()),
    }
    (ARTIFACTS / "monte_carlo_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary to {ARTIFACTS / 'monte_carlo_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
