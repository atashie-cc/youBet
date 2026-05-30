"""Phase 4 — macro x sector conditioning, at the survivorship-clean SPDR level.

CORE (gate-tested): vrp_defensive_tilt — overweight defensive sector SPDRs
(XLV/XLP/XLU) vs cyclicals (XLK/XLF/XLE/XLB/XLI/XLY) when the variance risk
premium (VIX - trailing realized vol) is elevated (above its expanding
top-tercile, PIT). Confirmatory inputs are market-derived not-revised ONLY
(VIX, realized vol). Monthly rebalance, T+1, flat ETF cost, benchmark SPY.

DESCRIPTIVE cross (NOT gated): the full 5 macro x 9 sector x 3 threshold grid,
reported as a whole-grid excess-Sharpe distribution with a max-statistic
permutation null (Romano-Wolf / White reality-check). No single cell is a
"finding" unless the best cell beats the permutation max-null. This is the
antidote to the Molchanov-Stangl sector-rotation data-snooping trap.

Pre-registered expectation: vrp_defensive_tilt FAILS the gate; the descriptive
best cell does NOT beat the max-null.

Usage:
    python workflows/individual-stocks-snp500/experiments/phase4_macro_sector.py
"""

from __future__ import annotations

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

from _shared import evaluate_gate, load_config, print_gate_table, save_phase_returns  # noqa: E402
from youbet.etf.data import fetch_prices  # noqa: E402
from youbet.etf.macro.fetchers import fetch_vix  # noqa: E402
from youbet.stock import macro_sector as ms  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = WORKFLOW_ROOT / "data" / "snapshots" / "sectors"
DEFENSIVE = ["XLV", "XLP", "XLU"]
CYCLICAL = ["XLK", "XLF", "XLE", "XLB", "XLI", "XLY"]
SECTORS = DEFENSIVE + CYCLICAL          # the original 9 SPDRs (full 2004+ history)
BENCH = "SPY"
COST_BPS = 2.0                          # one-way bps on rebalance turnover (ultra-liquid SPDRs)
DEFENSIVE_OVERWEIGHT = 0.60             # elevated-regime defensive total weight (pre-committed)


def _load_sector_prices(start="2004-01-01") -> pd.DataFrame:
    px = fetch_prices(SECTORS + [BENCH], start=start, snapshot_dir=DATA_DIR)
    return px.dropna(how="all")


def _monthly_rebal(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = index.to_series()
    return pd.DatetimeIndex(s.groupby(index.to_period("M")).first().values)


def _vrp_series(spy_close: pd.Series) -> pd.Series:
    vix = fetch_vix(start="2004-01-01").values
    spy_ret = spy_close.pct_change()
    return ms.variance_risk_premium(vix, spy_ret, realized_window=21).dropna()


def _equal_weights() -> pd.Series:
    return pd.Series(1.0 / len(SECTORS), index=SECTORS)


def _defensive_tilt_weights() -> pd.Series:
    w = pd.Series(0.0, index=SECTORS)
    w[DEFENSIVE] = DEFENSIVE_OVERWEIGHT / len(DEFENSIVE)
    w[CYCLICAL] = (1.0 - DEFENSIVE_OVERWEIGHT) / len(CYCLICAL)
    return w


def _overweight_one_weights(sector: str, w_target: float = 0.50) -> pd.Series:
    w = pd.Series((1.0 - w_target) / (len(SECTORS) - 1), index=SECTORS)
    w[sector] = w_target
    return w


def simulate(weights_by_rebal: dict[pd.Timestamp, pd.Series],
             returns: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.Series:
    """Daily strategy returns from a monthly weight schedule, T+1 + turnover cost.

    Weights decided at rebal date d are applied to returns from d+1 onward
    (strict T+1). Turnover cost charged on the rebal day.
    """
    daily_w = pd.DataFrame(index=returns.index, columns=SECTORS, dtype=float)
    for d, w in weights_by_rebal.items():
        if d in daily_w.index:
            daily_w.loc[d] = w.reindex(SECTORS).values
    daily_w = daily_w.ffill().shift(1)            # T+1: yesterday's weights drive today
    invested = daily_w.notna().any(axis=1)        # exclude pre-first-rebal days
    port = (daily_w * returns[SECTORS]).sum(axis=1).where(invested)
    # turnover cost on rebal days
    rebal_dates = sorted(weights_by_rebal)
    prev = pd.Series(0.0, index=SECTORS)
    for d in rebal_dates:
        w = weights_by_rebal[d].reindex(SECTORS).fillna(0.0)
        turnover = (w - prev).abs().sum()
        if d in port.index:
            port.loc[d] -= turnover * cost_bps / 1e4
        prev = w
    return port.dropna()


def _regime_elevated(pf_values: pd.Series, release: pd.Series,
                     rebal_dates, quantile: float) -> dict[pd.Timestamp, bool]:
    """PIT regime: at each rebal d, is the latest released value above the
    expanding `quantile` of values released strictly before d?"""
    out = {}
    rel = release.sort_index()
    for d in rebal_dates:
        avail_idx = rel.index[rel.values < np.datetime64(pd.Timestamp(d))]
        avail = pf_values.reindex(avail_idx).dropna()
        if len(avail) < 60:
            out[d] = False
            continue
        thr = avail.quantile(quantile)
        out[d] = bool(avail.iloc[-1] > thr)
    return out


def run_core(returns: pd.DataFrame, vrp: pd.Series, rebal_dates) -> pd.Series:
    # VRP as its own PITFeatureSeries-like (lag 0): release == observation date
    release = pd.Series(vrp.index, index=vrp.index)
    elevated = _regime_elevated(vrp, release, rebal_dates, quantile=2.0 / 3.0)
    n_elev = sum(elevated.values())
    logger.info("vrp_defensive_tilt: %d/%d rebal dates elevated", n_elev, len(rebal_dates))
    tilt, eq = _defensive_tilt_weights(), _equal_weights()
    wbr = {d: (tilt if elevated[d] else eq) for d in rebal_dates}
    return simulate(wbr, returns)


def run_descriptive_cross(returns: pd.DataFrame, signals: dict, rebal_dates,
                          spy_ret: pd.Series, seed: int = 42) -> dict:
    """5 macro x 9 sector x 3 threshold grid + max-statistic permutation null.

    Vectorized (gross, T+1): every cell's daily return is
    where(regime_elevated, overweight-S basket, equal-weight basket), so the
    whole grid + permutation null is O(cells) numpy ops. The descriptive
    layer is reported GROSS — the max-statistic null is apples-to-apples
    (same cost structure), so turnover cost does not change the relative
    comparison that guards against the 135-way search.
    """
    thresholds = {"top_tercile": 2.0 / 3.0, "median": 0.5, "top_quartile": 0.75}
    sec = returns[SECTORS]
    ew_ret = sec.mean(axis=1)                       # equal-weight basket (neutral)
    sum_all = sec.sum(axis=1)
    n_other = len(SECTORS) - 1
    ow_ret = {s: 0.5 * sec[s] + (0.5 / n_other) * (sum_all - sec[s])
              for s in SECTORS}                     # overweight-S basket
    common = returns.index.intersection(spy_ret.index)
    bench = spy_ret.reindex(common)

    def exsharpe(cell_ret: pd.Series) -> float:
        ex = (cell_ret.reindex(common) - bench).dropna()
        if len(ex) < 252 or ex.std() == 0:
            return float("nan")
        return float(ex.mean() / ex.std() * np.sqrt(252))

    rd = list(rebal_dates)

    def daily_mask(elev: dict) -> pd.Series:
        s = pd.Series(np.nan, index=returns.index)
        for d, e in elev.items():
            if d in s.index:
                s.loc[d] = 1.0 if e else 0.0
        return s.ffill().shift(1).fillna(0.0) > 0.5   # T+1

    monthly = {}                                      # (M,T) -> monthly label array
    masks = {}                                        # (M,T) -> daily bool mask
    for mname, (vals, rel) in signals.items():
        for tname, q in thresholds.items():
            elev = _regime_elevated(vals, rel, rd, q)
            monthly[(mname, tname)] = np.array([elev[d] for d in rd])
            masks[(mname, tname)] = daily_mask(elev)

    cells = []
    for (mname, tname), mask in masks.items():
        for s in SECTORS:
            cell_ret = ow_ret[s].where(mask, ew_ret)
            cells.append({"macro": mname, "threshold": tname, "sector": s,
                          "excess_sharpe": exsharpe(cell_ret)})
    grid = pd.DataFrame(cells).dropna(subset=["excess_sharpe"])
    observed_max = grid["excess_sharpe"].max()

    rng = np.random.default_rng(seed)
    n_perm = 200
    null_max = []
    for _ in range(n_perm):
        best = -np.inf
        for key, labels in monthly.items():
            perm = rng.permutation(labels)
            mask = daily_mask({rd[i]: bool(perm[i]) for i in range(len(rd))})
            for s in SECTORS:
                es = exsharpe(ow_ret[s].where(mask, ew_ret))
                if np.isfinite(es) and es > best:
                    best = es
        null_max.append(best)
    null_max = np.array(null_max)
    p_max = float((null_max >= observed_max).mean())
    return {"grid": grid, "observed_max": observed_max,
            "null_max_mean": float(null_max.mean()),
            "null_max_95": float(np.quantile(null_max, 0.95)), "p_max": p_max}


def main() -> None:
    config = load_config()
    t0 = time.monotonic()
    px = _load_sector_prices()
    returns = px.pct_change().dropna(how="all")
    spy_ret = returns[BENCH]
    rebal_dates = [d for d in _monthly_rebal(returns.index)
                   if d >= pd.Timestamp("2006-01-01")]  # ~1yr burn-in for expanding tercile
    logger.info("Sector prices %s, %d monthly rebal dates from %s",
                px.shape, len(rebal_dates), rebal_dates[0].date())

    vrp = _vrp_series(px[BENCH])
    core = run_core(returns, vrp, rebal_dates)
    save_phase_returns("phase4", {"vrp_defensive_tilt": core}, spy_ret.reindex(core.index))
    results = evaluate_gate({"vrp_defensive_tilt": core}, spy_ret.reindex(core.index))
    print_gate_table(results, "Phase 4 — vrp_defensive_tilt (CORE, gate member)")

    # Descriptive cross
    logger.info("Running descriptive macro x sector cross (135 cells + max-null)...")
    signals = {"vrp": (vrp, pd.Series(vrp.index, index=vrp.index))}
    for mname, sid in [("yield_3m10y", "T10Y3M"), ("breakeven_5y5y", "T5YIFR"),
                       ("real_rate_10y", "DFII10")]:
        pf = ms.fetch_market_series(mname, sid, start="2004-01-01")
        signals[mname] = (pf.values, pf.release_dates)
    baa = ms.fetch_credit_spread_baa10y(start="2004-01-01")
    signals["credit_baa10y"] = (baa.values, baa.release_dates)

    cross = run_descriptive_cross(returns, signals, rebal_dates, spy_ret)
    print("\n" + "=" * 90)
    print("Phase 4 — DESCRIPTIVE macro x sector cross (135 cells; NOT gated)")
    print("=" * 90)
    top = cross["grid"].sort_values("excess_sharpe", ascending=False).head(8)
    print(top.to_string(index=False))
    print(f"\n  Observed best-cell excess Sharpe : {cross['observed_max']:+.3f}")
    print(f"  Permutation max-null mean / 95%   : {cross['null_max_mean']:+.3f} / "
          f"{cross['null_max_95']:+.3f}")
    print(f"  Max-statistic p (best beats null) : {cross['p_max']:.3f}  "
          f"-> {'SIGNAL' if cross['p_max'] < 0.05 else 'indistinguishable from data-mining'}")
    logger.info("Phase 4 wall time %.1fs", time.monotonic() - t0)


if __name__ == "__main__":
    main()
