"""Phase 13: Cross-Factor Rotation.

Instead of binary factor-vs-cash timing, allocate the factor budget
ACROSS factors based on trend state. When a factor is below its SMA,
its weight goes to the factors that are still trending — not to cash.

This addresses the key weakness of independent timing: bull-regime drag
from sitting in cash. By rotating among factors instead of exiting to
cash, the risk budget stays deployed in factor space.

Strategies tested:
  1. Equal-weight B&H: 25% each HML/SMB/RMW/CMA (benchmark)
  2. Independent SMA timing: each factor independently timed vs cash (current approach)
  3. SMA rotation: above-trend factors get equal weight, below-trend get zero
  4. Momentum rotation: rank by trailing 6m return, top-2 get 50% each
  5. Trend + momentum combo: above-trend factors ranked by momentum
  6. Inverse-vol rotation: above-trend factors weighted by 1/trailing_vol

Also tests: what fraction of alpha comes from timing (exiting losers)
vs selection (concentrating in winners)?
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import load_factors

from youbet.factor.data import fetch_international_factors, INTL_REGION_NAMES
from youbet.etf.risk import sharpe_ratio as compute_sharpe, cagr_from_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252
FACTOR_NAMES = ["HML", "SMB", "RMW", "CMA"]
SNAP_DIR = WORKFLOW_ROOT / "data" / "snapshots"


# ---------------------------------------------------------------------------
# Portfolio strategies (operate on multi-factor return DataFrame)
# ---------------------------------------------------------------------------

def equal_weight_bh(factors: pd.DataFrame, rf: pd.Series) -> pd.Series:
    """Equal-weight buy-and-hold across all 4 factors."""
    return factors[FACTOR_NAMES].mean(axis=1)


def independent_sma_timing(
    factors: pd.DataFrame, rf: pd.Series, window: int = 100
) -> pd.Series:
    """Each factor independently timed: above SMA → exposed, below → cash (RF).
    Portfolio = equal-weight average of timed factor returns."""
    timed = pd.DataFrame(index=factors.index)
    for f in FACTOR_NAMES:
        cum = (1 + factors[f]).cumprod()
        sma = cum.rolling(window, min_periods=window).mean()
        signal = (cum > sma).astype(float).shift(1).fillna(0)
        timed[f] = signal * factors[f] + (1 - signal) * rf
    return timed.mean(axis=1)


def sma_rotation(
    factors: pd.DataFrame, rf: pd.Series, window: int = 100
) -> pd.Series:
    """Above-trend factors get equal weight; below-trend get ZERO (not cash).
    Budget reallocates to trending factors. If none trending, go to cash."""
    daily_returns = pd.Series(0.0, index=factors.index)

    for f in FACTOR_NAMES:
        factors[f"_cum_{f}"] = (1 + factors[f]).cumprod()
        factors[f"_sma_{f}"] = factors[f"_cum_{f}"].rolling(window, min_periods=window).mean()

    for i in range(1, len(factors)):
        above_trend = []
        for f in FACTOR_NAMES:
            cum_val = factors[f"_cum_{f}"].iloc[i-1]
            sma_val = factors[f"_sma_{f}"].iloc[i-1]
            if not pd.isna(sma_val) and cum_val > sma_val:
                above_trend.append(f)

        if above_trend:
            weight = 1.0 / len(above_trend)
            ret = sum(factors[f].iloc[i] * weight for f in above_trend)
        else:
            ret = rf.iloc[i]  # all below trend → cash

        daily_returns.iloc[i] = ret

    # Clean up temp columns
    for f in FACTOR_NAMES:
        factors.drop(columns=[f"_cum_{f}", f"_sma_{f}"], inplace=True, errors="ignore")

    return daily_returns


def momentum_rotation(
    factors: pd.DataFrame, rf: pd.Series, lookback: int = 126, top_k: int = 2
) -> pd.Series:
    """Rank factors by trailing return, top-k get equal weight."""
    daily_returns = pd.Series(0.0, index=factors.index)

    for i in range(lookback + 1, len(factors)):
        trailing = {}
        for f in FACTOR_NAMES:
            trailing[f] = factors[f].iloc[i-lookback:i].sum()

        ranked = sorted(trailing.keys(), key=lambda x: trailing[x], reverse=True)
        selected = ranked[:top_k]
        weight = 1.0 / len(selected)
        daily_returns.iloc[i] = sum(factors[f].iloc[i] * weight for f in selected)

    return daily_returns


def trend_momentum_rotation(
    factors: pd.DataFrame, rf: pd.Series, sma_window: int = 100,
    mom_lookback: int = 126, top_k: int = 2
) -> pd.Series:
    """Filter by SMA trend, then rank by momentum. Top-k trending factors."""
    daily_returns = pd.Series(0.0, index=factors.index)

    for f in FACTOR_NAMES:
        factors[f"_cum_{f}"] = (1 + factors[f]).cumprod()
        factors[f"_sma_{f}"] = factors[f"_cum_{f}"].rolling(sma_window, min_periods=sma_window).mean()

    for i in range(max(sma_window, mom_lookback) + 1, len(factors)):
        # Filter: above SMA trend
        above_trend = []
        for f in FACTOR_NAMES:
            cum_val = factors[f"_cum_{f}"].iloc[i-1]
            sma_val = factors[f"_sma_{f}"].iloc[i-1]
            if not pd.isna(sma_val) and cum_val > sma_val:
                above_trend.append(f)

        if not above_trend:
            daily_returns.iloc[i] = rf.iloc[i]
            continue

        # Rank by momentum among trending factors
        trailing = {f: factors[f].iloc[i-mom_lookback:i].sum() for f in above_trend}
        ranked = sorted(trailing.keys(), key=lambda x: trailing[x], reverse=True)
        selected = ranked[:min(top_k, len(ranked))]
        weight = 1.0 / len(selected)
        daily_returns.iloc[i] = sum(factors[f].iloc[i] * weight for f in selected)

    for f in FACTOR_NAMES:
        factors.drop(columns=[f"_cum_{f}", f"_sma_{f}"], inplace=True, errors="ignore")

    return daily_returns


def invvol_rotation(
    factors: pd.DataFrame, rf: pd.Series, sma_window: int = 100,
    vol_lookback: int = 63
) -> pd.Series:
    """Above-trend factors weighted by inverse trailing vol."""
    daily_returns = pd.Series(0.0, index=factors.index)

    for f in FACTOR_NAMES:
        factors[f"_cum_{f}"] = (1 + factors[f]).cumprod()
        factors[f"_sma_{f}"] = factors[f"_cum_{f}"].rolling(sma_window, min_periods=sma_window).mean()
        factors[f"_vol_{f}"] = factors[f].rolling(vol_lookback, min_periods=vol_lookback // 2).std()

    for i in range(max(sma_window, vol_lookback) + 1, len(factors)):
        above_trend = {}
        for f in FACTOR_NAMES:
            cum_val = factors[f"_cum_{f}"].iloc[i-1]
            sma_val = factors[f"_sma_{f}"].iloc[i-1]
            vol_val = factors[f"_vol_{f}"].iloc[i-1]
            if not pd.isna(sma_val) and cum_val > sma_val and vol_val > 1e-8:
                above_trend[f] = 1.0 / vol_val

        if not above_trend:
            daily_returns.iloc[i] = rf.iloc[i]
            continue

        total_inv_vol = sum(above_trend.values())
        daily_returns.iloc[i] = sum(
            factors[f].iloc[i] * (w / total_inv_vol)
            for f, w in above_trend.items()
        )

    for f in FACTOR_NAMES:
        factors.drop(columns=[f"_cum_{f}", f"_sma_{f}", f"_vol_{f}"], inplace=True, errors="ignore")

    return daily_returns


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_max_dd(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return float(((cum - cum.cummax()) / cum.cummax()).min())


def full_metrics(returns: pd.Series, name: str) -> dict:
    r = returns.dropna()
    n = len(r)
    if n < 252:
        return {"name": name}
    sharpe = compute_sharpe(r)
    cagr = cagr_from_returns(r)
    dd = compute_max_dd(r)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    calmar = cagr / abs(dd) if abs(dd) > 1e-6 else 0
    return {"name": name, "sharpe": sharpe, "cagr": cagr, "max_dd": dd, "vol": vol, "calmar": calmar}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print("PHASE 13: CROSS-FACTOR ROTATION")
    print("=" * 110)
    print("\nQuestion: Does rotating among trending factors beat independent timing to cash?")

    factors = load_factors()
    rf = factors["RF"]

    # =====================================================================
    # US RESULTS
    # =====================================================================
    print("\n" + "=" * 110)
    print("US FACTOR ROTATION (1963-2026, 62 years)")
    print("=" * 110)

    strategies = {
        "EW Buy-and-Hold": equal_weight_bh(factors.copy(), rf),
        "Independent SMA Timing": independent_sma_timing(factors.copy(), rf),
        "SMA Rotation": sma_rotation(factors.copy(), rf),
        "Momentum Rotation (top-2)": momentum_rotation(factors.copy(), rf),
        "Trend+Momentum (top-2)": trend_momentum_rotation(factors.copy(), rf),
        "InvVol Rotation": invvol_rotation(factors.copy(), rf),
    }

    print(f"\n{'Strategy':<30} {'CAGR':>7} {'Sharpe':>8} {'Vol':>7} {'MaxDD':>8} {'Calmar':>8}")
    print("-" * 75)
    for name, ret in strategies.items():
        m = full_metrics(ret, name)
        if "sharpe" in m:
            print(f"{name:<30} {m['cagr']:>6.1%} {m['sharpe']:>7.3f} {m['vol']:>6.1%} "
                  f"{m['max_dd']:>7.1%} {m['calmar']:>7.3f}")

    # Excess vs EW B&H
    print(f"\n--- Excess Sharpe vs Equal-Weight Buy-and-Hold ---")
    bh_ret = strategies["EW Buy-and-Hold"]
    for name, ret in strategies.items():
        if name == "EW Buy-and-Hold":
            continue
        common = ret.index.intersection(bh_ret.index)
        excess = ret[common] - bh_ret[common]
        ex_sharpe = compute_sharpe(excess)
        print(f"  {name:<30}: ExSharpe = {ex_sharpe:>+.3f}")

    # =====================================================================
    # TIMING VS SELECTION DECOMPOSITION
    # =====================================================================
    print(f"\n" + "=" * 110)
    print("TIMING VS SELECTION DECOMPOSITION")
    print("=" * 110)
    print("\nTiming effect = going to cash when ALL factors below trend")
    print("Selection effect = concentrating in trending factors vs equal-weight")

    # Decompose SMA Rotation into timing and selection components
    # Timing-only: equal-weight among ALL factors when any is trending, else cash
    # Selection-only: always invest in factors, but overweight trending ones

    sma_rot_ret = strategies["SMA Rotation"]
    indep_ret = strategies["Independent SMA Timing"]
    bh_ret_full = strategies["EW Buy-and-Hold"]

    # Compute: what fraction of time is the rotation fully in cash?
    n_trending = pd.Series(0, index=factors.index, dtype=int)
    for f in FACTOR_NAMES:
        cum = (1 + factors[f]).cumprod()
        sma = cum.rolling(100, min_periods=100).mean()
        above = (cum > sma).shift(1).fillna(False)
        n_trending += above.astype(int)

    pct_all_cash = (n_trending == 0).mean()
    pct_some_trending = (n_trending > 0).mean()

    print(f"\n  % days with 0 factors trending (all cash): {pct_all_cash:.1%}")
    print(f"  % days with 1+ factors trending:           {pct_some_trending:.1%}")
    print(f"\n  Distribution of trending factor count:")
    for k in range(5):
        pct = (n_trending == k).mean()
        print(f"    {k} factors trending: {pct:.1%}")

    # Compute selection alpha: rotation vs independent timing
    common = sma_rot_ret.index.intersection(indep_ret.index)
    selection_excess = sma_rot_ret[common] - indep_ret[common]
    selection_sharpe = compute_sharpe(selection_excess)

    # Compute timing alpha: independent timing vs B&H
    common2 = indep_ret.index.intersection(bh_ret_full.index)
    timing_excess = indep_ret[common2] - bh_ret_full[common2]
    timing_sharpe = compute_sharpe(timing_excess)

    # Compute total alpha: rotation vs B&H
    common3 = sma_rot_ret.index.intersection(bh_ret_full.index)
    total_excess = sma_rot_ret[common3] - bh_ret_full[common3]
    total_sharpe = compute_sharpe(total_excess)

    print(f"\n  Timing alpha (indep timing vs B&H):     ExSharpe = {timing_sharpe:>+.3f}")
    print(f"  Selection alpha (rotation vs indep):     ExSharpe = {selection_sharpe:>+.3f}")
    print(f"  Total alpha (rotation vs B&H):           ExSharpe = {total_sharpe:>+.3f}")

    # =====================================================================
    # YEAR-BY-YEAR COMPARISON (key strategies)
    # =====================================================================
    print(f"\n" + "=" * 110)
    print("YEAR-BY-YEAR: KEY STRATEGIES")
    print("=" * 110)

    key_strats = {
        "EW B&H": strategies["EW Buy-and-Hold"],
        "Indep SMA": strategies["Independent SMA Timing"],
        "SMA Rot": strategies["SMA Rotation"],
        "Trend+Mom": strategies["Trend+Momentum (top-2)"],
    }

    print(f"\n{'Year':<6}", end="")
    for name in key_strats:
        print(f" {name:>12}", end="")
    print()
    print("-" * (6 + 13 * len(key_strats)))

    for year in range(1970, 2027, 5):
        print(f"{year:<6}", end="")
        for name, ret in key_strats.items():
            yr_mask = ret.index.year == year
            if yr_mask.sum() > 50:
                yr_ret = float((1 + ret[yr_mask]).prod() - 1)
                print(f" {yr_ret:>+11.1%}", end="")
            else:
                print(f" {'':>12}", end="")
        print()

    # =====================================================================
    # INTERNATIONAL REPLICATION
    # =====================================================================
    print(f"\n" + "=" * 110)
    print("INTERNATIONAL ROTATION (Developed ex-US, Europe, Japan)")
    print("=" * 110)

    for region in ["developed_ex_us", "europe", "japan"]:
        try:
            intl = fetch_international_factors(region, snapshot_dir=SNAP_DIR)
        except Exception as e:
            print(f"  {region}: FAILED - {e}")
            continue

        rf_intl = intl.get("RF", pd.Series(0.0, index=intl.index))
        available = [f for f in FACTOR_NAMES if f in intl.columns]
        if len(available) < 3:
            print(f"  {region}: insufficient factors ({available})")
            continue

        intl_strats = {
            "EW B&H": intl[available].mean(axis=1),
            "SMA Rotation": sma_rotation(intl.copy(), rf_intl),
            "Indep SMA": independent_sma_timing(intl.copy(), rf_intl),
        }

        print(f"\n  --- {region} ---")
        for name, ret in intl_strats.items():
            m = full_metrics(ret, name)
            if "sharpe" in m:
                common = ret.index.intersection(intl_strats["EW B&H"].index)
                excess = ret[common] - intl_strats["EW B&H"][common]
                ex_sh = compute_sharpe(excess)
                print(f"    {name:<25}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.1%}, "
                      f"MaxDD={m['max_dd']:.1%}, ExSharpe={ex_sh:+.3f}")

    # =====================================================================
    # KEY QUESTION: Does rotation beat independent timing?
    # =====================================================================
    print(f"\n" + "=" * 110)
    print("KEY QUESTION: ROTATION vs INDEPENDENT TIMING")
    print("=" * 110)

    for name in ["SMA Rotation", "Trend+Momentum (top-2)", "InvVol Rotation"]:
        ret = strategies[name]
        indep = strategies["Independent SMA Timing"]
        common = ret.index.intersection(indep.index)
        excess = ret[common] - indep[common]
        ex_sh = compute_sharpe(excess)
        rot_dd = compute_max_dd(ret[common])
        indep_dd = compute_max_dd(indep[common])
        print(f"  {name:<30} vs Independent SMA: ExSharpe={ex_sh:>+.3f}, "
              f"DD: {rot_dd:.1%} vs {indep_dd:.1%}")

    print(f"\n{'=' * 110}")
    print("PHASE 13 COMPLETE")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
