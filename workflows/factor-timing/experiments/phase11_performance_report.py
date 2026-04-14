"""Phase 11: 10-Year Historical Performance Assessment.

Concrete performance report showing how each strategy would have
performed over recent history, with full signal logs, year-by-year
returns, event analysis, and cross-strategy comparison.

Section A (Investable): VTI SMA100, Hedged VLUE, 60/40, VTI B&H
Section B (Paper Reference): Multi-region factor timing (not investable)
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

from youbet.etf.data import fetch_prices, fetch_tbill_rates
from youbet.etf.risk import sharpe_ratio as compute_sharpe, cagr_from_returns
from youbet.factor.data import load_french_snapshot, fetch_international_factors

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SNAP_DIR = WORKFLOW_ROOT / "data" / "snapshots"
TRADING_DAYS = 252

# Cost parameters
VTI_ER = 0.0003   # 3 bps annual
VGSH_ER = 0.0004  # 4 bps annual
BND_ER = 0.0003
TRADE_COST_BPS = 1.0  # VTI/VGSH one-way
HEDGE_TRADE_BPS = 3.0  # factor ETF one-way
BORROW_ANNUAL = 0.0014  # 14 bps (exposure-weighted from Phase 10)
MARGIN_ANNUAL = 0.0099  # 99 bps


# ---------------------------------------------------------------------------
# Strategy simulators (straight simulation, no walk-forward)
# ---------------------------------------------------------------------------

def simulate_vti_sma(
    prices: pd.DataFrame,
    sma_window: int = 100,
    check_period: str = "W",
) -> tuple[pd.Series, list[dict]]:
    """Simulate VTI SMA overlay with weekly checking.

    Returns (daily_returns, signal_log).
    """
    vti = prices["VTI"].dropna()
    vgsh = prices["VGSH"].dropna()
    common = vti.index.intersection(vgsh.index)
    vti = vti[common]
    vgsh = vgsh[common]

    vti_ret = vti.pct_change()
    vgsh_ret = vgsh.pct_change()

    # SMA on VTI price
    sma = vti.rolling(sma_window, min_periods=sma_window).mean()

    # Weekly check dates
    check_dates = set(
        vti.index.to_series().groupby(vti.index.to_period(check_period)).first().values
    )

    # Simulate
    position = "VTI"  # start long
    signal_log = []
    returns = pd.Series(0.0, index=common)
    last_switch_date = common[0]

    for i, date in enumerate(common):
        if i == 0:
            continue

        # Apply return based on position
        if position == "VTI":
            returns.iloc[i] = vti_ret.iloc[i] - VTI_ER / TRADING_DAYS
        else:
            returns.iloc[i] = vgsh_ret.iloc[i] - VGSH_ER / TRADING_DAYS

        # Check signal on check dates
        if date in check_dates and not pd.isna(sma.iloc[i]):
            current_price = vti.iloc[i]
            current_sma = sma.iloc[i]
            daily_agrees = (current_price > current_sma) == (position == "VTI")

            old_position = position
            if current_price > current_sma:
                new_position = "VTI"
            else:
                new_position = "VGSH"

            if new_position != position:
                # State change — apply switch cost
                returns.iloc[i] -= 2 * TRADE_COST_BPS / 10_000  # sell old + buy new
                days_held = (date - last_switch_date).days
                signal_log.append({
                    "date": date, "price": current_price, "sma": current_sma,
                    "action": f"EXIT {old_position} -> ENTER {new_position}",
                    "type": "SWITCH", "days_since_last": days_held,
                    "daily_agrees": True,
                })
                position = new_position
                last_switch_date = date
            else:
                # Weekly check, no change
                signal_log.append({
                    "date": date, "price": current_price, "sma": current_sma,
                    "action": f"HOLD {position}",
                    "type": "CHECK", "days_since_last": (date - last_switch_date).days,
                    "daily_agrees": daily_agrees,
                })

    return returns.iloc[1:], signal_log


def simulate_hedged_vlue(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    sma_window: int = 100,
    check_period: str = "W",
) -> tuple[pd.Series, list[dict]]:
    """Simulate hedged VLUE timing with hybrid cost model."""
    from phase3_etf_bridge import compute_hedged_returns

    hedged = compute_hedged_returns(prices, factors, "VLUE", "HML")
    rf = factors["RF"].reindex(hedged.index, method="ffill").fillna(0.0)

    # Cumulative hedged return for SMA
    cum = (1 + hedged).cumprod()
    sma = cum.rolling(sma_window, min_periods=sma_window).mean()

    # Weekly check dates
    check_dates = set(
        hedged.index.to_series().groupby(hedged.index.to_period(check_period)).first().values
    )

    position = "EXPOSED"
    signal_log = []
    returns = pd.Series(0.0, index=hedged.index)
    last_switch_date = hedged.index[0]

    # Daily costs when exposed
    daily_borrow = BORROW_ANNUAL / TRADING_DAYS
    daily_margin = MARGIN_ANNUAL / TRADING_DAYS

    for i, date in enumerate(hedged.index):
        if i == 0:
            continue

        if position == "EXPOSED":
            returns.iloc[i] = hedged.iloc[i] - daily_borrow - daily_margin
        else:
            returns.iloc[i] = rf.iloc[i]

        if date in check_dates and not pd.isna(sma.iloc[i]):
            current_cum = cum.iloc[i]
            current_sma = sma.iloc[i]
            daily_agrees = (current_cum > current_sma) == (position == "EXPOSED")

            old_pos = position
            new_pos = "EXPOSED" if current_cum > current_sma else "CASH"

            if new_pos != position:
                returns.iloc[i] -= 2 * HEDGE_TRADE_BPS / 10_000
                days_held = (date - last_switch_date).days
                signal_log.append({
                    "date": date, "level": float(current_cum), "sma": float(current_sma),
                    "action": f"{old_pos} -> {new_pos}",
                    "type": "SWITCH", "days_since_last": days_held,
                })
                position = new_pos
                last_switch_date = date
            else:
                signal_log.append({
                    "date": date, "level": float(current_cum), "sma": float(current_sma),
                    "action": f"HOLD {position}",
                    "type": "CHECK", "days_since_last": (date - last_switch_date).days,
                    "daily_agrees": daily_agrees,
                })

    return returns.iloc[1:], signal_log


def simulate_60_40(prices: pd.DataFrame) -> pd.Series:
    """60% VTI / 40% BND, monthly rebalanced."""
    vti_ret = prices["VTI"].pct_change().dropna()
    bnd_ret = prices["BND"].pct_change().dropna()
    common = vti_ret.index.intersection(bnd_ret.index)

    w_vti, w_bnd = 0.60, 0.40
    returns = pd.Series(0.0, index=common)

    for i, date in enumerate(common):
        if i == 0:
            continue
        # Monthly rebalance: reset weights on first of month
        if date.month != common[i-1].month:
            w_vti, w_bnd = 0.60, 0.40
        ret = w_vti * vti_ret[date] + w_bnd * bnd_ret[date]
        # Drift weights
        w_vti *= (1 + vti_ret[date])
        w_bnd *= (1 + bnd_ret[date])
        total = w_vti + w_bnd
        w_vti /= total
        w_bnd /= total
        returns.iloc[i] = ret - (0.60 * VTI_ER + 0.40 * BND_ER) / TRADING_DAYS

    return returns.iloc[1:]


def simulate_paper_factor(factors_by_region: dict, factor_names: list[str]) -> pd.Series:
    """Equal-weight SMA100 daily timing on multi-region factors (paper, no costs)."""
    all_excess = []
    for region, factors in factors_by_region.items():
        for factor in factor_names:
            if factor not in factors.columns:
                continue
            ret = factors[factor]
            cum = (1 + ret).cumprod()
            sma = cum.rolling(100, min_periods=100).mean()
            signal = (cum > sma).astype(float).shift(1).fillna(0)
            rf = factors.get("RF", pd.Series(0.0, index=ret.index))
            timed = signal * ret + (1 - signal) * rf
            excess = timed - ret  # excess vs buy-and-hold
            all_excess.append(excess)

    # Equal-weight average on common dates
    combined = pd.DataFrame(all_excess).T
    return combined.mean(axis=1).dropna()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_full_metrics(returns: pd.Series, name: str) -> dict:
    r = returns.dropna()
    n = len(r)
    if n < 10:
        return {"name": name}
    n_years = n / TRADING_DAYS
    cum = (1 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1 / max(n_years, 0.01)) - 1)
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / max(r.std(), 1e-10) * np.sqrt(TRADING_DAYS))
    downside = r[r < 0]
    down_std = float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 1e-10
    sortino = float(r.mean() / max(down_std, 1e-10) * np.sqrt(TRADING_DAYS))
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else 0
    return {"name": name, "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "sortino": sortino, "max_dd": max_dd, "calmar": calmar, "total_ret": float(cum.iloc[-1] - 1)}


def year_by_year(returns: pd.Series) -> dict[int, float]:
    annual = {}
    for year in sorted(returns.index.year.unique()):
        yr_ret = returns[returns.index.year == year]
        annual[year] = float((1 + yr_ret).prod() - 1)
    return annual


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 110)
    print("PHASE 11: HISTORICAL PERFORMANCE ASSESSMENT")
    print("=" * 110)

    # Load data
    prices = fetch_prices(
        tickers=["VTI", "VGSH", "BND", "VLUE"],
        start="2005-01-01",
        snapshot_dir=SNAP_DIR / "etf",
    )
    factors_us = load_french_snapshot(SNAP_DIR)

    intl_regions = {}
    for region in ["europe", "japan"]:
        try:
            intl_regions[region] = fetch_international_factors(region, snapshot_dir=SNAP_DIR)
        except Exception as e:
            logger.warning("Could not load %s: %s", region, e)
    intl_regions["us"] = factors_us

    print(f"\nPrice data: {prices.index[0].date()} to {prices.index[-1].date()}")

    # =====================================================================
    # RUN ALL STRATEGIES
    # =====================================================================

    # 1. VTI SMA100 (2006-2026)
    print("\n--- Simulating VTI SMA100 Overlay ---")
    vti_sma_ret, vti_sma_log = simulate_vti_sma(prices, sma_window=100)
    vti_sma_ret = vti_sma_ret[vti_sma_ret.index >= "2006-01-01"]

    # 2. Hedged VLUE (2012-2026)
    print("--- Simulating Hedged VLUE Timing ---")
    vlue_ret, vlue_log = simulate_hedged_vlue(prices, factors_us)

    # 3. 60/40
    print("--- Simulating 60/40 VTI/BND ---")
    sixty_forty_ret = simulate_60_40(prices)
    sixty_forty_ret = sixty_forty_ret[sixty_forty_ret.index >= "2006-01-01"]

    # 4. VTI B&H
    vti_bh_ret = prices["VTI"].pct_change().dropna() - VTI_ER / TRADING_DAYS
    vti_bh_ret = vti_bh_ret[vti_bh_ret.index >= "2006-01-01"]

    # 5. Paper factor timing
    print("--- Simulating Multi-Region Paper Factor Timing ---")
    paper_ret = simulate_paper_factor(intl_regions, ["HML", "CMA"])
    paper_ret = paper_ret[paper_ret.index >= "2006-01-01"]

    # =====================================================================
    # SIGNAL LOGS (state changes only for readability)
    # =====================================================================
    print("\n" + "=" * 110)
    print("SECTION 1: SIGNAL LOGS (State Changes)")
    print("=" * 110)

    print("\n--- VTI SMA100 Signal Changes ---")
    print(f"{'Date':<12} {'VTI Price':>10} {'SMA100':>10} {'Action':<30} {'Days Held':>10}")
    print("-" * 75)
    switches_vti = [e for e in vti_sma_log if e["type"] == "SWITCH"]
    for e in switches_vti:
        print(f"{e['date'].strftime('%Y-%m-%d'):<12} {e['price']:>9.2f} {e['sma']:>9.2f} "
              f"{e['action']:<30} {e['days_since_last']:>9}d")
    print(f"\nTotal switches: {len(switches_vti)} ({len(switches_vti) / max(len(vti_sma_ret)/TRADING_DAYS, 0.01):.1f}/yr)")

    print("\n--- Hedged VLUE Signal Changes ---")
    print(f"{'Date':<12} {'Cum Level':>10} {'SMA100':>10} {'Action':<25} {'Days Held':>10}")
    print("-" * 70)
    switches_vlue = [e for e in vlue_log if e["type"] == "SWITCH"]
    for e in switches_vlue:
        print(f"{e['date'].strftime('%Y-%m-%d'):<12} {e['level']:>9.4f} {e['sma']:>9.4f} "
              f"{e['action']:<25} {e['days_since_last']:>9}d")
    print(f"\nTotal switches: {len(switches_vlue)} ({len(switches_vlue) / max(len(vlue_ret)/TRADING_DAYS, 0.01):.1f}/yr)")

    # =====================================================================
    # KEY EVENT BEHAVIOR
    # =====================================================================
    print("\n" + "=" * 110)
    print("SECTION 2: KEY EVENT BEHAVIOR")
    print("=" * 110)

    events = [
        ("GFC (Sep 2008 - Mar 2009)", "2008-09-01", "2009-04-01"),
        ("COVID Crash (Feb-Apr 2020)", "2020-02-01", "2020-05-01"),
        ("2022 Bear Market (Jan-Oct 2022)", "2022-01-01", "2022-11-01"),
        ("2025 Tariff Volatility (Mar-Apr 2025)", "2025-03-01", "2025-05-01"),
    ]

    strategies = {
        "VTI B&H": vti_bh_ret,
        "VTI SMA100": vti_sma_ret,
        "60/40": sixty_forty_ret,
        "Hedged VLUE": vlue_ret,
    }

    for event_name, start, end in events:
        print(f"\n--- {event_name} ---")
        print(f"{'Strategy':<18} {'Period Return':>14} {'MaxDD in Period':>16}")
        print("-" * 52)
        for name, ret in strategies.items():
            mask = (ret.index >= start) & (ret.index < end)
            if mask.sum() < 5:
                print(f"{name:<18} {'N/A':>14} {'N/A':>16}")
                continue
            period_ret = ret[mask]
            period_total = float((1 + period_ret).prod() - 1)
            cum = (1 + period_ret).cumprod()
            period_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
            print(f"{name:<18} {period_total:>+13.1%} {period_dd:>15.1%}")

        # Show signal switches during this event
        for slog_name, slog in [("VTI SMA", switches_vti), ("Hedged VLUE", switches_vlue)]:
            event_switches = [e for e in slog if start <= e["date"].strftime("%Y-%m-%d") < end]
            if event_switches:
                for e in event_switches:
                    date_str = e["date"].strftime("%Y-%m-%d")
                    price_str = f"price={e.get('price', e.get('level', 0)):.2f}"
                    print(f"  >> {slog_name} switch: {date_str} {price_str} {e['action']}")

    # =====================================================================
    # YEAR-BY-YEAR RETURNS
    # =====================================================================
    print("\n" + "=" * 110)
    print("SECTION 3: YEAR-BY-YEAR RETURNS")
    print("=" * 110)

    yoy = {name: year_by_year(ret) for name, ret in strategies.items()}
    yoy["Paper Factor"] = year_by_year(paper_ret)

    all_years = sorted(set().union(*[y.keys() for y in yoy.values()]))
    print(f"\n{'Year':<6}", end="")
    for name in list(strategies.keys()) + ["Paper Factor"]:
        print(f" {name:>14}", end="")
    print()
    print("-" * (6 + 15 * (len(strategies) + 1)))

    for year in all_years:
        print(f"{year:<6}", end="")
        for name in list(strategies.keys()) + ["Paper Factor"]:
            val = yoy[name].get(year)
            if val is not None:
                print(f" {val:>+13.1%}", end="")
            else:
                print(f" {'':>14}", end="")
        print()

    # =====================================================================
    # UNIFIED COMPARISON TABLE
    # =====================================================================
    print("\n" + "=" * 110)
    print("SECTION 4: UNIFIED COMPARISON (Common Period 2016-2026)")
    print("=" * 110)

    print("\n--- Section A: Investable Strategies ---")
    print(f"{'Strategy':<18} {'CAGR':>7} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'Total':>8}")
    print("-" * 70)

    common_start = "2016-01-01"
    for name, ret in strategies.items():
        r = ret[ret.index >= common_start]
        m = compute_full_metrics(r, name)
        if "cagr" in m:
            print(f"{name:<18} {m['cagr']:>6.1%} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} "
                  f"{m['max_dd']:>7.1%} {m['calmar']:>7.3f} {m['total_ret']:>7.0%}")

    print(f"\n--- Section B: Paper Factor Reference (NOT investable) ---")
    r = paper_ret[paper_ret.index >= common_start]
    m = compute_full_metrics(r, "Paper Factor")
    if "cagr" in m:
        print(f"{'Paper Factor':<18} {m['cagr']:>6.1%} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} "
              f"{m['max_dd']:>7.1%} {m['calmar']:>7.3f} {m['total_ret']:>7.0%}")
    print("(Represents theoretical ceiling — equal-weight SMA100 on US+Europe+Japan HML+CMA paper factors)")

    # =====================================================================
    # CROSS-STRATEGY ANALYSIS
    # =====================================================================
    print("\n" + "=" * 110)
    print("SECTION 5: CROSS-STRATEGY ANALYSIS")
    print("=" * 110)

    # Correlation of excess returns vs VTI B&H
    common_idx = vti_bh_ret.index
    for name in strategies:
        common_idx = common_idx.intersection(strategies[name].index)
    common_idx = common_idx[common_idx >= common_start]

    excess = {}
    for name, ret in strategies.items():
        if name == "VTI B&H":
            continue
        excess[name] = ret.reindex(common_idx).fillna(0) - vti_bh_ret.reindex(common_idx).fillna(0)

    print("\n--- Excess Return Correlation Matrix (vs VTI B&H) ---")
    corr_df = pd.DataFrame(excess).corr()
    print(corr_df.to_string(float_format=lambda x: f"{x:+.3f}"))

    # Signal concordance: VTI SMA vs Hedged VLUE
    print("\n--- Signal Concordance: VTI SMA vs Hedged VLUE ---")
    # Build daily signal series from logs
    vti_signal = pd.Series(1.0, index=vti_sma_ret.index)  # start long
    for e in switches_vti:
        if "VGSH" in e["action"]:
            vti_signal[vti_signal.index >= e["date"]] = 0.0
        elif "VTI" in e["action"]:
            vti_signal[vti_signal.index >= e["date"]] = 1.0

    vlue_signal = pd.Series(1.0, index=vlue_ret.index)
    for e in switches_vlue:
        if "CASH" in e["action"]:
            vlue_signal[vlue_signal.index >= e["date"]] = 0.0
        elif "EXPOSED" in e["action"]:
            vlue_signal[vlue_signal.index >= e["date"]] = 1.0

    common_sig = vti_signal.index.intersection(vlue_signal.index)
    common_sig = common_sig[common_sig >= common_start]
    if len(common_sig) > 100:
        vs = vti_signal[common_sig]
        vl = vlue_signal[common_sig]
        both_on = ((vs == 1) & (vl == 1)).mean()
        both_off = ((vs == 0) & (vl == 0)).mean()
        disagree = 1 - both_on - both_off
        print(f"  Both risk-on:  {both_on:.0%}")
        print(f"  Both risk-off: {both_off:.0%}")
        print(f"  Disagree:      {disagree:.0%}")

    # Combined portfolio: 70% VTI-SMA + 30% Hedged VLUE
    print("\n--- Combined Portfolio: 70% VTI-SMA + 30% Hedged VLUE ---")
    common_combo = vti_sma_ret.index.intersection(vlue_ret.index)
    common_combo = common_combo[common_combo >= common_start]
    if len(common_combo) > 100:
        combo_ret = 0.70 * vti_sma_ret.reindex(common_combo).fillna(0) + \
                    0.30 * vlue_ret.reindex(common_combo).fillna(0)
        m = compute_full_metrics(combo_ret, "70/30 Blend")
        vti_m = compute_full_metrics(vti_bh_ret[vti_bh_ret.index.isin(common_combo)], "VTI B&H")
        if "cagr" in m and "cagr" in vti_m:
            print(f"  Blend:  CAGR={m['cagr']:.1%}, Sharpe={m['sharpe']:.3f}, MaxDD={m['max_dd']:.1%}")
            print(f"  VTI BH: CAGR={vti_m['cagr']:.1%}, Sharpe={vti_m['sharpe']:.3f}, MaxDD={vti_m['max_dd']:.1%}")

    # =====================================================================
    # SMA WINDOW SENSITIVITY (2016-2026)
    # =====================================================================
    print("\n" + "=" * 110)
    print("SECTION 6: SMA WINDOW SENSITIVITY (VTI, 2016-2026)")
    print("=" * 110)

    print(f"\n{'Window':>8} {'CAGR':>7} {'Sharpe':>8} {'MaxDD':>8} {'Switches':>9}")
    print("-" * 45)
    for window in [50, 75, 100, 150, 200]:
        sma_ret, sma_log = simulate_vti_sma(prices, sma_window=window)
        sma_ret = sma_ret[sma_ret.index >= common_start]
        m = compute_full_metrics(sma_ret, f"SMA{window}")
        n_sw = len([e for e in sma_log if e["type"] == "SWITCH" and e["date"] >= pd.Timestamp(common_start)])
        if "cagr" in m:
            print(f"{window:>8} {m['cagr']:>6.1%} {m['sharpe']:>7.3f} {m['max_dd']:>7.1%} {n_sw:>8}")

    print(f"\n{'=' * 110}")
    print("PHASE 11 COMPLETE")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
