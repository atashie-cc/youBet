"""Phase 3: Novel Constructions (E10-E13).

E10: LEAPS-based leverage vs LETF leverage
E11: Lifecycle leverage glidepath with SMA100
E12: UPRO + TMF leveraged pairs ("Hedgefundie")
E13: Volatility-conditioned leverage with SMA

Codex R4 fixes:
- All experiments apply switching costs consistently
- E11 uses corrected glidepath_exposure (respects total_years)
- E12 SMA overlay tracks portfolio state correctly (no drift during off)
- E12 includes quarterly rebalance transaction costs
- E10 LEAPS model avoids double-counting financing

Usage:
    python experiments/phase3_novel.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _shared import (
    fetch_letf_prices,
    compute_metrics,
    print_table,
    sma_leveraged_returns,
    apply_switching_costs,
    glidepath_exposure,
    save_phase_returns,
    run_cagr_tests,
)
from youbet.etf.data import fetch_tbill_rates
from youbet.etf.synthetic_leverage import (
    sma_signal,
    conditional_leveraged_return,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SWITCHING_COST_BPS = 10.0
REBALANCE_COST_BPS = 5.0


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    tickers = [
        "VTI", "SPY", "QQQ", "XLK",
        "UPRO", "TQQQ", "SSO", "QLD",
        "TMF", "TLT", "UBT",
        "BIL",
    ]
    prices = fetch_letf_prices(tickers, start="1998-01-01")
    tbill = fetch_tbill_rates()
    return prices, tbill


def e10_leaps_vs_letf(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E10: LEAPS-based leverage vs LETF leverage.

    SIMPLIFIED MODEL: constant 2x exposure with fixed 4% annual theta drag.
    Codex R4 fix: theta drag is ALL-IN (no additional financing charge).
    Uses conditional_leveraged_return with borrow_spread=0 to avoid
    double-counting.
    """
    print("\n" + "=" * 70)
    print("E10: LEAPS-BASED LEVERAGE vs LETF LEVERAGE (SIMPLIFIED MODEL)")
    print("=" * 70)
    print("  CAVEAT: LEAPS modeled as constant 2x with fixed 4% theta drag.")
    print("  Theta drag is all-in — no separate financing charge.")

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    for underlying, letf_2x in [("SPY", "SSO"), ("QQQ", "QLD")]:
        if underlying not in returns.columns:
            continue

        und_ret = returns[underlying].dropna()
        und_prices = prices[underlying].dropna()

        # LEAPS: constant 2x, theta drag is all-in cost (no borrow spread)
        theta_drag_annual = 0.04
        sig = sma_signal(und_prices, 100)
        leaps_exposure = sig * 2.0

        # Codex R4 fix: borrow_spread=0 since theta is all-in
        leaps_sma_ret = conditional_leveraged_return(
            und_ret, leaps_exposure, tbill_daily,
            borrow_spread_bps=0.0,
            expense_ratio=theta_drag_annual,
        )
        leaps_sma_ret = apply_switching_costs(leaps_sma_ret, sig, SWITCHING_COST_BPS)

        name = f"LEAPS_2x_{underlying}_SMA100"
        strategy_returns[name] = leaps_sma_ret
        all_metrics.append(compute_metrics(leaps_sma_ret, name))

        # Compare with real 2x LETF via sma_leveraged_returns (includes costs)
        if letf_2x in returns.columns:
            real_2x_sma = sma_leveraged_returns(
                und_prices, und_ret, tbill_daily,
                leverage=2.0, sma_window=100,
                use_real_letf=returns[letf_2x].dropna(),
            )
            name_real = f"{letf_2x}_SMA100_real"
            strategy_returns[name_real] = real_2x_sma
            all_metrics.append(compute_metrics(real_2x_sma, name_real))

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E10: LEAPS vs LETF Leverage"
    )
    return strategy_returns


def e11_lifecycle_glidepath(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E11: Lifecycle leverage glidepath with SMA100.

    Codex R4 fix: glidepath_exposure now respects total_years parameter.
    Switching costs applied to all strategies.
    """
    print("\n" + "=" * 70)
    print("E11: LIFECYCLE LEVERAGE GLIDEPATH + SMA100")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    for underlying, er in [("SPY", 0.0091), ("QQQ", 0.0086)]:
        if underlying not in returns.columns:
            continue

        und_ret = returns[underlying].dropna()
        und_prices = prices[underlying].dropna()
        sig = sma_signal(und_prices, 100)

        for shape in ["linear", "convex", "step5"]:
            glide = glidepath_exposure(
                und_ret.index,
                start_leverage=3.0,
                end_leverage=1.0,
                total_years=20.0,
                shape=shape,
            )

            exposure = sig.reindex(und_ret.index).fillna(0.0) * glide
            strat_ret = conditional_leveraged_return(
                und_ret, exposure, tbill_daily,
                borrow_spread_bps=50.0,
                expense_ratio=er,
            )
            strat_ret = apply_switching_costs(strat_ret, sig, SWITCHING_COST_BPS)

            name = f"lifecycle_{shape}_{underlying}_SMA100"
            strategy_returns[name] = strat_ret
            all_metrics.append(compute_metrics(strat_ret, name))

        # Static comparisons (already include switching costs via sma_leveraged_returns)
        for static_lev in [1.0, 2.0, 3.0]:
            strat = sma_leveraged_returns(
                und_prices, und_ret, tbill_daily,
                leverage=static_lev, sma_window=100, expense_ratio=er,
            )
            all_metrics.append(compute_metrics(strat, f"static_{static_lev:.0f}x_{underlying}_SMA100"))

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E11: Lifecycle Glidepath Results"
    )
    return strategy_returns


def e12_upro_tmf(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E12: UPRO + TMF leveraged pairs ('Hedgefundie').

    Codex R4 fixes:
    - SMA overlay integrated into the rebalancing loop (not applied after)
    - Portfolio weights reset to target on SMA re-entry
    - Quarterly rebalance transaction costs deducted
    - SMA switching costs deducted
    """
    print("\n" + "=" * 70)
    print("E12: UPRO + TMF LEVERAGED PAIRS ('HEDGEFUNDIE')")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    if "UPRO" not in returns.columns or "TMF" not in returns.columns:
        print("  UPRO or TMF data not available. Skipping E12.")
        return {}

    upro_ret = returns["UPRO"].dropna()
    tmf_ret = returns["TMF"].dropna()
    common = upro_ret.index.intersection(tmf_ret.index)
    upro_ret = upro_ret.loc[common]
    tmf_ret = tmf_ret.loc[common]

    # SMA signal on SPY for overlay
    spy_sig = None
    if "SPY" in prices.columns:
        spy_sig = sma_signal(prices["SPY"], 100)

    allocations = [
        (0.55, 0.45, "55_45"),
        (0.60, 0.40, "60_40"),
        (0.70, 0.30, "70_30"),
    ]

    rebal_freq = 63
    rebal_cost_decimal = REBALANCE_COST_BPS / 10000.0

    for w_upro, w_tmf, label in allocations:
        n = len(common)

        # --- Baseline (no SMA overlay) ---
        daily_ret_base = np.zeros(n)
        uw, tw = w_upro, w_tmf
        for i in range(1, n):
            daily_ret_base[i] = uw * upro_ret.iloc[i] + tw * tmf_ret.iloc[i]
            uw_new = uw * (1 + upro_ret.iloc[i])
            tw_new = tw * (1 + tmf_ret.iloc[i])
            total = uw_new + tw_new
            if total > 0:
                uw, tw = uw_new / total, tw_new / total
            if i % rebal_freq == 0:
                turnover = abs(uw - w_upro) + abs(tw - w_tmf)
                daily_ret_base[i] -= turnover * rebal_cost_decimal
                uw, tw = w_upro, w_tmf

        port_base = pd.Series(daily_ret_base, index=common)
        name_base = f"UPRO_TMF_{label}_qtr"
        strategy_returns[name_base] = port_base
        all_metrics.append(compute_metrics(port_base, name_base))

        # --- With SMA100 overlay (Codex R4 fix: gate inside loop) ---
        if spy_sig is not None:
            sig_shifted = spy_sig.shift(1).reindex(common).fillna(0.0)
            rf = tbill_daily.reindex(common).fillna(0.0)

            daily_ret_sma = np.zeros(n)
            uw, tw = w_upro, w_tmf
            was_in = False
            for i in range(1, n):
                is_in = sig_shifted.iloc[i] > 0.5
                if is_in:
                    if not was_in:
                        uw, tw = w_upro, w_tmf
                    daily_ret_sma[i] = uw * upro_ret.iloc[i] + tw * tmf_ret.iloc[i]
                    uw_new = uw * (1 + upro_ret.iloc[i])
                    tw_new = tw * (1 + tmf_ret.iloc[i])
                    total = uw_new + tw_new
                    if total > 0:
                        uw, tw = uw_new / total, tw_new / total
                    if i % rebal_freq == 0:
                        turnover = abs(uw - w_upro) + abs(tw - w_tmf)
                        daily_ret_sma[i] -= turnover * rebal_cost_decimal
                        uw, tw = w_upro, w_tmf
                else:
                    daily_ret_sma[i] = rf.iloc[i]
                was_in = is_in

            port_sma = pd.Series(daily_ret_sma, index=common)
            port_sma = apply_switching_costs(port_sma, spy_sig, SWITCHING_COST_BPS)
            sma_name = f"UPRO_TMF_{label}_qtr_SMA100"
            strategy_returns[sma_name] = port_sma
            all_metrics.append(compute_metrics(port_sma, sma_name))

    # UPRO-only for comparison
    if "SPY" in returns.columns:
        upro_only = sma_leveraged_returns(
            prices["SPY"], returns["SPY"], tbill_daily,
            leverage=3.0, sma_window=100,
            use_real_letf=returns["UPRO"].dropna(),
        )
        all_metrics.append(compute_metrics(upro_only, "UPRO_only_SMA100"))

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E12: UPRO + TMF Pairs"
    )
    return strategy_returns


def e13_vol_conditioned(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E13: Volatility-conditioned leverage with SMA.

    Codex R4 fix: switching costs applied to all strategies.
    Vol-conditioning modulates leverage during 'in' periods only.
    """
    print("\n" + "=" * 70)
    print("E13: VOL-CONDITIONED LEVERAGE + SMA100")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    for underlying, er in [("SPY", 0.0091), ("QQQ", 0.0086)]:
        if underlying not in returns.columns:
            continue

        und_ret = returns[underlying].dropna()
        und_prices = prices[underlying].dropna()
        sig = sma_signal(und_prices, 100)
        trailing_vol = und_ret.rolling(63).std() * np.sqrt(252)

        for target_vol_pct, max_lev in [(0.30, 3.0), (0.45, 3.0), (0.30, 2.0)]:
            vol_lev = target_vol_pct / trailing_vol.clip(lower=0.05)
            vol_lev = vol_lev.clip(lower=1.0, upper=max_lev)

            exposure = sig.reindex(und_ret.index).fillna(0.0) * vol_lev.reindex(und_ret.index).fillna(1.0)

            strat_ret = conditional_leveraged_return(
                und_ret, exposure, tbill_daily,
                borrow_spread_bps=50.0,
                expense_ratio=er,
            )
            strat_ret = apply_switching_costs(strat_ret, sig, SWITCHING_COST_BPS)

            name = f"volcond_{target_vol_pct:.0%}tv_{max_lev:.0f}xmax_{underlying}"
            strategy_returns[name] = strat_ret
            all_metrics.append(compute_metrics(strat_ret, name))

        # Static comparisons (include costs via sma_leveraged_returns)
        for static_lev in [2.0, 3.0]:
            strat = sma_leveraged_returns(
                und_prices, und_ret, tbill_daily,
                leverage=static_lev, sma_window=100, expense_ratio=er,
            )
            all_metrics.append(compute_metrics(strat, f"static_{static_lev:.0f}x_{underlying}_SMA100"))

    print("\n  SMA gatekeeps: exposure = 0 when SMA negative (always).")
    print("  Vol only modulates HOW MUCH leverage during 'in' periods.")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E13: Vol-Conditioned Leverage"
    )
    return strategy_returns


def main():
    print("Loading data...", flush=True)
    prices, tbill = load_data()

    strats_e10 = e10_leaps_vs_letf(prices, tbill)
    strats_e11 = e11_lifecycle_glidepath(prices, tbill)
    strats_e12 = e12_upro_tmf(prices, tbill)
    strats_e13 = e13_vol_conditioned(prices, tbill)

    all_strats = {**strats_e10, **strats_e11, **strats_e12, **strats_e13}

    returns = prices.pct_change().dropna(how="all")
    vti_ret = returns["VTI"].dropna()
    save_phase_returns("phase3", all_strats, vti_ret)

    run_cagr_tests(all_strats, vti_ret, "Phase 3: Novel Constructions")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print("\nNext: Phase 4 — Signal Refinement (E14-E15)")


if __name__ == "__main__":
    main()
