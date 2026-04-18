"""Phase 5: Satellite & Tax (E16-E17).

E16: Crypto satellite with SMA100 (BTC ETF)
E17: Tax drag quantification (IRA vs taxable)

Usage:
    python experiments/phase5_satellite.py
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
    save_phase_returns,
    run_cagr_tests,
)
from youbet.etf.data import fetch_tbill_rates
from youbet.etf.synthetic_leverage import sma_signal

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    tickers = [
        "VTI", "SPY", "QQQ",
        "UPRO", "TQQQ", "SSO", "QLD",
        "SPXL",  # For TLH pair with UPRO
        "BIL",
        "BTC-USD",  # Bitcoin via yfinance (proxy for pre-ETF period)
        "IBIT", "FBTC", "BITO",  # Real crypto ETFs (short history)
    ]
    prices = fetch_letf_prices(tickers, start="2010-01-01")
    tbill = fetch_tbill_rates()
    return prices, tbill


def e16_crypto_satellite(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E16: Crypto satellite with SMA100.

    EXPLORATORY: Uses BTC-USD as proxy for pre-2024 period. Real spot
    BTC ETFs (IBIT, FBTC) launched Jan 2024 with only ~2yr history.
    Pre-2024 BTC-USD is NOT a directly investable product — no expense
    ratio, no bid-ask spread, no NAV tracking error. Results overstate
    implementable returns for the pre-ETF period.

    Codex R1 note: crypto ETFs (IBIT, FBTC, BITO) added to
    extended_universe.csv but insufficient history for walk-forward.
    """
    print("\n" + "=" * 70)
    print("E16: CRYPTO SATELLITE WITH SMA100 (EXPLORATORY)")
    print("=" * 70)
    print("\n  CAVEAT: Pre-2024 data uses BTC-USD (not investable).")
    print("  Real BTC ETFs (IBIT/FBTC) have only ~2yr history.")

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    btc_col = "BTC-USD"
    if btc_col not in prices.columns:
        print("  BTC-USD data not available. Skipping E16.")
        return {}

    btc_prices = prices[btc_col].dropna()
    btc_ret = returns[btc_col].dropna()

    # BTC standalone metrics
    all_metrics.append(compute_metrics(btc_ret, "BTC_buy_hold"))

    # BTC with SMA100
    btc_sig = sma_signal(btc_prices, 100)
    btc_sig_shifted = btc_sig.shift(1).reindex(btc_ret.index).fillna(0.0)
    rf = tbill_daily.reindex(btc_ret.index).fillna(0.0)
    btc_sma_ret = btc_sig_shifted * btc_ret + (1.0 - btc_sig_shifted) * rf
    all_metrics.append(compute_metrics(btc_sma_ret, "BTC_SMA100"))
    strategy_returns["BTC_SMA100"] = btc_sma_ret

    # Core: leveraged VTI SMA100
    if "SPY" in returns.columns:
        core_ret = sma_leveraged_returns(
            prices["SPY"], returns["SPY"], tbill_daily,
            leverage=3.0, sma_window=100, expense_ratio=0.0091,
        )

        # Blend: core + BTC satellite at various allocations
        for btc_pct in [0.05, 0.10, 0.15]:
            core_pct = 1.0 - btc_pct
            common = core_ret.index.intersection(btc_sma_ret.index)
            blend = core_pct * core_ret.loc[common] + btc_pct * btc_sma_ret.loc[common]
            name = f"3x_SPY_SMA100_{int(btc_pct*100)}pct_BTC"
            strategy_returns[name] = blend
            all_metrics.append(compute_metrics(blend, name))

        # Core-only for comparison
        all_metrics.append(compute_metrics(core_ret, "3x_SPY_SMA100_core_only"))

    print("\n  NOTE: BTC-USD prices are used as proxy for BTC ETF returns.")
    print("  Real BTC ETFs (IBIT, FBTC) launched Jan 2024 — only ~2yr history.")
    print("  Pre-2024 backtest uses BTC-USD which is NOT a directly investable product")
    print("  (no expense ratio, no bid-ask spread, no NAV tracking error).")

    # BTC signal statistics
    switches = (btc_sig.diff().abs() > 0.5).sum()
    pct_in = btc_sig.mean()
    print(f"\n  BTC SMA100 statistics:")
    print(f"    Signal switches: {switches}")
    print(f"    Time in market:  {pct_in:.1%}")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E16: Crypto Satellite"
    )

    return strategy_returns


def e17_tax_drag(prices: pd.DataFrame, tbill: pd.Series) -> dict[str, pd.Series]:
    """E17: Tax drag quantification (IRA vs taxable).

    Models the tax impact of SMA100 switching strategies.
    SMA100 switches ~4-6x/year, generating short-term capital gains.
    """
    print("\n" + "=" * 70)
    print("E17: TAX DRAG QUANTIFICATION (IRA vs TAXABLE)")
    print("=" * 70)

    returns = prices.pct_change().dropna(how="all")
    tbill_daily = tbill.reindex(returns.index, method="ffill").fillna(0.0) / 252

    strategy_returns = {}
    all_metrics = []

    # Tax parameters
    stcg_rate = 0.37  # Federal short-term capital gains (ordinary income)
    ltcg_rate = 0.20  # Federal long-term capital gains
    state_rate = 0.05  # Approximate state tax

    for underlying, letf, er in [("SPY", "UPRO", 0.0091), ("QQQ", "TQQQ", 0.0086)]:
        if underlying not in returns.columns:
            continue

        und_prices = prices[underlying].dropna()
        und_ret = returns[underlying].dropna()

        # Pre-tax strategy returns (tax-deferred / IRA)
        pretax_ret = sma_leveraged_returns(
            und_prices, und_ret, tbill_daily,
            leverage=3.0, sma_window=100, expense_ratio=er,
        )

        # Analyze holding periods
        sig = sma_signal(und_prices, 100)
        switches = sig.diff().abs() > 0.5
        switch_dates = sig.index[switches]
        n_switches = len(switch_dates)
        avg_switches_per_year = n_switches / (len(sig) / 252)

        # Estimate holding period distribution
        holding_periods = []
        in_position = False
        entry_date = None
        for i, (date, val) in enumerate(sig.items()):
            if val > 0.5 and not in_position:
                in_position = True
                entry_date = date
            elif val < 0.5 and in_position:
                in_position = False
                if entry_date is not None:
                    days_held = (date - entry_date).days
                    holding_periods.append(days_held)

        if holding_periods:
            avg_hold = np.mean(holding_periods)
            pct_short_term = sum(1 for h in holding_periods if h < 365) / len(holding_periods)
            pct_long_term = 1 - pct_short_term
        else:
            avg_hold = 0
            pct_short_term = 0.5
            pct_long_term = 0.5

        # Effective tax rate (weighted by holding period distribution)
        effective_tax = pct_short_term * (stcg_rate + state_rate) + \
                        pct_long_term * (ltcg_rate + state_rate)

        # Model after-tax returns (simplified: apply tax drag to positive returns)
        after_tax_ret = pretax_ret.copy()
        positive_mask = after_tax_ret > 0
        after_tax_ret[positive_mask] *= (1 - effective_tax)

        pretax_name = f"3x_{underlying}_SMA100_pretax"
        aftertax_name = f"3x_{underlying}_SMA100_aftertax"
        strategy_returns[pretax_name] = pretax_ret
        strategy_returns[aftertax_name] = after_tax_ret

        pretax_m = compute_metrics(pretax_ret, pretax_name)
        aftertax_m = compute_metrics(after_tax_ret, aftertax_name)
        all_metrics.extend([pretax_m, aftertax_m])

        print(f"\n  {underlying} ({letf}) SMA100 Tax Analysis:")
        print(f"    Avg switches/year:       {avg_switches_per_year:.1f}")
        print(f"    Avg holding period:      {avg_hold:.0f} days")
        print(f"    Short-term positions:    {pct_short_term:.1%}")
        print(f"    Long-term positions:     {pct_long_term:.1%}")
        print(f"    Effective tax rate:       {effective_tax:.1%}")
        print(f"    Pre-tax CAGR:            {pretax_m['cagr']:.1%}")
        print(f"    After-tax CAGR:          {aftertax_m['cagr']:.1%}")
        print(f"    Tax drag:                {pretax_m['cagr'] - aftertax_m['cagr']:.1%}")

    # TLH pair analysis
    print("\n--- Tax-Loss Harvesting Pairs ---")
    tlh_pairs = [
        ("UPRO", "SPXL", "3x S&P 500 (different providers)"),
        ("TQQQ", "QLD", "Nasdaq-100 at different leverage (3x vs 2x)"),
    ]
    for ticker_a, ticker_b, desc in tlh_pairs:
        if ticker_a in returns.columns and ticker_b in returns.columns:
            common = returns[ticker_a].dropna().index.intersection(returns[ticker_b].dropna().index)
            corr = returns[ticker_a].loc[common].corr(returns[ticker_b].loc[common])
            print(f"  {ticker_a} / {ticker_b}: correlation = {corr:.4f} — {desc}")
            if corr > 0.99:
                print(f"    WARNING: Correlation > 0.99. IRS may consider 'substantially identical'.")
            else:
                print(f"    Likely NOT substantially identical. TLH viable.")

    print("\n  RECOMMENDATION: Run SMA100 leveraged strategies in tax-deferred accounts")
    print("  (IRA/401k/Roth IRA) to avoid 4-6% annual tax drag.")

    print_table(
        sorted(all_metrics, key=lambda x: x["cagr"], reverse=True),
        "E17: Tax Drag Analysis"
    )

    return strategy_returns


def main():
    print("Loading data...")
    prices, tbill = load_data()

    strats_e16 = e16_crypto_satellite(prices, tbill)
    strats_e17 = e17_tax_drag(prices, tbill)

    all_strats = {**strats_e16, **strats_e17}

    returns = prices.pct_change().dropna(how="all")
    vti_ret = returns["VTI"].dropna()
    save_phase_returns("phase5", all_strats, vti_ret)

    run_cagr_tests(all_strats, vti_ret, "Phase 5: Satellite & Tax")

    print("\n" + "=" * 70)
    print("PHASE 5 COMPLETE")
    print("=" * 70)
    print("\nNext: Phase 6 — Global Gate (E18-E19)")


if __name__ == "__main__":
    main()
