"""Phase 4: Calendar Effects — Sell-in-May, January, Turn-of-Month, Year-End.

Tests calendar-based trading rules on TOTAL MARKET RETURN (Mkt-RF + RF)
using 62 years of Ken French data. All rules have zero fitting parameters.

Primary test: full-sample block bootstrap (not walk-forward, since fit()
is a no-op and walk-forward wastes 3 years of data for no statistical gain).

Stability check: per-decade excess returns.
Diagnostic: correlation between calendar signals and SMA100 signal.
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

from _shared import (
    compute_metrics,
    load_factors,
    print_table,
    precommit_strategies,
)

from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci, holm_bonferroni
from youbet.etf.risk import sharpe_ratio as compute_sharpe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calendar signal generators
# ---------------------------------------------------------------------------

def sell_in_may_signal(dates: pd.DatetimeIndex) -> pd.Series:
    """Nov-Apr exposed (1.0), May-Oct to cash (0.0).

    No T-1 shift needed: calendar membership is known in advance
    (you know what month tomorrow is). The signal for day T's return
    is determined by day T's calendar month.
    """
    exposed = dates.month.isin([11, 12, 1, 2, 3, 4])
    return pd.Series(exposed.astype(float), index=dates)


def january_effect_signal(dates: pd.DatetimeIndex) -> pd.Series:
    """January exposed, rest to cash. No shift (calendar known in advance)."""
    exposed = dates.month == 1
    return pd.Series(exposed.astype(float), index=dates)


def turn_of_month_signal(dates: pd.DatetimeIndex) -> pd.Series:
    """Last 3 + first 3 trading days of each month exposed, rest to cash.

    Uses trading-day position within month, not calendar days.
    No shift needed: the position in the month is known in advance.
    """
    df = pd.DataFrame({"date": dates})
    df["month"] = dates.to_period("M")

    # Position within month (0-indexed from start)
    df["pos_from_start"] = df.groupby("month").cumcount()
    # Position from end (0-indexed from end)
    df["pos_from_end"] = df.groupby("month").cumcount(ascending=False)

    exposed = (df["pos_from_start"] < 3) | (df["pos_from_end"] < 3)
    return pd.Series(exposed.astype(float).values, index=dates)


def year_end_rally_signal(dates: pd.DatetimeIndex) -> pd.Series:
    """Dec-Jan exposed, rest to cash. No shift (calendar known in advance)."""
    exposed = dates.month.isin([12, 1])
    return pd.Series(exposed.astype(float), index=dates)


CALENDAR_STRATEGIES = {
    "sell_in_may": sell_in_may_signal,
    "january_effect": january_effect_signal,
    "turn_of_month": turn_of_month_signal,
    "year_end_rally": year_end_rally_signal,
}


# ---------------------------------------------------------------------------
# Full-sample simulation (no walk-forward)
# ---------------------------------------------------------------------------

def simulate_calendar(
    total_mkt: pd.Series,
    rf: pd.Series,
    signal_fn: callable,
    name: str,
) -> tuple[pd.Series, pd.Series]:
    """Simulate a calendar strategy on total market return.

    When signal = 1: earn total_mkt return
    When signal = 0: earn RF return

    Returns (strategy_returns, benchmark_returns).
    Benchmark is buy-and-hold total market.
    """
    signal = signal_fn(total_mkt.index)
    rf_aligned = rf.reindex(total_mkt.index, method="ffill").fillna(0.0)

    strategy_returns = signal * total_mkt + (1 - signal) * rf_aligned
    benchmark_returns = total_mkt

    return strategy_returns, benchmark_returns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 95)
    print("PHASE 4: CALENDAR EFFECTS")
    print("=" * 95)
    print("\nAll rules have ZERO fitting parameters.")
    print("Primary test: full-sample block bootstrap (not walk-forward).")
    print("Test series: TOTAL MARKET RETURN (Mkt-RF + RF)")

    factors = load_factors()

    # Construct total market return
    total_mkt = factors["Mkt-RF"] + factors["RF"]
    rf = factors["RF"]
    n_years = len(total_mkt) / 252

    print(f"\nData: {len(total_mkt)} days ({total_mkt.index[0].strftime('%Y')} to "
          f"{total_mkt.index[-1].strftime('%Y')}), {n_years:.1f} years")

    # Precommit
    strategy_labels = list(CALENDAR_STRATEGIES.keys())
    precommit_strategies(
        "phase4_calendar",
        strategy_labels,
        rationale="4 calendar rules (Sell-in-May, January, Turn-of-Month, Year-End) "
                  "on total market return. Zero parameters. Pre-committed.",
    )

    # =====================================================================
    # PRIMARY TEST: Full-sample on total market return
    # =====================================================================
    print("\n" + "=" * 95)
    print("PRIMARY TEST: CALENDAR STRATEGIES ON TOTAL MARKET RETURN")
    print("=" * 95)

    all_strat = {}
    all_bench = {}
    all_metrics = []

    # Benchmark metrics
    bh_m = compute_metrics(total_mkt, "buy_and_hold_market")
    all_metrics.append(bh_m)

    for name, signal_fn in CALENDAR_STRATEGIES.items():
        strat_ret, bench_ret = simulate_calendar(total_mkt, rf, signal_fn, name)
        all_strat[name] = strat_ret
        all_bench[name] = bench_ret

        m = compute_metrics(strat_ret, name)
        all_metrics.append(m)

        # Exposure stats
        signal = signal_fn(total_mkt.index)
        avg_exp = signal.mean()
        print(f"  {name}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.1%}, "
              f"MaxDD={m['max_dd']:.1%}, Avg Exposure={avg_exp:.0%}")

    print_table(all_metrics, "Calendar Strategies vs Buy-and-Hold Total Market")

    # =====================================================================
    # STATISTICAL TESTS
    # =====================================================================
    print("\n" + "=" * 95)
    print("STATISTICAL TESTS (full-sample block bootstrap, 10K replicates)")
    print("=" * 95)
    print("Gate: ExSharpe > 0.20, Holm p < 0.05, CI lower > 0")

    p_values = {}
    test_results = {}

    for name, strat_ret in all_strat.items():
        bench_ret = all_bench[name]
        test = block_bootstrap_test(strat_ret, bench_ret, n_bootstrap=2_000, seed=42)
        ci = excess_sharpe_ci(strat_ret, bench_ret, n_bootstrap=2_000, seed=42)
        p_values[name] = test["p_value"]
        test_results[name] = {**test, **ci}

    holm = holm_bonferroni(p_values)

    print(f"\n{'Strategy':<25} {'ExSharpe':>9} {'Raw p':>9} {'Holm p':>9} {'90% CI':>22} {'GATE':>8}")
    print("-" * 85)

    n_pass = 0
    for name in sorted(holm, key=lambda x: test_results[x]["observed_excess_sharpe"], reverse=True):
        h = holm[name]
        t = test_results[name]
        ci_lo = t["excess_sharpe_lower"]
        ci_hi = t["excess_sharpe_upper"]
        passes = h["significant_05"] and t["observed_excess_sharpe"] > 0.20 and ci_lo > 0
        if passes:
            n_pass += 1
        print(f"{name:<25} {t['observed_excess_sharpe']:>+8.3f} "
              f"{h['raw_p']:>9.4f} {h['adjusted_p']:>9.4f} "
              f"[{ci_lo:>+6.3f}, {ci_hi:>+6.3f}] "
              f"{'PASS' if passes else 'FAIL':>8}")

    print(f"\nGate result: {n_pass}/{len(holm)} PASS")

    # Beta-adjusted alpha (regression-based)
    print(f"\n--- Beta-Adjusted Alpha (CAPM regression) ---")
    print(f"{'Strategy':<25} {'Alpha (ann)':>12} {'Beta':>7} {'t(alpha)':>9}")
    print("-" * 55)
    for name, strat_ret in all_strat.items():
        bench = all_bench[name]
        common = strat_ret.index.intersection(bench.index)
        y = strat_ret[common].values
        X = np.column_stack([np.ones(len(y)), bench[common].values])
        try:
            beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta_hat
            n_obs = len(y)
            sigma2 = np.sum(resid**2) / (n_obs - 2)
            se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X).diagonal())
            alpha_ann = beta_hat[0] * 252
            t_alpha = beta_hat[0] / se[0]
            print(f"{name:<25} {alpha_ann:>+11.1%} {beta_hat[1]:>6.3f} {t_alpha:>+8.2f}")
        except Exception:
            print(f"{name:<25} REGRESSION FAILED")

    # =====================================================================
    # STABILITY: Per-decade excess returns
    # =====================================================================
    print("\n" + "=" * 95)
    print("STABILITY: PER-DECADE EXCESS SHARPE")
    print("=" * 95)

    decades = [
        ("1963-1972", "1963-01-01", "1972-12-31"),
        ("1973-1982", "1973-01-01", "1982-12-31"),
        ("1983-1992", "1983-01-01", "1992-12-31"),
        ("1993-2002", "1993-01-01", "2002-12-31"),
        ("2003-2012", "2003-01-01", "2012-12-31"),
        ("2013-2026", "2013-01-01", "2026-12-31"),
    ]

    print(f"\n{'Strategy':<25}", end="")
    for label, _, _ in decades:
        print(f" {label:>11}", end="")
    print()
    print("-" * (25 + 12 * len(decades)))

    for name, signal_fn in CALENDAR_STRATEGIES.items():
        print(f"{name:<25}", end="")
        for label, start, end in decades:
            mask = (total_mkt.index >= start) & (total_mkt.index <= end)
            if mask.sum() < 252:
                print(f" {'N/A':>11}", end="")
                continue
            period_mkt = total_mkt[mask]
            period_rf = rf[mask]
            s_ret, b_ret = simulate_calendar(period_mkt, period_rf, signal_fn, name)
            excess = s_ret - b_ret
            ex_sh = compute_sharpe(excess)
            print(f" {ex_sh:>+10.3f}", end="")
        print()

    # =====================================================================
    # DIAGNOSTIC: SMA100 signal correlation
    # =====================================================================
    print("\n" + "=" * 95)
    print("DIAGNOSTIC: CORRELATION WITH SMA100 SIGNAL")
    print("=" * 95)

    # Compute SMA100 signal on total market cumulative return
    cum_mkt = (1 + total_mkt).cumprod()
    sma100 = cum_mkt.rolling(100, min_periods=100).mean()
    sma_signal = (cum_mkt > sma100).astype(float).shift(1).fillna(0.0)

    print(f"\n{'Calendar Rule':<25} {'Corr with SMA100':>18} {'Overlap %':>10}")
    print("-" * 55)
    for name, signal_fn in CALENDAR_STRATEGIES.items():
        cal_signal = signal_fn(total_mkt.index)
        valid = sma_signal.index[sma100.notna()]
        corr = float(cal_signal[valid].corr(sma_signal[valid]))
        overlap = float((cal_signal[valid] == sma_signal[valid]).mean())
        print(f"{name:<25} {corr:>+17.3f} {overlap:>9.0%}")

    # =====================================================================
    # EXPLORATORY: Calendar effects on French factors
    # =====================================================================
    print("\n" + "=" * 95)
    print("EXPLORATORY: CALENDAR EFFECTS ON FRENCH FACTORS (no Holm, informational)")
    print("=" * 95)

    factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    print(f"\n{'Strategy':<25}", end="")
    for f in factor_names:
        print(f" {f:>8}", end="")
    print()
    print("-" * (25 + 9 * len(factor_names)))

    for name, signal_fn in CALENDAR_STRATEGIES.items():
        print(f"{name:<25}", end="")
        for factor in factor_names:
            signal = signal_fn(factors[factor].index)
            rf_aligned = rf.reindex(factors[factor].index, method="ffill").fillna(0.0)
            strat = signal * factors[factor] + (1 - signal) * rf_aligned
            excess = strat - factors[factor]
            ex_sh = compute_sharpe(excess)
            print(f" {ex_sh:>+7.3f}", end="")
        print()

    print(f"\n{'=' * 95}")
    print("PHASE 4 COMPLETE")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
