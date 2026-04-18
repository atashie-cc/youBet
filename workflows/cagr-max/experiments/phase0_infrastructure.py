"""Phase 0: Power analysis + LETF data availability audit.

E0: Calibrate minimum detectable CAGR for ~20 strategies with Holm correction.
    Codex R2 fix: uses state-dependent SMA-gated DGP, not homoskedastic noise.
E1: Catalog real LETF products, verify walk-forward fold counts, build data.

Usage:
    python experiments/phase0_infrastructure.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _shared import (
    load_letf_universe,
    load_extended_universe,
    fetch_letf_prices,
    compute_metrics,
    print_table,
    precommit_universe,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SMA-gated power analysis (Codex R2 fix)
# ---------------------------------------------------------------------------


def simulate_market_returns(
    n_days: int,
    annual_return: float = 0.11,
    annual_vol: float = 0.16,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic daily market returns with GARCH-like clustering."""
    rng = np.random.default_rng(seed)
    daily_vol = annual_vol / np.sqrt(252)
    daily_mean = annual_return / 252

    returns = np.empty(n_days)
    vol = daily_vol
    for i in range(n_days):
        shock = rng.normal(0, vol)
        returns[i] = daily_mean + shock
        vol = 0.94 * vol + 0.06 * daily_vol * (1 + 0.5 * abs(shock / daily_vol))

    return returns


def simulate_sma_gated_strategy(
    market_returns: np.ndarray,
    leverage: float = 3.0,
    sma_window: int = 100,
    expense_ratio: float = 0.0091,
    borrow_spread_bps: float = 50.0,
    excess_cagr: float = 0.0,
    tbill_annual: float = 0.04,
) -> np.ndarray:
    """Simulate an SMA-gated leveraged strategy return series.

    Codex R2 fix: models the actual regime-switching structure:
    - When SMA signal is ON: hold leveraged position (lev * market - costs)
    - When SMA signal is OFF: earn T-bill rate
    - The excess_cagr is added to the underlying to model a higher-return
      concentrated asset (e.g., QQQ vs SPY)

    This produces heteroskedastic excess returns with the correct
    correlation structure vs the VTI benchmark.
    """
    n = len(market_returns)
    daily_excess = excess_cagr / 252
    daily_expense = expense_ratio / 252
    borrow_daily = (borrow_spread_bps / 10000) / 252
    tbill_daily = tbill_annual / 252

    # Underlying for the strategy (market + excess alpha)
    underlying = market_returns + daily_excess

    # Compute prices and SMA signal from the underlying
    prices = np.cumprod(1 + underlying) * 100
    sma = np.convolve(prices, np.ones(sma_window) / sma_window, mode="full")[:n]

    # SMA signal: 1 when price > SMA, 0 when below
    signal = np.zeros(n)
    signal[sma_window:] = (prices[sma_window:] > sma[sma_window:]).astype(float)

    # T+1 execution: shift signal by 1 day
    signal_shifted = np.zeros(n)
    signal_shifted[1:] = signal[:-1]

    # Strategy returns: leveraged when in, T-bill when out
    leverage_increment = max(leverage - 1.0, 0.0)
    strat_returns = np.where(
        signal_shifted > 0.5,
        leverage * underlying - leverage_increment * (tbill_daily + borrow_daily) - daily_expense,
        tbill_daily,
    )

    return strat_returns


def sma_gated_bootstrap_test(
    strat_returns: np.ndarray,
    bench_returns: np.ndarray,
    n_bootstrap: int = 1000,
    expected_block_length: int = 22,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Block bootstrap test on CAGR difference with production-matching p-value.

    Codex R2 fix: uses (1 + count) / (n + 1) p-value estimator to prevent
    exact zeros, matching production src/youbet/etf/stats.py.

    Returns: (p_value, observed_excess_cagr, ci_lower)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(strat_returns)
    p = 1.0 / expected_block_length

    log_strat = np.log1p(strat_returns)
    log_bench = np.log1p(bench_returns)
    log_excess = log_strat - log_bench

    obs_stat = log_excess.mean() * 252

    # Null: center at zero
    centered = log_excess - log_excess.mean()

    # Block bootstrap indices
    indices = np.empty((n_bootstrap, n), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n, size=n_bootstrap)
    for i in range(1, n):
        continued = (indices[:, i - 1] + 1) % n
        jumped = rng.integers(0, n, size=n_bootstrap)
        do_jump = rng.random(n_bootstrap) < p
        indices[:, i] = np.where(do_jump, jumped, continued)

    boot_centered = centered[indices]
    null_stats = boot_centered.mean(axis=1) * 252

    # Production-matching p-value: (1 + count) / (n + 1)
    count_ge = int(np.sum(null_stats >= obs_stat))
    p_value = (1 + count_ge) / (n_bootstrap + 1)

    # CAGR difference (observed)
    obs_cagr_s = np.expm1(log_strat.mean() * 252)
    obs_cagr_b = np.expm1(log_bench.mean() * 252)
    obs_excess_cagr = obs_cagr_s - obs_cagr_b

    # CI: bootstrap CAGR differences
    boot_log_s = log_strat[indices]
    boot_log_b = log_bench[indices]
    boot_cagr_diff = np.expm1(boot_log_s.mean(axis=1) * 252) - np.expm1(boot_log_b.mean(axis=1) * 252)
    ci_lower = float(np.percentile(boot_cagr_diff, 5))

    return p_value, obs_excess_cagr, ci_lower


def run_sma_gated_power_analysis(
    n_days: int = 5040,
    n_strategies: int = 20,
    leverage: float = 3.0,
    excess_cagrs: list[float] | None = None,
    n_simulations: int = 100,
    n_bootstrap: int = 500,
    significance: float = 0.05,
    min_excess_cagr: float = 0.01,
    annual_vol: float = 0.16,
    sma_window: int = 100,
) -> dict:
    """Monte Carlo power analysis using SMA-gated DGP.

    Codex R2 fix: replaces homoskedastic tracking-noise DGP with actual
    SMA cash/invested regime structure.

    For each target excess CAGR, simulates n_simulations datasets:
    1. Generate VTI-like benchmark returns
    2. Generate SMA-gated leveraged strategy returns (with excess alpha)
    3. Test the CAGR gate (magnitude + Bonferroni-corrected p + CI)

    Note: Uses Bonferroni (p * n_strategies) as single-candidate
    approximation of Holm correction (conservative, labeled as such).
    """
    if excess_cagrs is None:
        excess_cagrs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

    results = {}

    for target_cagr in excess_cagrs:
        detections = 0

        for sim in range(n_simulations):
            seed = sim * 1000
            rng = np.random.default_rng(seed)

            benchmark = simulate_market_returns(
                n_days, annual_return=0.11, annual_vol=annual_vol, seed=seed,
            )

            strategy = simulate_sma_gated_strategy(
                benchmark,
                leverage=leverage,
                sma_window=sma_window,
                excess_cagr=target_cagr,
            )

            p_val, obs_excess, ci_lo = sma_gated_bootstrap_test(
                strategy, benchmark, n_bootstrap, 22, rng,
            )

            # Bonferroni correction (conservative approximation of Holm)
            corrected_p = min(p_val * n_strategies, 1.0)

            passes_gate = (
                corrected_p < significance
                and obs_excess > min_excess_cagr
                and ci_lo > 0
            )

            if passes_gate:
                detections += 1

        power = detections / n_simulations
        results[target_cagr] = {
            "power": power,
            "detections": detections,
            "simulations": n_simulations,
        }

        print(
            f"  Excess CAGR {target_cagr:.1%}: "
            f"power = {power:.3f} ({detections}/{n_simulations})",
            flush=True,
        )

    return results


def e0_power_analysis():
    """E0: Power analysis using SMA-gated DGP.

    Codex R2 fix: models the actual regime-switching structure of
    SMA-gated leveraged strategies rather than homoskedastic noise.

    Tests three scenarios:
    1. 3x leverage, SMA100, 20 strategies — primary scenario
    2. 3x leverage, SMA100, 50 strategies — conservative (more variants)
    3. 3x leverage, SMA100, 50 strategies, 4000 days — shorter TQQQ sample
    """
    print("=" * 70, flush=True)
    print("E0: CAGR POWER ANALYSIS — SMA-GATED DGP (Codex R2 fix)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Parameters:", flush=True)
    print(f"  DGP:           SMA-gated leveraged strategy vs VTI benchmark", flush=True)
    print(f"  Leverage:      3x with SMA100 (cash when signal off)", flush=True)
    print(f"  Benchmark vol: 16% (VTI-like)", flush=True)
    print(f"  Borrow spread: 50bps", flush=True)
    print(f"  T-bill:        4%", flush=True)
    print(f"  P-value:       (1+count)/(n+1) — matches production", flush=True)
    print(f"  Correction:    Bonferroni (conservative approx of Holm)", flush=True)
    print(f"  Simulations:   100 per effect size", flush=True)
    print(flush=True)

    cagrs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]

    print("--- 3x SMA100, 20 strategies, 5040 days ---", flush=True)
    r20 = run_sma_gated_power_analysis(
        n_days=5040, n_strategies=20, leverage=3.0,
        excess_cagrs=cagrs, n_simulations=100, n_bootstrap=500,
    )

    print("\n--- 3x SMA100, 50 strategies (conservative), 5040 days ---", flush=True)
    r50 = run_sma_gated_power_analysis(
        n_days=5040, n_strategies=50, leverage=3.0,
        excess_cagrs=cagrs, n_simulations=100, n_bootstrap=500,
    )

    print("\n--- 3x SMA100, 50 strategies, 4000 days (~16yr TQQQ) ---", flush=True)
    rsh = run_sma_gated_power_analysis(
        n_days=4000, n_strategies=50, leverage=3.0,
        excess_cagrs=cagrs, n_simulations=100, n_bootstrap=500,
    )

    print("\n" + "=" * 70, flush=True)
    print("POWER COMPARISON (SMA-gated DGP)", flush=True)
    print("=" * 70, flush=True)
    print(f"\n{'ExCAGR':>10} {'20 strats':>10} {'50 strats':>10} {'50@4000d':>10}", flush=True)
    print("-" * 45, flush=True)
    for cagr in sorted(r20.keys()):
        print(
            f"{cagr:>9.1%} {r20[cagr]['power']:>10.3f} "
            f"{r50[cagr]['power']:>10.3f} {rsh[cagr]['power']:>10.3f}",
            flush=True,
        )

    print(flush=True)
    for label, res in [("20 strats", r20), ("50 strats", r50), ("50@4000d", rsh)]:
        mde = None
        for c in sorted(res.keys()):
            if res[c]["power"] >= 0.80:
                mde = c
                break
        if mde:
            print(f"  MDE ({label}): {mde:.1%}", flush=True)
        else:
            print(f"  MDE ({label}): >15%", flush=True)

    print(flush=True)
    print("Note: DGP models actual SMA cash/invested regime structure.", flush=True)
    print("  Strategy: 3x leveraged underlying when price > SMA100, T-bill when below.", flush=True)
    print("  Benchmark: buy-and-hold VTI at 16% vol.", flush=True)
    print("  Excess returns are heteroskedastic (high vol when in, near-zero when out).", flush=True)
    print("  P-value estimator: (1+count)/(n+1) — matches production.", flush=True)
    print("  Bonferroni used as conservative approximation of Holm.", flush=True)

    return r20, r50, rsh


def e1_letf_data_audit():
    """E1: Catalog real LETF products and verify data availability.

    Checks inception dates, data quality, and walk-forward fold counts.
    """
    print("\n" + "=" * 70)
    print("E1: REAL LETF DATA AVAILABILITY AUDIT")
    print("=" * 70)

    letf_univ = load_letf_universe()
    ext_univ = load_extended_universe()

    print(f"\nLeveraged ETF universe: {len(letf_univ)} products")
    print(f"Extended ETF universe:  {len(ext_univ)} products")

    print(f"\n{'Ticker':<8} {'Lev':>4} {'Inception':>12} {'ER':>6} {'Underlying':>8}")
    print("-" * 45)
    for _, row in letf_univ.iterrows():
        print(
            f"{row['ticker']:<8} {row['leverage']:>3.0f}x "
            f"{row['inception_date']:>12} {row['expense_ratio']:>5.2%} "
            f"{row['underlying_ticker']:>8}"
        )

    all_tickers = list(letf_univ["ticker"]) + list(letf_univ["underlying_ticker"].unique())
    all_tickers = list(set(all_tickers))
    all_tickers.append("VTI")

    print(f"\nFetching prices for {len(all_tickers)} tickers...")

    try:
        prices = fetch_letf_prices(all_tickers, start="1998-01-01")
        print(f"Fetched {prices.shape[0]} days × {prices.shape[1]} tickers")

        print(f"\n{'Ticker':<8} {'Start':>12} {'End':>12} {'Days':>6} {'Years':>6} {'Folds':>6}")
        print("-" * 55)
        for ticker in sorted(prices.columns):
            ts = prices[ticker].dropna()
            if len(ts) == 0:
                print(f"{ticker:<8} {'NO DATA':>12}")
                continue
            start = ts.index[0].strftime("%Y-%m-%d")
            end = ts.index[-1].strftime("%Y-%m-%d")
            n_days = len(ts)
            n_years = n_days / 252
            n_folds = max(0, int((n_years - 3) / 1))
            print(
                f"{ticker:<8} {start:>12} {end:>12} {n_days:>6} "
                f"{n_years:>5.1f} {n_folds:>6}"
            )

        min_folds = 10
        print(f"\n--- Walk-Forward Fold Check (minimum {min_folds}) ---")
        for _, row in letf_univ.iterrows():
            ticker = row["ticker"]
            if ticker in prices.columns:
                ts = prices[ticker].dropna()
                n_years = len(ts) / 252
                n_folds = max(0, int((n_years - 3) / 1))
                status = "OK" if n_folds >= min_folds else "SHORT"
                print(f"  {ticker:<8}: {n_folds} folds — {status}")

    except Exception as e:
        logger.error("Price fetch failed: %s", e)
        print(f"\nERROR: Could not fetch prices. Ensure yfinance is installed.")
        print(f"  Error: {e}")
        print("\nTo proceed, install yfinance: pip install yfinance")

    letf_tickers = list(letf_univ["ticker"])
    precommit_universe(
        "phase0_letf",
        letf_tickers,
        "Real LETF products for validation. Selected based on AUM > $200M, "
        "inception before 2011, providing >= 14 years history for walk-forward.",
    )

    core_underlyings = ["VTI", "SPY", "QQQ", "XLK", "IWM", "TLT"]
    precommit_universe(
        "phase0_underlyings",
        core_underlyings,
        "Underlying indices for LETF signal computation and synthetic leverage.",
    )


def main():
    results = e0_power_analysis()
    e1_letf_data_audit()

    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    print("\nNext: Phase 1 — Real LETF Validation (E2-E5)")
    print("  Requires: price data fetched successfully in E1")


if __name__ == "__main__":
    main()
