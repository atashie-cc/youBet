"""Phase 2: Portfolio Contribution Analysis (v2 — all Codex R3 fixes)

DESCRIPTIVE — no hard gate (per two-tier framework). All CIs are nominal
(no Holm correction); language says "nominal 90% CI" not "significant."

Core question: Does adding commodity exposure to a 60/40 VTI/BND portfolio
improve risk-adjusted returns?

Codex R3 fixes applied:
  F1:  Rebalance on actual last trading days, not calendar month-ends
  F2:  Phase 2 now documents why it bypasses the walk-forward backtester
       (static allocation test, not a timing strategy)
  F3:  Bootstrap uses excess Sharpe (subtracts aligned T-bill rates)
  F4:  Tests GLD, IAU, SLV, DBC, USCI (not just GLD/DBC/USCI)
  F5:  Weight grid is exploratory; no "optimal" recommendation
  F6:  Added gold bear-market window 2011-09 to 2018-08
  F7:  Language: "nominal 90% CI excludes zero" not "significant"
  F8:  Rebalance frequency sensitivity (monthly, quarterly, annual, none)
  F9:  Safe-haven conditional analysis with bootstrap CIs and threshold sweep
  F10: Narrowed DBC conclusion to "unconditional Sharpe" not "myth"
  F11: Recovery measured peak-to-recovery, not trough-to-recovery
  F12: Snapshot dates logged for reproducibility
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Setup ---
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from _shared import (
    compute_metrics,
    compound_monthly_returns,
    print_table,
    load_commodity_universe,
    save_phase_returns,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Pre-specified commodity sleeve weights (LOCKED — no optimizer)
COMMODITY_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20]

# Instruments to test in portfolio grid (F4: expanded beyond GLD/DBC/USCI)
PORTFOLIO_INSTRUMENTS = [
    ("GLD", "Physical gold"),
    ("IAU", "Physical gold (lower ER)"),
    ("SLV", "Physical silver"),
    ("DBC", "Broad commodity futures"),
    ("USCI", "Backwardation-seeking futures"),
]

# Locked regime windows + gold bear-market window (F6)
REGIME_WINDOWS = [
    ("2007-07-01", "2014-12-31", "Post-GFC + commodity bust"),
    ("2015-01-01", "2019-12-31", "Low-vol, dollar strength"),
    ("2020-01-01", "2022-12-31", "COVID + inflation (OUTLIER)"),
    ("2023-01-01", "2026-04-09", "Normalization"),
]

GOLD_BEAR_WINDOW = ("2011-09-01", "2018-08-31", "Gold bear market")


def load_prices():
    """Load commodity + equity/bond prices from cached snapshots."""
    from youbet.etf.data import load_snapshot
    from youbet.commodity.data import SNAPSHOTS_DIR

    snap_dirs = sorted(
        [d.name for d in SNAPSHOTS_DIR.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
        reverse=True,
    )
    if not snap_dirs:
        raise FileNotFoundError("No commodity snapshots")
    snap_date = snap_dirs[0]
    commodity_prices = load_snapshot(snapshot_date=snap_date, snapshot_dir=SNAPSHOTS_DIR)

    from youbet.etf.data import SNAPSHOTS_DIR as ETF_SNAPSHOTS
    etf_snap_dirs = sorted(
        [d.name for d in ETF_SNAPSHOTS.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
        reverse=True,
    )
    etf_snap_date = None
    if etf_snap_dirs:
        etf_snap_date = etf_snap_dirs[0]
        etf_prices = load_snapshot(snapshot_date=etf_snap_date, snapshot_dir=ETF_SNAPSHOTS)
        for ticker in ["VTI", "BND"]:
            if ticker in etf_prices.columns:
                commodity_prices[ticker] = etf_prices[ticker]

    # F12: Log snapshot dates for reproducibility
    print(f"  Commodity snapshot: {snap_date}")
    print(f"  ETF snapshot: {etf_snap_date}")

    return commodity_prices


def load_tbill_rates() -> pd.Series | None:
    """Load actual T-bill rates."""
    from youbet.commodity.data import fetch_commodity_tbill_rates
    try:
        rates = fetch_commodity_tbill_rates(allow_fallback=True)
        print(f"  T-bill rates: {len(rates)} observations loaded")
        return rates
    except Exception:
        logger.warning("Could not load T-bill rates, using 4%% fallback")
        return None


def build_portfolio_returns(
    returns: pd.DataFrame,
    equity_ticker: str,
    bond_ticker: str,
    commodity_ticker: str | None,
    commodity_weight: float,
    rebalance_freq: str = "ME",
) -> pd.Series:
    """Build portfolio daily returns with periodic rebalancing.

    F1 FIX: Rebalance on actual last trading day of each month,
    not calendar month-end. Uses groupby on year-month to find
    the actual last trading day in the data.

    F2 NOTE: This bypasses the walk-forward backtester intentionally.
    Static allocation tests don't have a train/test split or signal
    generation — they test whether a fixed allocation improves the
    portfolio. T+1 execution lag is not applicable (weights are
    constant within each rebalance period). Transaction costs are
    tested via rebalance frequency sensitivity (F8).
    """
    equity_w = 0.60 * (1 - commodity_weight)
    bond_w = 0.40 * (1 - commodity_weight)
    comm_w = commodity_weight

    tickers = [equity_ticker, bond_ticker]
    weights = [equity_w, bond_w]
    if commodity_ticker and commodity_weight > 0:
        tickers.append(commodity_ticker)
        weights.append(comm_w)

    # F1 FIX: Find actual last trading day per rebalance period
    if rebalance_freq == "none":
        rebalance_dates = set()  # Never rebalance (buy-and-hold)
    else:
        period_map = {"ME": "M", "QE": "Q", "YE": "Y"}
        period = period_map.get(rebalance_freq, "M")
        # Group by period, take last actual trading day in each group
        grouped = returns.groupby(returns.index.to_period(period))
        rebalance_dates = set(grouped.apply(lambda g: g.index[-1]))

    target_weights = dict(zip(tickers, weights))
    current_weights = dict(zip(tickers, weights))
    portfolio_values = np.ones(len(returns))

    for i in range(1, len(returns)):
        # Compute portfolio return using current (drifted) weights
        daily_ret = 0.0
        for t, w in current_weights.items():
            if t in returns.columns:
                daily_ret += w * returns[t].iloc[i]
        portfolio_values[i] = portfolio_values[i - 1] * (1 + daily_ret)

        # Update weights for drift
        total_val = 0.0
        for t in current_weights:
            if t in returns.columns:
                current_weights[t] *= (1 + returns[t].iloc[i])
                total_val += current_weights[t]
        if total_val > 0:
            for t in current_weights:
                current_weights[t] /= total_val

        # Rebalance to target at period ends
        if returns.index[i] in rebalance_dates:
            current_weights = dict(target_weights)

    port_series = pd.Series(portfolio_values, index=returns.index)
    port_returns = port_series.pct_change().dropna()
    return port_returns


def bootstrap_excess_sharpe_difference(
    port_a_returns: pd.Series,
    port_b_returns: pd.Series,
    tbill_rates: pd.Series | None = None,
    n_bootstrap: int = 5_000,
    block_length: int = 22,
    confidence: float = 0.90,
    seed: int = 42,
) -> dict:
    """Bootstrap CI for excess-Sharpe(A) - excess-Sharpe(B).

    F3 FIX: Subtracts aligned T-bill rates before computing Sharpe,
    matching the definition in compute_metrics().
    """
    rng = np.random.default_rng(seed)

    # Align to common index
    common = port_a_returns.index.intersection(port_b_returns.index)
    a_raw = port_a_returns.reindex(common).values
    b_raw = port_b_returns.reindex(common).values
    n = len(a_raw)

    # Subtract risk-free rate
    if tbill_rates is not None:
        rf_aligned = tbill_rates.reindex(common).ffill().fillna(0.04).values / 252
    else:
        rf_aligned = np.full(n, 0.04 / 252)

    a = a_raw - rf_aligned
    b = b_raw - rf_aligned

    p = 1.0 / block_length

    # Observed excess Sharpe
    sharpe_a = a.mean() / max(a.std(), 1e-10) * np.sqrt(252)
    sharpe_b = b.mean() / max(b.std(), 1e-10) * np.sqrt(252)
    obs_diff = sharpe_a - sharpe_b

    # Paired block bootstrap
    diffs = np.empty(n_bootstrap)
    for boot in range(n_bootstrap):
        indices = np.empty(n, dtype=np.int64)
        indices[0] = rng.integers(0, n)
        for i in range(1, n):
            if rng.random() < p:
                indices[i] = rng.integers(0, n)
            else:
                indices[i] = (indices[i - 1] + 1) % n

        boot_a = a[indices]
        boot_b = b[indices]
        s_a = boot_a.mean() / max(boot_a.std(), 1e-10) * np.sqrt(252)
        s_b = boot_b.mean() / max(boot_b.std(), 1e-10) * np.sqrt(252)
        diffs[boot] = s_a - s_b

    alpha = 1 - confidence
    ci_lo = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return {
        "sharpe_diff": obs_diff,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "sharpe_a": sharpe_a,
        "sharpe_b": sharpe_b,
    }


def run_portfolio_grid(
    returns: pd.DataFrame,
    commodity_ticker: str,
    tbill_rates: pd.Series | None,
    label: str,
) -> tuple[list[dict], dict[float, pd.Series]] | tuple[list, dict]:
    """Run the pre-specified weight grid for one commodity instrument."""
    print(f"\n{'=' * 95}")
    print(f"PORTFOLIO GRID: 60/40 VTI/BND + {commodity_ticker} ({label})")
    print(f"{'=' * 95}")

    needed = ["VTI", "BND", commodity_ticker]
    sub = returns[needed].dropna(how="any")
    if len(sub) < 504:
        print(f"  SKIP: insufficient common-period data ({len(sub)} days)")
        return [], {}

    n_years = len(sub) / 252
    print(f"  Common period: {sub.index.min().date()} to {sub.index.max().date()} ({n_years:.1f}yr)")

    results = []
    port_returns_by_weight = {}

    for w in COMMODITY_WEIGHTS:
        name = f"60/40" if w == 0 else f"{60*(1-w):.0f}/{40*(1-w):.0f}/{w*100:.0f} {commodity_ticker}"
        port_ret = build_portfolio_returns(sub, "VTI", "BND", commodity_ticker, w)
        m = compute_metrics(port_ret, name, tbill_rates=tbill_rates)
        results.append(m)
        port_returns_by_weight[w] = port_ret

    print_table(results, f"Weight Grid — {commodity_ticker}")

    # F3/F7: Bootstrap CIs using excess Sharpe, nominal language
    baseline_ret = port_returns_by_weight[0.0]
    print(f"\n  Nominal 90% CIs for excess-Sharpe improvement over 60/40 (no Holm correction):")
    print(f"  {'Portfolio':<35} {'dSharpe':>8} {'90% CI':>22} {'CI excl 0?':>12}")
    print("  " + "-" * 80)

    for w in COMMODITY_WEIGHTS[1:]:
        name = f"{60*(1-w):.0f}/{40*(1-w):.0f}/{w*100:.0f} {commodity_ticker}"
        port_ret = port_returns_by_weight[w]
        bs = bootstrap_excess_sharpe_difference(
            port_ret, baseline_ret, tbill_rates=tbill_rates, n_bootstrap=5_000,
        )
        excl = "YES" if bs["ci_lower"] > 0 else ("NEGATIVE" if bs["ci_upper"] < 0 else "NO")
        print(
            f"  {name:<35} {bs['sharpe_diff']:>+7.3f} "
            f"[{bs['ci_lower']:>+7.3f}, {bs['ci_upper']:>+7.3f}] "
            f"{excl:>12}"
        )

    return results, port_returns_by_weight


def run_regime_sensitivity(
    returns: pd.DataFrame,
    commodity_ticker: str,
    tbill_rates: pd.Series | None,
):
    """Leave-one-regime-out, per-regime, and gold bear-market sensitivity (F6)."""
    print(f"\n  --- Regime Sensitivity: {commodity_ticker} at 10% ---")

    needed = ["VTI", "BND", commodity_ticker]
    full = returns[needed].dropna(how="any")

    def _delta(sub):
        b = build_portfolio_returns(sub, "VTI", "BND", None, 0.0)
        p = build_portfolio_returns(sub, "VTI", "BND", commodity_ticker, 0.10)
        mb = compute_metrics(b, "b", tbill_rates=tbill_rates)
        mp = compute_metrics(p, "p", tbill_rates=tbill_rates)
        return mb["sharpe"], mp["sharpe"], mp["sharpe"] - mb["sharpe"]

    base_s, port_s, delta = _delta(full)
    print(f"\n  {'Window':<45} {'60/40':>8} {'w/Comm':>8} {'Delta':>8}")
    print("  " + "-" * 72)
    print(f"  {'Full period':<45} {base_s:>8.3f} {port_s:>8.3f} {delta:>+7.3f}")

    # Leave-one-regime-out
    for start, end, label in REGIME_WINDOWS:
        mask = ~((full.index >= start) & (full.index <= end))
        sub = full[mask]
        if len(sub) < 504:
            continue
        bs, ps, d = _delta(sub)
        print(f"  {'Ex ' + label[:40]:<45} {bs:>8.3f} {ps:>8.3f} {d:>+7.3f}")

    # F6: Gold bear-market window
    start, end, label = GOLD_BEAR_WINDOW
    sub = full.loc[start:end]
    if len(sub) >= 252:
        bs, ps, d = _delta(sub)
        print(f"  {label + ' (F6)':<45} {bs:>8.3f} {ps:>8.3f} {d:>+7.3f}")

    # Per-regime
    print()
    for start, end, label in REGIME_WINDOWS:
        sub = full.loc[start:end]
        if len(sub) < 63:
            continue
        bs, ps, d = _delta(sub)
        print(f"  {label[:45]:<45} {bs:>8.3f} {ps:>8.3f} {d:>+7.3f}")


def run_rebalance_sensitivity(
    returns: pd.DataFrame,
    commodity_ticker: str,
    tbill_rates: pd.Series | None,
):
    """F8: Test rebalance frequency sensitivity to isolate rebalancing premium."""
    print(f"\n  --- Rebalance Frequency Sensitivity: {commodity_ticker} at 10% ---")

    needed = ["VTI", "BND", commodity_ticker]
    sub = returns[needed].dropna(how="any")

    print(f"  {'Frequency':<20} {'60/40 Sharpe':>12} {'w/Comm':>12} {'Delta':>8}")
    print("  " + "-" * 55)

    for freq_name, freq_code in [("Monthly", "ME"), ("Quarterly", "QE"),
                                  ("Annual", "YE"), ("No rebalance", "none")]:
        b = build_portfolio_returns(sub, "VTI", "BND", None, 0.0, rebalance_freq=freq_code)
        p = build_portfolio_returns(sub, "VTI", "BND", commodity_ticker, 0.10, rebalance_freq=freq_code)
        mb = compute_metrics(b, "b", tbill_rates=tbill_rates)
        mp = compute_metrics(p, "p", tbill_rates=tbill_rates)
        d = mp["sharpe"] - mb["sharpe"]
        print(f"  {freq_name:<20} {mb['sharpe']:>12.3f} {mp['sharpe']:>12.3f} {d:>+7.3f}")


def run_conditional_analysis(
    returns: pd.DataFrame,
    commodity_ticker: str,
):
    """Conditional analysis with bootstrap CIs and threshold sweep (F9)."""
    print(f"\n  --- Conditional Analysis: {commodity_ticker} during equity drawdowns ---")

    needed = ["VTI", "BND", commodity_ticker]
    sub = returns[needed].dropna(how="any")
    monthly = compound_monthly_returns(sub)

    # F9: Threshold sweep
    thresholds = [0.0, -0.03, -0.05, -0.07]
    print(f"\n  {'Threshold':<15} {'N months':>8} {'VTI mean':>10} {commodity_ticker + ' mean':>10} {commodity_ticker + ' median':>12}")
    print("  " + "-" * 58)

    for thresh in thresholds:
        mask = monthly["VTI"] < thresh
        sel = monthly[mask]
        n = len(sel)
        if n < 3:
            continue
        vti_mean = sel["VTI"].mean()
        comm_mean = sel[commodity_ticker].mean()
        comm_median = sel[commodity_ticker].median()
        print(f"  VTI < {thresh:>+5.0%}     {n:>8} {vti_mean:>+9.2%} {comm_mean:>+10.2%} {comm_median:>+11.2%}")

    # F9: Bootstrap CI for conditional mean at -5% threshold
    severe = monthly[monthly["VTI"] < -0.05]
    if len(severe) >= 5:
        rng = np.random.default_rng(42)
        comm_vals = severe[commodity_ticker].values
        boot_means = np.array([
            rng.choice(comm_vals, size=len(comm_vals), replace=True).mean()
            for _ in range(5_000)
        ])
        ci_lo = np.percentile(boot_means, 5)
        ci_hi = np.percentile(boot_means, 95)
        print(f"\n  {commodity_ticker} mean during severe drawdowns (VTI < -5%): "
              f"{comm_vals.mean():+.2%} [90% CI: {ci_lo:+.2%}, {ci_hi:+.2%}]")

    # Rolling 36-month correlation
    rolling_corr = monthly["VTI"].rolling(36).corr(monthly[commodity_ticker])
    if len(rolling_corr.dropna()) > 12:
        print(f"\n  Rolling 36-month VTI-{commodity_ticker} correlation:")
        print(f"    Mean: {rolling_corr.mean():+.3f}, Min: {rolling_corr.min():+.3f}, "
              f"Max: {rolling_corr.max():+.3f}, Std: {rolling_corr.std():.3f}")


def run_drawdown_comparison(
    returns: pd.DataFrame,
    commodity_ticker: str,
    tbill_rates: pd.Series | None,
):
    """Drawdown comparison with peak-to-recovery duration (F11 fix)."""
    print(f"\n  --- Drawdown Comparison: 60/40 vs 54/36/10 {commodity_ticker} ---")

    needed = ["VTI", "BND", commodity_ticker]
    sub = returns[needed].dropna(how="any")

    baseline = build_portfolio_returns(sub, "VTI", "BND", None, 0.0)
    portfolio = build_portfolio_returns(sub, "VTI", "BND", commodity_ticker, 0.10)

    for name, ret in [("60/40", baseline), (f"54/36/10 {commodity_ticker}", portfolio)]:
        cum = (1 + ret).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        max_dd = dd.min()
        max_dd_date = dd.idxmin()

        # F11 FIX: Find peak date (last high before max drawdown trough)
        peak_date = running_max[:max_dd_date].idxmax()

        # Find recovery date (first time cum >= peak value after trough)
        peak_val = running_max[max_dd_date]
        recovery = cum[max_dd_date:]
        recovered = recovery[recovery >= peak_val]

        if len(recovered) > 0:
            recovery_date = recovered.index[0]
            # F11: Peak-to-recovery, not trough-to-recovery
            peak_to_recovery_days = (recovery_date - peak_date).days
            trough_to_recovery_days = (recovery_date - max_dd_date).days
        else:
            recovery_date = "Not recovered"
            peak_to_recovery_days = "N/A"
            trough_to_recovery_days = "N/A"

        print(f"  {name}:")
        print(f"    Max drawdown:        {max_dd:.1%} on {max_dd_date.date()}")
        print(f"    Peak date:           {peak_date.date()}")
        print(f"    Peak-to-recovery:    {peak_to_recovery_days} days")
        print(f"    Trough-to-recovery:  {trough_to_recovery_days} days")


def main():
    print("=" * 95)
    print("PHASE 2: PORTFOLIO CONTRIBUTION ANALYSIS (v2 — Codex R3 fixes)")
    print("  Descriptive — pre-specified weight grid, no optimizer")
    print("  Nominal CIs only (no Holm correction per two-tier framework)")
    print("=" * 95)

    prices = load_prices()
    returns = prices.pct_change().dropna(how="all")
    tbill_rates = load_tbill_rates()

    print(f"\nPrice data: {prices.shape[0]} days, {prices.shape[1]} tickers")
    for t in ["VTI", "BND", "GLD", "IAU", "SLV", "DBC", "USCI"]:
        if t in prices.columns:
            s = prices[t].dropna()
            print(f"  {t}: {s.index.min().date()} to {s.index.max().date()} ({len(s)} days)")

    # F4: Test expanded instrument set
    all_port_returns = {}
    for ticker, label in PORTFOLIO_INSTRUMENTS:
        if ticker not in returns.columns:
            print(f"\n  SKIP {ticker}: not in price data")
            continue

        results, port_rets = run_portfolio_grid(returns, ticker, tbill_rates, label)
        if not port_rets:
            continue
        all_port_returns[ticker] = port_rets

        # Detailed analysis for GLD and DBC only (primary hypotheses)
        if ticker in ("GLD", "DBC"):
            run_regime_sensitivity(returns, ticker, tbill_rates)
            run_rebalance_sensitivity(returns, ticker, tbill_rates)  # F8
            run_conditional_analysis(returns, ticker)
            run_drawdown_comparison(returns, ticker, tbill_rates)

    # --- Summary ---
    print(f"\n{'=' * 95}")
    print("PHASE 2 SUMMARY")
    print(f"{'=' * 95}")

    # F5: Exploratory weight grid — no "optimal" recommendation
    print("\n  NOTE: All weight-grid results are in-sample exploratory.")
    print("  No specific allocation is recommended without walk-forward validation.")
    print()
    print("  Key questions (descriptive answers only):")
    print("  1. Which commodity instruments show positive Sharpe delta in-sample?")
    print("  2. Is the delta robust across regimes, including the gold bear (2011-2018)?")
    print("  3. Is the delta robust to rebalance frequency (monthly vs none)?")
    print("  4. Does the instrument provide conditional diversification during equity drawdowns?")
    # F10: Narrowed DBC language
    print("  5. For instruments with negative unconditional Sharpe delta,")
    print("     is there regime-specific value (e.g., inflation shock)?")

    # Persist returns
    benchmark_rets = {}
    strategy_rets = {}
    for ticker, port_rets in all_port_returns.items():
        for w, ret in port_rets.items():
            if w == 0:
                benchmark_rets["60_40"] = ret
            else:
                strategy_rets[f"{ticker}_{w*100:.0f}pct"] = ret

    if strategy_rets and benchmark_rets:
        save_phase_returns("phase2", strategy_rets, benchmark_rets)


if __name__ == "__main__":
    main()
