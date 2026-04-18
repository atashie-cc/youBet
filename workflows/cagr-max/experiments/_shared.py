"""Shared utilities for cagr-max experiments.

Extends etflab-max's shared module with support for:
- Extended (non-Vanguard) universe loading
- Real LETF product data alongside synthetic leverage
- Real-vs-synthetic gap analysis utilities
- Per-sleeve independent SMA timing
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
ETF_WORKFLOW = WORKFLOW_ROOT.parents[0] / "etf"
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.backtester import Backtester, BacktestConfig, BacktestResult
from youbet.etf.costs import CostModel
from youbet.etf.data import fetch_prices, fetch_tbill_rates, load_universe
from youbet.etf.risk import cagr_from_returns, kelly_optimal_leverage
from youbet.etf.stats import block_bootstrap_cagr_test, excess_cagr_ci, holm_bonferroni
from youbet.etf.strategy import BaseStrategy
from youbet.etf.synthetic_leverage import (
    sma_signal,
    multi_sma_vote,
    leveraged_long_cash,
    conditional_leveraged_return,
    synthetic_leveraged_returns,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = WORKFLOW_ROOT / "artifacts"
PRECOMMIT_DIR = WORKFLOW_ROOT / "precommit"
UNIVERSE_DIR = WORKFLOW_ROOT / "universe"

# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------


def load_extended_universe() -> pd.DataFrame:
    """Load the extended (non-Vanguard) ETF universe."""
    path = UNIVERSE_DIR / "extended_universe.csv"
    return pd.read_csv(path)


def load_letf_universe() -> pd.DataFrame:
    """Load the real leveraged ETF universe."""
    path = UNIVERSE_DIR / "letf_universe.csv"
    return pd.read_csv(path)


def fetch_letf_prices(
    tickers: list[str],
    start: str = "2003-01-01",
    end: str = "2026-04-17",
    snapshot_dir: str | None = None,
) -> pd.DataFrame:
    """Fetch adjusted close prices for real LETF products.

    Uses yfinance via the existing fetch_prices infrastructure.
    """
    return fetch_prices(
        tickers,
        start=start,
        end=end,
        snapshot_dir=snapshot_dir,
    )


# ---------------------------------------------------------------------------
# Real vs Synthetic comparison
# ---------------------------------------------------------------------------


def real_vs_synthetic_gap(
    real_returns: pd.Series,
    underlying_returns: pd.Series,
    leverage: float,
    expense_ratio: float,
) -> dict:
    """Quantify the gap between real LETF returns and synthetic model.

    Returns decomposition: total gap, expense component, tracking residual.
    """
    common = real_returns.index.intersection(underlying_returns.index)
    real = real_returns.reindex(common).dropna()
    underlying = underlying_returns.reindex(common).dropna()
    common = real.index.intersection(underlying.index)
    real = real.loc[common]
    underlying = underlying.loc[common]

    synthetic = synthetic_leveraged_returns(underlying, leverage, expense_ratio)

    n_years = len(common) / 252
    real_cagr = float((1 + real).prod() ** (1 / max(n_years, 1e-6)) - 1)
    synth_cagr = float((1 + synthetic).prod() ** (1 / max(n_years, 1e-6)) - 1)

    daily_diff = real - synthetic
    mean_daily_gap = float(daily_diff.mean())
    tracking_error = float(daily_diff.std() * np.sqrt(252))

    return {
        "real_cagr": real_cagr,
        "synthetic_cagr": synth_cagr,
        "cagr_gap": real_cagr - synth_cagr,
        "mean_daily_gap_bps": mean_daily_gap * 10000,
        "annualized_tracking_error": tracking_error,
        "n_days": len(common),
        "n_years": n_years,
    }


# ---------------------------------------------------------------------------
# Switching cost deduction
# ---------------------------------------------------------------------------


def apply_switching_costs(
    returns: pd.Series,
    signal: pd.Series,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Deduct switching costs from a strategy return series.

    Each SMA signal flip incurs a one-way cost (applied on the day after
    the switch, matching T+1 execution).

    Args:
        returns: Daily strategy returns.
        signal: Binary SMA signal (pre-shift). Switches are detected on
            the signal's own index, then mapped to the returns index.
        cost_bps: One-way switching cost in basis points.

    Returns:
        Returns with switching costs deducted on switch days.
    """
    # Detect switches on the signal's own index to avoid spurious
    # transitions from reindex fillna at weekend/holiday boundaries
    switch_dates = signal.index[signal.diff().abs() > 0.5]
    # T+1 execution: cost is paid the day after the signal flips
    cost_decimal = cost_bps / 10000.0
    adjusted = returns.copy()
    for sd in switch_dates:
        # Find the next trading day in returns after the switch date
        later = returns.index[returns.index > sd]
        if len(later) > 0:
            adjusted.iloc[returns.index.get_loc(later[0])] -= cost_decimal
    return adjusted


# ---------------------------------------------------------------------------
# SMA-based leveraged strategy returns
# ---------------------------------------------------------------------------


def sma_leveraged_returns(
    underlying_prices: pd.Series,
    underlying_returns: pd.Series,
    tbill_daily: pd.Series,
    leverage: float = 3.0,
    sma_window: int = 100,
    expense_ratio: float = 0.0091,
    borrow_spread_bps: float = 50.0,
    switching_cost_bps: float = 10.0,
    use_real_letf: pd.Series | None = None,
) -> pd.Series:
    """Compute returns for a leveraged SMA strategy.

    When use_real_letf is provided, uses actual LETF returns instead of
    synthetic leverage during "in" periods.

    Includes switching costs (Codex R1 fix) deducted on each SMA flip.

    Args:
        underlying_prices: Price series for SMA signal computation.
        underlying_returns: Daily return series of the underlying.
        tbill_daily: Daily T-bill rate.
        leverage: Leverage multiplier (for synthetic mode).
        sma_window: SMA window for trend signal.
        expense_ratio: Annual ER for synthetic leverage.
        borrow_spread_bps: Annual borrow spread in bps.
        switching_cost_bps: One-way cost per SMA signal flip in bps.
        use_real_letf: If provided, use these real LETF daily returns
            instead of synthetic leverage during "in" periods.

    Returns:
        Daily portfolio returns with T+1 execution and switching costs.
    """
    signal = sma_signal(underlying_prices, sma_window)

    if use_real_letf is not None:
        sig = signal.shift(1).reindex(use_real_letf.index).fillna(0.0)
        rf = tbill_daily.reindex(use_real_letf.index).fillna(0.0)
        raw = sig * use_real_letf + (1.0 - sig) * rf
        return apply_switching_costs(raw, signal, switching_cost_bps)

    # Use conditional_leveraged_return for proper financing cost on
    # the leverage increment (Codex R1 fix: leveraged_long_cash has
    # zero financing cost, systematically overstating synthetic returns)
    exposure = signal * leverage
    raw = conditional_leveraged_return(
        underlying_returns,
        exposure,
        tbill_daily,
        borrow_spread_bps=borrow_spread_bps,
        expense_ratio=expense_ratio,
    )
    return apply_switching_costs(raw, signal, switching_cost_bps)


def multi_sma_leveraged_returns(
    underlying_prices: pd.Series,
    underlying_returns: pd.Series,
    tbill_daily: pd.Series,
    windows: list[int],
    max_leverage: float = 3.0,
    borrow_spread_bps: float = 50.0,
    expense_ratio: float = 0.0,
) -> pd.Series:
    """Compute returns using multi-SMA vote with proportional leverage.

    Exposure = vote_fraction * max_leverage, applied via
    conditional_leveraged_return for proper financing.
    """
    vote = multi_sma_vote(underlying_prices, windows)
    exposure = vote * max_leverage
    return conditional_leveraged_return(
        underlying_returns,
        exposure,
        tbill_daily,
        borrow_spread_bps=borrow_spread_bps,
        expense_ratio=expense_ratio,
    )


def independent_sleeve_returns(
    sleeves: list[dict],
    tbill_daily: pd.Series,
    sma_window: int = 100,
) -> pd.Series:
    """Compute returns for a multi-sleeve portfolio with independent SMA timing.

    Each sleeve has its own SMA signal and leverage level. Equal-weighted.

    Args:
        sleeves: List of dicts with keys: 'prices', 'returns', 'leverage',
            'expense_ratio', 'name'. Each sleeve is independently timed.
        tbill_daily: Daily T-bill rate.
        sma_window: SMA window for each sleeve's trend signal.

    Returns:
        Equal-weighted daily portfolio returns.
    """
    sleeve_returns = []
    for s in sleeves:
        ret = sma_leveraged_returns(
            s["prices"], s["returns"], tbill_daily,
            leverage=s.get("leverage", 3.0),
            sma_window=sma_window,
            expense_ratio=s.get("expense_ratio", 0.0091),
        )
        sleeve_returns.append(ret)

    combined = pd.concat(sleeve_returns, axis=1).dropna()
    return combined.mean(axis=1)


# ---------------------------------------------------------------------------
# Lifecycle glidepath leverage
# ---------------------------------------------------------------------------


def glidepath_exposure(
    dates: pd.DatetimeIndex,
    start_leverage: float = 3.0,
    end_leverage: float = 1.0,
    total_years: float = 20.0,
    shape: str = "linear",
) -> pd.Series:
    """Generate time-varying leverage exposure for lifecycle strategies.

    Codex R4 fix: uses total_years to compute transition fraction.
    Leverage reaches end_leverage at year total_years and stays flat after.

    Args:
        dates: DatetimeIndex of trading days.
        start_leverage: Leverage at the start of the period.
        end_leverage: Leverage at the end of the period.
        total_years: Duration over which to glide (leverage stays at
            end_leverage for any remaining time).
        shape: "linear", "convex" (front-loaded), or "step5" (5-year steps).

    Returns:
        Series of target leverage per day.
    """
    n = len(dates)
    sample_years = n / 252
    # t is fraction of total_years elapsed, clamped to [0, 1]
    t = np.clip(np.linspace(0, sample_years / total_years, n), 0.0, 1.0)

    if shape == "linear":
        leverage = start_leverage + (end_leverage - start_leverage) * t
    elif shape == "convex":
        leverage = start_leverage + (end_leverage - start_leverage) * t**2
    elif shape == "step5":
        n_steps = int(total_years / 5)
        step_lev = np.linspace(start_leverage, end_leverage, n_steps + 1)
        leverage = np.full(n, end_leverage)
        for i in range(n_steps):
            frac_start = i / n_steps
            frac_end = (i + 1) / n_steps
            mask = (t >= frac_start) & (t < frac_end)
            leverage[mask] = step_lev[i]
    else:
        raise ValueError(f"Unknown glidepath shape: {shape}")

    return pd.Series(leverage, index=dates)


# ---------------------------------------------------------------------------
# Artifact persistence for global gate
# ---------------------------------------------------------------------------


def save_phase_returns(
    phase_name: str,
    returns_dict: dict[str, pd.Series],
    benchmark_returns: pd.Series,
) -> Path:
    """Persist strategy returns from a phase for the global CAGR gate."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{phase_name}_returns.parquet"

    df = pd.DataFrame(returns_dict)
    df["__benchmark__"] = benchmark_returns.reindex(df.index)
    df.to_parquet(path)
    logger.info("Saved %d strategy returns to %s", len(returns_dict), path)
    return path


def load_all_phase_returns() -> tuple[dict[str, pd.Series], pd.Series]:
    """Load all persisted phase returns for the global CAGR gate."""
    if not ARTIFACTS_DIR.exists():
        return {}, pd.Series(dtype=float)

    all_returns = {}
    benchmark = pd.Series(dtype=float)

    for path in sorted(ARTIFACTS_DIR.glob("*_returns.parquet")):
        df = pd.read_parquet(path)
        if "__benchmark__" in df.columns:
            bench_chunk = df["__benchmark__"].dropna()
            if len(bench_chunk) > len(benchmark):
                benchmark = bench_chunk
            df = df.drop(columns=["__benchmark__"])
        for col in df.columns:
            all_returns[col] = df[col].dropna()

    return all_returns, benchmark


# ---------------------------------------------------------------------------
# Precommitment enforcement
# ---------------------------------------------------------------------------


def precommit_universe(phase_name: str, tickers: list[str], rationale: str) -> str:
    """Lock a universe selection with a hash for audit trail.

    IMMUTABLE: If a precommitment file already exists for this phase,
    verifies tickers match. Raises ValueError on mismatch.
    """
    PRECOMMIT_DIR.mkdir(parents=True, exist_ok=True)

    commitment = {
        "phase": phase_name,
        "tickers": sorted(tickers),
        "rationale": rationale,
    }
    content = json.dumps(commitment, sort_keys=True, indent=2)
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]

    path = PRECOMMIT_DIR / f"{phase_name}_universe.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if sorted(existing["tickers"]) != sorted(tickers):
            raise ValueError(
                f"Precommitment already exists for {phase_name} with DIFFERENT tickers!\n"
                f"  Existing: {sorted(existing['tickers'])}\n"
                f"  Requested: {sorted(tickers)}\n"
                f"Delete {path} manually if you intentionally want to change the universe."
            )
        logger.info("Precommitment verified for %s (hash=%s)", phase_name, sha)
    else:
        path.write_text(content)
        logger.info("Precommitted %s universe (%d tickers, hash=%s)", phase_name, len(tickers), sha)

    return sha


def verify_precommitment(phase_name: str, tickers: list[str]) -> bool:
    """Verify that the universe matches the precommitted version."""
    path = PRECOMMIT_DIR / f"{phase_name}_universe.json"
    if not path.exists():
        raise ValueError(
            f"No precommitment found for {phase_name}. "
            f"Run precommit_universe() first."
        )

    commitment = json.loads(path.read_text())
    committed = sorted(commitment["tickers"])
    requested = sorted(tickers)

    if committed != requested:
        raise ValueError(
            f"Precommitment violation for {phase_name}!\n"
            f"  Committed: {committed}\n"
            f"  Requested: {requested}"
        )
    return True


# ---------------------------------------------------------------------------
# Backtesting wrapper
# ---------------------------------------------------------------------------

class BuyAndHold(BaseStrategy):
    """Simple buy-and-hold for a single ticker (benchmark)."""

    def __init__(self, ticker: str = "VTI"):
        self.ticker = ticker

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        return pd.Series({self.ticker: 1.0})

    @property
    def name(self) -> str:
        return f"buy_hold_{self.ticker}"

    @property
    def params(self) -> dict:
        return {"ticker": self.ticker}


def run_backtest(
    strategy: BaseStrategy,
    prices: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    tbill_rates: pd.Series | None = None,
    benchmark_ticker: str = "VTI",
) -> BacktestResult:
    """Run a strategy through the full Backtester with all safeguards."""
    config = BacktestConfig(
        train_months=36,
        test_months=12,
        step_months=12,
        rebalance_frequency="monthly",
    )

    cost_model = CostModel()
    if universe is not None:
        for _, row in universe.iterrows():
            ticker = row["ticker"]
            cost_model.expense_ratios[ticker] = row.get("expense_ratio", 0.0003)
            cost_model.ticker_categories[ticker] = row.get("category", "default")

    benchmark = BuyAndHold(benchmark_ticker)

    bt = Backtester(
        config=config,
        prices=prices,
        cost_model=cost_model,
        tbill_rates=tbill_rates,
        universe=universe,
    )

    return bt.run(strategy, benchmark)


# ---------------------------------------------------------------------------
# Metrics and reporting
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, name: str) -> dict:
    """Compute CAGR-focused metrics from daily returns."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {"name": name, "cagr": 0, "ann_vol": 0, "sharpe": 0,
                "max_dd": 0, "calmar": 0, "terminal_wealth": 1.0,
                "kelly_leverage": 0}

    n_years = n / 252
    cum = (1 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1 / max(n_years, 1e-6)) - 1)
    ann_vol = float(r.std() * np.sqrt(252))
    daily_rf = 0.04 / 252
    excess = r - daily_rf
    sharpe = float(excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252))

    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    mu_arith = float(r.mean() * 252)
    variance = ann_vol ** 2
    rf = 0.04
    kelly_lev = kelly_optimal_leverage(mu_arith, variance, rf)

    return {
        "name": name,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "terminal_wealth": float(cum.iloc[-1]),
        "kelly_leverage": kelly_lev,
    }


def print_table(results: list[dict], title: str):
    """Print a formatted CAGR-focused comparison table."""
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    print(f"{'Strategy':<35} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'Calmar':>7} {'Kelly':>6} {'Final$':>8}")
    print("-" * 100)
    for r in results:
        kelly = r.get("kelly_leverage", 0)
        print(
            f"{r['name']:<35} {r['cagr']:>7.1%} {r['ann_vol']:>7.1%} "
            f"{r['sharpe']:>7.3f} {r['max_dd']:>8.1%} "
            f"{r['calmar']:>7.3f} {kelly:>5.1f}x "
            f"{r['terminal_wealth']:>7.1f}x"
        )


def run_cagr_tests(
    returns_dict: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    title: str,
    n_bootstrap: int = 5_000,
) -> dict[str, dict]:
    """Run CAGR bootstrap tests and Holm correction."""
    print(f"\n--- Statistical Tests: {title} ---")

    p_values = {}
    test_results = {}

    for name, ret in returns_dict.items():
        test = block_bootstrap_cagr_test(ret, benchmark_returns, n_bootstrap=n_bootstrap, seed=42)
        ci = excess_cagr_ci(ret, benchmark_returns, n_bootstrap=n_bootstrap, seed=42)
        p_values[name] = test["p_value"]
        test_results[name] = {**test, **ci}

    holm = holm_bonferroni(p_values)

    print(f"\n{'Strategy':<35} {'ExCAGR':>8} {'Raw p':>9} {'Adj p':>9} {'90% CI':>22} {'GATE':>8}")
    print("-" * 95)
    for name in sorted(holm, key=lambda x: test_results[x]["observed_excess_cagr"], reverse=True):
        h = holm[name]
        t = test_results[name]
        passes = h["significant_05"] and t["observed_excess_cagr"] > 0.01 and t["ci_lower"] > 0
        print(
            f"{name:<35} {t['observed_excess_cagr']:>+7.1%} "
            f"{h['raw_p']:>9.4f} {h['adjusted_p']:>9.4f} "
            f"[{t['ci_lower']:>+6.1%}, {t['ci_upper']:>+6.1%}] "
            f"{'PASS' if passes else 'FAIL':>8}"
        )
        test_results[name]["holm_adjusted_p"] = h["adjusted_p"]
        test_results[name]["passes_gate"] = passes

    return test_results
