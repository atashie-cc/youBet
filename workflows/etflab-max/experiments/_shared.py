"""Shared utilities for etflab-max experiments.

Provides a proper backtesting wrapper using the full Backtester engine
with PIT enforcement, transaction costs, survivorship checks, and
T-bill cash accrual. Also provides artifact persistence for the
global CAGR gate.
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Artifact persistence for global gate
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = WORKFLOW_ROOT / "artifacts"


def save_phase_returns(
    phase_name: str,
    returns_dict: dict[str, pd.Series],
    benchmark_returns: pd.Series,
) -> Path:
    """Persist strategy returns from a phase for the global CAGR gate.

    Saves as parquet for reproducibility and fast loading.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{phase_name}_returns.parquet"

    df = pd.DataFrame(returns_dict)
    df["__benchmark__"] = benchmark_returns.reindex(df.index)
    df.to_parquet(path)
    logger.info("Saved %d strategy returns to %s", len(returns_dict), path)
    return path


def load_all_phase_returns() -> tuple[dict[str, pd.Series], pd.Series]:
    """Load all persisted phase returns for the global CAGR gate.

    Returns:
        (strategy_returns_dict, benchmark_returns)
    """
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

PRECOMMIT_DIR = WORKFLOW_ROOT / "precommit"


def precommit_universe(phase_name: str, tickers: list[str], rationale: str) -> str:
    """Lock a universe selection with a hash for audit trail.

    IMMUTABLE: If a precommitment file already exists for this phase,
    this function will NOT overwrite it. Instead it verifies the
    requested tickers match the existing commitment and raises
    ValueError on mismatch. This prevents post-hoc universe shopping.

    Returns the SHA-256 hash of the commitment.
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
        # File exists — verify it matches, do NOT overwrite
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
    """Verify that the universe matches the precommitted version.

    Raises ValueError if the precommitment file doesn't exist or
    the tickers don't match.
    """
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
            f"  Requested: {requested}\n"
            f"To change the universe, update the precommitment BEFORE running experiments."
        )
    return True


# ---------------------------------------------------------------------------
# Proper backtesting wrapper
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
    """Run a strategy through the full Backtester with all safeguards.

    Includes: PIT enforcement, survivorship checks, transaction costs,
    expense ratio drag, T-bill cash accrual, T+1 execution.
    """
    config = BacktestConfig(
        train_months=36,
        test_months=12,
        step_months=12,
        rebalance_frequency="monthly",
    )

    # Build cost model from universe if available
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
    # Proper Sharpe: excess return over risk-free / volatility (F7 fix)
    daily_rf = 0.04 / 252
    excess = r - daily_rf
    sharpe = float(excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252))

    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # Kelly using arithmetic mean (not geometric CAGR)
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
