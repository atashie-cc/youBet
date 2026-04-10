"""Shared utilities for commodity workflow experiments.

Provides a proper backtesting wrapper using the full Backtester engine
with PIT enforcement, transaction costs, survivorship checks, and
T-bill cash accrual. Also provides artifact persistence for the
global gate, benchmark family routing, and commodity-specific
sector/instrument type maps.
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
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# --- Generic engine imports (from youbet.etf) ---
from youbet.etf.backtester import Backtester, BacktestConfig, BacktestResult
from youbet.etf.costs import CostModel
from youbet.etf.risk import cagr_from_returns, kelly_optimal_leverage
from youbet.etf.stats import (
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)
from youbet.etf.strategy import BaseStrategy

# --- Commodity-specific imports ---
from youbet.commodity.costs import register_commodity_costs
from youbet.commodity.data import (
    fetch_commodity_prices,
    fetch_commodity_tbill_rates,
    filter_universe_alive_at,
    load_commodity_universe,
)
from youbet.commodity.pit import register_commodity_lags

# Register commodity cost categories and PIT lags at import time.
register_commodity_costs()
register_commodity_lags()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commodity sector and instrument type maps (mutually exclusive)
# ---------------------------------------------------------------------------

COMMODITY_SECTOR_MAP: dict[str, list[str]] = {
    "precious_metals": ["GLD", "IAU", "SGOL", "SLV", "PPLT", "PALL"],
    "energy_futures": ["USO", "UNG", "DBO"],
    "broad_commodity": ["DBC", "GSG", "PDBC", "USCI"],
    "agriculture": ["DBA"],
    "industrial_metals": ["DBB", "CPER"],
    "miners": ["GDX", "GDXJ", "SIL", "XME", "COPX", "PICK"],
    "energy_infra": ["AMLP"],
}

INSTRUMENT_TYPE_MAP: dict[str, list[str]] = {
    "physical": ["GLD", "IAU", "SGOL", "SLV", "PPLT", "PALL"],
    "futures": [
        "DBC", "GSG", "PDBC", "USCI", "USO", "UNG", "DBO",
        "DBA", "DBB", "CPER",
    ],
    "equity": ["GDX", "GDXJ", "SIL", "XME", "COPX", "PICK", "AMLP"],
}

# ---------------------------------------------------------------------------
# Same-exposure clusters for Phase 1 wrapper-efficiency tests
# Each cluster compares instruments with the same (or very similar) underlying
# ---------------------------------------------------------------------------

SAME_EXPOSURE_CLUSTERS: dict[str, dict] = {
    "gold_wrappers": {
        "benchmark": "GLD",
        "instruments": ["IAU", "SGOL"],
        "description": "Same underlying (physical gold), different wrapper/ER",
    },
    "broad_baskets": {
        "benchmark": "DBC",
        "instruments": ["GSG", "PDBC", "USCI"],
        "description": "Diversified commodity futures, different index methodology",
    },
    "oil_futures": {
        "benchmark": "USO",
        "instruments": ["DBO"],
        "description": "Crude oil futures, different roll methodology",
    },
    "gold_miners": {
        "benchmark": "GDX",
        "instruments": ["GDXJ"],
        "description": "Gold mining equities, large-cap vs junior",
    },
    "pm_miners": {
        "benchmark": "GDX",
        "instruments": ["SIL"],
        "description": "Precious metals miners: silver vs gold",
    },
    "diversified_miners": {
        "benchmark": "XME",
        "instruments": ["COPX", "PICK"],
        "description": "Diversified metals/mining equities, different sub-sectors",
    },
}

# Standalone instruments: descriptive profile only, no within-cluster test
STANDALONE_INSTRUMENTS: list[str] = [
    "SLV", "PPLT", "PALL", "UNG", "DBA", "DBB", "CPER", "AMLP",
]

# Cross-cluster comparisons (descriptive, not gated)
CROSS_CLUSTER_COMPARISONS: list[dict] = [
    {
        "name": "Asset class comparison",
        "tickers": ["GLD", "DBC", "GDX"],
        "description": "Physical gold vs broad futures vs gold miners",
    },
    {
        "name": "Precious metals comparison",
        "tickers": ["GLD", "SLV", "PPLT", "PALL"],
        "description": "Gold vs silver vs platinum vs palladium",
    },
    {
        "name": "Commodity vs equity",
        "tickers": ["DBC", "VTI"],
        "description": "Should commodities be in the portfolio at all?",
    },
]

# Reverse lookup: ticker → instrument type (kept for general use)
TICKER_TO_TYPE: dict[str, str] = {}
for itype, tickers in INSTRUMENT_TYPE_MAP.items():
    for t in tickers:
        TICKER_TO_TYPE[t] = itype


def get_cluster_benchmark(ticker: str) -> str | None:
    """Return the same-exposure cluster benchmark for a ticker, or None if standalone."""
    for cluster in SAME_EXPOSURE_CLUSTERS.values():
        if ticker in cluster["instruments"]:
            return cluster["benchmark"]
        if ticker == cluster["benchmark"]:
            return None  # Is itself a benchmark
    return None  # Standalone


# ---------------------------------------------------------------------------
# Artifact persistence for global gate
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = WORKFLOW_ROOT / "artifacts"


def save_phase_returns(
    phase_name: str,
    returns_dict: dict[str, pd.Series],
    benchmark_returns: dict[str, pd.Series],
) -> Path:
    """Persist strategy returns from a phase for the global gate.

    Args:
        phase_name: e.g. "phase1", "phase3_trend".
        returns_dict: {strategy_name: daily_returns_series}.
        benchmark_returns: {benchmark_ticker: daily_returns_series}.
            Supports multiple benchmarks (DBC, GLD, GDX).
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{phase_name}_returns.parquet"

    df = pd.DataFrame(returns_dict)
    for bench_ticker, bench_ret in benchmark_returns.items():
        df[f"__benchmark_{bench_ticker}__"] = bench_ret.reindex(df.index)
    df.to_parquet(path)
    logger.info("Saved %d strategy returns to %s", len(returns_dict), path)
    return path


def load_all_phase_returns() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Load all persisted phase returns for the global gate.

    Returns:
        (strategy_returns_dict, benchmark_returns_dict)
    """
    if not ARTIFACTS_DIR.exists():
        return {}, {}

    all_returns: dict[str, pd.Series] = {}
    benchmarks: dict[str, pd.Series] = {}

    for path in sorted(ARTIFACTS_DIR.glob("*_returns.parquet")):
        df = pd.read_parquet(path)
        for col in df.columns:
            if col.startswith("__benchmark_") and col.endswith("__"):
                bench_ticker = col[len("__benchmark_"):-2]
                chunk = df[col].dropna()
                if bench_ticker not in benchmarks or len(chunk) > len(benchmarks[bench_ticker]):
                    benchmarks[bench_ticker] = chunk
            else:
                all_returns[col] = df[col].dropna()

    return all_returns, benchmarks


# ---------------------------------------------------------------------------
# Precommitment enforcement
# ---------------------------------------------------------------------------

PRECOMMIT_DIR = WORKFLOW_ROOT / "precommit"


def precommit_universe(phase_name: str, tickers: list[str], rationale: str) -> str:
    """Lock a universe selection with a hash for audit trail.

    IMMUTABLE: If a precommitment file already exists for this phase,
    this function will NOT overwrite it. Instead it verifies the
    requested tickers match the existing commitment and raises
    ValueError on mismatch.
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
        logger.info(
            "Precommitted %s universe (%d tickers, hash=%s)",
            phase_name, len(tickers), sha,
        )

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
            f"  Requested: {requested}\n"
            f"To change the universe, update the precommitment BEFORE running experiments."
        )
    return True


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark strategy
# ---------------------------------------------------------------------------

class BuyAndHold(BaseStrategy):
    """Simple buy-and-hold for a single ticker (benchmark)."""

    def __init__(self, ticker: str = "DBC"):
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


# ---------------------------------------------------------------------------
# Proper backtesting wrapper
# ---------------------------------------------------------------------------

def run_backtest(
    strategy: BaseStrategy,
    prices: pd.DataFrame,
    universe: pd.DataFrame | None = None,
    tbill_rates: pd.Series | None = None,
    benchmark_ticker: str = "DBC",
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

    cost_model = CostModel()
    if universe is not None:
        for _, row in universe.iterrows():
            ticker = row["ticker"]
            cost_model.expense_ratios[ticker] = row.get("expense_ratio", 0.0008)
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

def compute_metrics(
    returns: pd.Series,
    name: str,
    tbill_rates: pd.Series | None = None,
) -> dict:
    """Compute Sharpe-focused metrics from daily returns.

    Args:
        returns: Daily simple returns.
        name: Strategy/ticker name.
        tbill_rates: Actual daily T-bill rates (annualized, as decimal).
            If None, falls back to 4% constant (non-authoritative).
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {
            "name": name, "cagr": 0, "ann_vol": 0, "sharpe": 0,
            "max_dd": 0, "calmar": 0, "terminal_wealth": 1.0,
            "kelly_leverage": 0,
        }

    n_years = n / 252
    cum = (1 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1 / max(n_years, 1e-6)) - 1)
    ann_vol = float(r.std() * np.sqrt(252))

    # Sharpe: use actual T-bill rates if available, else 4% constant
    if tbill_rates is not None:
        aligned_rf = tbill_rates.reindex(r.index).ffill().fillna(0.04)
        daily_rf = aligned_rf / 252
    else:
        daily_rf = 0.04 / 252
    excess = r - daily_rf
    sharpe = float(excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252))

    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # Kelly using arithmetic mean and average rf
    mu_arith = float(r.mean() * 252)
    variance = ann_vol ** 2
    rf = float(daily_rf.mean() * 252) if isinstance(daily_rf, pd.Series) else 0.04
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


def compound_monthly_returns(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute compounded monthly returns from daily returns.

    Uses geometric compounding: (1+r1)*(1+r2)*...*(1+rn) - 1
    NOT arithmetic sum.
    """
    return (1 + daily_returns).resample("ME").prod() - 1


def print_table(results: list[dict], title: str) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 105}")
    print(title)
    print(f"{'=' * 105}")
    print(
        f"{'Strategy':<35} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'MaxDD':>8} "
        f"{'Calmar':>7} {'Kelly':>6} {'Final$':>8}"
    )
    print("-" * 105)
    for r in results:
        kelly = r.get("kelly_leverage", 0)
        print(
            f"{r['name']:<35} {r['cagr']:>7.1%} {r['ann_vol']:>7.1%} "
            f"{r['sharpe']:>7.3f} {r['max_dd']:>8.1%} "
            f"{r['calmar']:>7.3f} {kelly:>5.1f}x "
            f"{r['terminal_wealth']:>7.1f}x"
        )


def run_sharpe_tests(
    returns_dict: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    title: str,
    n_bootstrap: int = 5_000,
) -> dict[str, dict]:
    """Run Sharpe bootstrap tests and Holm correction.

    Primary gate: excess Sharpe > 0.20, Holm p < 0.05, CI lower > 0.
    """
    print(f"\n--- Statistical Tests: {title} ---")

    p_values: dict[str, float] = {}
    test_results: dict[str, dict] = {}

    for name, ret in returns_dict.items():
        test = block_bootstrap_test(ret, benchmark_returns, n_bootstrap=n_bootstrap, seed=42)
        ci = excess_sharpe_ci(ret, benchmark_returns, n_bootstrap=n_bootstrap, seed=42)
        p_values[name] = test["p_value"]
        test_results[name] = {**test, **ci}

    holm = holm_bonferroni(p_values)

    print(
        f"\n{'Strategy':<35} {'ExSharpe':>9} {'Raw p':>9} {'Adj p':>9} "
        f"{'90% CI':>22} {'GATE':>8}"
    )
    print("-" * 95)
    for name in sorted(
        holm, key=lambda x: test_results[x].get("observed_excess_sharpe", 0), reverse=True
    ):
        h = holm[name]
        t = test_results[name]
        ex_sharpe = t.get("observed_excess_sharpe", 0)
        ci_lo = t.get("ci_lower", 0)
        ci_hi = t.get("ci_upper", 0)
        passes = (
            h["significant_05"]
            and ex_sharpe > 0.20
            and ci_lo > 0
        )
        print(
            f"{name:<35} {ex_sharpe:>+8.3f} "
            f"{h['raw_p']:>9.4f} {h['adjusted_p']:>9.4f} "
            f"[{ci_lo:>+7.3f}, {ci_hi:>+7.3f}] "
            f"{'PASS' if passes else 'FAIL':>8}"
        )
        test_results[name]["holm_adjusted_p"] = h["adjusted_p"]
        test_results[name]["passes_gate"] = passes

    return test_results
