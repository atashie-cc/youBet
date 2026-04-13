"""Shared utilities for factor-timing experiments.

Provides metrics, reporting, bootstrap tests, Holm correction,
artifact persistence, and precommitment enforcement.

NOTE: Factor portfolios are PAPER portfolios (hypothetical long-short
from CRSP). No transaction costs apply. Results must be labeled as
such and cannot be used as direct strategy recommendations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.risk import (
    cagr_from_returns,
    compute_risk_metrics,
    kelly_optimal_leverage,
    sharpe_ratio,
)
from youbet.etf.stats import (
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)
from youbet.factor.data import (
    FACTOR_NAMES,
    PUBLICATION_DATES,
    fetch_french_factors,
    load_french_snapshot,
)
from youbet.factor.simulator import (
    BuyAndHoldFactor,
    FactorStrategy,
    SMATrendFilter,
    SimulationConfig,
    SimulationResult,
    VolTargeting,
    simulate_factor_timing,
    simulate_multi_factor,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load locked workflow configuration."""
    config_path = WORKFLOW_ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_factors() -> pd.DataFrame:
    """Load French factor data (from cache or download)."""
    config = load_config()
    snap_dir = WORKFLOW_ROOT / config["data"]["snapshot_dir"]
    freq = config["data"]["frequency"]

    try:
        return load_french_snapshot(snap_dir, freq)
    except FileNotFoundError:
        logger.info("No snapshot found, downloading fresh data...")
        return fetch_french_factors(frequency=freq, snapshot_dir=snap_dir)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategies(config: dict | None = None) -> dict[str, FactorStrategy]:
    """Build all pre-committed timing strategies from config.

    Returns dict of {name: FactorStrategy}.
    """
    if config is None:
        config = load_config()

    strategies = {}

    # Buy-and-hold benchmark
    strategies["buy_and_hold"] = BuyAndHoldFactor()

    # SMA trend filters
    for window in config["strategies"]["sma_windows"]:
        strategies[f"sma_{window}"] = SMATrendFilter(window=window)

    # Constant vol-targeting
    strategies["vol_target"] = VolTargeting(
        target_vol=config["strategies"]["vol_target_pct"] / 100.0,
        lookback_days=config["strategies"]["vol_target_lookback"],
        max_leverage=config["strategies"]["vol_target_max_leverage"],
    )

    return strategies


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = WORKFLOW_ROOT / "artifacts"


def save_phase_returns(
    phase_name: str,
    returns_dict: dict[str, pd.Series],
    benchmark_dict: dict[str, pd.Series],
) -> Path:
    """Persist strategy and benchmark returns from a phase.

    For factor timing, each factor has its own benchmark (buy-and-hold
    that factor), so we store benchmark returns per factor.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{phase_name}_returns.parquet"

    df = pd.DataFrame(returns_dict)
    for name, bench in benchmark_dict.items():
        df[f"__bench_{name}__"] = bench.reindex(df.index)
    df.to_parquet(path)
    logger.info("Saved %d strategy returns to %s", len(returns_dict), path)
    return path


def load_all_phase_returns() -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Load all persisted phase returns for the global gate."""
    if not ARTIFACTS_DIR.exists():
        return {}, {}

    all_returns = {}
    all_benchmarks = {}

    for path in sorted(ARTIFACTS_DIR.glob("*_returns.parquet")):
        df = pd.read_parquet(path)
        bench_cols = [c for c in df.columns if c.startswith("__bench_") and c.endswith("__")]
        for col in bench_cols:
            name = col[8:-2]  # strip __bench_ and __
            all_benchmarks[name] = df[col].dropna()
        data_cols = [c for c in df.columns if not c.startswith("__")]
        for col in data_cols:
            all_returns[col] = df[col].dropna()

    return all_returns, all_benchmarks


# ---------------------------------------------------------------------------
# Precommitment enforcement
# ---------------------------------------------------------------------------

PRECOMMIT_DIR = WORKFLOW_ROOT / "precommit"


def precommit_strategies(
    phase_name: str,
    strategy_labels: list[str],
    rationale: str,
) -> str:
    """Lock a strategy list with a hash for audit trail."""
    PRECOMMIT_DIR.mkdir(parents=True, exist_ok=True)

    commitment = {
        "phase": phase_name,
        "strategies": sorted(strategy_labels),
        "rationale": rationale,
    }
    content = json.dumps(commitment, sort_keys=True, indent=2)
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]

    path = PRECOMMIT_DIR / f"{phase_name}_strategies.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if sorted(existing["strategies"]) != sorted(strategy_labels):
            raise ValueError(
                f"Precommitment already exists for {phase_name} with DIFFERENT strategies!\n"
                f"  Existing: {sorted(existing['strategies'])}\n"
                f"  Requested: {sorted(strategy_labels)}\n"
                f"Delete {path} manually if you intentionally want to change."
            )
        logger.info("Precommitment verified for %s (hash=%s)", phase_name, sha)
    else:
        path.write_text(content)
        logger.info(
            "Precommitted %s strategies (%d, hash=%s)",
            phase_name, len(strategy_labels), sha,
        )

    return sha


# ---------------------------------------------------------------------------
# Metrics and reporting
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, name: str, rf_rate: float = 0.0) -> dict:
    """Compute Sharpe-focused metrics from daily returns.

    NOTE: rf_rate defaults to 0.0 because Ken French factor returns are
    already excess returns (Mkt-RF = market minus RF; other factors are
    zero-cost long-short portfolios). Subtracting RF again would be
    double-counting.
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {"name": name, "cagr": 0, "ann_vol": 0, "sharpe": 0,
                "max_dd": 0, "calmar": 0, "sortino": 0}

    n_years = n / 252
    cum = (1 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1 / max(n_years, 1e-6)) - 1)
    ann_vol = float(r.std() * np.sqrt(252))

    # Sharpe with RF
    daily_rf = rf_rate / 252
    excess = r - daily_rf
    sharpe_val = float(excess.mean() / max(excess.std(), 1e-10) * np.sqrt(252))

    # Sortino
    downside = excess[excess < 0]
    downside_std = float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 1e-10
    sortino = float(excess.mean() / max(downside_std, 1e-10) * np.sqrt(252))

    # Max drawdown
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    return {
        "name": name,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe_val,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
    }


def print_table(results: list[dict], title: str):
    """Print a formatted Sharpe-focused comparison table."""
    print(f"\n{'=' * 95}")
    print(title)
    print(f"{'=' * 95}")
    print(f"{'Strategy':<40} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD':>8} {'Calmar':>7}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['name']:<40} {r['cagr']:>7.1%} {r['ann_vol']:>7.1%} "
            f"{r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['max_dd']:>8.1%} "
            f"{r['calmar']:>7.3f}"
        )


def run_sharpe_tests(
    returns_dict: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    title: str,
    n_bootstrap: int = 10_000,
) -> dict[str, dict]:
    """Run Sharpe bootstrap tests and Holm correction.

    Tests each strategy vs the benchmark (buy-and-hold factor exposure).
    Returns dict of {name: {test_results + holm_results}}.
    """
    config = load_config()
    gate = config["gate"]
    min_excess = gate["min_excess_sharpe"]

    print(f"\n--- Statistical Tests: {title} ---")
    print(f"    Gate: ExSharpe > {min_excess}, Holm p < {gate['significance']}, CI lower > {gate['ci_lower_threshold']}")

    p_values = {}
    test_results = {}

    for name, ret in returns_dict.items():
        test = block_bootstrap_test(ret, benchmark_returns, n_bootstrap=n_bootstrap, seed=42)
        ci = excess_sharpe_ci(ret, benchmark_returns, n_bootstrap=n_bootstrap, seed=42)
        p_values[name] = test["p_value"]
        test_results[name] = {**test, **ci}

    holm = holm_bonferroni(p_values)

    print(f"\n{'Strategy':<40} {'ExSharpe':>9} {'Raw p':>9} {'Adj p':>9} {'90% CI':>22} {'GATE':>8}")
    print("-" * 100)

    for name in sorted(holm, key=lambda x: test_results[x]["observed_excess_sharpe"], reverse=True):
        h = holm[name]
        t = test_results[name]
        # Use Sharpe-of-excess CI (same estimand as p-value from block_bootstrap_test)
        ci_lo = t["excess_sharpe_lower"]
        ci_hi = t["excess_sharpe_upper"]
        passes = (
            h["significant_05"]
            and t["observed_excess_sharpe"] > min_excess
            and ci_lo > gate["ci_lower_threshold"]
        )
        print(
            f"{name:<40} {t['observed_excess_sharpe']:>+8.3f} "
            f"{h['raw_p']:>9.4f} {h['adjusted_p']:>9.4f} "
            f"[{ci_lo:>+6.3f}, {ci_hi:>+6.3f}] "
            f"{'PASS' if passes else 'FAIL':>8}"
        )
        test_results[name]["holm_adjusted_p"] = h["adjusted_p"]
        test_results[name]["passes_gate"] = passes

    n_pass = sum(1 for t in test_results.values() if t["passes_gate"])
    n_total = len(test_results)
    print(f"\nGate result: {n_pass}/{n_total} PASS")

    return test_results
