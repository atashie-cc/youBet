"""Shared utilities for stock-selection experiments.

Path helpers, config loading, precommit enforcement, and a thin
`run_backtest` wrapper around `StockBacktester`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.data import fetch_tbill_rates
from youbet.etf.stats import (
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)
from youbet.stock.backtester import (
    StockBacktestConfig,
    StockBacktester,
    StockBacktestResult,
)
from youbet.stock.costs import StockCostModel
from youbet.stock.data import fetch_stock_prices
from youbet.stock.strategies.base import BuyAndHoldETF, CrossSectionalStrategy
from youbet.stock.universe import Universe

logger = logging.getLogger(__name__)

CONFIG_PATH = WORKFLOW_ROOT / "config.yaml"
UNIVERSE_DIR = WORKFLOW_ROOT / "universe"
PRECOMMIT_DIR = WORKFLOW_ROOT / "precommit"
ARTIFACTS_DIR = WORKFLOW_ROOT / "artifacts"


def load_config() -> dict:
    """Load workflow config.yaml."""
    with CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


def load_sp500_universe() -> Universe:
    """Load the seed S&P 500 membership with delisting returns."""
    return Universe.from_csv(
        UNIVERSE_DIR / "sp500_membership.csv",
        UNIVERSE_DIR / "delisting_returns.csv",
        index_name="S&P 500",
    )


def load_prices_with_benchmark(
    universe: Universe,
    start: str = "1990-01-01",
    end: str | None = None,
    include_delisted: bool = True,
    config: dict | None = None,
) -> pd.DataFrame:
    """Fetch member prices + config-configured benchmark ticker.

    The benchmark (e.g., SPY) is not an index member but MUST be present
    in the price DataFrame for the backtester's benchmark comparison to
    work. This helper enforces that invariant.
    """
    config = config or load_config()
    bench_ticker = config["benchmark"]["ticker"]
    return fetch_stock_prices(
        universe=universe,
        start=start,
        end=end,
        include_delisted=include_delisted,
        extra_tickers=[bench_ticker],
    )


def make_cost_model(config: dict | None = None) -> StockCostModel:
    config = config or load_config()
    return StockCostModel.from_config(config)


def make_backtest_config(
    config: dict | None = None,
    first_test_start_min: str | pd.Timestamp | None = None,
) -> StockBacktestConfig:
    """Build StockBacktestConfig from workflow config.yaml.

    `first_test_start_min` (R7-HIGH-1): if set, the backtester skips any
    fold whose test_start is earlier than this date. Pass the precommit's
    `walk_forward.first_test_start_min` to enforce per-phase floors.
    """
    config = config or load_config()
    bt = config["backtest"]
    return StockBacktestConfig(
        train_months=int(bt["train_months"]),
        test_months=int(bt["test_months"]),
        step_months=int(bt["step_months"]),
        rebalance_frequency=str(bt["rebalance_frequency"]),
        initial_capital=float(bt["initial_capital"]),
        first_test_start_min=first_test_start_min,
    )


def run_backtest(
    strategy: CrossSectionalStrategy,
    prices: pd.DataFrame,
    universe: Universe,
    config: dict | None = None,
    facts_by_ticker: dict | None = None,
    benchmark_ticker: str | None = None,
    tbill_rates: pd.Series | None = None,
) -> StockBacktestResult:
    """Run a strategy through StockBacktester with config-driven defaults."""
    config = config or load_config()
    bench_ticker = benchmark_ticker or config["benchmark"]["ticker"]
    bt_cfg = make_backtest_config(config)
    cost = make_cost_model(config)

    # H5 guard: benchmark ticker MUST be in the price DataFrame.
    # A silent absence makes the benchmark effectively 100% T-bill.
    if bench_ticker not in prices.columns or prices[bench_ticker].notna().sum() == 0:
        raise RuntimeError(
            f"Benchmark ticker {bench_ticker!r} is not in prices. "
            f"Use load_prices_with_benchmark() or pass extra_tickers=[{bench_ticker!r}] "
            f"to fetch_stock_prices."
        )

    if tbill_rates is None:
        try:
            tbill_rates = fetch_tbill_rates(
                start=prices.index.min().strftime("%Y-%m-%d"),
                end=prices.index.max().strftime("%Y-%m-%d"),
                allow_fallback=True,
            )
        except Exception as exc:
            logger.warning("T-bill fetch failed (%s); using constant 4%%", exc)
            tbill_rates = pd.Series(0.04, index=prices.index, name="tbill_3m")

    bt = StockBacktester(
        config=bt_cfg,
        prices=prices,
        universe=universe,
        cost_model=cost,
        tbill_rates=tbill_rates,
        facts_by_ticker=facts_by_ticker,
    )
    return bt.run(strategy=strategy, benchmark=BuyAndHoldETF(bench_ticker))


# ---------------------------------------------------------------------------
# Precommit enforcement
# ---------------------------------------------------------------------------


def precommit_phase(
    phase_name: str,
    strategies: list[dict],
    rationale: str,
) -> str:
    """Lock a phase's confirmatory strategy set with a hash for audit.

    IMMUTABLE. If the file exists and differs, raises ValueError.
    """
    PRECOMMIT_DIR.mkdir(parents=True, exist_ok=True)
    commitment = {
        "phase": phase_name,
        "strategies": strategies,
        "rationale": rationale,
    }
    content = json.dumps(commitment, sort_keys=True, indent=2)
    sha = hashlib.sha256(content.encode()).hexdigest()[:16]

    path = PRECOMMIT_DIR / f"{phase_name}_confirmatory.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("strategies") != strategies:
            raise ValueError(
                f"Precommit for {phase_name} already exists with DIFFERENT "
                f"strategies. Delete {path} manually to change."
            )
        logger.info("Precommit verified for %s (hash=%s)", phase_name, sha)
    else:
        path.write_text(content)
        logger.info(
            "Precommitted %s: %d strategies (hash=%s)",
            phase_name, len(strategies), sha,
        )
    return sha


# ---------------------------------------------------------------------------
# Statistical evaluation
# ---------------------------------------------------------------------------


def load_canonical_benchmark(
    artifact_paths: list[Path] | None = None,
    overlap_tolerance: float = 1e-4,
) -> pd.Series:
    """R9-HIGH-3: load and reconcile saved `__benchmark__` series across
    multiple phase artifacts.

    Returns the LONGEST benchmark series (by length), after asserting all
    sourced series are identical on their overlapping date range. This
    avoids the prior bug where Joint Holm reused a runtime benchmark
    (e.g., Phase 4's 2012+ slice) and silently truncated returns from
    strategies that ran on the longer 2010+ range.

    Args:
        artifact_paths: list of parquet files containing `__benchmark__`
            columns. Defaults to all phase*_returns.parquet under
            ARTIFACTS_DIR (NOT smoke / contaminated / pre_filter / v2_off
            variants).
        overlap_tolerance: max allowed |diff| on overlapping dates.
    """
    if artifact_paths is None:
        artifact_paths = sorted(
            p for p in ARTIFACTS_DIR.glob("phase*_returns.parquet")
            if not any(tag in p.name for tag in (
                "smoke", "contaminated", "pre_filter", "v2_off"
            ))
        )
    benches = {}
    for p in artifact_paths:
        df = pd.read_parquet(p)
        if "__benchmark__" not in df.columns:
            continue
        ser = df["__benchmark__"].dropna()
        if not ser.empty:
            benches[p.name] = ser

    if not benches:
        raise ValueError("No __benchmark__ columns found in any artifact.")

    # Assert pairwise consistency on overlap
    items = list(benches.items())
    for i, (n_i, b_i) in enumerate(items):
        for n_j, b_j in items[i + 1:]:
            common = b_i.index.intersection(b_j.index)
            if len(common) == 0:
                continue
            diff = (b_i.loc[common] - b_j.loc[common]).abs()
            if diff.max() > overlap_tolerance:
                raise ValueError(
                    f"Benchmark mismatch on overlap between {n_i} and {n_j}: "
                    f"max |diff|={diff.max():.2e} on {len(common)} shared dates."
                )

    longest_name, longest = max(benches.items(), key=lambda kv: len(kv[1]))
    logger.info(
        "Canonical benchmark: %s (%d days, %s to %s); "
        "consistent with %d other artifact benches on overlap.",
        longest_name, len(longest),
        longest.index.min().date(), longest.index.max().date(),
        len(benches) - 1,
    )
    return longest


def evaluate_gate(
    strat_returns: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    config: dict | None = None,
) -> dict[str, dict]:
    """Run block-bootstrap tests + Holm on a set of strategies against benchmark.

    Estimand — pre-committed in CLAUDE.md #1: **Sharpe of excess returns**,
    i.e., Sharpe(strat - bench). The p-value from `block_bootstrap_test` is
    on this same quantity (paired bootstrap of the excess-return series
    under the null that mean excess = 0). The CI is the bootstrap CI of
    that same Sharpe-of-excess (not the diff-of-Sharpes, which is a
    different statistic and is kept for information only).

    Returns dict per strategy with:
        observed_excess_sharpe (Sharpe of excess, gate metric)
        p_value, holm_adjusted_p
        gate_ci_lower, gate_ci_upper (CI on gate metric)
        diff_of_sharpes_point, diff_of_sharpes_lower, diff_of_sharpes_upper
            (alternative diff-of-Sharpes metric, informational only)
        passes_gate (bool)
    """
    config = config or load_config()
    gate = config["gate"]
    boot = config["bootstrap"]

    min_excess = float(gate["min_excess_sharpe"])
    ci_lower_thr = float(gate["ci_lower_threshold"])

    p_values: dict[str, float] = {}
    test_results: dict[str, dict] = {}

    for name, ret in strat_returns.items():
        test = block_bootstrap_test(
            ret, benchmark_returns,
            n_bootstrap=int(boot["n_replicates"]),
            expected_block_length=int(boot["block_length"]),
            seed=int(boot["seed"]),
        )
        ci = excess_sharpe_ci(
            ret, benchmark_returns,
            n_bootstrap=int(boot["n_replicates"]),
            confidence=float(gate["confidence"]),
            expected_block_length=int(boot["block_length"]),
            seed=int(boot["seed"]),
        )
        p_values[name] = test["p_value"]
        # Normalize the result dict: gate uses Sharpe-of-excess throughout.
        test_results[name] = {
            "observed_excess_sharpe": test["observed_excess_sharpe"],
            "p_value": test["p_value"],
            "p_mc_se": test.get("p_mc_se"),
            "null_95th": test.get("null_95th"),
            # Gate CI: on Sharpe-of-excess (matches the p-value)
            "gate_ci_lower": ci["excess_sharpe_lower"],
            "gate_ci_upper": ci["excess_sharpe_upper"],
            # Informational only: diff-of-Sharpes (different estimand)
            "diff_of_sharpes_point": ci["point_estimate"],
            "diff_of_sharpes_lower": ci["ci_lower"],
            "diff_of_sharpes_upper": ci["ci_upper"],
            "strategy_sharpe": ci["strategy_sharpe"],
            "benchmark_sharpe": ci["benchmark_sharpe"],
            "n_bootstrap": boot["n_replicates"],
        }

    holm = holm_bonferroni(p_values)

    for name in test_results:
        t = test_results[name]
        h = holm[name]
        t["holm_adjusted_p"] = h["adjusted_p"]
        t["passes_gate"] = (
            h["significant_05"]
            and t["observed_excess_sharpe"] > min_excess
            and t["gate_ci_lower"] > ci_lower_thr
        )
    return test_results


def print_gate_table(results: dict[str, dict], title: str) -> None:
    print(f"\n{'=' * 110}")
    print(title)
    print("  Gate metric: Sharpe of excess returns (Sharpe(strat - bench)).")
    print("  diff-of-Sharpes shown for reference only; not part of gate.")
    print("=" * 110)
    print(
        f"{'Strategy':<35} {'ExSharpe':>10} {'Raw p':>9} {'Adj p':>9} "
        f"{'90% CI':>24} {'GATE':>8}"
    )
    print("-" * 110)
    for name in sorted(
        results,
        key=lambda k: -results[k]["observed_excess_sharpe"],
    ):
        r = results[name]
        ci_str = f"[{r['gate_ci_lower']:+.3f}, {r['gate_ci_upper']:+.3f}]"
        gate_str = "PASS" if r["passes_gate"] else "FAIL"
        print(
            f"{name:<35} {r['observed_excess_sharpe']:>+10.3f} "
            f"{r['p_value']:>9.4f} {r['holm_adjusted_p']:>9.4f} "
            f"{ci_str:>24} {gate_str:>8}"
        )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def save_phase_returns(
    phase_name: str,
    returns: dict[str, pd.Series],
    benchmark: pd.Series,
) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{phase_name}_returns.parquet"
    df = pd.DataFrame(returns)
    df["__benchmark__"] = benchmark.reindex(df.index)
    df.to_parquet(path)
    logger.info("Saved %d strategy returns to %s", len(returns), path)
    return path
