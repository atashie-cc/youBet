"""Shared utilities for individual-stocks-snp500 experiments.

Reuses the stock-selection engine AND its already-downloaded data (universe
CSVs, EDGAR cache, price snapshots) to avoid re-fetching. Config inherits
stock-selection's locked thresholds via an `inherits:` key.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.data import fetch_tbill_rates  # noqa: E402
from youbet.etf.stats import (  # noqa: E402
    block_bootstrap_test,
    excess_sharpe_ci,
    holm_bonferroni,
)
from youbet.stock.backtester import StockBacktestConfig  # noqa: E402
from youbet.stock.costs import StockCostModel  # noqa: E402
from youbet.stock.data import fetch_stock_prices  # noqa: E402
from youbet.stock.edgar import EdgarConfig, IndexedFacts, get_company_facts  # noqa: E402
from youbet.stock.fundamentals import TickerFundamentalsPanel, _clear_caches  # noqa: E402
from youbet.stock.universe import Universe  # noqa: E402

logger = logging.getLogger(__name__)

CONFIG_PATH = WORKFLOW_ROOT / "config.yaml"
PRECOMMIT_DIR = WORKFLOW_ROOT / "precommit"
ARTIFACTS_DIR = WORKFLOW_ROOT / "artifacts"

# Reuse stock-selection's downloaded data (do NOT duplicate the EDGAR cache).
STOCK_SELECTION_ROOT = REPO_ROOT / "workflows" / "stock-selection"
SHARED_UNIVERSE_DIR = STOCK_SELECTION_ROOT / "universe"
SHARED_PRICE_SNAPSHOTS = STOCK_SELECTION_ROOT / "data" / "snapshots" / "prices"
SHARED_EDGAR_CACHE = STOCK_SELECTION_ROOT / "data" / "snapshots" / "edgar"


# --------------------------------------------------------------------------
# Config (with inherits merge)
# --------------------------------------------------------------------------

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k == "inherits":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    with CONFIG_PATH.open() as fh:
        child = yaml.safe_load(fh) or {}
    parent_ref = child.get("inherits")
    if parent_ref:
        parent_path = (CONFIG_PATH.parent / parent_ref).resolve()
        with parent_path.open() as fh:
            parent = yaml.safe_load(fh) or {}
        return _deep_merge(parent, child)
    return child


def load_sp500_universe() -> Universe:
    return Universe.from_csv(
        SHARED_UNIVERSE_DIR / "sp500_membership.csv",
        SHARED_UNIVERSE_DIR / "delisting_returns.csv",
        index_name="S&P 500",
    )


# --------------------------------------------------------------------------
# Backtester construction
# --------------------------------------------------------------------------

def make_backtest_config(config: dict | None = None,
                         first_test_start_min=None) -> StockBacktestConfig:
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


def make_cost_model(config: dict | None = None) -> StockCostModel:
    return StockCostModel.from_config(config or load_config())


def load_prices_with_benchmark(universe: Universe, start: str = "2005-01-01",
                               config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    bench = config["benchmark"]["ticker"]
    prices = fetch_stock_prices(universe=universe, start=start,
                                snapshot_dir=SHARED_PRICE_SNAPSHOTS,
                                extra_tickers=[bench])
    return prices.loc[prices.index >= pd.Timestamp(start)]


def load_facts_by_ticker(universe: Universe) -> dict:
    """Build TickerFundamentalsPanel per ticker from the reused EDGAR cache."""
    cfg = EdgarConfig(cache_dir=SHARED_EDGAR_CACHE)
    ticker_to_cik = {}
    for _, row in universe.membership.sort_values("start_date").iterrows():
        c = str(row["cik"]).strip() if pd.notna(row["cik"]) else ""
        if c and c != "nan":
            ticker_to_cik[row["ticker"]] = c.zfill(10)
    facts: dict[str, TickerFundamentalsPanel] = {}
    loaded = missing = 0
    for t, cik in ticker_to_cik.items():
        path = SHARED_EDGAR_CACHE / f"CIK{cik}.parquet"
        if not path.exists():
            missing += 1
            continue
        try:
            indexed = IndexedFacts(get_company_facts(cik, cfg))
            facts[t] = TickerFundamentalsPanel.build(t, indexed)
            loaded += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("facts load failed %s (CIK %s): %s", t, cik, exc)
            missing += 1
    logger.info("EDGAR panels: %d built, %d missing/no-cache", loaded, missing)
    return facts


def get_tbill(prices: pd.DataFrame) -> pd.Series:
    try:
        return fetch_tbill_rates(
            start=prices.index.min().strftime("%Y-%m-%d"),
            end=prices.index.max().strftime("%Y-%m-%d"), allow_fallback=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("T-bill fetch failed (%s); 4%% constant", exc)
        return pd.Series(0.04, index=prices.index, name="tbill_3m")


# --------------------------------------------------------------------------
# Gate evaluation (Sharpe-of-excess estimand + Holm) — copied from
# stock-selection _shared to keep the locked gate identical.
# --------------------------------------------------------------------------

def evaluate_gate(strat_returns: dict[str, pd.Series],
                  benchmark_returns: pd.Series,
                  config: dict | None = None) -> dict[str, dict]:
    config = config or load_config()
    gate, boot = config["gate"], config["bootstrap"]
    min_excess = float(gate["min_excess_sharpe"])
    ci_lower_thr = float(gate["ci_lower_threshold"])
    p_values, results = {}, {}
    for name, ret in strat_returns.items():
        test = block_bootstrap_test(
            ret, benchmark_returns, n_bootstrap=int(boot["n_replicates"]),
            expected_block_length=int(boot["block_length"]), seed=int(boot["seed"]))
        ci = excess_sharpe_ci(
            ret, benchmark_returns, n_bootstrap=int(boot["n_replicates"]),
            confidence=float(gate["confidence"]),
            expected_block_length=int(boot["block_length"]), seed=int(boot["seed"]))
        p_values[name] = test["p_value"]
        results[name] = {
            "observed_excess_sharpe": test["observed_excess_sharpe"],
            "p_value": test["p_value"],
            "gate_ci_lower": ci["excess_sharpe_lower"],
            "gate_ci_upper": ci["excess_sharpe_upper"],
            "diff_of_sharpes_point": ci["point_estimate"],
            "strategy_sharpe": ci["strategy_sharpe"],
            "benchmark_sharpe": ci["benchmark_sharpe"],
        }
    holm = holm_bonferroni(p_values)
    for name, r in results.items():
        h = holm[name]
        r["holm_adjusted_p"] = h["adjusted_p"]
        r["passes_gate"] = (h["significant_05"]
                            and r["observed_excess_sharpe"] > min_excess
                            and r["gate_ci_lower"] > ci_lower_thr)
    return results


def print_gate_table(results: dict[str, dict], title: str) -> None:
    print("\n" + "=" * 104)
    print(title)
    print("  Gate metric: Sharpe of excess returns (Sharpe(strat - SPY)).")
    print("=" * 104)
    print(f"{'Strategy':<34}{'ExSharpe':>10}{'Raw p':>9}{'Adj p':>9}"
          f"{'90% CI (excess)':>26}{'GATE':>8}")
    print("-" * 104)
    for name in sorted(results, key=lambda k: -results[k]["observed_excess_sharpe"]):
        r = results[name]
        ci = f"[{r['gate_ci_lower']:+.3f}, {r['gate_ci_upper']:+.3f}]"
        print(f"{name:<34}{r['observed_excess_sharpe']:>+10.3f}{r['p_value']:>9.4f}"
              f"{r['holm_adjusted_p']:>9.4f}{ci:>26}"
              f"{'PASS' if r['passes_gate'] else 'FAIL':>8}")


def save_phase_returns(phase_name: str, returns: dict[str, pd.Series],
                       benchmark: pd.Series) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{phase_name}_returns.parquet"
    df = pd.DataFrame(returns)
    df["__benchmark__"] = benchmark.reindex(df.index)
    df.to_parquet(path)
    logger.info("Saved %d strategy returns -> %s", len(returns), path)
    return path


def clear_fundamentals_cache() -> None:
    _clear_caches()
