"""Phase 3: Trend Following — Confirmatory + Descriptive

CONFIRMATORY (1 hypothesis, Holm N=1):
  Strategy: 54/36 VTI/BND + 10% sleeve that holds IAU when IAU > SMA100,
            otherwise holds VGSH (short-term Treasuries)
  Benchmark: Static 54/36/10 VTI/BND/IAU (Phase 2B validated strategy)
  Gate: excess Sharpe > 0.20, Holm p < 0.05, CI lower > 0
  Walk-forward: 36/12/12, full backtester with T+1, costs, PIT

DESCRIPTIVE (no gate):
  1. IAU standalone: buy-and-hold IAU vs SMA100 IAU/VGSH
  2. DBC standalone: buy-and-hold DBC vs SMA100 DBC/VGSH
  3. MaxDD, drawdown duration, regime-specific analysis
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

from youbet.commodity.costs import register_commodity_costs
from youbet.commodity.pit import register_commodity_lags

register_commodity_costs()
register_commodity_lags()

from youbet.etf.backtester import Backtester, BacktestConfig, BacktestResult
from youbet.etf.costs import CostModel
from youbet.etf.strategy import BaseStrategy
from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci

from _shared import (
    load_commodity_universe,
    compute_metrics,
    print_table,
    save_phase_returns,
)
from youbet.commodity.data import fetch_commodity_tbill_rates

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

class StaticAllocation(BaseStrategy):
    """Fixed-weight allocation (benchmark)."""

    def __init__(self, weights: dict[str, float], name_label: str):
        self._weights = weights
        self._name = name_label

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        available = {t: w for t, w in self._weights.items() if t in prices.columns}
        return pd.Series(available)

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {"weights": self._weights}


class SMASleeveTimingStrategy(BaseStrategy):
    """Portfolio with a trend-timed commodity sleeve.

    Holds equity_weight in VTI, bond_weight in BND.
    The commodity sleeve (sleeve_weight) switches between:
      - commodity_ticker when price > SMA(lookback)
      - cash_ticker (VGSH) when price <= SMA(lookback)

    Signal uses data strictly before as_of_date (T+1 enforced by backtester).
    """

    def __init__(
        self,
        commodity_ticker: str = "IAU",
        cash_ticker: str = "VGSH",
        equity_ticker: str = "VTI",
        bond_ticker: str = "BND",
        equity_weight: float = 0.54,
        bond_weight: float = 0.36,
        sleeve_weight: float = 0.10,
        sma_lookback: int = 100,
        name_label: str = "sma_sleeve",
    ):
        self.commodity_ticker = commodity_ticker
        self.cash_ticker = cash_ticker
        self.equity_ticker = equity_ticker
        self.bond_ticker = bond_ticker
        self.equity_weight = equity_weight
        self.bond_weight = bond_weight
        self.sleeve_weight = sleeve_weight
        self.sma_lookback = sma_lookback
        self._name = name_label

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass  # No parameters to estimate — rule-based

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        weights = {
            self.equity_ticker: self.equity_weight,
            self.bond_ticker: self.bond_weight,
        }

        # Check if commodity is above SMA using data strictly before as_of_date
        if self.commodity_ticker in prices.columns:
            hist = prices[self.commodity_ticker].loc[:as_of_date].dropna()
            if len(hist) >= self.sma_lookback:
                sma = hist.iloc[-self.sma_lookback:].mean()
                current_price = hist.iloc[-1]
                if current_price > sma:
                    weights[self.commodity_ticker] = self.sleeve_weight
                else:
                    weights[self.cash_ticker] = self.sleeve_weight
            else:
                # Not enough history for SMA — default to cash
                weights[self.cash_ticker] = self.sleeve_weight
        else:
            weights[self.cash_ticker] = self.sleeve_weight

        return pd.Series(weights)

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {
            "commodity": self.commodity_ticker,
            "sma_lookback": self.sma_lookback,
            "sleeve_weight": self.sleeve_weight,
        }


class StandaloneSMAStrategy(BaseStrategy):
    """Standalone SMA timing: 100% in ticker when above SMA, 100% in cash when below."""

    def __init__(
        self,
        ticker: str,
        cash_ticker: str = "VGSH",
        sma_lookback: int = 100,
        name_label: str = "sma_standalone",
    ):
        self.ticker = ticker
        self.cash_ticker = cash_ticker
        self.sma_lookback = sma_lookback
        self._name = name_label

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        pass

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        if self.ticker in prices.columns:
            hist = prices[self.ticker].loc[:as_of_date].dropna()
            if len(hist) >= self.sma_lookback:
                sma = hist.iloc[-self.sma_lookback:].mean()
                if hist.iloc[-1] > sma:
                    return pd.Series({self.ticker: 1.0})
                else:
                    return pd.Series({self.cash_ticker: 1.0})
        return pd.Series({self.cash_ticker: 1.0})

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {"ticker": self.ticker, "sma_lookback": self.sma_lookback}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prices_and_universe():
    """Load prices and universe."""
    from youbet.etf.data import load_snapshot
    from youbet.commodity.data import SNAPSHOTS_DIR

    snap_dirs = sorted(
        [d.name for d in SNAPSHOTS_DIR.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
        reverse=True,
    )
    prices = load_snapshot(snapshot_date=snap_dirs[0], snapshot_dir=SNAPSHOTS_DIR)
    print(f"  Commodity snapshot: {snap_dirs[0]}")

    from youbet.etf.data import SNAPSHOTS_DIR as ETF_SNAPSHOTS
    etf_snap_dirs = sorted(
        [d.name for d in ETF_SNAPSHOTS.iterdir()
         if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"],
        reverse=True,
    )
    if etf_snap_dirs:
        etf_prices = load_snapshot(snapshot_date=etf_snap_dirs[0], snapshot_dir=ETF_SNAPSHOTS)
        for t in ["VTI", "BND", "VGSH"]:
            if t in etf_prices.columns:
                prices[t] = etf_prices[t]
        print(f"  ETF snapshot: {etf_snap_dirs[0]}")

    universe = load_commodity_universe()
    extra = pd.DataFrame([
        {"ticker": "VTI", "name": "Vanguard Total Stock Market",
         "inception_date": pd.Timestamp("2001-05-24"), "expense_ratio": 0.0003,
         "category": "broad_us_equity", "aum_billions": 427.0},
        {"ticker": "BND", "name": "Vanguard Total Bond Market",
         "inception_date": pd.Timestamp("2007-04-03"), "expense_ratio": 0.0003,
         "category": "broad_us_bond", "aum_billions": 113.0},
    ])
    universe = pd.concat([universe, extra], ignore_index=True)

    return prices, universe


def build_cost_model(universe: pd.DataFrame) -> CostModel:
    """Build cost model from universe."""
    cost_model = CostModel()
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        cost_model.expense_ratios[ticker] = row.get("expense_ratio", 0.0008)
        cat = row.get("category", "default")
        if pd.isna(cat):
            cat = "default"
        cost_model.ticker_categories[ticker] = cat
    return cost_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("PHASE 3: TREND FOLLOWING")
    print("  1 confirmatory hypothesis (Holm N=1) + descriptive secondaries")
    print("  Full backtester: T+1, costs, walk-forward 36/12/12")
    print("=" * 90)

    # Precommitment
    print("\n  PRECOMMITTED CONFIRMATORY HYPOTHESIS (1 of 1):")
    print("    Strategy:  54/36 VTI/BND + 10% sleeve IAU>SMA100 ? IAU : VGSH")
    print("    Benchmark: Static 54/36/10 VTI/BND/IAU")
    print("    Gate:      excess Sharpe > 0.20, p < 0.05, CI lower > 0")
    print("    Holm N=1")
    print()

    # Load data
    prices, universe = load_prices_and_universe()
    tbill_rates = fetch_commodity_tbill_rates(allow_fallback=True)

    config = BacktestConfig(
        train_months=36, test_months=12, step_months=12,
        rebalance_frequency="monthly",
    )
    cost_model = build_cost_model(universe)

    # =====================================================================
    # CONFIRMATORY: IAU SMA100 sleeve timing vs static allocation
    # =====================================================================
    print(f"\n{'=' * 90}")
    print("CONFIRMATORY TEST: IAU SMA100 Sleeve Timing")
    print(f"{'=' * 90}")

    timing_strategy = SMASleeveTimingStrategy(
        commodity_ticker="IAU", cash_ticker="VGSH",
        equity_ticker="VTI", bond_ticker="BND",
        equity_weight=0.54, bond_weight=0.36, sleeve_weight=0.10,
        sma_lookback=100,
        name_label="54_36_10_IAU_SMA100",
    )
    static_benchmark = StaticAllocation(
        weights={"VTI": 0.54, "BND": 0.36, "IAU": 0.10},
        name_label="54_36_10_IAU_static",
    )

    bt = Backtester(config=config, prices=prices, cost_model=cost_model,
                    tbill_rates=tbill_rates, universe=universe)
    result = bt.run(timing_strategy, static_benchmark)

    print(f"\n{result.summary()}")

    # Fold details
    print(f"\n  {'Fold':<10} {'Test Period':<25} {'Strat':>7} {'Bench':>7} {'Delta':>7} {'Turnover':>9}")
    print("  " + "-" * 70)
    for fold in result.fold_results:
        period = f"{fold.test_start.date()} to {fold.test_end.date()}"
        s = fold.metrics.sharpe_ratio if fold.metrics else 0
        b_ret = fold.benchmark_returns
        b_s = float(b_ret.mean() / max(b_ret.std(), 1e-10) * np.sqrt(252)) if len(b_ret) > 0 else 0
        print(f"  {fold.fold_name:<10} {period:<25} {s:>7.3f} {b_s:>7.3f} {s-b_s:>+6.3f} {fold.total_turnover:>9.4f}")

    # Statistical test
    print(f"\n{'=' * 90}")
    print("CONFIRMATORY STATISTICAL TEST (Holm N=1)")
    print(f"{'=' * 90}")

    strat_ret = result.overall_returns
    bench_ret = result.benchmark_returns

    test = block_bootstrap_test(strat_ret, bench_ret, n_bootstrap=10_000, seed=42)
    ci = excess_sharpe_ci(strat_ret, bench_ret, n_bootstrap=10_000, seed=42)

    p_raw = test["p_value"]
    obs_excess = test.get("observed_excess_sharpe", result.excess_sharpe)
    ci_lo = ci.get("ci_lower", 0)
    ci_hi = ci.get("ci_upper", 0)

    passes = obs_excess > 0.20 and p_raw < 0.05 and ci_lo > 0

    print(f"\n  Observed excess Sharpe:  {obs_excess:+.4f}")
    print(f"  p-value:                {p_raw:.4f}")
    print(f"  90% CI:                 [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()
    print(f"  Excess Sharpe > 0.20:   {obs_excess:.4f} {'PASS' if obs_excess > 0.20 else 'FAIL'}")
    print(f"  p < 0.05:               {p_raw:.4f} {'PASS' if p_raw < 0.05 else 'FAIL'}")
    print(f"  CI lower > 0:           {ci_lo:.4f} {'PASS' if ci_lo > 0 else 'FAIL'}")
    print(f"\n  *** STRICT GATE: {'PASS' if passes else 'FAIL'} ***")

    # Regime breakdown
    print(f"\n  Regime breakdown (walk-forward returns):")
    print(f"  {'Regime':<35} {'Strat':>7} {'Bench':>7} {'Delta':>7}")
    print("  " + "-" * 58)
    for start, end, label in [
        ("2007-07-01", "2014-12-31", "Post-GFC + commodity bust"),
        ("2011-09-01", "2018-08-31", "Gold bear market"),
        ("2015-01-01", "2019-12-31", "Low-vol, dollar strength"),
        ("2020-01-01", "2022-12-31", "COVID + inflation"),
        ("2023-01-01", "2026-04-09", "Normalization"),
    ]:
        sw = strat_ret.loc[start:end]
        bw = bench_ret.loc[start:end]
        if len(sw) < 63:
            continue
        ss = float(sw.mean() / max(sw.std(), 1e-10) * np.sqrt(252))
        bs = float(bw.mean() / max(bw.std(), 1e-10) * np.sqrt(252))
        print(f"  {label:<35} {ss:>7.3f} {bs:>7.3f} {ss-bs:>+6.3f}")

    # =====================================================================
    # DESCRIPTIVE: Standalone timing tests
    # =====================================================================
    for ticker, label in [("IAU", "Gold"), ("DBC", "Broad commodity")]:
        print(f"\n{'=' * 90}")
        print(f"DESCRIPTIVE: {label} Standalone SMA100 Timing ({ticker})")
        print(f"{'=' * 90}")

        timing = StandaloneSMAStrategy(
            ticker=ticker, cash_ticker="VGSH", sma_lookback=100,
            name_label=f"{ticker}_SMA100",
        )
        buyhold = StaticAllocation(
            weights={ticker: 1.0},
            name_label=f"{ticker}_buyhold",
        )

        bt2 = Backtester(config=config, prices=prices, cost_model=cost_model,
                         tbill_rates=tbill_rates, universe=universe)
        res = bt2.run(timing, buyhold)

        print(f"\n{res.summary()}")

        # Regime breakdown
        sr = res.overall_returns
        br = res.benchmark_returns
        print(f"\n  Regime breakdown:")
        print(f"  {'Regime':<35} {'SMA100':>7} {'B&H':>7} {'Delta':>7}")
        print("  " + "-" * 58)
        for start, end, rlabel in [
            ("2007-07-01", "2014-12-31", "Post-GFC + commodity bust"),
            ("2011-09-01", "2018-08-31", "Gold bear market"),
            ("2015-01-01", "2019-12-31", "Low-vol, dollar strength"),
            ("2020-01-01", "2022-12-31", "COVID + inflation"),
            ("2023-01-01", "2026-04-09", "Normalization"),
        ]:
            sw = sr.loc[start:end]
            bw = br.loc[start:end]
            if len(sw) < 63:
                continue
            ss = float(sw.mean() / max(sw.std(), 1e-10) * np.sqrt(252))
            bs = float(bw.mean() / max(bw.std(), 1e-10) * np.sqrt(252))
            print(f"  {rlabel:<35} {ss:>7.3f} {bs:>7.3f} {ss-bs:>+6.3f}")

        # Drawdown comparison
        for name_tag, ret in [(f"{ticker}_SMA100", sr), (f"{ticker}_B&H", br)]:
            cum = (1 + ret).cumprod()
            rm = cum.cummax()
            dd = (cum - rm) / rm
            print(f"  {name_tag}: MaxDD {dd.min():.1%}, duration {(dd == 0).astype(int).groupby((dd != 0).cumsum()).sum().max()} days recovery")

    # =====================================================================
    # Summary
    # =====================================================================
    print(f"\n{'=' * 90}")
    print("PHASE 3 SUMMARY")
    print(f"{'=' * 90}")
    print(f"\n  Confirmatory: IAU SMA100 sleeve timing vs static 54/36/10")
    print(f"    Gate result: {'PASS' if passes else 'FAIL'}")
    print(f"    Excess Sharpe: {obs_excess:+.4f}, p={p_raw:.4f}, CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()
    print("  Descriptive: standalone timing tests (IAU, DBC) reported above")

    # Persist
    save_phase_returns(
        "phase3",
        {"IAU_SMA100_sleeve": strat_ret},
        {"static_54_36_10_IAU": bench_ret},
    )

    return result, passes


if __name__ == "__main__":
    main()
