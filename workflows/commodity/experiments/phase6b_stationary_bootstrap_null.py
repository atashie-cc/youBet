"""Phase 6B: Stationary Bootstrap Null for Macro-Gated DBC

Codex R8 recommended a more defensible null test: stationary bootstrap
of BOTH macro series (T10YIE and DXY) with matched block length, paired
with the true DBC price series. This tests whether the observed
Sharpe diff is distinguishable from what random blocks of the two
macro series would produce, preserving each series' autocorrelation
structure while breaking their joint alignment with DBC.

This is a stronger null than Phase 6.8 which only shifted T10YIE
and kept DXY real. If the observed effect survives this null, the
dollar-linked commodity timing idea has stronger empirical support.

Design:
  - 200 stationary block bootstrap draws (more than Phase 6.8's 100)
  - 22-day expected block length (matches project standard)
  - Both T10YIE and DXY resampled independently
  - Full walk-forward backtest per draw
  - PIT-safe (uses as_of() method)
  - Observed effect (Phase 6.1 result) compared to null distribution
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

from youbet.commodity.costs import register_commodity_costs
from youbet.commodity.pit import register_commodity_lags
register_commodity_costs()
register_commodity_lags()

from youbet.etf.backtester import Backtester, BacktestConfig
from youbet.etf.costs import CostModel
from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy
from youbet.etf.stats import excess_sharpe_ci

from _shared import load_commodity_universe
from youbet.commodity.data import fetch_commodity_tbill_rates
from youbet.commodity.macro.fetchers import fetch_breakeven_inflation, fetch_dollar_index

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class StaticAllocation(BaseStrategy):
    def __init__(self, weights: dict, name_label: str):
        self._weights = weights
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        return pd.Series({t: w for t, w in self._weights.items() if t in prices.columns})

    @property
    def name(self): return self._name

    @property
    def params(self): return {"weights": self._weights}


class MacroGatedAND(BaseStrategy):
    """PIT-safe macro-gated DBC strategy (frozen 126-day AND rule)."""
    def __init__(self, breakeven, dxy, lookback=126, name_label="macro_DBC_AND"):
        self.breakeven = breakeven
        self.dxy = dxy
        self.lookback = lookback
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        if "DBC" not in prices.columns:
            return pd.Series(dtype=float)

        be = self.breakeven.as_of(as_of_date).dropna()
        dx = self.dxy.as_of(as_of_date).dropna()

        if len(be) < self.lookback or len(dx) < self.lookback:
            return pd.Series(dtype=float)

        be_rising = be.iloc[-1] > be.iloc[-self.lookback:].mean()
        dx_weak = dx.iloc[-1] < dx.iloc[-self.lookback:].mean()

        if be_rising and dx_weak:
            return pd.Series({"DBC": 1.0})
        return pd.Series(dtype=float)

    @property
    def name(self): return self._name

    @property
    def params(self): return {"lookback": self.lookback}


def stationary_block_bootstrap_series(
    series: pd.Series,
    expected_block_length: int = 22,
    rng: np.random.Generator = None,
) -> pd.Series:
    """Generate one stationary block bootstrap draw of a time series.

    Preserves autocorrelation within random-length blocks. Keeps the
    original index so release_dates remain aligned with observation times.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(series)
    p = 1.0 / expected_block_length

    # Generate indices
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < p:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n

    # Create new series with bootstrapped values but original index
    new_values = series.values[indices]
    return pd.Series(new_values, index=series.index, name=series.name)


def load_all():
    from youbet.etf.data import load_snapshot
    from youbet.commodity.data import SNAPSHOTS_DIR
    from youbet.etf.data import SNAPSHOTS_DIR as ETF_SNAP

    snap = sorted([d.name for d in SNAPSHOTS_DIR.iterdir()
                   if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"], reverse=True)
    prices = load_snapshot(snapshot_date=snap[0], snapshot_dir=SNAPSHOTS_DIR)

    etf_snap = sorted([d.name for d in ETF_SNAP.iterdir()
                       if d.is_dir() and len(d.name) == 10 and d.name[4] == "-"], reverse=True)
    if etf_snap:
        ep = load_snapshot(snapshot_date=etf_snap[0], snapshot_dir=ETF_SNAP)
        for t in ["VTI", "BND", "VGSH"]:
            if t in ep.columns:
                prices[t] = ep[t]

    universe = load_commodity_universe()
    extra = pd.DataFrame([
        {"ticker": "VTI", "name": "VTI", "inception_date": pd.Timestamp("2001-05-24"),
         "expense_ratio": 0.0003, "category": "broad_us_equity", "aum_billions": 427.0},
        {"ticker": "BND", "name": "BND", "inception_date": pd.Timestamp("2007-04-03"),
         "expense_ratio": 0.0003, "category": "broad_us_bond", "aum_billions": 113.0},
    ])
    universe = pd.concat([universe, extra], ignore_index=True)

    tbill = fetch_commodity_tbill_rates(allow_fallback=True)
    cost = CostModel()
    for _, r in universe.iterrows():
        cost.expense_ratios[r["ticker"]] = r.get("expense_ratio", 0.0008)
        cat = r.get("category", "default")
        cost.ticker_categories[r["ticker"]] = cat if not pd.isna(cat) else "default"

    return prices, universe, tbill, cost


def main():
    print("=" * 95)
    print("PHASE 6B: STATIONARY BOOTSTRAP NULL FOR MACRO-GATED DBC")
    print("  Both T10YIE and DXY resampled with 22-day blocks")
    print("  Frozen 126-day AND rule, PIT-safe implementation")
    print("=" * 95)

    prices, universe, tbill, cost = load_all()
    print(f"  Prices: {prices.index.min().date()} to {prices.index.max().date()}")

    be = fetch_breakeven_inflation()
    dxy = fetch_dollar_index()
    print(f"  Breakeven inflation: {len(be.values)} observations")
    print(f"  Dollar index: {len(dxy.values)} observations")

    config = BacktestConfig(train_months=36, test_months=12, step_months=12,
                            rebalance_frequency="monthly")
    benchmark = StaticAllocation({"DBC": 1.0}, "DBC_BH")

    # --- Observed Sharpe diff (Phase 6.1 PIT-safe rerun) ---
    print(f"\n{'=' * 95}")
    print("OBSERVED EFFECT")
    print(f"{'=' * 95}")

    observed_strategy = MacroGatedAND(be, dxy, 126, "observed")
    bt = Backtester(config=config, prices=prices, cost_model=cost,
                    tbill_rates=tbill, universe=universe)
    observed_result = bt.run(observed_strategy, benchmark)

    observed_ci = excess_sharpe_ci(
        observed_result.overall_returns,
        observed_result.benchmark_returns,
        n_bootstrap=10_000, seed=42,
    )
    observed_diff = observed_ci.get("point_estimate", 0)
    print(f"  Observed Sharpe diff: {observed_diff:+.4f}")
    print(f"  90% CI: [{observed_ci.get('ci_lower', 0):+.4f}, {observed_ci.get('ci_upper', 0):+.4f}]")

    # --- Stationary bootstrap null ---
    print(f"\n{'=' * 95}")
    print("STATIONARY BOOTSTRAP NULL (N=200, both series, 22-day blocks)")
    print(f"{'=' * 95}")

    n_shuffles = 200
    block_length = 22
    null_diffs = []

    rng = np.random.default_rng(42)

    for i in range(n_shuffles):
        # Resample both series independently with stationary block bootstrap
        be_boot_vals = stationary_block_bootstrap_series(
            be.values, block_length, rng,
        )
        dxy_boot_vals = stationary_block_bootstrap_series(
            dxy.values, block_length, rng,
        )

        # Create PITFeatureSeries with bootstrapped values but original release dates
        be_boot = PITFeatureSeries(
            values=be_boot_vals,
            release_dates=be.release_dates,
            feature_name="breakeven_inflation",
            lag_days=be.lag_days,
        )
        dxy_boot = PITFeatureSeries(
            values=dxy_boot_vals,
            release_dates=dxy.release_dates,
            feature_name="dollar_index",
            lag_days=dxy.lag_days,
        )

        strategy = MacroGatedAND(be_boot, dxy_boot, 126, f"null_{i}")

        try:
            bt_null = Backtester(config=config, prices=prices, cost_model=cost,
                                 tbill_rates=tbill, universe=universe)
            res = bt_null.run(strategy, benchmark)
            ci = excess_sharpe_ci(
                res.overall_returns, res.benchmark_returns,
                n_bootstrap=500, seed=42,  # Small inner bootstrap for speed
            )
            null_diffs.append(ci.get("point_estimate", 0))
        except Exception as e:
            logger.warning(f"Null shuffle {i} failed: {e}")

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_shuffles} shuffles, "
                  f"current null mean: {np.mean(null_diffs):+.4f}")

    null_diffs = np.array(null_diffs)

    # --- Analysis ---
    print(f"\n{'=' * 95}")
    print("NULL DISTRIBUTION ANALYSIS")
    print(f"{'=' * 95}")

    print(f"\n  Observed Sharpe diff:    {observed_diff:+.4f}")
    print(f"  Null shuffles:           {len(null_diffs)}")
    print(f"  Null mean:               {null_diffs.mean():+.4f}")
    print(f"  Null median:             {np.median(null_diffs):+.4f}")
    print(f"  Null std:                {null_diffs.std():+.4f}")
    print(f"  Null 5th percentile:     {np.percentile(null_diffs, 5):+.4f}")
    print(f"  Null 50th percentile:    {np.percentile(null_diffs, 50):+.4f}")
    print(f"  Null 90th percentile:    {np.percentile(null_diffs, 90):+.4f}")
    print(f"  Null 95th percentile:    {np.percentile(null_diffs, 95):+.4f}")
    print(f"  Null 99th percentile:    {np.percentile(null_diffs, 99):+.4f}")
    print(f"  Null min:                {null_diffs.min():+.4f}")
    print(f"  Null max:                {null_diffs.max():+.4f}")

    fraction_above = (null_diffs >= observed_diff).mean()
    p_permutation = (1 + (null_diffs >= observed_diff).sum()) / (n_shuffles + 1)

    print(f"\n  Fraction null >= observed: {fraction_above:.4f}")
    print(f"  Permutation p-value:       {p_permutation:.4f} (one-sided)")
    print(f"  Observed > null 90th:      {'PASS' if observed_diff > np.percentile(null_diffs, 90) else 'FAIL'}")
    print(f"  Observed > null 95th:      {'PASS' if observed_diff > np.percentile(null_diffs, 95) else 'FAIL'}")
    print(f"  Observed > null 99th:      {'PASS' if observed_diff > np.percentile(null_diffs, 99) else 'FAIL'}")

    # Save null distribution for inspection
    out_path = WORKFLOW_ROOT / "artifacts" / "phase6b_null_distribution.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(null_diffs, name="sharpe_diff").to_csv(out_path, index=False)
    print(f"\n  Null distribution saved to {out_path}")

    print(f"\n{'=' * 95}")
    print("INTERPRETATION")
    print(f"{'=' * 95}")
    if p_permutation < 0.05:
        print("  Null REJECTED at p<0.05 — observed effect is distinguishable from")
        print("  random resamples of the macro series.")
    elif p_permutation < 0.10:
        print("  Null REJECTED at p<0.10 only — moderate evidence against noise.")
    else:
        print("  Null NOT REJECTED — observed effect is not distinguishable from")
        print("  what stationary bootstrap resamples of the macro series would produce.")

    print(f"\n  This null is stricter than Phase 6.8 (which shifted only T10YIE).")
    print(f"  It preserves each series' autocorrelation but breaks joint alignment")
    print(f"  with the true DBC return path.")


if __name__ == "__main__":
    main()
