"""Phases 3R + 4D + 4C + 5D: Exhaustive Commodity Testing

Incorporates all Codex R5 fixes and runs the full remaining test battery.

PHASE 3R: Fix and rerun Phase 3 (estimand fix, VGSH inception handling)
PHASE 4D: Descriptive diagnostics (gold SMA200/252, static 15%/20% WF, TIPS)
PHASE 4C: Confirmatory (1 hypothesis): VTI SMA100 + static IAU vs static 54/36/10
PHASE 5D: Descriptive (inflation-conditional DBC, TIPS portfolio)

Fixes from Codex R5:
  - Consistent Sharpe estimand: use excess_sharpe_ci for point estimate + CI,
    block_bootstrap_test for p-value only
  - Start walk-forward at 2009-12-01 (after VGSH inception 2009-11-19)
    to avoid pre-inception contamination
  - Narrow SMA conclusions to tested lookbacks only
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
from youbet.etf.strategy import BaseStrategy
from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci

from _shared import (
    load_commodity_universe, compute_metrics, print_table, save_phase_returns,
)
from youbet.commodity.data import fetch_commodity_tbill_rates

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REGIME_WINDOWS = [
    ("2009-12-01", "2014-12-31", "Post-GFC recovery"),
    ("2011-09-01", "2018-08-31", "Gold bear market"),
    ("2015-01-01", "2019-12-31", "Low-vol, dollar strength"),
    ("2020-01-01", "2022-12-31", "COVID + inflation"),
    ("2023-01-01", "2026-04-09", "Normalization"),
]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class StaticAllocation(BaseStrategy):
    def __init__(self, weights: dict[str, float], name_label: str):
        self._weights = weights
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        return pd.Series({t: w for t, w in self._weights.items() if t in prices.columns})

    @property
    def name(self): return self._name

    @property
    def params(self): return {"weights": self._weights}


class SMASleeveStrategy(BaseStrategy):
    """Portfolio with SMA-timed commodity sleeve."""
    def __init__(self, commodity: str, cash: str, equity: str, bond: str,
                 eq_w: float, bond_w: float, sleeve_w: float,
                 sma: int, name_label: str):
        self.commodity = commodity
        self.cash = cash
        self.equity = equity
        self.bond = bond
        self.eq_w = eq_w
        self.bond_w = bond_w
        self.sleeve_w = sleeve_w
        self.sma = sma
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        w = {self.equity: self.eq_w, self.bond: self.bond_w}
        if self.commodity in prices.columns:
            h = prices[self.commodity].loc[:as_of_date].dropna()
            if len(h) >= self.sma and h.iloc[-1] > h.iloc[-self.sma:].mean():
                w[self.commodity] = self.sleeve_w
            else:
                w[self.cash] = self.sleeve_w
        else:
            w[self.cash] = self.sleeve_w
        return pd.Series(w)

    @property
    def name(self): return self._name

    @property
    def params(self): return {"commodity": self.commodity, "sma": self.sma}


class VTITimingWithStaticGold(BaseStrategy):
    """Time VTI with SMA, hold static gold. Bond/cash absorbs equity risk-off."""
    def __init__(self, sma: int = 100, gold_w: float = 0.10, name_label: str = "vti_sma_gold"):
        self.sma = sma
        self.gold_w = gold_w
        self._name = name_label
        self.eq_target = 0.60 * (1 - gold_w)  # 0.54
        self.bond_target = 0.40 * (1 - gold_w)  # 0.36

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        w = {"IAU": self.gold_w}
        if "VTI" in prices.columns:
            h = prices["VTI"].loc[:as_of_date].dropna()
            if len(h) >= self.sma and h.iloc[-1] > h.iloc[-self.sma:].mean():
                w["VTI"] = self.eq_target
                w["BND"] = self.bond_target
            else:
                # Risk-off: move equity to bonds
                w["BND"] = self.eq_target + self.bond_target
        else:
            w["BND"] = self.eq_target + self.bond_target
        return pd.Series(w)

    @property
    def name(self): return self._name

    @property
    def params(self): return {"sma": self.sma, "gold_w": self.gold_w}


class StandaloneSMA(BaseStrategy):
    def __init__(self, ticker: str, cash: str = "VGSH", sma: int = 100, name_label: str = "sma"):
        self.ticker = ticker
        self.cash = cash
        self.sma = sma
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        if self.ticker in prices.columns:
            h = prices[self.ticker].loc[:as_of_date].dropna()
            if len(h) >= self.sma and h.iloc[-1] > h.iloc[-self.sma:].mean():
                return pd.Series({self.ticker: 1.0})
        return pd.Series({self.cash: 1.0})

    @property
    def name(self): return self._name

    @property
    def params(self): return {"ticker": self.ticker, "sma": self.sma}


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

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

    # Trim to post-VGSH inception to avoid survivorship contamination
    prices = prices.loc["2009-12-01":]
    print(f"  Prices trimmed to post-VGSH: {prices.index.min().date()} to {prices.index.max().date()}")

    universe = load_commodity_universe()
    # Note: VGSH is already in commodity_universe.csv as cash_equivalent.
    # Phase 4 originally omitted adding VTI/BND but the universe already had VGSH.
    # The bug was that VGSH was NOT in the combined universe here — fixed by
    # verifying VGSH is in commodity universe and adding VTI/BND.
    extra = pd.DataFrame([
        {"ticker": "VTI", "name": "VTI", "inception_date": pd.Timestamp("2001-05-24"),
         "expense_ratio": 0.0003, "category": "broad_us_equity", "aum_billions": 427.0},
        {"ticker": "BND", "name": "BND", "inception_date": pd.Timestamp("2007-04-03"),
         "expense_ratio": 0.0003, "category": "broad_us_bond", "aum_billions": 113.0},
    ])
    universe = pd.concat([universe, extra], ignore_index=True)

    # Verify VGSH is in universe (should be from commodity_universe.csv)
    assert "VGSH" in universe["ticker"].values, \
        "VGSH must be in universe — required as cash sleeve for SMA strategies"

    tbill = fetch_commodity_tbill_rates(allow_fallback=True)
    cost = CostModel()
    for _, r in universe.iterrows():
        cost.expense_ratios[r["ticker"]] = r.get("expense_ratio", 0.0008)
        cat = r.get("category", "default")
        cost.ticker_categories[r["ticker"]] = cat if not pd.isna(cat) else "default"

    return prices, universe, tbill, cost


def run_test(strategy, benchmark, prices, cost, tbill, universe, label,
             confirmatory=False, n_boot=10_000):
    """Run a walk-forward test and print results."""
    config = BacktestConfig(train_months=36, test_months=12, step_months=12,
                            rebalance_frequency="monthly")
    bt = Backtester(config=config, prices=prices, cost_model=cost,
                    tbill_rates=tbill, universe=universe)
    result = bt.run(strategy, benchmark)

    print(f"\n  {result.strategy_name} vs {benchmark.name}")
    print(f"  Period: {result.overall_returns.index[0].date()} to {result.overall_returns.index[-1].date()}")
    print(f"  Folds: {len(result.fold_results)}")

    # Metrics
    sm = result.overall_metrics
    bm = result.benchmark_metrics
    print(f"  Strategy: Sharpe {sm.sharpe_ratio:.3f}, CAGR {sm.annualized_return:.1%}, "
          f"MaxDD {sm.max_drawdown:.1%}, Vol {sm.annualized_volatility:.1%}")
    print(f"  Benchmark: Sharpe {bm.sharpe_ratio:.3f}, CAGR {bm.annualized_return:.1%}, "
          f"MaxDD {bm.max_drawdown:.1%}, Vol {bm.annualized_volatility:.1%}")
    print(f"  Turnover: {result.total_turnover:.1f}, Cost drag: {result.total_cost_drag:.4f}")

    # Consistent estimand: CI from excess_sharpe_ci, p from block_bootstrap_test
    sr = result.overall_returns
    br = result.benchmark_returns
    ci = excess_sharpe_ci(sr, br, n_bootstrap=n_boot, seed=42)
    test = block_bootstrap_test(sr, br, n_bootstrap=n_boot, seed=42)

    # Use Sharpe difference (not Sharpe-of-excess) for point estimate
    sharpe_diff = ci.get("point_estimate", sm.sharpe_ratio - bm.sharpe_ratio)
    ci_lo = ci.get("ci_lower", 0)
    ci_hi = ci.get("ci_upper", 0)
    p_val = test["p_value"]

    print(f"\n  Sharpe difference: {sharpe_diff:+.4f}")
    print(f"  p-value (one-sided): {p_val:.4f}")
    print(f"  90% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    if confirmatory:
        passes = sharpe_diff > 0.20 and p_val < 0.05 and ci_lo > 0
        print(f"\n  GATE: Sharpe diff > 0.20: {sharpe_diff:.4f} {'PASS' if sharpe_diff > 0.20 else 'FAIL'}")
        print(f"  GATE: p < 0.05: {p_val:.4f} {'PASS' if p_val < 0.05 else 'FAIL'}")
        print(f"  GATE: CI lower > 0: {ci_lo:.4f} {'PASS' if ci_lo > 0 else 'FAIL'}")
        print(f"  *** STRICT GATE: {'PASS' if passes else 'FAIL'} ***")
    else:
        excl = "YES" if ci_lo > 0 else ("NEG" if ci_hi < 0 else "NO")
        print(f"  Nominal CI excludes zero: {excl}")

    # Regime breakdown
    print(f"\n  Regime breakdown:")
    print(f"  {'Regime':<35} {'Strat':>7} {'Bench':>7} {'Delta':>7}")
    print("  " + "-" * 58)
    for start, end, rlabel in REGIME_WINDOWS:
        sw = sr.loc[start:end]
        bw = br.loc[start:end]
        if len(sw) < 63: continue
        ss = float(sw.mean() / max(sw.std(), 1e-10) * np.sqrt(252))
        bs = float(bw.mean() / max(bw.std(), 1e-10) * np.sqrt(252))
        print(f"  {rlabel:<35} {ss:>7.3f} {bs:>7.3f} {ss-bs:>+6.3f}")

    return result, sharpe_diff, p_val, ci_lo, ci_hi


def main():
    print("=" * 90)
    print("PHASES 3R + 4D + 4C + 5D: EXHAUSTIVE COMMODITY TESTING")
    print("  Post-VGSH inception (2009-12+), consistent Sharpe estimand")
    print("=" * 90)

    prices, universe, tbill, cost = load_all()

    all_results = {}

    # =====================================================================
    # PHASE 3R: Rerun gold SMA100 sleeve timing with fixes
    # =====================================================================
    print(f"\n{'=' * 90}")
    print("PHASE 3R: IAU SMA100 Sleeve Timing (fixed)")
    print(f"{'=' * 90}")

    r, *_ = run_test(
        SMASleeveStrategy("IAU", "VGSH", "VTI", "BND", 0.54, 0.36, 0.10, 100,
                          "IAU_SMA100_sleeve"),
        StaticAllocation({"VTI": 0.54, "BND": 0.36, "IAU": 0.10}, "static_54_36_10"),
        prices, cost, tbill, universe, "Phase 3R", confirmatory=False,
    )
    all_results["IAU_SMA100_sleeve"] = r

    # =====================================================================
    # PHASE 4D: Descriptive diagnostics
    # =====================================================================
    print(f"\n{'=' * 90}")
    print("PHASE 4D: DESCRIPTIVE DIAGNOSTICS")
    print(f"{'=' * 90}")

    # 4D.1: Gold SMA200 sleeve
    print(f"\n--- 4D.1: IAU SMA200 Sleeve ---")
    r, *_ = run_test(
        SMASleeveStrategy("IAU", "VGSH", "VTI", "BND", 0.54, 0.36, 0.10, 200,
                          "IAU_SMA200_sleeve"),
        StaticAllocation({"VTI": 0.54, "BND": 0.36, "IAU": 0.10}, "static_54_36_10"),
        prices, cost, tbill, universe, "4D.1",
    )
    all_results["IAU_SMA200_sleeve"] = r

    # 4D.2: Gold SMA252 sleeve
    print(f"\n--- 4D.2: IAU SMA252 Sleeve ---")
    r, *_ = run_test(
        SMASleeveStrategy("IAU", "VGSH", "VTI", "BND", 0.54, 0.36, 0.10, 252,
                          "IAU_SMA252_sleeve"),
        StaticAllocation({"VTI": 0.54, "BND": 0.36, "IAU": 0.10}, "static_54_36_10"),
        prices, cost, tbill, universe, "4D.2",
    )
    all_results["IAU_SMA252_sleeve"] = r

    # 4D.3: Standalone gold SMA200 vs B&H
    print(f"\n--- 4D.3: IAU Standalone SMA200 vs B&H ---")
    r, *_ = run_test(
        StandaloneSMA("IAU", "VGSH", 200, "IAU_SMA200"),
        StaticAllocation({"IAU": 1.0}, "IAU_BH"),
        prices, cost, tbill, universe, "4D.3",
    )
    all_results["IAU_SMA200_standalone"] = r

    # 4D.4: DBC SMA100 standalone (rerun with fixes)
    print(f"\n--- 4D.4: DBC Standalone SMA100 vs B&H (fixed) ---")
    r, *_ = run_test(
        StandaloneSMA("DBC", "VGSH", 100, "DBC_SMA100"),
        StaticAllocation({"DBC": 1.0}, "DBC_BH"),
        prices, cost, tbill, universe, "4D.4",
    )
    all_results["DBC_SMA100_standalone"] = r

    # 4D.5: DBC SMA200 standalone
    print(f"\n--- 4D.5: DBC Standalone SMA200 vs B&H ---")
    r, *_ = run_test(
        StandaloneSMA("DBC", "VGSH", 200, "DBC_SMA200"),
        StaticAllocation({"DBC": 1.0}, "DBC_BH"),
        prices, cost, tbill, universe, "4D.5",
    )
    all_results["DBC_SMA200_standalone"] = r

    # 4D.6: Static gold 15% walk-forward
    print(f"\n--- 4D.6: Static 51/34/15 VTI/BND/IAU vs 60/40 ---")
    r, *_ = run_test(
        StaticAllocation({"VTI": 0.51, "BND": 0.34, "IAU": 0.15}, "static_51_34_15"),
        StaticAllocation({"VTI": 0.60, "BND": 0.40}, "static_60_40"),
        prices, cost, tbill, universe, "4D.6",
    )
    all_results["static_15pct_IAU"] = r

    # 4D.7: Static gold 20% walk-forward
    print(f"\n--- 4D.7: Static 48/32/20 VTI/BND/IAU vs 60/40 ---")
    r, *_ = run_test(
        StaticAllocation({"VTI": 0.48, "BND": 0.32, "IAU": 0.20}, "static_48_32_20"),
        StaticAllocation({"VTI": 0.60, "BND": 0.40}, "static_60_40"),
        prices, cost, tbill, universe, "4D.7",
    )
    all_results["static_20pct_IAU"] = r

    # =====================================================================
    # PHASE 4C: CONFIRMATORY — VTI timing + static gold
    # =====================================================================
    print(f"\n{'=' * 90}")
    print("PHASE 4C: CONFIRMATORY — VTI SMA100 Timing + Static 10% IAU")
    print("  Precommitted: 1 hypothesis, Holm N=1")
    print("  Benchmark: Static 54/36/10 VTI/BND/IAU")
    print(f"{'=' * 90}")

    r, sd, pv, cl, ch = run_test(
        VTITimingWithStaticGold(sma=100, gold_w=0.10, name_label="VTI_SMA100_gold10"),
        StaticAllocation({"VTI": 0.54, "BND": 0.36, "IAU": 0.10}, "static_54_36_10"),
        prices, cost, tbill, universe, "Phase 4C", confirmatory=True,
    )
    all_results["VTI_SMA100_gold"] = r

    # =====================================================================
    # PHASE 5D: Descriptive — additional tests
    # =====================================================================
    print(f"\n{'=' * 90}")
    print("PHASE 5D: DESCRIPTIVE — ADDITIONAL TESTS")
    print(f"{'=' * 90}")

    # 5D.1: VTI SMA100 + static gold vs plain 60/40 (different benchmark)
    print(f"\n--- 5D.1: VTI SMA100 + 10% IAU vs 60/40 ---")
    r, *_ = run_test(
        VTITimingWithStaticGold(sma=100, gold_w=0.10, name_label="VTI_SMA100_gold10"),
        StaticAllocation({"VTI": 0.60, "BND": 0.40}, "static_60_40"),
        prices, cost, tbill, universe, "5D.1",
    )
    all_results["VTI_SMA100_gold_vs_6040"] = r

    # 5D.2: VTI SMA200 + static gold vs static 54/36/10
    print(f"\n--- 5D.2: VTI SMA200 + 10% IAU vs static 54/36/10 ---")
    r, *_ = run_test(
        VTITimingWithStaticGold(sma=200, gold_w=0.10, name_label="VTI_SMA200_gold10"),
        StaticAllocation({"VTI": 0.54, "BND": 0.36, "IAU": 0.10}, "static_54_36_10"),
        prices, cost, tbill, universe, "5D.2",
    )
    all_results["VTI_SMA200_gold"] = r

    # 5D.3: DBC SMA100 in portfolio (10% sleeve) vs 60/40
    print(f"\n--- 5D.3: DBC SMA100 Sleeve in Portfolio vs 60/40 ---")
    r, *_ = run_test(
        SMASleeveStrategy("DBC", "VGSH", "VTI", "BND", 0.54, 0.36, 0.10, 100,
                          "DBC_SMA100_sleeve"),
        StaticAllocation({"VTI": 0.60, "BND": 0.40}, "static_60_40"),
        prices, cost, tbill, universe, "5D.3",
    )
    all_results["DBC_SMA100_sleeve_vs_6040"] = r

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'=' * 90}")
    print("EXHAUSTIVE TEST SUMMARY")
    print(f"{'=' * 90}")
    print(f"\n  Tests run: {len(all_results)}")
    print(f"  All walk-forward with T+1, costs, post-VGSH inception (2009-12+)")
    print(f"  Consistent estimand: Sharpe(strat) - Sharpe(bench) for point + CI")


if __name__ == "__main__":
    main()
