"""Phase 5: Exhaustive Commodity Strategy Testing

5 tests covering the Codex-recommended ranked battery:

1. DBC SMA200 CONFIRMATORY RERUN (precommitted, Holm N=1)
   - Uses T-bill cash accrual (empty weights), not VGSH
   - Recovers pre-2009 history → ~19 folds
   - Primary hypothesis: gate = excess Sharpe > 0.20, p < 0.05, CI lower > 0

2. DBC 12-month Time-Series Momentum (descriptive)
   - Long DBC if trailing 252-day return > 0, else cash
   - Canonical Moskowitz-Ooi-Pedersen form
   - Benchmark: DBC buy-and-hold

3. Macro-gated DBC (descriptive)
   - Long DBC when T10YIE > 6mo avg AND DXY < 6mo avg
   - Uses fetch_breakeven_inflation + fetch_dollar_index
   - Benchmark: DBC buy-and-hold

4. Cross-sectional rotation DBC/PDBC/USCI (descriptive)
   - Monthly select top-1 by trailing 6mo return
   - Absolute momentum filter (cash if winner's return < 0)
   - Benchmark: DBC buy-and-hold

5. Dynamic 10% commodity sleeve (descriptive)
   - Core: 54/36 VTI/BND static
   - Sleeve: IAU by default, DBC when inflation rising, cash when neither
   - Benchmark: static 54/36/10 VTI/BND/IAU
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

from _shared import load_commodity_universe, save_phase_returns
from youbet.commodity.data import fetch_commodity_tbill_rates

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REGIME_WINDOWS = [
    ("2007-07-01", "2014-12-31", "Post-GFC + commodity bust"),
    ("2011-09-01", "2018-08-31", "Gold bear market"),
    ("2015-01-01", "2019-12-31", "Low-vol, dollar strength"),
    ("2020-01-01", "2022-12-31", "COVID + inflation"),
    ("2023-01-01", "2026-04-09", "Normalization"),
]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

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


class SMAStandaloneTBillCash(BaseStrategy):
    """SMA timing with T-bill cash via empty weights.

    Returns empty Series when below SMA → backtester parks in cash at T-bill rate.
    This lets us use the full pre-2009 history without needing VGSH.
    """
    def __init__(self, ticker: str, sma: int, name_label: str):
        self.ticker = ticker
        self.sma = sma
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        if self.ticker in prices.columns:
            h = prices[self.ticker].loc[:as_of_date].dropna()
            if len(h) >= self.sma and h.iloc[-1] > h.iloc[-self.sma:].mean():
                return pd.Series({self.ticker: 1.0})
        return pd.Series(dtype=float)  # Empty = 100% cash at T-bill rate

    @property
    def name(self): return self._name

    @property
    def params(self): return {"ticker": self.ticker, "sma": self.sma}


class TSMStrategy(BaseStrategy):
    """Time-series momentum: long if trailing N-day return > 0, else cash."""
    def __init__(self, ticker: str, lookback: int, name_label: str):
        self.ticker = ticker
        self.lookback = lookback
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        if self.ticker in prices.columns:
            h = prices[self.ticker].loc[:as_of_date].dropna()
            if len(h) >= self.lookback + 1:
                ret = h.iloc[-1] / h.iloc[-self.lookback - 1] - 1
                if ret > 0:
                    return pd.Series({self.ticker: 1.0})
        return pd.Series(dtype=float)

    @property
    def name(self): return self._name

    @property
    def params(self): return {"ticker": self.ticker, "lookback": self.lookback}


class MacroGatedDBC(BaseStrategy):
    """DBC long when inflation rising AND dollar weakening; else cash.

    Uses daily price data to compute inflation proxy (cached breakeven series)
    and DXY index. Both are compared to 6-month trailing averages.
    """
    def __init__(self, breakeven: pd.Series, dxy: pd.Series,
                 lookback_days: int = 126, name_label: str = "macro_DBC"):
        self.breakeven = breakeven
        self.dxy = dxy
        self.lookback_days = lookback_days
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        if "DBC" not in prices.columns:
            return pd.Series(dtype=float)

        # Breakeven: rising if current > 6mo avg (strict inequality)
        be = self.breakeven.loc[:as_of_date].dropna()
        if len(be) < self.lookback_days:
            return pd.Series(dtype=float)
        be_current = be.iloc[-1]
        be_avg = be.iloc[-self.lookback_days:].mean()
        inflation_rising = be_current > be_avg

        # Dollar: weakening if current < 6mo avg
        dx = self.dxy.loc[:as_of_date].dropna()
        if len(dx) < self.lookback_days:
            return pd.Series(dtype=float)
        dx_current = dx.iloc[-1]
        dx_avg = dx.iloc[-self.lookback_days:].mean()
        dollar_weakening = dx_current < dx_avg

        if inflation_rising and dollar_weakening:
            return pd.Series({"DBC": 1.0})
        return pd.Series(dtype=float)

    @property
    def name(self): return self._name

    @property
    def params(self): return {"lookback_days": self.lookback_days}


class CrossSectionalRotation(BaseStrategy):
    """Top-1 rotation among broad commodity wrappers with absolute momentum filter."""
    def __init__(self, candidates: list, lookback_days: int,
                 abs_momentum: bool = True, name_label: str = "xs_rotation"):
        self.candidates = candidates
        self.lookback_days = lookback_days
        self.abs_momentum = abs_momentum
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        returns = {}
        for t in self.candidates:
            if t not in prices.columns:
                continue
            h = prices[t].loc[:as_of_date].dropna()
            if len(h) < self.lookback_days + 1:
                continue
            ret = h.iloc[-1] / h.iloc[-self.lookback_days - 1] - 1
            returns[t] = ret

        if not returns:
            return pd.Series(dtype=float)

        # Top-1
        winner = max(returns, key=returns.get)
        winner_ret = returns[winner]

        # Absolute momentum filter
        if self.abs_momentum and winner_ret <= 0:
            return pd.Series(dtype=float)

        return pd.Series({winner: 1.0})

    @property
    def name(self): return self._name

    @property
    def params(self): return {"candidates": self.candidates, "lookback": self.lookback_days}


class DynamicSleeveStrategy(BaseStrategy):
    """Core 54/36 VTI/BND + 10% dynamic sleeve: IAU default, DBC in inflation, cash otherwise."""
    def __init__(self, breakeven: pd.Series, dxy: pd.Series,
                 lookback_days: int = 126, name_label: str = "dynamic_sleeve"):
        self.breakeven = breakeven
        self.dxy = dxy
        self.lookback_days = lookback_days
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        w = {"VTI": 0.54, "BND": 0.36}

        # Determine sleeve asset based on macro regime
        be = self.breakeven.loc[:as_of_date].dropna()
        dx = self.dxy.loc[:as_of_date].dropna()

        if len(be) >= self.lookback_days and len(dx) >= self.lookback_days:
            be_rising = be.iloc[-1] > be.iloc[-self.lookback_days:].mean()
            dx_weak = dx.iloc[-1] < dx.iloc[-self.lookback_days:].mean()

            if be_rising and dx_weak and "DBC" in prices.columns:
                # Inflation regime → DBC sleeve
                w["DBC"] = 0.10
            elif "IAU" in prices.columns:
                # Default → gold sleeve
                w["IAU"] = 0.10
            elif "VGSH" in prices.columns:
                # Fallback → cash
                w["VGSH"] = 0.10
        elif "IAU" in prices.columns:
            # Before macro data available → default to gold
            w["IAU"] = 0.10

        return pd.Series(w)

    @property
    def name(self): return self._name

    @property
    def params(self): return {"lookback_days": self.lookback_days}


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def load_all(trim_to_vgsh: bool = False):
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

    if trim_to_vgsh:
        prices = prices.loc["2009-12-01":]
        print(f"  Trimmed to post-VGSH: {prices.index.min().date()} to {prices.index.max().date()}")
    else:
        # Use full history — T-bill cash accrual via empty weights
        print(f"  Full history: {prices.index.min().date()} to {prices.index.max().date()}")

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


def load_macro_series():
    """Load breakeven inflation and dollar index for macro-gated strategies."""
    from youbet.commodity.macro.fetchers import fetch_breakeven_inflation, fetch_dollar_index

    try:
        be = fetch_breakeven_inflation()
        be_series = be.values if hasattr(be, "values") else be
        print(f"  Breakeven inflation: {len(be_series)} observations")
    except Exception as e:
        logger.warning(f"Could not fetch breakeven inflation: {e}")
        be_series = None

    try:
        dxy = fetch_dollar_index()
        dxy_series = dxy.values if hasattr(dxy, "values") else dxy
        print(f"  Dollar index: {len(dxy_series)} observations")
    except Exception as e:
        logger.warning(f"Could not fetch dollar index: {e}")
        dxy_series = None

    return be_series, dxy_series


def run_test(strategy, benchmark, prices, cost, tbill, universe, label,
             confirmatory: bool = False, n_boot: int = 10_000):
    config = BacktestConfig(train_months=36, test_months=12, step_months=12,
                            rebalance_frequency="monthly")
    bt = Backtester(config=config, prices=prices, cost_model=cost,
                    tbill_rates=tbill, universe=universe)
    result = bt.run(strategy, benchmark)

    sm = result.overall_metrics
    bm = result.benchmark_metrics
    print(f"\n  {result.strategy_name} vs {benchmark.name}")
    print(f"  Period: {result.overall_returns.index[0].date()} to {result.overall_returns.index[-1].date()}")
    print(f"  Folds: {len(result.fold_results)}")
    print(f"  Strategy: Sharpe {sm.sharpe_ratio:.3f}, CAGR {sm.annualized_return:.1%}, "
          f"MaxDD {sm.max_drawdown:.1%}, Vol {sm.annualized_volatility:.1%}")
    print(f"  Benchmark: Sharpe {bm.sharpe_ratio:.3f}, CAGR {bm.annualized_return:.1%}, "
          f"MaxDD {bm.max_drawdown:.1%}, Vol {bm.annualized_volatility:.1%}")
    print(f"  Turnover: {result.total_turnover:.1f}, Cost drag: {result.total_cost_drag:.4f}")

    sr = result.overall_returns
    br = result.benchmark_returns
    ci = excess_sharpe_ci(sr, br, n_bootstrap=n_boot, seed=42)
    test = block_bootstrap_test(sr, br, n_bootstrap=n_boot, seed=42)

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
    print("=" * 95)
    print("PHASE 5: EXHAUSTIVE COMMODITY STRATEGY TESTING")
    print("  5 tests: 1 confirmatory (DBC SMA200) + 4 descriptive")
    print("=" * 95)

    # Full history load for standalone tests (T-bill cash via empty weights)
    prices_full, universe, tbill, cost = load_all(trim_to_vgsh=False)

    # Trimmed load for sleeve tests (need VGSH)
    prices_trim, _, _, _ = load_all(trim_to_vgsh=True)

    all_results = {}

    # =====================================================================
    # 5.1: DBC SMA200 CONFIRMATORY RERUN (full history, T-bill cash)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 5.1: DBC SMA200 CONFIRMATORY (full history, T-bill cash)")
    print("  Precommitted hypothesis, Holm N=1")
    print("  Gate: Sharpe diff > 0.20, p < 0.05, CI lower > 0")
    print(f"{'=' * 95}")

    r, sd, pv, cl, ch = run_test(
        SMAStandaloneTBillCash("DBC", 200, "DBC_SMA200_conf"),
        StaticAllocation({"DBC": 1.0}, "DBC_BH"),
        prices_full, cost, tbill, universe, "Phase 5.1",
        confirmatory=True,
    )
    all_results["DBC_SMA200_conf"] = (r, sd, pv, cl, ch)

    # =====================================================================
    # 5.2: DBC 12-month TSMOM (descriptive)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 5.2: DBC 12-month Time-Series Momentum (descriptive)")
    print(f"{'=' * 95}")

    r, sd, pv, cl, ch = run_test(
        TSMStrategy("DBC", 252, "DBC_TSM12"),
        StaticAllocation({"DBC": 1.0}, "DBC_BH"),
        prices_full, cost, tbill, universe, "Phase 5.2",
    )
    all_results["DBC_TSM12"] = (r, sd, pv, cl, ch)

    # =====================================================================
    # 5.3: Macro-gated DBC (descriptive)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 5.3: Macro-gated DBC — T10YIE rising AND DXY weakening")
    print(f"{'=' * 95}")

    be_series, dxy_series = load_macro_series()

    if be_series is not None and dxy_series is not None:
        r, sd, pv, cl, ch = run_test(
            MacroGatedDBC(be_series, dxy_series, lookback_days=126,
                          name_label="macro_DBC_6mo"),
            StaticAllocation({"DBC": 1.0}, "DBC_BH"),
            prices_full, cost, tbill, universe, "Phase 5.3",
        )
        all_results["macro_DBC"] = (r, sd, pv, cl, ch)
    else:
        print("  SKIP: macro series unavailable")

    # =====================================================================
    # 5.4: Cross-sectional rotation DBC/PDBC/USCI (descriptive)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 5.4: Cross-Sectional Rotation DBC/PDBC/USCI (top-1 by 6mo momentum)")
    print(f"{'=' * 95}")

    r, sd, pv, cl, ch = run_test(
        CrossSectionalRotation(["DBC", "PDBC", "USCI"], lookback_days=126,
                               abs_momentum=True, name_label="xs_rotation_6mo"),
        StaticAllocation({"DBC": 1.0}, "DBC_BH"),
        prices_full, cost, tbill, universe, "Phase 5.4",
    )
    all_results["xs_rotation"] = (r, sd, pv, cl, ch)

    # =====================================================================
    # 5.5: Dynamic commodity sleeve (descriptive)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 5.5: Dynamic Sleeve — IAU default, DBC in inflation regime")
    print(f"{'=' * 95}")

    if be_series is not None and dxy_series is not None:
        r, sd, pv, cl, ch = run_test(
            DynamicSleeveStrategy(be_series, dxy_series, lookback_days=126,
                                  name_label="dynamic_sleeve_6mo"),
            StaticAllocation({"VTI": 0.54, "BND": 0.36, "IAU": 0.10}, "static_54_36_10"),
            prices_trim, cost, tbill, universe, "Phase 5.5",
        )
        all_results["dynamic_sleeve"] = (r, sd, pv, cl, ch)
    else:
        print("  SKIP: macro series unavailable")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 5 SUMMARY")
    print(f"{'=' * 95}")
    print()
    print(f"  {'Test':<25} {'Sharpe Diff':>12} {'p-value':>9} {'90% CI':>22}")
    print("  " + "-" * 70)
    for name, (r, sd, pv, cl, ch) in all_results.items():
        print(f"  {name:<25} {sd:>+11.4f} {pv:>9.4f} [{cl:>+6.3f}, {ch:>+6.3f}]")

    # Persist
    returns = {name: r.overall_returns for name, (r, *_) in all_results.items()}
    bench = {"DBC_BH": all_results["DBC_SMA200_conf"][0].benchmark_returns}
    save_phase_returns("phase5", returns, bench)


if __name__ == "__main__":
    main()
