"""Phase 6: Macro-Gated DBC Validation Battery

CRITICAL: Fixes PIT leakage bug from Phase 5.3 (confirmed by Codex R7).
  Bug: load_macro_series() stripped PITFeatureSeries metadata by extracting
  .values, then MacroGatedDBC used .loc[:as_of_date] which includes same-day.
  Fix: Keep PITFeatureSeries intact; use .as_of(as_of_date) which applies
  release_date < decision_date (strict inequality).

Validation tests (from Codex R7 ranked recommendations):
  6.1 PIT-SAFE RERUN of frozen 126-day AND rule (CRITICAL)
  6.2 Leave-2015-2019-out rerun
  6.3 Lookback sensitivity sweep (63, 126, 189, 252)
  6.4 Logic ablation: inflation-only, dollar-only, OR, AND, additive
  6.5 External replication on PDBC, USCI, GSG
  6.6 Block length sensitivity (22, 44, 66 days)
  6.7 Portfolio sleeve test (10% macro-gated DBC in 60/40)
  6.8 Null test (circular time-shift of macro signals)
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
from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci

from _shared import load_commodity_universe, save_phase_returns
from youbet.commodity.data import fetch_commodity_tbill_rates
from youbet.commodity.macro.fetchers import fetch_breakeven_inflation, fetch_dollar_index

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
# Strategies — PIT-safe macro gating
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


class MacroGatedStrategy(BaseStrategy):
    """PIT-SAFE macro-gated long-or-cash strategy.

    Holds `ticker` when macro conditions are met, otherwise cash (empty weights).

    Parameters control the signal logic:
      - logic: 'AND', 'OR', 'INFLATION_ONLY', 'DOLLAR_ONLY', 'ADDITIVE'
      - lookback: days for trailing average
      - ticker: tradeable ETF

    PIT-safe: uses PITFeatureSeries.as_of(as_of_date) which enforces
    release_date < decision_date (strict inequality).
    """
    def __init__(
        self,
        ticker: str,
        breakeven: PITFeatureSeries,
        dxy: PITFeatureSeries,
        lookback_days: int = 126,
        logic: str = "AND",
        name_label: str = "macro_gated",
    ):
        self.ticker = ticker
        self.breakeven = breakeven
        self.dxy = dxy
        self.lookback_days = lookback_days
        self.logic = logic
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def _compute_signals(self, as_of_date):
        """Returns (inflation_rising, dollar_weakening) booleans, PIT-safe."""
        # PIT-safe: .as_of() excludes observations released >= decision_date
        be = self.breakeven.as_of(as_of_date).dropna()
        dx = self.dxy.as_of(as_of_date).dropna()

        if len(be) < self.lookback_days or len(dx) < self.lookback_days:
            return None, None

        be_current = be.iloc[-1]
        be_avg = be.iloc[-self.lookback_days:].mean()
        inflation_rising = be_current > be_avg

        dx_current = dx.iloc[-1]
        dx_avg = dx.iloc[-self.lookback_days:].mean()
        dollar_weakening = dx_current < dx_avg

        return inflation_rising, dollar_weakening

    def generate_weights(self, prices, as_of_date):
        if self.ticker not in prices.columns:
            return pd.Series(dtype=float)

        inflation_rising, dollar_weakening = self._compute_signals(as_of_date)
        if inflation_rising is None:
            return pd.Series(dtype=float)

        if self.logic == "AND":
            hold = inflation_rising and dollar_weakening
            weight = 1.0 if hold else 0.0
        elif self.logic == "OR":
            hold = inflation_rising or dollar_weakening
            weight = 1.0 if hold else 0.0
        elif self.logic == "INFLATION_ONLY":
            weight = 1.0 if inflation_rising else 0.0
        elif self.logic == "DOLLAR_ONLY":
            weight = 1.0 if dollar_weakening else 0.0
        elif self.logic == "ADDITIVE":
            # 0/0.5/1.0 based on 0/1/2 conditions met
            score = int(inflation_rising) + int(dollar_weakening)
            weight = score / 2.0
        else:
            raise ValueError(f"Unknown logic: {self.logic}")

        if weight > 0:
            return pd.Series({self.ticker: weight})
        return pd.Series(dtype=float)

    @property
    def name(self): return self._name

    @property
    def params(self): return {
        "ticker": self.ticker, "lookback": self.lookback_days, "logic": self.logic,
    }


class MacroGatedSleeveStrategy(BaseStrategy):
    """10% macro-gated sleeve in 60/40 VTI/BND.

    Core: 54% VTI, 36% BND always.
    Sleeve: 10% in `ticker` when both macro conditions met, else 0% (cash).
    """
    def __init__(
        self,
        ticker: str,
        breakeven: PITFeatureSeries,
        dxy: PITFeatureSeries,
        lookback_days: int = 126,
        name_label: str = "macro_sleeve",
    ):
        self.ticker = ticker
        self.breakeven = breakeven
        self.dxy = dxy
        self.lookback_days = lookback_days
        self._name = name_label

    def fit(self, prices, as_of_date): pass

    def generate_weights(self, prices, as_of_date):
        w = {"VTI": 0.54, "BND": 0.36}

        be = self.breakeven.as_of(as_of_date).dropna()
        dx = self.dxy.as_of(as_of_date).dropna()

        if (len(be) >= self.lookback_days and len(dx) >= self.lookback_days
                and self.ticker in prices.columns):
            be_rising = be.iloc[-1] > be.iloc[-self.lookback_days:].mean()
            dx_weak = dx.iloc[-1] < dx.iloc[-self.lookback_days:].mean()
            if be_rising and dx_weak:
                w[self.ticker] = 0.10
        # Else: 90% invested (remaining 10% is cash at T-bill rate via backtester)

        return pd.Series(w)

    @property
    def name(self): return self._name

    @property
    def params(self): return {"ticker": self.ticker, "lookback": self.lookback_days}


# ---------------------------------------------------------------------------
# Data loading
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

    print(f"  Commodity snapshot: {snap[0]}")
    print(f"  Prices: {prices.index.min().date()} to {prices.index.max().date()}")

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


def load_macro_pit():
    """Load macro series as PITFeatureSeries (do NOT strip metadata)."""
    be = fetch_breakeven_inflation()
    dxy = fetch_dollar_index()
    print(f"  Breakeven inflation: {len(be.values)} observations, lag={be.lag_days}d")
    print(f"  Dollar index: {len(dxy.values)} observations, lag={dxy.lag_days}d")
    return be, dxy


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(
    strategy, benchmark, prices, cost, tbill, universe, label,
    n_boot: int = 10_000, block_length: int = 22,
    confirmatory: bool = False,
    exclude_windows: list = None,
):
    """Run a walk-forward test and return key metrics."""
    config = BacktestConfig(train_months=36, test_months=12, step_months=12,
                            rebalance_frequency="monthly")
    bt = Backtester(config=config, prices=prices, cost_model=cost,
                    tbill_rates=tbill, universe=universe)
    result = bt.run(strategy, benchmark)

    sr = result.overall_returns
    br = result.benchmark_returns

    # Apply exclusion windows if specified
    if exclude_windows:
        mask = pd.Series(True, index=sr.index)
        for start, end in exclude_windows:
            mask &= ~((sr.index >= start) & (sr.index <= end))
        sr = sr[mask]
        br = br[mask]

    sm = result.overall_metrics
    bm = result.benchmark_metrics
    print(f"\n  {label}: {result.strategy_name} vs {benchmark.name}")
    print(f"  Folds: {len(result.fold_results)}, Period: {sr.index.min().date()} to {sr.index.max().date()}")
    print(f"  Strategy Sharpe: {sm.sharpe_ratio:.3f}, CAGR: {sm.annualized_return:.1%}, MaxDD: {sm.max_drawdown:.1%}")
    print(f"  Benchmark Sharpe: {bm.sharpe_ratio:.3f}, CAGR: {bm.annualized_return:.1%}, MaxDD: {bm.max_drawdown:.1%}")
    print(f"  Turnover: {result.total_turnover:.1f}")

    ci = excess_sharpe_ci(sr, br, n_bootstrap=n_boot,
                          expected_block_length=block_length, seed=42)
    test = block_bootstrap_test(sr, br, n_bootstrap=n_boot,
                                expected_block_length=block_length, seed=42)

    sharpe_diff = ci.get("point_estimate", sm.sharpe_ratio - bm.sharpe_ratio)
    ci_lo = ci.get("ci_lower", 0)
    ci_hi = ci.get("ci_upper", 0)
    p_val = test["p_value"]

    print(f"  Sharpe diff: {sharpe_diff:+.4f}, p={p_val:.4f}, 90% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    if confirmatory:
        passes = sharpe_diff > 0.20 and p_val < 0.05 and ci_lo > 0
        print(f"  GATE: diff>0.20 {'PASS' if sharpe_diff > 0.20 else 'FAIL'}, "
              f"p<0.05 {'PASS' if p_val < 0.05 else 'FAIL'}, "
              f"CI_lo>0 {'PASS' if ci_lo > 0 else 'FAIL'} "
              f"=> {'*** PASS ***' if passes else 'FAIL'}")

    return result, sharpe_diff, p_val, ci_lo, ci_hi


def print_regime_breakdown(result, label):
    sr = result.overall_returns
    br = result.benchmark_returns
    print(f"\n  Regime breakdown for {label}:")
    print(f"  {'Regime':<35} {'Strat':>7} {'Bench':>7} {'Delta':>7}")
    print("  " + "-" * 58)
    for start, end, rlabel in REGIME_WINDOWS:
        sw = sr.loc[start:end]
        bw = br.loc[start:end]
        if len(sw) < 63: continue
        ss = float(sw.mean() / max(sw.std(), 1e-10) * np.sqrt(252))
        bs = float(bw.mean() / max(bw.std(), 1e-10) * np.sqrt(252))
        print(f"  {rlabel:<35} {ss:>7.3f} {bs:>7.3f} {ss-bs:>+6.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 95)
    print("PHASE 6: MACRO-GATED DBC VALIDATION BATTERY")
    print("  PIT-safe reimplementation (Codex R7 fix)")
    print("  Tests: PIT rerun, regime leave-out, lookback sweep, logic ablation,")
    print("         external replication, block length sensitivity, sleeve test")
    print("=" * 95)

    prices, universe, tbill, cost = load_all()
    be, dxy = load_macro_pit()

    all_results = {}

    # =====================================================================
    # 6.1: PIT-SAFE RERUN of frozen 126-day AND rule (CRITICAL)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.1: PIT-SAFE RERUN of frozen Phase 5.3 rule")
    print("  Hypothesis: frozen 126-day AND rule on DBC vs DBC B&H")
    print("  This is the critical validation — does the effect survive PIT fix?")
    print(f"{'=' * 95}")

    r, sd, pv, cl, ch = run_test(
        MacroGatedStrategy("DBC", be, dxy, 126, "AND", "DBC_macro_AND_126_pitsafe"),
        StaticAllocation({"DBC": 1.0}, "DBC_BH"),
        prices, cost, tbill, universe, "6.1 PIT-safe",
        confirmatory=True,
    )
    all_results["6.1_pit_safe"] = (r, sd, pv, cl, ch)
    print_regime_breakdown(r, "6.1")

    # =====================================================================
    # 6.2: Leave-2015-2019-out rerun
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.2: Leave-2015-2019-out rerun")
    print("  Check if suspicious +1.371 regime drives the result")
    print(f"{'=' * 95}")

    r, sd, pv, cl, ch = run_test(
        MacroGatedStrategy("DBC", be, dxy, 126, "AND", "DBC_macro_ex_2015_2019"),
        StaticAllocation({"DBC": 1.0}, "DBC_BH"),
        prices, cost, tbill, universe, "6.2 ex-2015-2019",
        exclude_windows=[("2015-01-01", "2019-12-31")],
    )
    all_results["6.2_ex_lowvol"] = (r, sd, pv, cl, ch)

    # =====================================================================
    # 6.3: Lookback sensitivity (63, 126, 189, 252)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.3: Lookback sensitivity sweep")
    print(f"{'=' * 95}")

    for lb in [63, 126, 189, 252]:
        r, sd, pv, cl, ch = run_test(
            MacroGatedStrategy("DBC", be, dxy, lb, "AND", f"DBC_macro_AND_{lb}"),
            StaticAllocation({"DBC": 1.0}, "DBC_BH"),
            prices, cost, tbill, universe, f"6.3 lookback={lb}",
            n_boot=5_000,  # Smaller bootstrap for sweeps
        )
        all_results[f"6.3_lb_{lb}"] = (r, sd, pv, cl, ch)

    # =====================================================================
    # 6.4: Logic ablation
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.4: Logic ablation (inflation-only, dollar-only, OR, additive)")
    print(f"{'=' * 95}")

    for logic in ["INFLATION_ONLY", "DOLLAR_ONLY", "OR", "ADDITIVE"]:
        r, sd, pv, cl, ch = run_test(
            MacroGatedStrategy("DBC", be, dxy, 126, logic, f"DBC_macro_{logic}_126"),
            StaticAllocation({"DBC": 1.0}, "DBC_BH"),
            prices, cost, tbill, universe, f"6.4 logic={logic}",
            n_boot=5_000,
        )
        all_results[f"6.4_{logic}"] = (r, sd, pv, cl, ch)

    # =====================================================================
    # 6.5: External replication on PDBC, USCI, GSG
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.5: External replication on PDBC, USCI, GSG")
    print("  Same frozen rule applied to different broad commodity ETPs")
    print(f"{'=' * 95}")

    for ticker in ["PDBC", "USCI", "GSG"]:
        if ticker not in prices.columns:
            print(f"  SKIP {ticker}: not in prices")
            continue
        r, sd, pv, cl, ch = run_test(
            MacroGatedStrategy(ticker, be, dxy, 126, "AND", f"{ticker}_macro_AND_126"),
            StaticAllocation({ticker: 1.0}, f"{ticker}_BH"),
            prices, cost, tbill, universe, f"6.5 {ticker}",
            n_boot=5_000,
        )
        all_results[f"6.5_{ticker}"] = (r, sd, pv, cl, ch)

    # =====================================================================
    # 6.6: Block length sensitivity
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.6: Block length sensitivity (22, 44, 66 days)")
    print("  Recomputes CI on the Phase 6.1 return series with different block lengths")
    print(f"{'=' * 95}")

    sr = all_results["6.1_pit_safe"][0].overall_returns
    br = all_results["6.1_pit_safe"][0].benchmark_returns
    print(f"\n  {'Block length':<15} {'Sharpe diff':>12} {'p-value':>9} {'90% CI':>22}")
    print("  " + "-" * 60)
    for bl in [22, 44, 66]:
        ci = excess_sharpe_ci(sr, br, n_bootstrap=10_000, expected_block_length=bl, seed=42)
        test = block_bootstrap_test(sr, br, n_bootstrap=10_000, expected_block_length=bl, seed=42)
        sd = ci.get("point_estimate", 0)
        cl = ci.get("ci_lower", 0)
        ch = ci.get("ci_upper", 0)
        pv = test["p_value"]
        print(f"  {bl:>3} days       {sd:>+12.4f} {pv:>9.4f} [{cl:>+7.4f}, {ch:>+7.4f}]")

    # =====================================================================
    # 6.7: Portfolio sleeve test (10% macro-DBC in 60/40)
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.7: Portfolio sleeve test — 10% macro-gated DBC in 60/40")
    print(f"{'=' * 95}")

    r, sd, pv, cl, ch = run_test(
        MacroGatedSleeveStrategy("DBC", be, dxy, 126, "DBC_macro_sleeve_60_40"),
        StaticAllocation({"VTI": 0.60, "BND": 0.40}, "static_60_40"),
        prices, cost, tbill, universe, "6.7 sleeve",
    )
    all_results["6.7_sleeve"] = (r, sd, pv, cl, ch)
    print_regime_breakdown(r, "6.7")

    # =====================================================================
    # 6.8: Null test — circular time-shift of macro signals
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6.8: Null test — circular time-shift of macro signals")
    print("  Shift macro series by random amounts, recompute Sharpe diff")
    print("  Observed result should exceed 95th percentile of null distribution")
    print(f"{'=' * 95}")

    from copy import copy

    observed_diff = all_results["6.1_pit_safe"][1]
    rng = np.random.default_rng(42)
    n_shuffles = 100  # Keep moderate for compute time
    null_diffs = []

    # Get full common date range for shifting
    be_dates = be.values.index
    dx_dates = dxy.values.index

    for i in range(n_shuffles):
        shift = rng.integers(252, len(be_dates) - 252)  # Avoid boundary
        # Shift breakeven values only (keep dxy aligned)
        shifted_be_vals = pd.Series(
            np.roll(be.values.values, shift),
            index=be.values.index, name="breakeven_inflation",
        )
        be_shifted = PITFeatureSeries(
            values=shifted_be_vals,
            release_dates=be.release_dates,
            feature_name="breakeven_inflation",
            lag_days=be.lag_days,
        )
        strat = MacroGatedStrategy("DBC", be_shifted, dxy, 126, "AND",
                                    f"null_shift_{i}")
        bench = StaticAllocation({"DBC": 1.0}, "DBC_BH")
        config = BacktestConfig(train_months=36, test_months=12, step_months=12,
                                rebalance_frequency="monthly")
        bt = Backtester(config=config, prices=prices, cost_model=cost,
                        tbill_rates=tbill, universe=universe)
        try:
            res = bt.run(strat, bench)
            ci = excess_sharpe_ci(res.overall_returns, res.benchmark_returns,
                                  n_bootstrap=500, seed=42)
            null_diffs.append(ci.get("point_estimate", 0))
        except Exception as e:
            logger.warning(f"Null shuffle {i} failed: {e}")

    null_diffs = np.array(null_diffs)
    if len(null_diffs) > 10:
        pct_above = (null_diffs >= observed_diff).mean()
        null_95 = np.percentile(null_diffs, 95)
        null_99 = np.percentile(null_diffs, 99)
        print(f"\n  Observed Sharpe diff: {observed_diff:+.4f}")
        print(f"  Null shuffles: {len(null_diffs)}")
        print(f"  Null mean: {null_diffs.mean():+.4f}")
        print(f"  Null 95th percentile: {null_95:+.4f}")
        print(f"  Null 99th percentile: {null_99:+.4f}")
        print(f"  Fraction of null ≥ observed: {pct_above:.4f}")
        print(f"  Observed > null 95th: {'PASS' if observed_diff > null_95 else 'FAIL'}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'=' * 95}")
    print("PHASE 6 VALIDATION BATTERY — SUMMARY")
    print(f"{'=' * 95}")
    print()
    print(f"  {'Test':<35} {'Sharpe Diff':>12} {'p-value':>9} {'90% CI':>22}")
    print("  " + "-" * 85)
    for name, (r, sd, pv, cl, ch) in all_results.items():
        print(f"  {name:<35} {sd:>+11.4f} {pv:>9.4f} [{cl:>+7.4f}, {ch:>+7.4f}]")

    # Persist
    returns = {name: r.overall_returns for name, (r, *_) in all_results.items()}
    bench = {"DBC_BH": all_results["6.1_pit_safe"][0].benchmark_returns}
    save_phase_returns("phase6", returns, bench)


if __name__ == "__main__":
    main()
