"""Phase 3: Factor ETF Bridge Test — Regression Qualification + Hedged Timing.

Revised after Codex review (round 2). Two blockers fixed:
1. Proxy qualification now uses multi-factor regression (not raw correlation)
2. Tests both unhedged AND hedged (market-neutralized) timing

Three stages:
  A. Factor Loading Test: Regress ETF excess returns on Mkt-RF + target factor.
     Check that b_factor > 0 and significant.
  B. Unhedged Timing: SMA100 on raw ETF price (vs buy-and-hold ETF)
  C. Hedged Timing: SMA100 on market-neutralized ETF spread (ETF - beta*VTI)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (
    compute_metrics,
    load_config,
    load_factors,
    precommit_strategies,
    print_table,
)

from youbet.etf.backtester import Backtester, BacktestConfig, BacktestResult
from youbet.etf.costs import CostModel
from youbet.etf.data import fetch_prices, fetch_tbill_rates
from youbet.etf.risk import compute_risk_metrics, sharpe_ratio as compute_sharpe, cagr_from_returns
from youbet.etf.stats import block_bootstrap_test, excess_sharpe_ci, holm_bonferroni
from youbet.etf.strategy import BaseStrategy
from youbet.factor.data import load_french_snapshot

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ETF_FACTOR_MAP = {
    "VLUE": "HML",
    "QUAL": "RMW",
    "SIZE": "SMB",
}
RISK_OFF = "VGSH"
ETF_DATA_START = "2011-01-01"

EXPENSE_RATIOS = {"VLUE": 0.0015, "QUAL": 0.0015, "SIZE": 0.0015,
                  "VGSH": 0.0004, "VTI": 0.0003}
TICKER_CATEGORIES = {"VLUE": "factor", "QUAL": "factor", "SIZE": "factor",
                     "VGSH": "cash_equivalent", "VTI": "broad_us_equity"}


# ---------------------------------------------------------------------------
# Strategy classes
# ---------------------------------------------------------------------------

class SMATimedETF(BaseStrategy):
    """SMA100 trend filter on a single ETF. Risk-off to VGSH."""
    def __init__(self, ticker: str, risk_off: str = "VGSH", sma_window: int = 100):
        self.ticker = ticker
        self.risk_off = risk_off
        self.sma_window = sma_window

    def fit(self, prices, as_of_date):
        pass

    def generate_weights(self, prices, as_of_date):
        if self.ticker not in prices.columns:
            return pd.Series({self.risk_off: 1.0})
        p = prices[self.ticker].dropna()
        if len(p) < self.sma_window:
            return pd.Series({self.ticker: 1.0})
        sma = p.rolling(self.sma_window, min_periods=self.sma_window).mean()
        if pd.isna(sma.iloc[-1]) or p.iloc[-1] > sma.iloc[-1]:
            return pd.Series({self.ticker: 1.0})
        return pd.Series({self.risk_off: 1.0})

    @property
    def name(self): return f"sma{self.sma_window}_{self.ticker}"
    @property
    def params(self): return {"ticker": self.ticker, "sma_window": self.sma_window}


class BuyAndHoldETF(BaseStrategy):
    def __init__(self, ticker: str): self.ticker = ticker
    def fit(self, prices, as_of_date): pass
    def generate_weights(self, prices, as_of_date):
        return pd.Series({self.ticker: 1.0})
    @property
    def name(self): return f"bh_{self.ticker}"
    @property
    def params(self): return {"ticker": self.ticker}


# ---------------------------------------------------------------------------
# Stage A: Multi-Factor Regression
# ---------------------------------------------------------------------------

def factor_loading_test(etf_prices: pd.DataFrame, factors: pd.DataFrame) -> dict[str, dict]:
    """Regress ETF excess returns on Mkt-RF + target factor.

    r_ETF - RF = alpha + b_mkt*(Mkt-RF) + b_factor*FACTOR + eps

    If b_factor > 0 and t-stat > 2, the ETF captures the factor.
    """
    results = {}
    for etf, factor_name in ETF_FACTOR_MAP.items():
        if etf not in etf_prices.columns:
            continue

        etf_ret = etf_prices[etf].pct_change().dropna()
        common = etf_ret.index.intersection(factors.index)
        common = common[common >= etf_ret.index[1]]  # skip first day

        e = etf_ret[common].values
        mkt = factors["Mkt-RF"][common].values
        rf = factors["RF"][common].values
        fac = factors[factor_name][common].values

        # Dependent: ETF excess return
        y = e - rf

        # Regressors: constant, Mkt-RF, target factor
        X = np.column_stack([np.ones(len(y)), mkt, fac])

        # OLS
        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            results[etf] = {"factor": factor_name, "status": "REGRESSION_FAILED"}
            continue

        y_hat = X @ beta
        resid = y - y_hat
        n, k = X.shape
        sigma2 = np.sum(resid**2) / (n - k)
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov_beta))
        t_stats = beta / se

        alpha_ann = beta[0] * 252
        b_mkt = beta[1]
        b_factor = beta[2]
        t_factor = t_stats[2]
        p_factor = 2 * (1 - scipy_stats.t.cdf(abs(t_factor), df=n - k))
        r_squared = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)

        # Also compute raw and partial correlation
        raw_corr = float(np.corrcoef(e, fac)[0, 1])

        results[etf] = {
            "factor": factor_name,
            "n_days": len(y),
            "n_years": len(y) / 252,
            "alpha_ann": float(alpha_ann),
            "b_mkt": float(b_mkt),
            "b_factor": float(b_factor),
            "t_factor": float(t_factor),
            "p_factor": float(p_factor),
            "r_squared": float(r_squared),
            "raw_corr": raw_corr,
            "factor_significant": abs(t_factor) > 2.0 and b_factor > 0,
        }

    return results


# ---------------------------------------------------------------------------
# Stage C: Hedged (Market-Neutralized) Timing
# ---------------------------------------------------------------------------

def compute_hedged_returns(
    etf_prices: pd.DataFrame,
    factors: pd.DataFrame,
    etf: str,
    factor_name: str,
    hedge_ticker: str = "VTI",
) -> pd.Series:
    """Create market-neutralized factor return from ETF.

    hedged_return = ETF_return - beta_mkt * VTI_return

    Beta estimated from trailing 252-day rolling regression.
    """
    etf_ret = etf_prices[etf].pct_change().dropna()
    vti_ret = etf_prices[hedge_ticker].pct_change().dropna()
    common = etf_ret.index.intersection(vti_ret.index)
    etf_r = etf_ret[common]
    vti_r = vti_ret[common]

    # Rolling beta (252-day), lagged by 1 day for PIT safety.
    # Beta estimated from days [t-252, t-1] is used to hedge day t's return.
    rolling_cov = etf_r.rolling(252).cov(vti_r)
    rolling_var = vti_r.rolling(252).var()
    rolling_beta = (rolling_cov / rolling_var.clip(lower=1e-10)).fillna(1.0).shift(1)

    hedged = etf_r - rolling_beta * vti_r
    return hedged


def run_hedged_timing(
    hedged_returns: pd.Series,
    rf: pd.Series,
    factor_name: str,
    sma_window: int = 100,
) -> dict:
    """Apply SMA100 timing to hedged (market-neutral) return series.

    Uses factor simulator approach (return-series, not price-based backtester)
    since hedged returns aren't a tradeable price series.
    """
    from youbet.factor.simulator import (
        BuyAndHoldFactor, SMATrendFilter, SimulationConfig, simulate_factor_timing
    )

    config = SimulationConfig(train_months=36, test_months=12, step_months=12)
    rf_aligned = rf.reindex(hedged_returns.index, method="ffill").fillna(0.0)

    bh = simulate_factor_timing(hedged_returns, rf_aligned, BuyAndHoldFactor(), config, factor_name)
    sma = simulate_factor_timing(hedged_returns, rf_aligned, SMATrendFilter(sma_window), config, factor_name)

    bh_sharpe = compute_sharpe(bh.overall_returns)
    sma_sharpe = compute_sharpe(sma.overall_returns)

    excess = sma.overall_returns - bh.overall_returns.reindex(sma.overall_returns.index).fillna(0)
    ex_sharpe = compute_sharpe(excess)

    bh_dd = float((1 + bh.overall_returns).cumprod().pipe(lambda c: ((c - c.cummax()) / c.cummax()).min()))
    sma_dd = float((1 + sma.overall_returns).cumprod().pipe(lambda c: ((c - c.cummax()) / c.cummax()).min()))

    return {
        "factor": factor_name,
        "n_folds": sma.n_folds,
        "bh_sharpe": bh_sharpe,
        "sma_sharpe": sma_sharpe,
        "excess_sharpe": ex_sharpe,
        "bh_max_dd": bh_dd,
        "sma_max_dd": sma_dd,
        "dd_reduction": 1 - sma_dd / bh_dd if bh_dd < 0 else 0,
        "overall_strat": sma.overall_returns,
        "overall_bench": bh.overall_returns,
    }


# ---------------------------------------------------------------------------
# Unhedged timing (same as before but cleaner)
# ---------------------------------------------------------------------------

def run_unhedged_timing(etf: str, prices: pd.DataFrame, tbill: pd.Series) -> dict:
    """SMA100 on raw ETF using full backtester with costs."""
    config = BacktestConfig(train_months=36, test_months=12, step_months=12, rebalance_frequency="monthly")
    cost_model = CostModel(expense_ratios=EXPENSE_RATIOS, ticker_categories=TICKER_CATEGORIES)
    strategy = SMATimedETF(etf, risk_off=RISK_OFF, sma_window=100)
    benchmark = BuyAndHoldETF(etf)
    bt = Backtester(config=config, prices=prices, cost_model=cost_model, tbill_rates=tbill)
    result = bt.run(strategy, benchmark)
    return {
        "n_folds": len(result.fold_results),
        "strat_sharpe": result.overall_metrics.sharpe_ratio,
        "bench_sharpe": result.benchmark_metrics.sharpe_ratio,
        "excess_sharpe": result.excess_sharpe,
        "strat_dd": result.overall_metrics.max_drawdown,
        "bench_dd": result.benchmark_metrics.max_drawdown,
        "dd_reduction": 1 - result.overall_metrics.max_drawdown / result.benchmark_metrics.max_drawdown if result.benchmark_metrics.max_drawdown < 0 else 0,
        "turnover": result.total_turnover,
        "cost_drag": result.total_cost_drag,
        "overall_strat": result.overall_returns,
        "overall_bench": result.benchmark_returns,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 95)
    print("PHASE 3: FACTOR ETF BRIDGE TEST (Revised — Regression + Hedged)")
    print("=" * 95)

    factors = load_factors()
    all_tickers = list(ETF_FACTOR_MAP.keys()) + [RISK_OFF, "VTI"]
    prices = fetch_prices(tickers=all_tickers, start=ETF_DATA_START,
                          snapshot_dir=WORKFLOW_ROOT / "data" / "snapshots" / "etf")
    tbill = fetch_tbill_rates(start=ETF_DATA_START)
    rf = factors["RF"]

    print(f"\nPrice data: {prices.index[0].date()} to {prices.index[-1].date()}")

    # =====================================================================
    # STAGE A: MULTI-FACTOR REGRESSION
    # =====================================================================
    print("\n" + "=" * 95)
    print("STAGE A: FACTOR LOADING TEST (Multi-Factor Regression)")
    print("=" * 95)
    print("r_ETF - RF = alpha + b_mkt*(Mkt-RF) + b_factor*FACTOR + eps")
    print("Qualification: b_factor > 0 with |t| > 2.0\n")

    reg_results = factor_loading_test(prices, factors)

    print(f"{'ETF':<6} {'Factor':<8} {'Years':>6} {'b_mkt':>7} {'b_factor':>9} {'t_stat':>7} "
          f"{'p_val':>7} {'R2':>6} {'Raw Corr':>9} {'Qualified':>10}")
    print("-" * 85)
    for etf, r in reg_results.items():
        if r.get("status"):
            print(f"{etf:<6} {r['factor']:<8} {r['status']}")
            continue
        print(f"{etf:<6} {r['factor']:<8} {r['n_years']:>5.1f} {r['b_mkt']:>6.3f} "
              f"{r['b_factor']:>+8.3f} {r['t_factor']:>6.2f} {r['p_factor']:>6.4f} "
              f"{r['r_squared']:>5.3f} {r['raw_corr']:>+8.3f} "
              f"{'YES' if r['factor_significant'] else 'NO':>10}")

    # =====================================================================
    # STAGE B: UNHEDGED TIMING (SMA on raw ETF)
    # =====================================================================
    print("\n" + "=" * 95)
    print("STAGE B: UNHEDGED TIMING (SMA100 on raw ETF, with costs)")
    print("=" * 95)

    print(f"\n{'ETF':<6} {'Factor':<8} {'Folds':>6} {'B&H Sh':>8} {'SMA Sh':>8} "
          f"{'ExSharpe':>9} {'B&H DD':>8} {'SMA DD':>8} {'DD Red':>7}")
    print("-" * 80)

    unhedged = {}
    for etf, factor in ETF_FACTOR_MAP.items():
        r = run_unhedged_timing(etf, prices, tbill)
        unhedged[etf] = r
        print(f"{etf:<6} {factor:<8} {r['n_folds']:>5} {r['bench_sharpe']:>7.3f} "
              f"{r['strat_sharpe']:>7.3f} {r['excess_sharpe']:>+8.3f} "
              f"{r['bench_dd']:>7.1%} {r['strat_dd']:>7.1%} {r['dd_reduction']:>6.0%}")

    # =====================================================================
    # STAGE C: HEDGED TIMING (SMA on market-neutralized spread)
    # =====================================================================
    print("\n" + "=" * 95)
    print("STAGE C: HEDGED TIMING (SMA100 on ETF - beta*VTI spread)")
    print("=" * 95)
    print("Uses rolling 252-day beta for market neutralization.\n")

    print(f"{'ETF':<6} {'Factor':<8} {'Folds':>6} {'B&H Sh':>8} {'SMA Sh':>8} "
          f"{'ExSharpe':>9} {'B&H DD':>8} {'SMA DD':>8} {'DD Red':>7}")
    print("-" * 80)

    hedged = {}
    for etf, factor in ETF_FACTOR_MAP.items():
        hedged_ret = compute_hedged_returns(prices, factors, etf, factor)
        r = run_hedged_timing(hedged_ret, rf, factor)
        hedged[etf] = r
        print(f"{etf:<6} {factor:<8} {r['n_folds']:>5} {r['bh_sharpe']:>7.3f} "
              f"{r['sma_sharpe']:>7.3f} {r['excess_sharpe']:>+8.3f} "
              f"{r['bh_max_dd']:>7.1%} {r['sma_max_dd']:>7.1%} {r['dd_reduction']:>6.0%}")

    # =====================================================================
    # STATISTICAL TESTS (all strategies)
    # =====================================================================
    print("\n" + "=" * 95)
    print("STATISTICAL TESTS (descriptive)")
    print("=" * 95)

    all_p = {}
    all_tests = {}
    for etf in ETF_FACTOR_MAP:
        # Unhedged
        label_u = f"unhedged_{etf}"
        test_u = block_bootstrap_test(unhedged[etf]["overall_strat"], unhedged[etf]["overall_bench"],
                                       n_bootstrap=2_000, seed=42)
        ci_u = excess_sharpe_ci(unhedged[etf]["overall_strat"], unhedged[etf]["overall_bench"],
                                 n_bootstrap=2_000, seed=42)
        all_p[label_u] = test_u["p_value"]
        all_tests[label_u] = {**test_u, **ci_u}

        # Hedged
        label_h = f"hedged_{etf}"
        test_h = block_bootstrap_test(hedged[etf]["overall_strat"], hedged[etf]["overall_bench"],
                                       n_bootstrap=2_000, seed=42)
        ci_h = excess_sharpe_ci(hedged[etf]["overall_strat"], hedged[etf]["overall_bench"],
                                 n_bootstrap=2_000, seed=42)
        all_p[label_h] = test_h["p_value"]
        all_tests[label_h] = {**test_h, **ci_h}

    holm = holm_bonferroni(all_p)

    print(f"\n{'Strategy':<25} {'ExSharpe':>9} {'Raw p':>9} {'Holm p':>9} {'90% CI':>22}")
    print("-" * 80)
    for label in sorted(holm, key=lambda x: all_tests[x]["observed_excess_sharpe"], reverse=True):
        h = holm[label]
        t = all_tests[label]
        print(f"{label:<25} {t['observed_excess_sharpe']:>+8.3f} "
              f"{h['raw_p']:>9.4f} {h['adjusted_p']:>9.4f} "
              f"[{t['excess_sharpe_lower']:>+6.3f}, {t['excess_sharpe_upper']:>+6.3f}]")

    # =====================================================================
    # COMPARISON: Paper vs Unhedged vs Hedged
    # =====================================================================
    print("\n" + "=" * 95)
    print("COMPARISON: PAPER vs UNHEDGED vs HEDGED")
    print("=" * 95)

    paper_excess = {"HML": 0.535, "RMW": 0.582, "SMB": 0.641}
    print(f"\n{'Factor':<8} {'ETF':<6} {'Paper ExSh':>11} {'Unhedged ExSh':>14} {'Hedged ExSh':>12} {'Factor Load':>12}")
    print("-" * 70)
    for etf, factor in ETF_FACTOR_MAP.items():
        paper = paper_excess.get(factor, 0)
        u_ex = unhedged[etf]["excess_sharpe"]
        h_ex = hedged[etf]["excess_sharpe"]
        b_fac = reg_results[etf]["b_factor"] if etf in reg_results else 0
        print(f"{factor:<8} {etf:<6} {paper:>+10.3f} {u_ex:>+13.3f} {h_ex:>+11.3f} {b_fac:>+11.3f}")

    print(f"\n{'=' * 95}")
    print("PHASE 3 COMPLETE")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
