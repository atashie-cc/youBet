"""Phase 1C: SMA Deep Dive — Window Sensitivity + Drawdown Decomposition.

Two questions:
1. Is the SMA timing effect parameter-robust across windows (50-250)?
   If similar excess Sharpe across windows → genuine trend effect.
   If concentrated at exactly 100 → possible overfitting.

2. How much of the timing alpha comes from avoiding crashes vs
   capturing uptrends? Decompose returns into bear/bull regimes
   and measure SMA contribution in each.

Also: test drawdown reduction on ALL 6 factors (including Mkt-RF and
UMD where Sharpe improvement failed) to connect back to the ETF
workflow's finding that trend following provides genuine drawdown
reduction even when Sharpe improvement is inconclusive.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (
    load_config,
    load_factors,
)

from youbet.factor.simulator import (
    BuyAndHoldFactor,
    SMATrendFilter,
    SimulationConfig,
    simulate_factor_timing,
)
from youbet.etf.risk import sharpe_ratio as compute_sharpe, cagr_from_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def compute_max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(dd.min())


def compute_metrics_quick(returns: pd.Series) -> dict:
    """Lightweight metrics for sweep tables."""
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {"sharpe": 0, "cagr": 0, "vol": 0, "max_dd": 0}
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = float(r.mean() / max(r.std(), 1e-10) * np.sqrt(252))
    cagr = cagr_from_returns(r)
    max_dd = compute_max_drawdown(r)
    return {"sharpe": sharpe, "cagr": cagr, "vol": ann_vol, "max_dd": max_dd}


def regime_decompose(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    factor_returns: pd.Series,
) -> dict:
    """Decompose excess returns into bear and bull factor regimes.

    Bear regime: trailing 6-month factor return < 0
    Bull regime: trailing 6-month factor return >= 0
    """
    # Trailing 6-month return on the factor
    trailing_6m = factor_returns.rolling(126).sum()

    common = strategy_returns.index.intersection(benchmark_returns.index).intersection(trailing_6m.dropna().index)
    strat = strategy_returns[common]
    bench = benchmark_returns[common]
    trail = trailing_6m[common]

    bear_mask = trail < 0
    bull_mask = trail >= 0

    excess = strat - bench

    bear_excess = excess[bear_mask]
    bull_excess = excess[bull_mask]

    def ann_mean(s):
        return float(s.mean() * 252) if len(s) > 0 else 0.0

    return {
        "bear_days": int(bear_mask.sum()),
        "bull_days": int(bull_mask.sum()),
        "bear_pct": float(bear_mask.mean()),
        "bear_ann_excess": ann_mean(bear_excess),
        "bull_ann_excess": ann_mean(bull_excess),
        "bear_contribution": ann_mean(bear_excess) * float(bear_mask.mean()),
        "bull_contribution": ann_mean(bull_excess) * float(bull_mask.mean()),
    }


def main():
    print("=" * 95)
    print("PHASE 1C: SMA DEEP DIVE — Window Sensitivity + Drawdown Decomposition")
    print("=" * 95)

    config = load_config()
    factors = load_factors()
    rf = factors["RF"]
    sim_config = SimulationConfig(
        train_months=config["simulation"]["train_months"],
        test_months=config["simulation"]["test_months"],
        step_months=config["simulation"]["step_months"],
    )

    all_factors = config["factors"]
    passing_factors = ["SMB", "HML", "RMW", "CMA"]
    sma_windows = [50, 75, 100, 125, 150, 175, 200, 250]

    # =====================================================================
    # PART 1: SMA Window Sensitivity Sweep
    # =====================================================================
    print("\n" + "=" * 95)
    print("PART 1: SMA WINDOW SENSITIVITY (on 4 passing factors)")
    print("=" * 95)
    print(f"\nWindows tested: {sma_windows}")
    print("If effect is flat across windows -> genuine trend signal")
    print("If peaked at 100 -> possible parameter sensitivity\n")

    # Compute buy-and-hold benchmarks
    bh_results = {}
    for factor in all_factors:
        bh = simulate_factor_timing(
            factors[factor], rf, BuyAndHoldFactor(), sim_config, factor
        )
        bh_results[factor] = bh

    # Sweep for passing factors
    sweep_data = {}  # factor -> window -> metrics dict
    for factor in passing_factors:
        print(f"\n--- {factor} ---")
        print(f"{'Window':>8} {'ExSharpe':>9} {'Sharpe':>8} {'CAGR':>7} {'Vol':>7} "
              f"{'MaxDD':>8} {'B&H DD':>8} {'DD Redn':>8} {'Exposure':>9}")
        print("-" * 85)

        bh_metrics = compute_metrics_quick(bh_results[factor].overall_returns)
        sweep_data[factor] = {}

        for window in sma_windows:
            sma = simulate_factor_timing(
                factors[factor], rf, SMATrendFilter(window), sim_config, factor
            )
            m = compute_metrics_quick(sma.overall_returns)

            # Excess Sharpe (Sharpe of excess returns)
            excess_ret = sma.overall_returns - bh_results[factor].overall_returns.reindex(
                sma.overall_returns.index
            ).fillna(0)
            ex_sharpe = compute_sharpe(excess_ret)

            # Exposure
            all_exp = pd.concat([f.exposure for f in sma.fold_results])
            mean_exp = all_exp.mean()

            # Drawdown reduction
            dd_reduction = 1.0 - (m["max_dd"] / bh_metrics["max_dd"]) if bh_metrics["max_dd"] < 0 else 0

            sweep_data[factor][window] = {
                "excess_sharpe": ex_sharpe,
                "sharpe": m["sharpe"],
                "cagr": m["cagr"],
                "vol": m["vol"],
                "max_dd": m["max_dd"],
                "dd_reduction": dd_reduction,
                "exposure": mean_exp,
            }

            print(f"{window:>8} {ex_sharpe:>+8.3f} {m['sharpe']:>7.3f} {m['cagr']:>6.1%} "
                  f"{m['vol']:>6.1%} {m['max_dd']:>7.1%} {bh_metrics['max_dd']:>7.1%} "
                  f"{dd_reduction:>7.0%} {mean_exp:>8.0%}")

    # Robustness summary: coefficient of variation of excess Sharpe across windows
    print(f"\n--- Parameter Robustness Summary ---")
    print(f"{'Factor':<8} {'Mean ExSh':>10} {'Std ExSh':>9} {'CV':>6} {'Min':>7} {'Max':>7} {'Range':>7} {'Verdict':>15}")
    print("-" * 75)
    for factor in passing_factors:
        values = [sweep_data[factor][w]["excess_sharpe"] for w in sma_windows]
        mean_v = np.mean(values)
        std_v = np.std(values)
        cv = std_v / abs(mean_v) if abs(mean_v) > 1e-6 else float("inf")
        min_v = min(values)
        max_v = max(values)

        if cv < 0.20:
            verdict = "ROBUST"
        elif cv < 0.40:
            verdict = "MODERATE"
        else:
            verdict = "SENSITIVE"

        print(f"{factor:<8} {mean_v:>+9.3f} {std_v:>8.3f} {cv:>5.0%} "
              f"{min_v:>+6.3f} {max_v:>+6.3f} {max_v - min_v:>6.3f} {verdict:>15}")

    # =====================================================================
    # PART 2: Drawdown Reduction on ALL 6 Factors
    # =====================================================================
    print("\n" + "=" * 95)
    print("PART 2: DRAWDOWN REDUCTION (all 6 factors, SMA100)")
    print("=" * 95)
    print("\nConnecting to ETF workflow finding: trend following provides genuine")
    print("drawdown reduction even when Sharpe improvement is inconclusive.\n")

    print(f"{'Factor':<10} {'B&H DD':>8} {'SMA DD':>8} {'Reduction':>10} {'B&H Sharpe':>11} {'SMA Sharpe':>11} {'ExSharpe':>9}")
    print("-" * 75)

    for factor in all_factors:
        sma = simulate_factor_timing(
            factors[factor], rf, SMATrendFilter(100), sim_config, factor
        )
        bh_m = compute_metrics_quick(bh_results[factor].overall_returns)
        sma_m = compute_metrics_quick(sma.overall_returns)

        excess_ret = sma.overall_returns - bh_results[factor].overall_returns.reindex(
            sma.overall_returns.index
        ).fillna(0)
        ex_sharpe = compute_sharpe(excess_ret)

        dd_red = 1.0 - (sma_m["max_dd"] / bh_m["max_dd"]) if bh_m["max_dd"] < 0 else 0

        print(f"{factor:<10} {bh_m['max_dd']:>7.1%} {sma_m['max_dd']:>7.1%} "
              f"{dd_red:>9.0%} {bh_m['sharpe']:>10.3f} {sma_m['sharpe']:>10.3f} {ex_sharpe:>+8.3f}")

    # =====================================================================
    # PART 3: Bear/Bull Regime Decomposition
    # =====================================================================
    print("\n" + "=" * 95)
    print("PART 3: REGIME DECOMPOSITION (Bear vs Bull Factor Regimes)")
    print("=" * 95)
    print("\nBear = trailing 6-month factor return < 0")
    print("Bull = trailing 6-month factor return >= 0")
    print("Question: Does SMA alpha come from avoiding crashes or capturing uptrends?\n")

    print(f"{'Factor':<10} {'Bear%':>6} {'Bear ExRet':>11} {'Bull ExRet':>11} "
          f"{'Bear Contr':>11} {'Bull Contr':>11} {'Source':>12}")
    print("-" * 80)

    for factor in all_factors:
        sma = simulate_factor_timing(
            factors[factor], rf, SMATrendFilter(100), sim_config, factor
        )
        bh = bh_results[factor]

        decomp = regime_decompose(
            sma.overall_returns,
            bh.overall_returns.reindex(sma.overall_returns.index).fillna(0),
            factors[factor],
        )

        total = decomp["bear_contribution"] + decomp["bull_contribution"]
        if abs(total) > 1e-6:
            bear_pct = decomp["bear_contribution"] / total
        else:
            bear_pct = 0.5

        if bear_pct > 0.65:
            source = "BEAR-DRIVEN"
        elif bear_pct < 0.35:
            source = "BULL-DRIVEN"
        else:
            source = "BALANCED"

        print(f"{factor:<10} {decomp['bear_pct']:>5.0%} "
              f"{decomp['bear_ann_excess']:>+10.1%} {decomp['bull_ann_excess']:>+10.1%} "
              f"{decomp['bear_contribution']:>+10.1%} {decomp['bull_contribution']:>+10.1%} "
              f"{source:>12}")

    # =====================================================================
    # PART 4: Optimal Window Analysis
    # =====================================================================
    print("\n" + "=" * 95)
    print("PART 4: OPTIMAL WINDOW PER FACTOR")
    print("=" * 95)

    for factor in passing_factors:
        best_w = max(sma_windows, key=lambda w: sweep_data[factor][w]["excess_sharpe"])
        worst_w = min(sma_windows, key=lambda w: sweep_data[factor][w]["excess_sharpe"])
        best_v = sweep_data[factor][best_w]["excess_sharpe"]
        worst_v = sweep_data[factor][worst_w]["excess_sharpe"]
        at_100 = sweep_data[factor][100]["excess_sharpe"]

        print(f"  {factor}: best={best_w} ({best_v:+.3f}), worst={worst_w} ({worst_v:+.3f}), "
              f"at_100={at_100:+.3f}, all_positive={all(sweep_data[factor][w]['excess_sharpe'] > 0 for w in sma_windows)}")


if __name__ == "__main__":
    main()
