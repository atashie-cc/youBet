"""Phase 4 robustness for C1b (60/40 VTI/HEFA) — the only finding with a positive CI upper bound.

Round 2 review (codex_review_round2.md) found:
  H3: Phase 3 sample misalignment between C1a/C1b/C3 (FIXED in phase3_hedging.py).
  Recommended SHOULD-DO: P1a placebo + P2 linear-scaling sweep on C1b before any claim.

Per the project memory `feedback_source_period_bias`:
  > Bootstrap MC mechanically reproduces source-period asset means. Run placebo
  > + linear-scaling + sub-period checks before claiming any positive finding.

This script runs:
  P1a — USD-return mean-shifted placebo: shift HEFA's daily USD return so its
        mean equals VTI's mean. Re-run 60/40 VTI/HEFA. If the +0.109 CI upper
        bound disappears, the finding was source-period bias (a higher-mean
        HEFA in the 2017-2026 sample was driving it). If the CI upper bound
        survives, there's a structural rebalancing-premium / vol-pumping effect.

  P2  — Linear-scaling sweep: HEFA at {1, 5, 10, 30, 50}% within VTI core,
        annual rebalance, same shared sample window as Phase 3. If SharpeDiff
        scales monotonically with weight, structural; if peak-then-collapse,
        artifact.

  P3 (sub-period 3 only) — re-run C1b on 2018-05 to 2026-04 only. The mechanical-
        equal-thirds locks for the workflow (2001-2009 / 2010-2018 / 2018-2026)
        cannot test C1b on sub-periods 1-2 (HEFA inception 2014). Sub-period 3
        is 8 years entirely within the USD-bull regime — provides limited
        additional evidence but is the only available cut.

Output:
  artifacts/phase4_c1b_robustness.json
  research/phase4_c1b_robustness.md
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_DIR))
sys.path.insert(0, str(WORKFLOW_DIR / "experiments"))

from youbet.etf.backtester import Backtester, BacktestConfig            # noqa: E402
from youbet.etf.benchmark import BuyAndHold                              # noqa: E402

from phase0_efficiency import (                                          # noqa: E402
    build_cost_model,
    compute_inferential_stats,
    load_gate,
    load_prices,
    load_tbill_daily,
    load_workflow_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


PHASE3_START = "2014-01-31"
PHASE3_END = "2026-04-30"
PHASE4_P3_START = "2018-05-01"
PHASE4_P3_END = "2026-04-30"
GATING_COST_BPS = 10.0
LINEAR_SCALING_WEIGHTS = [0.01, 0.05, 0.10, 0.30, 0.50]


def mean_shift_returns(
    target_returns: pd.Series, reference_returns: pd.Series
) -> pd.Series:
    """Shift target's daily returns so its mean matches reference's mean.

    Preserves variance, autocorrelation, and cross-correlation structure.
    Implements the gold-mean-shifted placebo pattern from real-world-test
    workflow (per `feedback_source_period_bias` memory).
    """
    common = target_returns.index.intersection(reference_returns.index)
    target = target_returns.loc[common]
    reference = reference_returns.loc[common]
    delta = reference.mean() - target.mean()
    return target_returns + delta


def _backtest_static_mix(
    weight_ex_us: float,
    ex_us_ticker: str,
    prices_in: pd.DataFrame,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
    intl_cost_bps: float,
) -> dict:
    """Annual-rebalanced 1-w VTI / w ex_us_ticker static mix vs BuyAndHold(VTI)."""
    cost_model = build_cost_model(universe, intl_cost_bps)

    bt = Backtester(
        config=bt_config,
        prices=prices_in,
        cost_model=cost_model,
        tbill_rates=tbill,
        universe=universe,
    )
    weights = {"VTI": 1 - weight_ex_us, ex_us_ticker: weight_ex_us}
    weights = {k: v for k, v in weights.items() if v > 0}
    strategy = BuyAndHold(weights)
    benchmark = BuyAndHold({"VTI": 1.0})
    result = bt.run(strategy, benchmark)

    return {
        "weight_ex_us": weight_ex_us,
        "ex_us_ticker": ex_us_ticker,
        "intl_cost_bps": intl_cost_bps,
        "n_folds": len(result.fold_results),
        "test_start": result.overall_returns.index[0].date().isoformat(),
        "test_end": result.overall_returns.index[-1].date().isoformat(),
        "n_test_days": len(result.overall_returns),
        "strat_sharpe": result.overall_metrics.sharpe_ratio,
        "bench_sharpe": result.benchmark_metrics.sharpe_ratio,
        "strat_vol_annual": result.overall_metrics.annualized_volatility,
        "bench_vol_annual": result.benchmark_metrics.annualized_volatility,
        "excess_sharpe": result.excess_sharpe,
        "_strategy_returns": result.overall_returns,
        "_benchmark_returns": result.benchmark_returns,
    }


def run_p1a_placebo(
    prices_full: pd.DataFrame,
    shared_index: pd.DatetimeIndex,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
) -> dict:
    """P1a — mean-shift HEFA's daily return to match VTI's mean. Re-run C1b.

    If +0.109 CI upper bound goes away, the finding was source-period bias.
    """
    prices = prices_full[["VTI", "VEA", "HEFA"]].dropna(how="any")
    prices = prices.loc[prices.index.intersection(shared_index)]
    prices = prices.loc[(prices.index >= PHASE3_START) & (prices.index <= PHASE3_END)]

    # Compute returns, mean-shift HEFA to match VTI mean, reconstruct prices
    rets = prices.pct_change().dropna(how="any")
    hefa_shifted = mean_shift_returns(rets["HEFA"], rets["VTI"])
    delta_per_day = (rets["VTI"].mean() - rets["HEFA"].mean())

    rets_shifted = rets.copy()
    rets_shifted["HEFA"] = hefa_shifted

    # Reconstruct HEFA prices from shifted returns starting at the same base
    base_price = prices["HEFA"].iloc[0]
    hefa_shifted_prices = pd.Series(
        base_price * (1 + rets_shifted["HEFA"]).cumprod().reindex(prices.index, fill_value=1.0).values,
        index=prices.index,
        name="HEFA",
    )
    # Splice back: keep first row at base, then accumulate
    hefa_recon = prices["HEFA"].copy()
    hefa_recon.iloc[1:] = base_price * (1 + rets_shifted["HEFA"]).cumprod().values
    prices_placebo = prices.copy()
    prices_placebo["HEFA"] = hefa_recon

    run = _backtest_static_mix(
        weight_ex_us=0.40,
        ex_us_ticker="HEFA",
        prices_in=prices_placebo,
        universe=universe,
        tbill=tbill,
        bt_config=bt_config,
        intl_cost_bps=GATING_COST_BPS,
    )
    inf = compute_inferential_stats(run, n_bootstrap=10_000)

    return {
        "placebo_method": "mean_shift_HEFA_to_VTI_mean",
        "delta_applied_per_day": float(delta_per_day),
        "delta_applied_annualized": float(delta_per_day * 252),
        "raw_hefa_mean_daily": float(rets["HEFA"].mean()),
        "raw_vti_mean_daily": float(rets["VTI"].mean()),
        "shifted_hefa_mean_daily": float(hefa_shifted.mean()),
        **{k: v for k, v in run.items() if not k.startswith("_")},
        **inf,
    }


def run_p2_linear_scaling(
    prices_full: pd.DataFrame,
    shared_index: pd.DatetimeIndex,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
) -> list[dict]:
    """P2 — sweep HEFA weight {1,5,10,30,50}% in VTI, same window."""
    prices = prices_full[["VTI", "VEA", "HEFA"]].dropna(how="any")
    prices = prices.loc[prices.index.intersection(shared_index)]
    prices = prices.loc[(prices.index >= PHASE3_START) & (prices.index <= PHASE3_END)]

    out = []
    for w in LINEAR_SCALING_WEIGHTS:
        run = _backtest_static_mix(
            weight_ex_us=w,
            ex_us_ticker="HEFA",
            prices_in=prices[["VTI", "HEFA"]],
            universe=universe,
            tbill=tbill,
            bt_config=bt_config,
            intl_cost_bps=GATING_COST_BPS,
        )
        inf = compute_inferential_stats(run, n_bootstrap=10_000)
        out.append({
            **{k: v for k, v in run.items() if not k.startswith("_")},
            **inf,
        })
        print(f"  HEFA {int(w*100):>3}%: sharpe_diff={inf['sharpe_diff_point']:+.3f}, "
              f"ci=[{inf['sharpe_diff_ci_lower']:+.3f}, {inf['sharpe_diff_ci_upper']:+.3f}]")
    return out


def run_p3_subperiod3_c1b(
    prices_full: pd.DataFrame,
    shared_index: pd.DatetimeIndex,
    universe: pd.DataFrame,
    tbill: pd.Series,
    bt_config: BacktestConfig,
) -> dict:
    """P3 sub-period 3 — re-run C1b on 2018-05 to 2026-04 only.

    Sub-periods 1+2 unavailable: HEFA inception 2014-01. Sub-period 3 is
    fully within USD-bull era so adds limited independent evidence; reported
    for completeness only.
    """
    prices = prices_full[["VTI", "HEFA"]].dropna(how="any")
    prices = prices.loc[(prices.index >= PHASE4_P3_START) & (prices.index <= PHASE4_P3_END)]
    if prices.empty:
        raise RuntimeError(f"Empty price slice for sub-period 3")

    run = _backtest_static_mix(
        weight_ex_us=0.40,
        ex_us_ticker="HEFA",
        prices_in=prices,
        universe=universe,
        tbill=tbill,
        bt_config=bt_config,
        intl_cost_bps=GATING_COST_BPS,
    )
    inf = compute_inferential_stats(run, n_bootstrap=10_000)
    return {
        "sub_period": "3 of 3 (2018-05 to 2026-04)",
        "caveat": "Sub-periods 1+2 untestable (HEFA inception 2014). Sub-period 3 is fully within USD-bull era; not an independent cross-validation.",
        **{k: v for k, v in run.items() if not k.startswith("_")},
        **inf,
    }


def main() -> None:
    print("=" * 70)
    print("PHASE 4 ROBUSTNESS for C1b (60/40 VTI/HEFA)")
    print("Round 2 codex review SHOULD-DO: P1a placebo + P2 linear-scaling + P3 sub-period 3")
    print("=" * 70)

    cfg = load_workflow_config()
    gate = load_gate(cfg)
    universe = pd.read_csv(WORKFLOW_DIR / "data" / "reference" / "international_universe.csv")
    universe["inception_date"] = pd.to_datetime(universe["inception_date"])

    prices = load_prices()
    tbill = load_tbill_daily(prices.index)

    bt_config = BacktestConfig(
        train_months=cfg["walk_forward"]["train_months"],
        test_months=cfg["walk_forward"]["test_months"],
        step_months=cfg["walk_forward"]["step_months"],
        rebalance_frequency="annual",
        initial_capital=cfg["walk_forward"]["initial_capital"],
    )

    shared_index = prices[["VTI", "VEA", "HEFA"]].dropna(how="any").index
    print(f"Shared Phase 3 sample window: {shared_index.min().date()} to {shared_index.max().date()}, n={len(shared_index)}")
    print()

    print("--- P1a: USD-return mean-shifted placebo ---")
    p1a = run_p1a_placebo(prices, shared_index, universe, tbill, bt_config)
    print(f"  Delta applied: {p1a['delta_applied_per_day']:+.6f}/day = {p1a['delta_applied_annualized']:+.4f}/yr")
    print(f"  Raw HEFA mean: {p1a['raw_hefa_mean_daily']:+.6f} | Raw VTI mean: {p1a['raw_vti_mean_daily']:+.6f}")
    print(f"  After shift: HEFA mean = {p1a['shifted_hefa_mean_daily']:+.6f}")
    print(f"  Placebo sharpe_diff: {p1a['sharpe_diff_point']:+.3f}, "
          f"ci=[{p1a['sharpe_diff_ci_lower']:+.3f}, {p1a['sharpe_diff_ci_upper']:+.3f}]")
    print()

    print("--- P2: Linear-scaling sweep {1, 5, 10, 30, 50}% HEFA ---")
    p2 = run_p2_linear_scaling(prices, shared_index, universe, tbill, bt_config)
    print()

    print("--- P3 sub-period 3 (2018-05 to 2026-04, only sub-period testable) ---")
    p3 = run_p3_subperiod3_c1b(prices, shared_index, universe, tbill, bt_config)
    print(f"  C1b in sub-period 3: sharpe_diff={p3['sharpe_diff_point']:+.3f}, "
          f"ci=[{p3['sharpe_diff_ci_lower']:+.3f}, {p3['sharpe_diff_ci_upper']:+.3f}]")
    print()

    # Verdict
    print("=" * 70)
    print("ROBUSTNESS VERDICT for C1b")
    print("=" * 70)
    p1a_passes = (
        p1a["log_excess_mean_daily"] > 0
        and p1a["log_excess_ci_lower"] > 0
    )
    p2_monotonic = all(
        p2[i + 1]["sharpe_diff_point"] >= p2[i]["sharpe_diff_point"] - 0.02
        for i in range(len(p2) - 1)
    )
    p3_positive_sign = p3["sharpe_diff_point"] > 0

    print(f"P1a placebo passes (log-excess > 0 with CI lower > 0): {'YES' if p1a_passes else 'NO'}")
    print(f"  -> If NO: original +0.109 finding was source-period bias (HEFA mean drove it).")
    print(f"P2 linear-scaling roughly monotonic in weight: {'YES' if p2_monotonic else 'NO'}")
    print(f"  -> If NO: peak-then-collapse pattern indicates artifact.")
    print(f"P3 sub-period 3 sign positive: {'YES' if p3_positive_sign else 'NO'}")
    print(f"  -> Caveat: only 1 of 3 sub-periods testable; not a true cross-validation.")

    # Persist
    artifacts_dir = WORKFLOW_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_path = artifacts_dir / "phase4_c1b_robustness.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "run_date": date.today().isoformat(),
                "subject": "C1b 60/40 VTI/HEFA — only Phase 0-3 finding with positive CI upper bound",
                "p1a_mean_shifted_placebo": p1a,
                "p2_linear_scaling_sweep": p2,
                "p3_sub_period_3_only": p3,
                "verdict": {
                    "p1a_passes": bool(p1a_passes),
                    "p2_monotonic": bool(p2_monotonic),
                    "p3_positive_sign": bool(p3_positive_sign),
                },
                "p1a_pass_meaning": "If passes, +0.109 finding has structural component beyond source-period bias.",
                "p2_pass_meaning": "If monotonic in weight, structural rebalancing-premium support; if peak-collapse, artifact.",
                "p3_pass_meaning": "Limited evidence — sub-periods 1+2 untestable due to HEFA inception 2014.",
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("Wrote %s", out_path)

    # Markdown
    md = ["# Phase 4 Robustness — C1b (60/40 VTI/HEFA) only", ""]
    md.append(f"- Run date: {date.today()}")
    md.append("- Subject: the only Phase 0-3 finding whose CI upper bound is positive (+0.109 in unfixed Phase 3)")
    md.append("- Per Round 2 review SHOULD-DO + project memory `feedback_source_period_bias`")
    md.append("")
    md.append("## P1a — USD-return mean-shifted placebo")
    md.append("")
    md.append(f"- Delta applied: {p1a['delta_applied_per_day']:+.6f}/day ({p1a['delta_applied_annualized']:+.4f}/yr annualized)")
    md.append(f"- Raw HEFA mean: {p1a['raw_hefa_mean_daily']:+.6f}/day; raw VTI mean: {p1a['raw_vti_mean_daily']:+.6f}/day")
    md.append(f"- Placebo Sharpe-diff: **{p1a['sharpe_diff_point']:+.3f}** (CI [{p1a['sharpe_diff_ci_lower']:+.3f}, {p1a['sharpe_diff_ci_upper']:+.3f}])")
    md.append(f"- Placebo log-excess daily: {p1a['log_excess_mean_daily']:+.6f} (CI [{p1a['log_excess_ci_lower']:+.6f}, {p1a['log_excess_ci_upper']:+.6f}])")
    md.append(f"- **P1a verdict:** {'PASS — finding has structural component' if p1a_passes else 'FAIL — finding was source-period bias'}")
    md.append("")
    md.append("## P2 — Linear-scaling sweep")
    md.append("")
    md.append("| Weight HEFA | SharpeDiff | 90% CI |")
    md.append("|---|---|---|")
    for r in p2:
        md.append(f"| {int(r['weight_ex_us']*100)}% | {r['sharpe_diff_point']:+.3f} | "
                  f"[{r['sharpe_diff_ci_lower']:+.3f}, {r['sharpe_diff_ci_upper']:+.3f}] |")
    md.append("")
    md.append(f"**P2 verdict:** {'PASS — monotonic in weight (structural rebalancing-premium support)' if p2_monotonic else 'FAIL — non-monotonic (artifact)'}")
    md.append("")
    md.append("## P3 — Sub-period 3 only (2018-05 to 2026-04)")
    md.append("")
    md.append(f"- Sub-period 3 SharpeDiff: **{p3['sharpe_diff_point']:+.3f}** (CI [{p3['sharpe_diff_ci_lower']:+.3f}, {p3['sharpe_diff_ci_upper']:+.3f}])")
    md.append(f"- **CAVEAT:** Sub-periods 1+2 not testable (HEFA inception 2014-01-31). Sub-period 3 is fully within USD-bull era; provides limited independent evidence.")
    md.append("")
    md.append("## Combined verdict")
    md.append("")
    md.append(f"- P1a: {'PASS' if p1a_passes else 'FAIL'}")
    md.append(f"- P2:  {'PASS' if p2_monotonic else 'FAIL'}")
    md.append(f"- P3:  {'positive' if p3_positive_sign else 'negative'} (limited evidence)")
    md.append("")

    md_path = WORKFLOW_DIR / "research" / "phase4_c1b_robustness.md"
    md_path.write_text("\n".join(md))
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
