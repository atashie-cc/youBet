"""Phase 2: Publication-Lag Alpha Decay Measurement.

Measure how factor premiums decay after academic publication.
McLean & Pontiff (2016) found 26% OOS decay and 58% post-publication
decay ON AVERAGE. This study measures FACTOR-SPECIFIC decay using
the Ken French daily data.

For each factor:
  1. Split returns into pre-publication and post-publication periods
  2. Compute Sharpe ratio for each period
  3. Block bootstrap CI for the Sharpe difference
  4. Report factor-specific decay rates

Publication dates:
  - Mkt-RF: 1964 (Sharpe/CAPM)
  - HML: 1992-06 (Fama-French 1992)
  - SMB: 1992-06 (Fama-French 1992)
  - UMD: 1993-03 (Jegadeesh-Titman 1993)
  - RMW: 2013-09 (Novy-Marx 2013 / Fama-French 2015)
  - CMA: 2013-09 (Fama-French 2015)

This is a CALIBRATION study, not a strategy search. No Holm correction
is applied because we are measuring decay, not testing for significance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Setup paths
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOW_ROOT / "experiments"))

from _shared import (
    compute_metrics,
    load_config,
    load_factors,
    print_table,
)

from youbet.factor.data import PUBLICATION_DATES
from youbet.etf.stats import bootstrap_confidence_interval
from youbet.etf.risk import sharpe_ratio as compute_sharpe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def sharpe_from_returns(returns: np.ndarray) -> float:
    """Annualized Sharpe from daily returns (no RF adjustment)."""
    if len(returns) < 22:
        return 0.0
    return float(returns.mean() / max(returns.std(), 1e-10) * np.sqrt(252))


def measure_decay(
    factor_returns: pd.Series,
    publication_date: str,
    factor_name: str,
    n_bootstrap: int = 10_000,
) -> dict:
    """Measure Sharpe decay around publication date.

    Returns dict with pre/post Sharpe, decay rate, and bootstrap CI.
    """
    pub_date = pd.Timestamp(publication_date)

    pre = factor_returns[factor_returns.index < pub_date]
    post = factor_returns[factor_returns.index >= pub_date]

    if len(pre) < 252 or len(post) < 252:
        logger.warning(
            "%s: insufficient data (pre=%d, post=%d days)",
            factor_name, len(pre), len(post),
        )
        return {
            "factor": factor_name,
            "pub_date": publication_date,
            "pre_sharpe": None,
            "post_sharpe": None,
            "decay_pct": None,
            "pre_years": len(pre) / 252,
            "post_years": len(post) / 252,
        }

    pre_sharpe = sharpe_from_returns(pre.values)
    post_sharpe = sharpe_from_returns(post.values)

    # Decay: how much of the pre-publication Sharpe is lost?
    if abs(pre_sharpe) > 1e-6:
        decay_pct = 1.0 - (post_sharpe / pre_sharpe)
    else:
        decay_pct = 0.0

    # Bootstrap CIs for pre and post Sharpe individually
    pre_ci = bootstrap_confidence_interval(
        pre, sharpe_from_returns, n_bootstrap=n_bootstrap, confidence=0.90
    )
    post_ci = bootstrap_confidence_interval(
        post, sharpe_from_returns, n_bootstrap=n_bootstrap, confidence=0.90
    )

    # Bootstrap CI for the Sharpe DIFFERENCE (pre - post)
    # This requires a paired approach, but since pre and post are
    # non-overlapping time periods, we bootstrap each independently
    # and compute the difference distribution.
    from youbet.etf.stats import stationary_block_bootstrap
    pre_boot = stationary_block_bootstrap(pre.values, n_bootstrap, seed=42)
    post_boot = stationary_block_bootstrap(post.values, n_bootstrap, seed=43)
    diff_boot = pre_boot - post_boot
    diff_ci = (
        float(np.percentile(diff_boot, 5)),
        float(np.percentile(diff_boot, 95)),
    )

    return {
        "factor": factor_name,
        "pub_date": publication_date,
        "pre_sharpe": pre_sharpe,
        "post_sharpe": post_sharpe,
        "decay_pct": decay_pct,
        "pre_years": len(pre) / 252,
        "post_years": len(post) / 252,
        "pre_ci_90": pre_ci,
        "post_ci_90": post_ci,
        "diff_point": pre_sharpe - post_sharpe,
        "diff_ci_90": diff_ci,
    }


def main():
    print("=" * 80)
    print("PHASE 2: PUBLICATION-LAG ALPHA DECAY MEASUREMENT")
    print("=" * 80)
    print("\nMethodology: Compare Sharpe ratio pre- vs post-publication")
    print("for each factor using block bootstrap CIs.")
    print("Reference: McLean & Pontiff (2016) — 26% OOS, 58% post-pub decay.")

    config = load_config()
    factors = load_factors()

    # Full-sample descriptive stats
    factor_names = config["factors"]
    all_metrics = []
    for factor in factor_names:
        m = compute_metrics(factors[factor], factor)
        all_metrics.append(m)
    print_table(all_metrics, "Full-Sample Factor Statistics (1963-2026)")

    # Measure decay for each factor
    print(f"\n{'=' * 100}")
    print("DECAY MEASUREMENTS")
    print(f"{'=' * 100}")

    results = []
    for factor in factor_names:
        if factor not in PUBLICATION_DATES:
            logger.warning("No publication date for %s, skipping", factor)
            continue

        pub_date = PUBLICATION_DATES[factor]
        logger.info("Measuring decay for %s (published %s)", factor, pub_date)
        result = measure_decay(factors[factor], pub_date, factor)
        results.append(result)

    # Report
    print(f"\n{'Factor':<10} {'Pub Date':>12} {'Pre Sharpe':>12} {'Post Sharpe':>12} "
          f"{'Decay':>8} {'Pre Yrs':>8} {'Post Yrs':>9} {'Diff 90% CI':>24}")
    print("-" * 100)

    for r in results:
        if r["pre_sharpe"] is None:
            print(f"{r['factor']:<10} {r['pub_date']:>12}  INSUFFICIENT DATA")
            continue

        diff_ci = r.get("diff_ci_90", (0, 0))
        print(
            f"{r['factor']:<10} {r['pub_date']:>12} "
            f"{r['pre_sharpe']:>+11.3f} {r['post_sharpe']:>+11.3f} "
            f"{r['decay_pct']:>7.0%} "
            f"{r['pre_years']:>7.1f} {r['post_years']:>8.1f} "
            f"[{diff_ci[0]:>+6.3f}, {diff_ci[1]:>+6.3f}]"
        )

    # Summary statistics
    valid = [r for r in results if r["pre_sharpe"] is not None]
    if valid:
        decay_rates = [r["decay_pct"] for r in valid]
        avg_decay = np.mean(decay_rates)
        median_decay = np.median(decay_rates)

        print(f"\n--- Summary ---")
        print(f"Average decay:  {avg_decay:.0%}")
        print(f"Median decay:   {median_decay:.0%}")
        print(f"McLean-Pontiff: 58% (post-publication reference)")

        # Factor-specific haircuts
        print(f"\n--- Factor-Specific Haircuts ---")
        print("Use these to discount published Sharpe claims:")
        for r in valid:
            if r["decay_pct"] is not None:
                remaining = 1.0 - r["decay_pct"]
                print(f"  {r['factor']:<8}: {r['decay_pct']:>5.0%} decay -> "
                      f"multiply published Sharpe by {max(remaining, 0):.2f}")

        # Power assessment
        print(f"\n--- Power Assessment ---")
        for r in valid:
            diff_ci = r.get("diff_ci_90", (0, 0))
            ci_excludes_zero = diff_ci[0] > 0 or diff_ci[1] < 0
            if diff_ci[0] > 0:
                verdict = "SIGNIFICANT DECAY (CI excludes zero, pre > post)"
            elif diff_ci[1] < 0:
                verdict = "SIGNIFICANT IMPROVEMENT (CI excludes zero, post > pre)"
            else:
                verdict = "INCONCLUSIVE (CI spans zero)"
            print(f"  {r['factor']:<8}: {verdict}")

    print(f"\n{'=' * 80}")
    print("PHASE 2 COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
