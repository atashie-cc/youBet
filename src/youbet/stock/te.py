"""Empirical tracking error + MDE re-computation for Phase 2.

Phase 0 used a literature-anchored TE (8%/yr) to compute MDE; Phase 2
replaces that with the empirically realized TE from Phase 1 return
series and reports the revised power picture. If the empirical TE is
materially lower than the anchor, MDE improves (often halves); if
higher, MDE worsens.

Also exposes a lightweight `power_at_target` helper that computes
empirical power on a given (target_sharpe, n_years, te) triplet so the
orchestrator can emit a "power-if-this-TE" table without re-running the
full Phase 0 simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from youbet.etf.stats import block_bootstrap_test

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class TrackingErrorReport:
    annualized_te: float
    annualized_ir: float   # information ratio = mean_excess_annual / TE
    n_days: int
    mean_daily_excess: float
    mean_annual_excess: float


def empirical_tracking_error(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
) -> TrackingErrorReport:
    """Annualized tracking error = stdev of daily excess returns × √252.

    Uses the intersection of the two indices. The excess mean is reported
    so callers can recover the information ratio (mean_excess / TE).
    """
    common = strat_returns.index.intersection(bench_returns.index)
    s = strat_returns.loc[common].astype(float)
    b = bench_returns.loc[common].astype(float)
    excess = s - b

    # Drop any leading NaNs from misaligned rebalancing
    excess = excess.dropna()
    if len(excess) < 30:
        raise ValueError(
            f"Insufficient overlap to estimate TE (n={len(excess)} < 30)"
        )

    sd_daily = float(excess.std())
    mean_daily = float(excess.mean())
    te_ann = sd_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
    mean_ann = mean_daily * TRADING_DAYS_PER_YEAR
    ir = mean_ann / te_ann if te_ann > 0 else 0.0
    return TrackingErrorReport(
        annualized_te=te_ann,
        annualized_ir=ir,
        n_days=len(excess),
        mean_daily_excess=mean_daily,
        mean_annual_excess=mean_ann,
    )


def recompute_power(
    n_years: int,
    target_sharpes: list[float],
    tracking_error_annual: float,
    n_sims: int = 100,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict[float, float]:
    """Empirical power at (n_years, TE) for each target Sharpe.

    Same construction as `phase0_infrastructure.test_power_analysis` —
    duplicated here intentionally so Phase 2 can re-run with an empirical
    TE without touching the frozen Phase 0 script.
    """
    rng = np.random.default_rng(seed)
    n = n_years * TRADING_DAYS_PER_YEAR
    sd_daily = tracking_error_annual / np.sqrt(TRADING_DAYS_PER_YEAR)

    powers: dict[float, float] = {}
    for target in target_sharpes:
        mu_daily = target * sd_daily / np.sqrt(TRADING_DAYS_PER_YEAR)
        rejects = 0
        for _ in range(n_sims):
            bench = rng.normal(0.0003, 0.01, n)
            excess = rng.normal(mu_daily, sd_daily, n)
            strat = bench + excess
            idx = pd.bdate_range("2010-01-01", periods=n)
            res = block_bootstrap_test(
                pd.Series(strat, index=idx),
                pd.Series(bench, index=idx),
                n_bootstrap=n_bootstrap, expected_block_length=22, seed=42,
            )
            if res["p_value"] < alpha:
                rejects += 1
        powers[target] = rejects / n_sims
    return powers


def mde_at_80_power(
    powers: dict[float, float],
    threshold: float = 0.80,
) -> float | None:
    """Smallest target with empirical power ≥ threshold, or None."""
    for t in sorted(powers):
        if powers[t] >= threshold:
            return t
    return None


def power_sensitivity_table(
    n_years: int,
    target_sharpes: list[float],
    te_anchors: list[float],
    n_sims: int = 50,
    n_bootstrap: int = 300,
    seed: int = 13,
) -> pd.DataFrame:
    """Grid of empirical power over (target_sharpe × te_anchor).

    Compact sensitivity report: how does MDE shift as our TE assumption
    changes? Used by Phase 2 orchestrator to produce the `power-if-TE-is-X`
    table that informs the reframing decision.

    Returns DataFrame indexed by target_sharpe, columns per TE anchor.
    """
    rows = {}
    for te in te_anchors:
        rows[f"TE={te:.2f}"] = recompute_power(
            n_years=n_years,
            target_sharpes=target_sharpes,
            tracking_error_annual=te,
            n_sims=n_sims,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    df = pd.DataFrame(rows)
    df.index.name = "target_sharpe"
    return df


def mde_table(powers_table: pd.DataFrame, threshold: float = 0.80) -> pd.Series:
    """Given a power_sensitivity_table, compute MDE per column.

    None if no target reaches `threshold`.
    """
    out = {}
    for col in powers_table.columns:
        series = powers_table[col]
        hits = series[series >= threshold]
        out[col] = float(hits.index.min()) if len(hits) else None
    return pd.Series(out, name=f"mde_at_{int(threshold*100)}pct_power")
