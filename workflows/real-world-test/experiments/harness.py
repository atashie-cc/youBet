"""Shared Monte Carlo harness for Phase A.

Provides a single bootstrap-and-evaluate pipeline that any strategy function
can plug into. Key features:
  - Paired daily block bootstrap (22-day Politis-Romano blocks)
  - Cadence parameter: weekly (5d), biweekly (10d), monthly approx (21d)
  - Flip-aware rebalance for trend strategies (signal flips trigger trades)
  - Annual or drift-threshold rebalance for static-weight strategies
  - Vectorized strategy execution where possible; per-path fallback otherwise

Strategy contract:
  A strategy is a function f(panel_block: np.ndarray, cols: dict) -> np.ndarray
  returning daily portfolio returns of length n_days. `cols` maps ticker name to
  column index in panel_block.

Output: dict of per-path metrics (terminal wealth, CAGR, MaxDD) plus paired
benchmark metrics for comparison.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class HarnessConfig:
    n_paths: int = 10_000
    horizon_years: int = 25
    block_length_days: int = 22
    seed: int = 20260429

    @property
    def n_days(self) -> int:
        return self.horizon_years * TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------


def sample_block_path(
    rng: np.random.Generator, panel: np.ndarray, n_days: int, block_len: int
) -> np.ndarray:
    n_src = panel.shape[0]
    n_blocks = n_days // block_len + 1
    starts = rng.integers(0, n_src - block_len + 1, size=n_blocks)
    parts = [panel[s : s + block_len] for s in starts]
    return np.concatenate(parts, axis=0)[:n_days]


# ---------------------------------------------------------------------------
# Reusable signal builders
# ---------------------------------------------------------------------------


def sma_signal(prices: np.ndarray, window: int) -> np.ndarray:
    """Daily 0/1 above-SMA signal. NaN for first window-1 days; fill with 1 (in-market)."""
    n = len(prices)
    sma = np.full(n, np.nan)
    if n > window:
        csum = np.cumsum(prices)
        sma[window - 1 :] = (csum[window - 1 :] - np.concatenate([[0.0], csum[: n - window]])) / window
    above = (prices > sma).astype(float)
    above[: window - 1] = 1.0
    return above


def cadence_subsample(daily_signal: np.ndarray, cadence_days: int) -> np.ndarray:
    """Hold the last decision-day's signal between decisions. T+1 shift inside.

    cadence_days=5 → weekly Friday equivalent (every 5th trading day)
    cadence_days=21 → ~monthly approximation
    """
    n = len(daily_signal)
    is_decision = ((np.arange(n) % cadence_days) == (cadence_days - 1))
    masked = np.where(is_decision, daily_signal, np.nan)
    # ffill
    idx = np.where(~np.isnan(masked), np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    out = np.where(idx >= 0, masked[np.maximum(idx, 0)], 1.0)
    # T+1: signal effective starting next day
    return np.concatenate([[1.0], out[:-1]])


def switching_costs(sig: np.ndarray, cost_bps: float) -> np.ndarray:
    flips = np.abs(np.diff(sig, prepend=sig[0])) > 0.5
    return flips.astype(float) * (cost_bps / 10000.0)


# ---------------------------------------------------------------------------
# Portfolio rebalancing — vectorized within annual segments
# ---------------------------------------------------------------------------


def annual_rebalance_returns(
    sleeve_rets: np.ndarray, target_weights: np.ndarray
) -> np.ndarray:
    """Compute portfolio daily returns with annual rebalance to target weights.

    Within each year, sleeves drift; weights drift; portfolio return is computed
    from drifted weights. At year boundary, weights snap back to target.
    Vectorized per-segment. NO drift-threshold trigger.
    """
    n_days, n_sleeves = sleeve_rets.shape
    port_rets = np.empty(n_days)
    start = 0
    while start < n_days:
        end = min(start + TRADING_DAYS_PER_YEAR, n_days)
        seg = sleeve_rets[start:end]
        sleeve_cum = np.cumprod(1.0 + seg, axis=0)
        port_cum = sleeve_cum @ target_weights
        port_seg = np.empty(end - start)
        port_seg[0] = port_cum[0] - 1.0
        port_seg[1:] = port_cum[1:] / port_cum[:-1] - 1.0
        port_rets[start:end] = port_seg
        start = end
    return port_rets


def drift_aware_rebalance_returns(
    sleeve_rets: np.ndarray,
    target_weights: np.ndarray,
    drift_threshold: float = 0.05,
) -> np.ndarray:
    """Annual rebalance + intra-year drift-trigger rebalance.

    Honors the precommit specification (Test 2 / Test 4 / Test 7 RP component).
    Tracks running weights day-by-day. When any weight has drifted by more than
    `drift_threshold` from target, snap back to target. Otherwise compound forward.

    Vectorized within drift-free segments; explicit segment boundaries on triggers.
    """
    n_days, n_sleeves = sleeve_rets.shape
    port_rets = np.empty(n_days)
    w = target_weights.copy()
    seg_start = 0
    last_year_boundary = 0
    for i in range(n_days):
        # Update weights from today's sleeve returns
        new_vals = w * (1.0 + sleeve_rets[i])
        s = new_vals.sum()
        if s <= 0:
            port_rets[i] = -1.0
            w = target_weights.copy()
            seg_start = i + 1
            continue
        port_rets[i] = float(np.dot(w, sleeve_rets[i]))
        w = new_vals / s
        # Trigger checks
        triggered = False
        # Year-end boundary
        if (i + 1 - last_year_boundary) >= TRADING_DAYS_PER_YEAR:
            triggered = True
            last_year_boundary = i + 1
        # Drift threshold
        elif (np.abs(w - target_weights) > drift_threshold).any():
            triggered = True
        if triggered:
            w = target_weights.copy()
            seg_start = i + 1
    return port_rets


def flip_aware_rebalance_returns(
    sleeve_rets: np.ndarray,
    target_weights: np.ndarray,
    flip_events: np.ndarray,
    cost_bps_per_flip: float = 10.0,
) -> np.ndarray:
    """Rebalance whenever any flip_event occurs OR at year-end.

    flip_events: boolean array of length n_days, True on days where a sleeve
    flipped its signal (and a portfolio rebalance is therefore desired).

    At each rebalance point, weights snap back to target. Switching cost is
    applied at the rebalance day.
    """
    n_days = sleeve_rets.shape[0]
    port_rets = np.empty(n_days)
    # Combine flip events + annual boundaries
    annual_marks = np.zeros(n_days, dtype=bool)
    annual_marks[TRADING_DAYS_PER_YEAR - 1 :: TRADING_DAYS_PER_YEAR] = True
    rebalance_marks = flip_events | annual_marks

    start = 0
    for end_inclusive in np.where(rebalance_marks)[0]:
        end = end_inclusive + 1
        seg = sleeve_rets[start:end]
        sleeve_cum = np.cumprod(1.0 + seg, axis=0)
        port_cum = sleeve_cum @ target_weights
        port_seg = np.empty(end - start)
        port_seg[0] = port_cum[0] - 1.0
        port_seg[1:] = port_cum[1:] / port_cum[:-1] - 1.0
        # Charge a single switching cost at the rebalance day
        port_seg[-1] -= cost_bps_per_flip / 10000.0
        port_rets[start:end] = port_seg
        start = end
    if start < n_days:
        seg = sleeve_rets[start:]
        sleeve_cum = np.cumprod(1.0 + seg, axis=0)
        port_cum = sleeve_cum @ target_weights
        port_seg = np.empty(n_days - start)
        port_seg[0] = port_cum[0] - 1.0
        port_seg[1:] = port_cum[1:] / port_cum[:-1] - 1.0
        port_rets[start:] = port_seg
    return port_rets


# ---------------------------------------------------------------------------
# Path metrics
# ---------------------------------------------------------------------------


def path_metrics(returns: np.ndarray) -> tuple[float, float, float]:
    """Returns (terminal, cagr, max_dd)."""
    cum = np.cumprod(1.0 + returns)
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    terminal = float(cum[-1])
    cagr = float(terminal ** (1.0 / n_years) - 1.0) if terminal > 0 else -1.0
    running = np.maximum.accumulate(cum)
    max_dd = float((cum / running - 1.0).min())
    return terminal, cagr, max_dd


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_mc(
    panel: pd.DataFrame,
    strategy_fn: Callable[[np.ndarray, dict[str, int]], np.ndarray],
    config: HarnessConfig,
    extra_strategies: dict[str, Callable] | None = None,
) -> pd.DataFrame:
    """Run Monte Carlo on a primary strategy plus optional secondary strategies.

    All strategies see the SAME bootstrapped panel block on each path, so paired
    comparisons across strategies are valid (same source noise).
    """
    cols = {c: i for i, c in enumerate(panel.columns)}
    panel_arr = panel.to_numpy()
    n_days = config.n_days
    rng = np.random.default_rng(config.seed)
    log_every = max(config.n_paths // 10, 1)

    extras = extra_strategies or {}
    all_strategies = {"primary": strategy_fn, **extras}

    rows: list[dict] = []
    t0 = time.time()
    for i in range(config.n_paths):
        block = sample_block_path(rng, panel_arr, n_days, config.block_length_days)
        row: dict[str, float] = {}
        for name, fn in all_strategies.items():
            r = fn(block, cols)
            term, cagr, mdd = path_metrics(r)
            row[f"{name}_terminal"] = term
            row[f"{name}_cagr"] = cagr
            row[f"{name}_max_dd"] = mdd
        rows.append(row)
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (config.n_paths - i - 1) / max(rate, 1e-6)
            logger.info(
                "  Path %d/%d (%.0f%%)  elapsed %.1fs  rate %.1f/s  ETA %.0fs",
                i + 1, config.n_paths, 100 * (i + 1) / config.n_paths,
                elapsed, rate, eta,
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Standard comparators (used by all Phase A tests)
# ---------------------------------------------------------------------------


def vti_buy_hold(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    return block[:, cols["VTI"]]


def vti60_bnd40(block: np.ndarray, cols: dict[str, int]) -> np.ndarray:
    rets = np.column_stack([block[:, cols["VTI"]], block[:, cols["BND"]]])
    return annual_rebalance_returns(rets, np.array([0.60, 0.40]))


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def percentile_table(df: pd.DataFrame, names: list[tuple[str, str]], metric: str = "terminal") -> str:
    """Format a percentile table for terminal/CAGR/MaxDD."""
    header = f"{'Strategy':<28} {'p1':>9} {'p5':>9} {'p25':>9} {'p50':>9} {'p75':>9} {'p95':>9} {'p99':>9} {'mean':>9}"
    lines = [header, "-" * len(header)]
    for col_prefix, label in names:
        col = f"{col_prefix}_{metric}"
        if col not in df.columns:
            continue
        s = df[col].values
        ps = np.percentile(s, [1, 5, 25, 50, 75, 95, 99])
        if metric == "terminal":
            line = f"{label:<28} " + " ".join(f"{p:>8.2f}x" for p in ps) + f" {s.mean():>8.2f}x"
        else:
            line = f"{label:<28} " + " ".join(f"{p:>+8.2%}" for p in ps) + f" {s.mean():>+8.2%}"
        lines.append(line)
    return "\n".join(lines)


def head_to_head(df: pd.DataFrame, primary: str, comparator: str) -> dict:
    p_term = df[f"{primary}_terminal"]
    c_term = df[f"{comparator}_terminal"]
    log_excess = np.log(p_term) - np.log(c_term)
    cagr_excess = df[f"{primary}_cagr"] - df[f"{comparator}_cagr"]
    mdd_diff = df[f"{primary}_max_dd"] - df[f"{comparator}_max_dd"]
    return {
        "p_beats": float((p_term > c_term).mean()),
        "mean_log_excess": float(log_excess.mean()),
        "median_log_excess": float(log_excess.median()),
        "p5_log_excess": float(np.percentile(log_excess, 5)),
        "p95_log_excess": float(np.percentile(log_excess, 95)),
        "median_cagr_excess": float(cagr_excess.median()),
        "mean_cagr_excess": float(cagr_excess.mean()),
        "p_neg_cagr_primary": float((df[f"{primary}_cagr"] < 0).mean()),
        "p_neg_cagr_comparator": float((df[f"{comparator}_cagr"] < 0).mean()),
        "primary_p5_mdd": float(np.percentile(df[f"{primary}_max_dd"], 5)),
        "comparator_p5_mdd": float(np.percentile(df[f"{comparator}_max_dd"], 5)),
        "p5_mdd_diff_pp": float((np.percentile(df[f"{primary}_max_dd"], 5) -
                                  np.percentile(df[f"{comparator}_max_dd"], 5)) * 100),
    }
