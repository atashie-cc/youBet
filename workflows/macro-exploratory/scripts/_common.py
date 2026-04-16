"""Shared helpers for macro-exploratory experiments.

Loads data, computes descriptive metrics + block-bootstrap CIs, and handles
the three sub-period consistency check used by the elevation rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "workflows" / "etf"))

from youbet.etf.risk import cagr_from_returns, sharpe_ratio  # noqa: E402
from youbet.etf.stats import excess_sharpe_ci  # noqa: E402
from youbet.utils.io import load_config  # noqa: E402

TRADING_DAYS = 252
RESULTS_DIR = WORKFLOW_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_workflow_config() -> dict:
    return load_config(WORKFLOW_ROOT / "config.yaml")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    """Max drawdown magnitude and duration in calendar days."""
    if len(returns) == 0:
        return 0.0, 0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = float(dd.min())

    is_dd = dd < -0.001
    max_dur = 0
    if is_dd.any():
        groups = (~is_dd).cumsum()
        dd_groups = groups[is_dd]
        for _, group in dd_groups.groupby(dd_groups):
            dur_days = (group.index[-1] - group.index[0]).days
            max_dur = max(max_dur, dur_days)
    return max_dd, max_dur


def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    """Core descriptive metrics for a daily return series."""
    if len(returns) == 0:
        return {"name": name, "error": "empty return series"}
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    cagr = cagr_from_returns(returns)
    sharpe = sharpe_ratio(returns)
    max_dd, max_dd_dur = max_drawdown(returns)
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0
    cum = (1 + returns).cumprod()
    final_wealth = float(cum.iloc[-1]) if len(cum) else 1.0
    return {
        "name": name,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "max_dd_duration_days": max_dd_dur,
        "calmar": calmar,
        "final_wealth": final_wealth,
        "n_days": int(len(returns)),
    }


def subperiod_consistency(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    windows: list[tuple[str, str]],
    periods_per_year: int = 252,
    min_obs: int = 50,
) -> dict:
    """Per-sub-period sign-consistency check using the Sharpe-of-excess estimand.

    Uses `Sharpe(strat - bench)` (the same metric as
    `excess_sharpe_ci["excess_sharpe_point"]`) for the sign test, so the
    sub-period consistency leg of the elevation gate is on the SAME estimand
    as the CI leg. Also reports the difference-of-Sharpes as diagnostic.
    """
    ann = np.sqrt(periods_per_year)

    def _sharpe(x):
        return float(x.mean() / max(x.std(), 1e-10) * ann)

    def _cagr(x):
        cum = (1 + x).cumprod()
        n_years = len(x) / periods_per_year
        return float(cum.iloc[-1] ** (1.0 / max(n_years, 0.01)) - 1.0) if len(x) else 0.0

    per_window = []
    signs = []
    for start, end in windows:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask_s = (strat_returns.index >= start_ts) & (strat_returns.index <= end_ts)
        mask_b = (bench_returns.index >= start_ts) & (bench_returns.index <= end_ts)
        s = strat_returns[mask_s]
        b = bench_returns[mask_b]
        if len(s) < min_obs or len(b) < min_obs:
            per_window.append({
                "start": start, "end": end, "status": "insufficient_data",
                "n_obs_strategy": int(len(s)), "n_obs_benchmark": int(len(b)),
            })
            continue
        # Align both series on intersection to build paired excess
        common = s.index.intersection(b.index)
        s_aligned = s.loc[common]
        b_aligned = b.loc[common]
        excess_series = s_aligned - b_aligned
        sharpe_of_excess = _sharpe(excess_series)
        strat_sharpe = _sharpe(s_aligned)
        bench_sharpe = _sharpe(b_aligned)
        sharpe_diff = strat_sharpe - bench_sharpe
        signs.append(np.sign(sharpe_of_excess))
        per_window.append({
            "start": start,
            "end": end,
            "strat_sharpe": strat_sharpe,
            "bench_sharpe": bench_sharpe,
            "sharpe_diff": sharpe_diff,                 # diagnostic
            "excess_sharpe": sharpe_of_excess,          # primary — matches CI
            "strat_cagr": _cagr(s_aligned),
            "bench_cagr": _cagr(b_aligned),
            "n_obs": int(len(s_aligned)),
        })

    same_sign = bool(signs and len(set(signs)) == 1 and signs[0] > 0)
    return {"per_window": per_window, "same_sign_positive_excess_sharpe": same_sign}


def bootstrap_excess_sharpe(
    strat_returns: pd.Series,
    bench_returns: pd.Series,
    n_bootstrap: int = 10_000,
    confidence: float = 0.90,
    block_length: int = 22,
    periods_per_year: int = 252,
) -> dict:
    """Wrap etf/stats excess_sharpe_ci. Returns point estimate + CI."""
    return excess_sharpe_ci(
        strat_returns,
        bench_returns,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        expected_block_length=block_length,
        periods_per_year=periods_per_year,
    )


# ---------------------------------------------------------------------------
# Elevation check
# ---------------------------------------------------------------------------

ELEVATION_VERSION = 2


def check_elevation(
    excess_sharpe_point: float,
    ci_lower: float,
    subperiod_same_sign: bool,
    sharpe_diff_point: float,
    threshold_excess_sharpe: float = 0.60,
) -> tuple[bool, list[str]]:
    """Apply the exploratory-to-confirmatory elevation rule (v2).

    v2 adds the Sharpe-difference positivity check (p4). Rationale: E2 showed
    that `Sharpe(strat - bench)` (Sharpe-of-excess) can be > 0 while
    `Sharpe(strat) - Sharpe(bench)` (difference-of-Sharpes) is < 0 for the
    same pair — a strategy whose risk-adjusted return is objectively worse
    than the benchmark can still pass a Sharpe-of-excess-only gate. Requiring
    both estimands to be positive closes that hole. Direction-of-change is
    rejection of spurious passes, not acceptance of spurious passes.
    """
    reasons = []
    p1 = excess_sharpe_point > threshold_excess_sharpe
    reasons.append(f"ExSharpe > {threshold_excess_sharpe}: {'PASS' if p1 else 'FAIL'} ({excess_sharpe_point:+.3f})")
    p2 = ci_lower > 0
    reasons.append(f"CI lower > 0: {'PASS' if p2 else 'FAIL'} ({ci_lower:+.3f})")
    p3 = subperiod_same_sign
    reasons.append(f"Sub-periods same-sign positive: {'PASS' if p3 else 'FAIL'}")
    p4 = sharpe_diff_point > 0
    reasons.append(f"Sharpe-diff > 0: {'PASS' if p4 else 'FAIL'} ({sharpe_diff_point:+.3f})")
    return (p1 and p2 and p3 and p4), reasons


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(experiment_name: str, result: dict[str, Any]) -> Path:
    """Save experiment result dict as JSON in results/."""
    def _coerce(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _coerce(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_coerce(x) for x in obj]
        return obj

    path = RESULTS_DIR / f"{experiment_name}.json"
    with open(path, "w") as f:
        json.dump(_coerce(result), f, indent=2, default=str)
    return path


def format_report(experiment_name: str, result: dict) -> str:
    """Human-readable result summary."""
    lines = [
        "=" * 72,
        f"EXPERIMENT: {experiment_name}",
        "=" * 72,
    ]
    if "description" in result:
        lines.append(result["description"])
        lines.append("")
    for bench_name, cmp in result.get("comparisons", {}).items():
        lines.append(f"--- vs {bench_name} ---")
        strat = cmp.get("strategy_metrics", {})
        bench = cmp.get("benchmark_metrics", {})
        ci = cmp.get("excess_sharpe_ci", {})
        lines.append(
            f"  Strategy:  CAGR {strat.get('cagr', 0):.1%}  "
            f"Sharpe {strat.get('sharpe', 0):.3f}  MaxDD {strat.get('max_dd', 0):.1%}"
        )
        lines.append(
            f"  Benchmark: CAGR {bench.get('cagr', 0):.1%}  "
            f"Sharpe {bench.get('sharpe', 0):.3f}  MaxDD {bench.get('max_dd', 0):.1%}"
        )
        if ci:
            lines.append(
                f"  ExSharpe:  {ci.get('excess_sharpe_point', 0):+.3f}  "
                f"90% CI [{ci.get('excess_sharpe_lower', 0):+.3f}, "
                f"{ci.get('excess_sharpe_upper', 0):+.3f}]"
            )
        lines.append("")
    if "elevation" in result:
        elev = result["elevation"]
        lines.append(f"Elevation: {'PASS (candidate for confirmatory)' if elev['passed'] else 'FAIL'}")
        for r in elev.get("reasons", []):
            lines.append(f"    - {r}")
    if "subperiods" in result:
        lines.append("")
        lines.append("Sub-period consistency (vs primary benchmark):")
        for win in result["subperiods"].get("per_window", []):
            if "excess_sharpe" in win:
                lines.append(
                    f"    {win['start']} to {win['end']}: "
                    f"strat SR {win['strat_sharpe']:+.3f}  "
                    f"bench SR {win['bench_sharpe']:+.3f}  "
                    f"excess {win['excess_sharpe']:+.3f}"
                )
            else:
                lines.append(f"    {win['start']} to {win['end']}: {win.get('status', 'n/a')}")
        same_sign = result["subperiods"].get("same_sign_positive_excess_sharpe", False)
        lines.append(f"    Same-sign positive across all windows: {same_sign}")
    lines.append("=" * 72)
    return "\n".join(lines)
