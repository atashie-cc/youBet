"""Shared utilities for strategy-dashboard.

Provides:
- Roster + config loading
- Strategy daily-return loader (parquet col lookup OR daily-rebal weighted blend from panel)
- 16-metric battery via compute_full_metrics
- MAD-robust z-score + composite ranking
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as scipy_stats

THIS_FILE = Path(__file__).resolve()
WORKFLOW_DIR = THIS_FILE.parent.parent
REPO_ROOT = WORKFLOW_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.risk import compute_risk_metrics  # noqa: E402

CONFIG_PATH = WORKFLOW_DIR / "config.yaml"
ROSTER_PATH = WORKFLOW_DIR / "artifacts" / "roster.json"
ARTIFACTS_DIR = WORKFLOW_DIR / "artifacts"
RESEARCH_DIR = WORKFLOW_DIR / "research"

TRADING_DAYS = 252


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_roster() -> dict:
    with open(ROSTER_PATH) as f:
        return json.load(f)


def load_vti_benchmark(config: dict) -> pd.Series:
    """Load VTI daily returns for use as universal Sharpe-of-excess benchmark."""
    parquet_path = REPO_ROOT / config["benchmark"]["vti_parquet"]
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    series = df[config["benchmark"]["vti_column"]].dropna()
    series.name = "VTI"
    return series


def load_panel(config: dict) -> pd.DataFrame:
    """Load the panel parquet (multiple asset return columns) for reference baseline construction."""
    parquet_path = REPO_ROOT / config["benchmark"]["vti_parquet"]
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def load_strategy(entry: dict) -> pd.Series:
    """Load a strategy's daily returns from its parquet (column lookup or constructed blend).

    Schema for `entry`:
      - parquet (relative path)
      - candidate_column (str)  OR  construction (dict with weights, parquet_panel)
    Optional:
      - window_start, window_end (date strings to slice the native window)
    """
    parquet_path = REPO_ROOT / entry["parquet"]
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    construction = entry.get("construction")
    if construction is not None:
        weights = construction["weights"]
        for asset in weights:
            if asset not in df.columns:
                raise KeyError(f"asset column '{asset}' not in {parquet_path}")
        series = sum(w * df[asset] for asset, w in weights.items())
    else:
        col = entry["candidate_column"]
        if col not in df.columns:
            raise KeyError(f"column '{col}' not in {parquet_path}")
        series = df[col]

    series = series.dropna()
    series.name = entry["id"]
    if entry.get("window_start"):
        series = series[series.index >= pd.Timestamp(entry["window_start"])]
    if entry.get("window_end"):
        series = series[series.index <= pd.Timestamp(entry["window_end"])]
    return series


def restrict_window(series: pd.Series, start: str | None, end: str | None) -> pd.Series:
    if start:
        series = series[series.index >= pd.Timestamp(start)]
    if end:
        series = series[series.index <= pd.Timestamp(end)]
    return series


# ---------------------------------------------------------------------------
# Metric primitives (those NOT already in risk.py)


def _rolling_total_return(daily: pd.Series, window: int = 252) -> pd.Series:
    """Rolling-window total return: prod(1+r) - 1 over each `window`-day slice."""
    log_r = np.log1p(daily)
    rolling_log = log_r.rolling(window).sum()
    return np.expm1(rolling_log)


def _rolling_sharpe(daily: pd.Series, window: int = 252) -> pd.Series:
    """Rolling-window annualized Sharpe."""
    mean = daily.rolling(window).mean()
    std = daily.rolling(window).std()
    sharpe = mean / std.replace(0, np.nan) * np.sqrt(TRADING_DAYS)
    return sharpe


def _regime_period_return(daily: pd.Series, start: str, end: str) -> float:
    """Total return over a calendar slice. Returns NaN if no overlap."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    sub = daily.loc[(daily.index >= s) & (daily.index <= e)]
    if len(sub) < 5:  # require >=5 trading days of data
        return float("nan")
    return float((1 + sub).prod() - 1)


def compute_full_metrics(
    daily: pd.Series, benchmark: pd.Series, regime_periods: dict, label: str = ""
) -> dict:
    """Compute the 16-metric battery for one strategy.

    Args:
        daily: strategy daily simple returns
        benchmark: VTI daily simple returns (or other universal benchmark for IR)
        regime_periods: {regime_name: [start_iso, end_iso]} from config

    Returns flat dict of {metric_name: float}. Some entries may be NaN
    (e.g. regime returns outside strategy window).
    """
    if len(daily) < TRADING_DAYS:
        raise ValueError(f"{label}: need >= 252 days, got {len(daily)}")

    rm = compute_risk_metrics(daily, benchmark_returns=benchmark)

    rolling_1y = _rolling_total_return(daily, TRADING_DAYS)
    median_1y = float(np.nanmedian(rolling_1y)) if rolling_1y.notna().any() else float("nan")
    worst_1y = float(np.nanmin(rolling_1y)) if rolling_1y.notna().any() else float("nan")

    rolling_sh = _rolling_sharpe(daily, TRADING_DAYS)
    rolling_sh_std = float(rolling_sh.std()) if rolling_sh.notna().sum() > 50 else float("nan")

    skew = float(scipy_stats.skew(daily.values, bias=False))

    metrics: dict[str, float] = {
        # Group A — Return
        "cagr": rm.annualized_return,
        "annualized_return": rm.annualized_return,
        "median_rolling_1y_return": median_1y,
        # Group B — Risk-adjusted
        "sharpe": rm.sharpe_ratio,
        "sortino": rm.sortino_ratio,
        "info_ratio_vs_vti": rm.information_ratio,
        # Group C — Drawdown
        "max_drawdown": rm.max_drawdown,
        "calmar": rm.calmar_ratio,
        "longest_underwater_days": float(rm.max_drawdown_duration_days),
        # Group D — Tail / distribution
        "cvar_95": rm.cvar_95,
        "skew": skew,
        "worst_rolling_12mo": worst_1y,
        # Group E — Regime
        "return_gfc_2008": _regime_period_return(daily, *regime_periods["gfc_2008"]),
        "return_covid_2020": _regime_period_return(daily, *regime_periods["covid_2020"]),
        "return_stagflation_2022": _regime_period_return(daily, *regime_periods["stagflation_2022"]),
        # Group F — Stability
        "rolling_sharpe_std": rolling_sh_std,
        # Metadata (not in composite)
        "_n_obs": float(rm.n_observations),
        "_total_return": rm.total_return,
        "_annualized_vol": rm.annualized_volatility,
        "_correlation_to_vti": rm.correlation_to_benchmark,
        "_window_start": str(daily.index.min().date()),
        "_window_end": str(daily.index.max().date()),
    }
    return metrics


# ---------------------------------------------------------------------------
# Composite scoring


def mad_zscore(x: np.ndarray) -> np.ndarray:
    """MAD-robust z-score: (x - median) / MAD. NaN inputs propagate to NaN outputs."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        # If MAD is zero, all (non-NaN) values are equal; return zeros for those.
        out = np.zeros_like(x)
        out[~np.isfinite(x)] = np.nan
        return out
    return (x - med) / (1.4826 * mad)  # 1.4826 → MAD ~ σ for Gaussian


def compose_metric_matrix(
    metrics_df: pd.DataFrame,
    metric_names: list[str],
    lower_is_better: list[str],
    weights: dict[str, float],
    impute_nan: bool = True,
) -> pd.DataFrame:
    """Build a z-scored metric matrix and weighted composite.

    Args:
        metrics_df: rows=strategy_id, columns include `metric_names`
        metric_names: list of metric columns to include in composite
        lower_is_better: subset that should be sign-flipped before z-scoring
        weights: {metric: weight} (default 1.0)
        impute_nan: if True, NaN values are filled with the column median (z=0) before composite

    Returns DataFrame with z-score columns + composite column, indexed like metrics_df.
    """
    z_mat = pd.DataFrame(index=metrics_df.index)
    active_metrics: list[str] = []
    for m in metric_names:
        x = metrics_df[m].values.astype(float)
        if np.all(np.isnan(x)):
            # Whole-column NaN — skip this metric in this window's composite.
            # Common case: regime-period metrics whose epoch is outside the scoring window.
            z_mat[f"z__{m}"] = np.nan
            continue
        sign = -1.0 if m in lower_is_better else 1.0
        z = mad_zscore(sign * x)
        z_mat[f"z__{m}"] = z
        active_metrics.append(m)

    # Composite — only over metrics with at least some non-NaN values.
    # Each metric's z is clipped to [-3, +3] before summing so a single pathological
    # metric (e.g. rolling_sharpe_std for paper factor SMA200 which sees near-zero
    # daily returns and infinite Sharpes on some windows) cannot dominate the
    # composite. The unclipped z is preserved in `z__{m}` columns for per-metric
    # diagnostics.
    composite = pd.Series(0.0, index=metrics_df.index)
    total_w = 0.0
    for m in active_metrics:
        z_col = z_mat[f"z__{m}"].values.copy()
        if impute_nan:
            med = np.nanmedian(z_col)
            if np.isnan(med):
                med = 0.0
            z_col = np.where(np.isnan(z_col), med, z_col)
        # Clip to [-3, +3] for the composite contribution
        z_col_clipped = np.clip(z_col, -3.0, 3.0)
        w = float(weights.get(m, 1.0))
        composite = composite + w * pd.Series(z_col_clipped, index=metrics_df.index)
        total_w += w

    z_mat["composite_z"] = composite / max(total_w, 1e-9)
    return z_mat


def per_group_score(
    metrics_df: pd.DataFrame, metric_groups: dict[str, list[str]], lower_is_better: list[str]
) -> pd.DataFrame:
    """Compute per-group composite z-score (equal-weight within group, MAD-robust)."""
    out = pd.DataFrame(index=metrics_df.index)
    for group_name, group_metrics in metric_groups.items():
        z_cols = []
        for m in group_metrics:
            if m not in metrics_df.columns:
                continue
            x = metrics_df[m].values.astype(float)
            sign = -1.0 if m in lower_is_better else 1.0
            z = mad_zscore(sign * x)
            z_cols.append(pd.Series(z, index=metrics_df.index))
        if not z_cols:
            out[f"group__{group_name}"] = np.nan
            continue
        # Average z (NaN-aware), with same [-3, +3] clipping as composite
        stacked = pd.concat(z_cols, axis=1).clip(lower=-3.0, upper=3.0)
        out[f"group__{group_name}"] = stacked.mean(axis=1, skipna=True)
    return out
