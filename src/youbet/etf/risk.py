"""Risk metrics computation.

Sharpe, Sortino, MaxDD, CVaR, information ratio, Kelly sizing,
risk-of-ruin estimation. All metrics annualized from daily returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class RiskMetrics:
    """Container for risk-adjusted performance metrics."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    cvar_95: float
    cvar_99: float
    information_ratio: float  # vs benchmark
    correlation_to_benchmark: float
    annual_turnover: float
    n_observations: int

    def summary(self) -> str:
        lines = [
            f"Annualized Return:  {self.annualized_return:+.4f}",
            f"Annualized Vol:     {self.annualized_volatility:.4f}",
            f"Sharpe Ratio:       {self.sharpe_ratio:.4f}",
            f"Sortino Ratio:      {self.sortino_ratio:.4f}",
            f"Max Drawdown:       {self.max_drawdown:.4f}",
            f"Max DD Duration:    {self.max_drawdown_duration_days} days",
            f"Calmar Ratio:       {self.calmar_ratio:.4f}",
            f"CVaR 95%:           {self.cvar_95:.4f}",
            f"CVaR 99%:           {self.cvar_99:.4f}",
            f"Information Ratio:  {self.information_ratio:.4f}",
            f"Corr to Benchmark:  {self.correlation_to_benchmark:.4f}",
            f"Annual Turnover:    {self.annual_turnover:.4f}",
            f"N Observations:     {self.n_observations}",
        ]
        return "\n".join(lines)


def compute_risk_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float | pd.Series = 0.0,
    annual_turnover: float = 0.0,
) -> RiskMetrics:
    """Compute full risk metrics battery from daily returns.

    Args:
        returns: Daily simple returns of the strategy.
        benchmark_returns: Daily simple returns of benchmark (for IR, correlation).
        risk_free_rate: Annual risk-free rate (scalar) or daily rate series.
        annual_turnover: Pre-computed annual turnover.

    Returns:
        RiskMetrics dataclass.
    """
    r = returns.dropna()
    n = len(r)

    if isinstance(risk_free_rate, pd.Series):
        # Convert annual rate to daily
        daily_rf = risk_free_rate.reindex(r.index, method="ffill").fillna(0.0) / TRADING_DAYS
    else:
        daily_rf = risk_free_rate / TRADING_DAYS

    excess = r - daily_rf

    # Annualized return (geometric)
    total_ret = (1 + r).prod() - 1
    n_years = n / TRADING_DAYS
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 1e-6)) - 1

    # Annualized volatility
    ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))

    # Sharpe
    sharpe = float(excess.mean() / max(excess.std(), 1e-10) * np.sqrt(TRADING_DAYS))

    # Sortino (downside deviation)
    downside = excess[excess < 0]
    downside_std = float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 1e-10
    sortino = float(excess.mean() / max(downside_std, 1e-10) * np.sqrt(TRADING_DAYS))

    # Max drawdown
    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())
    dd_duration = _max_drawdown_duration(drawdown)

    # Calmar
    calmar = ann_ret / max(abs(max_dd), 1e-10)

    # CVaR
    cvar_95 = _cvar(r, 0.05)
    cvar_99 = _cvar(r, 0.01)

    # Information ratio and correlation (vs benchmark)
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(r.index).dropna()
        common = r.index.intersection(bench.index)
        if len(common) > 10:
            excess_vs_bench = r[common] - bench[common]
            te = float(excess_vs_bench.std() * np.sqrt(TRADING_DAYS))
            ir = float(excess_vs_bench.mean() * TRADING_DAYS / max(te, 1e-10))
            corr = float(r[common].corr(bench[common]))
        else:
            ir, corr = 0.0, 0.0
    else:
        ir, corr = 0.0, 0.0

    return RiskMetrics(
        total_return=float(total_ret),
        annualized_return=float(ann_ret),
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=dd_duration,
        calmar_ratio=float(calmar),
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        information_ratio=ir,
        correlation_to_benchmark=corr,
        annual_turnover=annual_turnover,
        n_observations=n,
    )


def _cvar(returns: pd.Series, alpha: float) -> float:
    """Conditional Value at Risk (Expected Shortfall) at given alpha."""
    if len(returns) == 0:
        return 0.0
    sorted_r = returns.sort_values()
    cutoff = max(1, int(np.ceil(len(sorted_r) * alpha)))
    return float(sorted_r.iloc[:cutoff].mean())


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Duration of the longest drawdown period in calendar days."""
    is_dd = drawdown < 0
    if not is_dd.any():
        return 0

    # Find streaks of drawdown
    groups = (~is_dd).cumsum()
    dd_groups = groups[is_dd]
    if len(dd_groups) == 0:
        return 0

    durations = []
    for _, group in dd_groups.groupby(dd_groups):
        start = group.index[0]
        end = group.index[-1]
        if hasattr(start, 'date'):
            durations.append((end - start).days)
        else:
            # Integer index — count positions
            durations.append(int(end - start))

    return max(durations) if durations else 0


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0
) -> float:
    """Quick Sharpe ratio from daily returns."""
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    return float(excess.mean() / max(excess.std(), 1e-10) * np.sqrt(TRADING_DAYS))


def kelly_optimal_weight(
    expected_excess_return: float,
    variance: float,
    fraction: float = 0.25,
    max_position: float = 0.30,
) -> float:
    """Single-asset Kelly-optimal weight.

    f* = mu / sigma^2, then apply fractional scaling and cap.

    Args:
        expected_excess_return: Expected excess return (annualized).
        variance: Return variance (annualized).
        fraction: Fractional Kelly (0.25 = quarter-Kelly).
        max_position: Maximum position size.
    """
    if variance < 1e-10:
        return 0.0
    full_kelly = expected_excess_return / variance
    return float(min(max(full_kelly * fraction, 0.0), max_position))


def cagr_from_returns(returns: pd.Series | np.ndarray) -> float:
    """Compute CAGR (Compound Annual Growth Rate) from daily returns.

    CAGR = (terminal_value / initial_value) ^ (1 / n_years) - 1

    Args:
        returns: Daily simple returns.

    Returns:
        Annualized CAGR as a float.
    """
    r = returns.values if isinstance(returns, pd.Series) else returns
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        return 0.0
    cum = np.prod(1 + r)
    if cum <= 0:
        return -1.0
    n_years = n / TRADING_DAYS
    return float(cum ** (1 / max(n_years, 1e-6)) - 1)


def kelly_optimal_leverage(
    mu_arithmetic: float,
    sigma2: float,
    rf: float = 0.0,
) -> float:
    """Kelly-optimal leverage for geometric growth rate maximization.

    The Kelly criterion for leverage: f* = (mu - rf) / sigma^2
    where mu is the expected ARITHMETIC annualized return (not CAGR/
    geometric return) and sigma^2 is the annualized variance.

    IMPORTANT: Do not pass CAGR (geometric return) as mu. CAGR already
    incorporates the volatility drag (Jensen's inequality), so using it
    would underestimate optimal leverage. Use annualized arithmetic mean
    return: daily_mean * 252.

    Args:
        mu_arithmetic: Expected annualized ARITHMETIC return.
        sigma2: Annualized return variance.
        rf: Risk-free rate (annualized).

    Returns:
        Kelly-optimal leverage ratio (e.g., 2.5 means 2.5x leverage).
    """
    if sigma2 < 1e-10:
        return 0.0
    return float((mu_arithmetic - rf) / sigma2)


def risk_of_ruin(
    sharpe: float,
    max_drawdown_tolerance: float,
    horizon_years: float,
) -> float:
    """Approximate probability of hitting max_drawdown_tolerance within horizon.

    Uses: P(ruin) ~ exp(-2 * SR^2 * |threshold|)
    where SR is annualized Sharpe and threshold is the drawdown tolerance.
    """
    if sharpe <= 0:
        return 1.0
    return float(np.exp(-2 * sharpe**2 * abs(max_drawdown_tolerance) * horizon_years))
