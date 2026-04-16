"""Synthetic leveraged return construction.

Promotes the synthetic-LETF pattern from workflows/etf/experiments/leveraged_strategies.py
into a reusable module. The ETF backtester enforces `weights.sum() <= 1` (backtester.py:333),
so strategies cannot express leverage via `weight > 1`. Instead, strategies operate on
synthetic leveraged return series constructed here.

Core identities:
    LETF_daily_return = leverage * underlying_daily_return - daily_expense
    (Volatility decay emerges from daily compounding — no separate drag term.)

Financing cost model:
    For conditional leverage (e.g., 1.5x when signal is on, 0 when off), the incremental
    leverage (0.5x) is financed at T-bill + borrow spread. Financing is charged only
    during on-periods, not cash periods.
"""

from __future__ import annotations

import pandas as pd

TRADING_DAYS = 252


def synthetic_leveraged_returns(
    index_returns: pd.Series,
    leverage: float = 3.0,
    expense_ratio: float = 0.0095,
) -> pd.Series:
    """Synthesize daily leveraged ETF returns from an underlying return series.

    Args:
        index_returns: Daily return series of the underlying asset.
        leverage: Leverage multiplier (negative for inverse).
        expense_ratio: Annualized expense ratio in decimal (default 0.0095 = 95 bps,
            typical for 3x LETFs like UPRO/TQQQ).

    Returns:
        Daily return series of the synthetic leveraged ETF. Volatility decay is not
        applied as a separate term — it emerges from daily compounding.
    """
    daily_expense = expense_ratio / TRADING_DAYS
    return leverage * index_returns - daily_expense


def leveraged_long_cash(
    underlying_returns: pd.Series,
    signal: pd.Series,
    leverage: float,
    tbill_daily: pd.Series,
    expense_ratio: float = 0.0095,
) -> pd.Series:
    """Leveraged long when signal=1, cash (T-bill) when signal=0.

    Applies T+1 execution: signal from day T drives exposure on day T+1.

    Args:
        underlying_returns: Daily returns of the underlying (e.g., VTI).
        signal: Binary or [0,1] exposure signal (same index as underlying or broader).
        leverage: Leverage multiplier (positive).
        tbill_daily: Daily T-bill rate (decimal, not annualized).
        expense_ratio: Annualized expense ratio on the leveraged position.

    Returns:
        Daily portfolio returns.
    """
    lev = synthetic_leveraged_returns(underlying_returns, leverage, expense_ratio)
    sig = signal.shift(1).reindex(lev.index).fillna(0.0)
    tbill = tbill_daily.reindex(lev.index).fillna(0.0)
    return sig * lev + (1.0 - sig) * tbill


def conditional_leveraged_return(
    underlying_returns: pd.Series,
    exposure: pd.Series,
    tbill_daily: pd.Series,
    borrow_spread_bps: float = 50.0,
    expense_ratio: float = 0.0,
) -> pd.Series:
    """Apply time-varying leverage with explicit financing cost on the leverage increment.

    This is the model E2 / E3 use. Leverage above 1.0 is financed at
    T-bill + borrow_spread_bps annualized, charged only on the leverage increment
    (exposure - 1.0) and only during periods when exposure > 1.

    When exposure is in [0, 1], behaves like a partial long/cash mix: the fraction
    (1 - exposure) earns T-bill (no borrow cost).

    Args:
        underlying_returns: Daily returns of the underlying asset.
        exposure: Daily exposure signal (can exceed 1 for leverage, or be below 1 for
            partial cash allocation). T+1 shift applied inside.
        tbill_daily: Daily T-bill rate (decimal).
        borrow_spread_bps: Annual borrow spread over T-bill in basis points (default 50).
        expense_ratio: Annualized expense drag applied to the full position (e.g.,
            an ETF expense ratio). Default 0 for paper portfolios.

    Returns:
        Daily portfolio returns.
    """
    exp_shifted = exposure.shift(1).reindex(underlying_returns.index).fillna(0.0)
    rf = tbill_daily.reindex(underlying_returns.index).fillna(0.0)
    daily_expense = expense_ratio / TRADING_DAYS
    borrow_daily = (borrow_spread_bps / 10000.0) / TRADING_DAYS

    # Partial long/cash (exposure <= 1): exposure * underlying + (1 - exposure) * rf
    # Leveraged (exposure > 1): exposure * underlying - (exposure - 1) * (rf + borrow)
    # These are equivalent up to a sign flip on (1 - exposure) — unified below.
    leverage_increment = (exp_shifted - 1.0).clip(lower=0.0)
    cash_fraction = (1.0 - exp_shifted).clip(lower=0.0)

    port = (
        exp_shifted * underlying_returns
        + cash_fraction * rf                               # cash earns T-bill
        - leverage_increment * (rf + borrow_daily)          # borrow increment pays T-bill + spread
        - daily_expense
    )
    return port


def sma_signal(prices: pd.Series, window: int = 100) -> pd.Series:
    """Binary signal: 1 when price > SMA, 0 when below.

    Returns a series aligned to the price index. Applies no T+1 shift; callers
    should shift downstream (or use `leveraged_long_cash` which shifts internally).
    """
    sma = prices.rolling(window).mean()
    return (prices > sma).astype(float)


def multi_sma_vote(prices: pd.Series, windows: list[int]) -> pd.Series:
    """Proportional exposure = fraction of SMAs below the price.

    For E7: smoother transitions than binary SMA. Returns values in [0, 1].
    """
    votes = pd.DataFrame(
        {w: (prices > prices.rolling(w).mean()).astype(float) for w in windows}
    )
    return votes.mean(axis=1)
