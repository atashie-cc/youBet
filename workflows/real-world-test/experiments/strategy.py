"""v3 portfolio strategy — three sleeves, two SMA100 weekly overlays.

Returns daily portfolio returns given pre-aligned daily price series.
The implementation:
  - Computes Friday-close SMA100 signals on VTI and SPY (separately).
  - Holds VTI when its signal is on, BIL otherwise (60% sleeve).
  - Holds UPRO (real or synthetic 3x SPY) when SPY signal is on, BIL otherwise (30% sleeve).
  - Holds IAU statically (10% sleeve).
  - T+1 execution: signal at Friday close drives Monday open exposure.
  - Annual rebalance OR drift > 7pp triggers reset to target weights.
  - 10 bps one-way switching cost per overlay flip.

This is a single-pass backtest, not a walk-forward harness — the strategy has
no fitted parameters, so walk-forward adds nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class V3Config:
    w_vti: float = 0.60
    w_upro: float = 0.30
    w_iau: float = 0.10
    sma_window: int = 100
    cadence: str = "W-FRI"          # weekly, Friday-anchored
    drift_threshold: float = 0.07
    switching_cost_bps: float = 10.0
    rebalance_freq: str = "A"        # annual

    def __post_init__(self) -> None:
        if abs(self.w_vti + self.w_upro + self.w_iau - 1.0) > 1e-9:
            raise ValueError("weights must sum to 1.0")


def _weekly_sma_signal(prices: pd.Series, window: int) -> pd.Series:
    """Friday-close SMA100 signal, ffilled to daily index.

    Returns a daily 0/1 series aligned to prices.index. Signal flips happen on
    the trading day after the Friday-close decision (T+1).
    """
    daily_sma = prices.rolling(window).mean()
    sig_daily = (prices > daily_sma).astype(float)

    # Subsample to Fridays (or last trading day of the week if Friday is a holiday)
    weekly = sig_daily.resample("W-FRI").last()
    # Reindex back to daily, ffill so the signal stays constant between Fridays
    sig = weekly.reindex(prices.index, method="ffill").fillna(0.0)
    # T+1: shift by one trading day so Friday's decision applies starting Monday
    return sig.shift(1).fillna(0.0)


def _switching_cost(sig: pd.Series, cost_bps: float) -> pd.Series:
    """Cost charged on the day of the flip (already T+1-shifted)."""
    flips = sig.diff().abs() > 0.5
    cost = flips.astype(float) * (cost_bps / 10000.0)
    return cost


def run_v3(
    vti_ret: pd.Series,
    spy_prices: pd.Series,
    vti_prices: pd.Series,
    upro_ret: pd.Series,
    iau_ret: pd.Series,
    bil_ret: pd.Series,
    config: V3Config | None = None,
) -> pd.DataFrame:
    """Run the v3 strategy on aligned daily series.

    All series must share the same daily DatetimeIndex (or be reindexable to it).
    Inputs are daily simple returns except for prices used for SMA (vti_prices, spy_prices).

    Returns a DataFrame with columns:
        port_return            — daily portfolio return after costs
        vti_sleeve_return      — daily return of the 60% VTI sleeve (post-overlay, post-cost)
        upro_sleeve_return     — daily return of the 30% UPRO sleeve (post-overlay, post-cost)
        iau_sleeve_return      — daily return of the IAU sleeve
        vti_signal             — daily 0/1 signal for VTI overlay
        spy_signal             — daily 0/1 signal for SPY overlay (drives UPRO)
        port_value             — cumulative portfolio value normalized to 1.0
        vti_weight, upro_weight, iau_weight — running weights (drift-tracked)
    """
    if config is None:
        config = V3Config()

    idx = vti_ret.index
    vti_ret = vti_ret.reindex(idx).fillna(0.0)
    upro_ret = upro_ret.reindex(idx).fillna(0.0)
    iau_ret = iau_ret.reindex(idx).fillna(0.0)
    bil_ret = bil_ret.reindex(idx).fillna(0.0)
    vti_prices = vti_prices.reindex(idx).ffill()
    spy_prices = spy_prices.reindex(idx).ffill()

    vti_sig = _weekly_sma_signal(vti_prices, config.sma_window)
    spy_sig = _weekly_sma_signal(spy_prices, config.sma_window)

    vti_cost = _switching_cost(vti_sig, config.switching_cost_bps)
    spy_cost = _switching_cost(spy_sig, config.switching_cost_bps)

    # Sleeve daily returns (after overlay and switching cost)
    vti_sleeve = vti_sig * vti_ret + (1.0 - vti_sig) * bil_ret - vti_cost
    upro_sleeve = spy_sig * upro_ret + (1.0 - spy_sig) * bil_ret - spy_cost
    iau_sleeve = iau_ret  # static, no overlay, no switch cost

    # Run drift-tracked portfolio with annual + drift-threshold rebalance
    n = len(idx)
    w = np.array([config.w_vti, config.w_upro, config.w_iau])
    weights_history = np.zeros((n, 3))
    port_returns = np.zeros(n)

    sleeve_rets = np.column_stack(
        [vti_sleeve.values, upro_sleeve.values, iau_sleeve.values]
    )
    targets = np.array([config.w_vti, config.w_upro, config.w_iau])

    last_year = idx[0].year
    for i in range(n):
        weights_history[i] = w
        # Today's portfolio return is dot product of last weights and today's sleeve returns
        port_ret = float(np.dot(w, sleeve_rets[i]))
        port_returns[i] = port_ret

        # Update weights from drift after today's returns
        new_vals = w * (1.0 + sleeve_rets[i])
        total = new_vals.sum()
        if total <= 0:
            w = targets.copy()
            continue
        w = new_vals / total

        # Year-end annual rebalance trigger
        next_idx = i + 1
        if next_idx < n and idx[next_idx].year != last_year:
            w = targets.copy()
            last_year = idx[next_idx].year
            continue

        # Drift-threshold rebalance trigger
        if (np.abs(w - targets) > config.drift_threshold).any():
            w = targets.copy()

    port_ret_s = pd.Series(port_returns, index=idx)
    port_value = (1.0 + port_ret_s).cumprod()

    df = pd.DataFrame(
        {
            "port_return": port_ret_s,
            "vti_sleeve_return": vti_sleeve,
            "upro_sleeve_return": upro_sleeve,
            "iau_sleeve_return": iau_sleeve,
            "vti_signal": vti_sig,
            "spy_signal": spy_sig,
            "vti_weight": weights_history[:, 0],
            "upro_weight": weights_history[:, 1],
            "iau_weight": weights_history[:, 2],
            "port_value": port_value,
        }
    )
    return df


def metrics(returns: pd.Series, name: str = "v3", trading_days: int = 252) -> dict:
    r = returns.dropna()
    if len(r) == 0:
        return {"name": name}
    n_years = len(r) / trading_days
    cum = (1.0 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1.0 / max(n_years, 1e-9)) - 1.0)
    vol = float(r.std() * np.sqrt(trading_days))
    rf_daily = 0.04 / trading_days
    excess = r - rf_daily
    sharpe = float(excess.mean() / max(excess.std(), 1e-12) * np.sqrt(trading_days))
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    max_dd = float(dd.min())
    underwater_days = int((dd < 0).sum())
    return {
        "name": name,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": cagr / abs(max_dd) if abs(max_dd) > 1e-9 else np.nan,
        "underwater_pct_days": underwater_days / max(len(r), 1),
        "terminal_wealth": float(cum.iloc[-1]),
        "n_years": n_years,
    }
