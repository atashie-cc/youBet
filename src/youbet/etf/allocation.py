"""Portfolio allocation utilities.

Used as building blocks by strategies. Strategies compose these
internally rather than the backtester wiring them together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --- Asset class hierarchy ---------------------------------------------------

ASSET_CLASS_MAP: dict[str, list[str]] = {
    "us_equity": [
        "VTI", "VOO", "VV", "VB", "VO", "VXF",
        "VUG", "VTV", "MGK", "MGV", "VBK", "VBR", "VOT", "VOE",
        "VIG", "VYM",
    ],
    "intl_equity": ["VXUS", "VEA", "VWO", "VGK", "VPL", "VSS", "VIGI", "VYMI"],
    "us_bond": [
        "BND", "BNDX", "VGIT", "VGLT", "BSV", "BIV", "BLV",
        "VCSH", "VCIT", "VCLT", "VMBS", "EDV",
    ],
    "sector": ["VGT", "VHT", "VDC", "VCR", "VFH", "VIS", "VAW", "VDE", "VPU", "VOX"],
    "real_assets": ["VNQ", "VNQI", "VTIP"],
    "cash": ["VGSH"],
}

ASSET_CLASS_REPS: dict[str, str] = {
    "us_equity": "VTI",
    "intl_equity": "VXUS",
    "us_bond": "BND",
    "real_assets": "VNQ",
    "cash": "VGSH",
}

# Fallback representatives for early periods when primary may not exist
ASSET_CLASS_FALLBACKS: dict[str, list[str]] = {
    "us_equity": ["VTI", "VOO", "VV"],
    "intl_equity": ["VXUS", "VEA", "VWO"],
    "us_bond": ["BND", "BSV", "BIV"],
    "real_assets": ["VNQ"],
    "cash": ["VGSH"],
}

TICKER_TO_CLASS: dict[str, str] = {}
for _cls, _tickers in ASSET_CLASS_MAP.items():
    for _t in _tickers:
        TICKER_TO_CLASS[_t] = _cls


def get_class_representative(
    asset_class: str,
    available_tickers: list[str],
) -> str | None:
    """Return the best available representative for an asset class."""
    for ticker in ASSET_CLASS_FALLBACKS.get(asset_class, []):
        if ticker in available_tickers:
            return ticker
    return None


def available_by_class(
    available_tickers: list[str],
) -> dict[str, list[str]]:
    """Group available tickers by asset class."""
    result: dict[str, list[str]] = {}
    for cls, members in ASSET_CLASS_MAP.items():
        avail = [t for t in members if t in available_tickers]
        if avail:
            result[cls] = avail
    return result


def enforce_class_concentration(
    weights: pd.Series,
    max_per_class: float = 0.30,
) -> pd.Series:
    """Cap total weight in any single asset class, redistribute excess."""
    class_weights: dict[str, float] = {}
    for ticker, w in weights.items():
        cls = TICKER_TO_CLASS.get(ticker, "other")
        class_weights[cls] = class_weights.get(cls, 0.0) + w

    # Find over-allocated classes
    excess = 0.0
    n_under = 0
    for cls, cw in class_weights.items():
        if cw > max_per_class:
            excess += cw - max_per_class
        else:
            n_under += 1

    if excess == 0.0:
        return weights

    # Scale down over-allocated classes, redistribute to under-allocated
    adjusted = weights.copy()
    for ticker in adjusted.index:
        cls = TICKER_TO_CLASS.get(ticker, "other")
        cw = class_weights[cls]
        if cw > max_per_class:
            adjusted[ticker] *= max_per_class / cw

    # Renormalize to sum to original total
    total = weights.sum()
    if adjusted.sum() > 0:
        adjusted = adjusted * total / adjusted.sum()
    return adjusted


def equal_weight(tickers: list[str]) -> pd.Series:
    """1/N allocation. Surprisingly hard to beat in practice."""
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=tickers)


def inverse_volatility(
    returns: pd.DataFrame,
    lookback: int | None = None,
) -> pd.Series:
    """Weight inversely proportional to historical volatility.

    Args:
        returns: Daily returns DataFrame (columns = tickers).
        lookback: Number of trailing days. None = use all data.

    Returns:
        Normalized weights summing to 1.0.
    """
    if lookback is not None:
        returns = returns.iloc[-lookback:]

    vol = returns.std()
    vol = vol.replace(0, np.nan).dropna()
    if len(vol) == 0:
        return pd.Series(dtype=float)

    inv_vol = 1.0 / vol
    return inv_vol / inv_vol.sum()


def momentum_rank(
    prices: pd.DataFrame,
    lookback_months: int = 6,
    top_k: int = 5,
) -> list[str]:
    """Rank tickers by trailing total return, return top K.

    Args:
        prices: Adjusted close prices.
        lookback_months: Trailing period in months (~21 trading days each).
        top_k: Number of top performers to select.

    Returns:
        List of top_k tickers sorted by trailing return (descending).
    """
    lookback_days = lookback_months * 21

    if len(prices) < lookback_days + 1:
        # Not enough data — return all available
        return prices.columns.tolist()[:top_k]

    recent = prices.iloc[-lookback_days:]
    total_returns = recent.iloc[-1] / recent.iloc[0] - 1
    total_returns = total_returns.dropna().sort_values(ascending=False)

    return total_returns.index.tolist()[:top_k]


def absolute_momentum_filter(
    prices: pd.DataFrame,
    tickers: list[str],
    lookback_months: int = 6,
) -> list[str]:
    """Filter tickers: only keep those with positive trailing return.

    If a ticker's trailing return is negative, it's excluded (go to cash).
    """
    lookback_days = lookback_months * 21

    if len(prices) < lookback_days + 1:
        return tickers

    recent = prices.iloc[-lookback_days:]
    passed = []
    for t in tickers:
        if t in recent.columns:
            ret = recent[t].iloc[-1] / recent[t].iloc[0] - 1
            if ret > 0:
                passed.append(t)

    return passed


def trailing_volatility(
    returns: pd.DataFrame,
    lookback_days: int = 63,
) -> pd.Series:
    """Annualized trailing volatility per ticker.

    Args:
        returns: Daily returns.
        lookback_days: Trailing window (default ~3 months).

    Returns:
        Annualized volatility per ticker.
    """
    recent = returns.iloc[-lookback_days:] if len(returns) > lookback_days else returns
    return recent.std() * np.sqrt(252)
