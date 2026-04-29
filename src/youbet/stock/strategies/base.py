"""Base class and primitives for cross-sectional stock strategies.

Each strategy produces per-stock SCORES from a PIT panel; selection +
weighting are separate responsibilities with sensible defaults.

    class MyStrategy(CrossSectionalStrategy):
        def score(self, panel) -> pd.Series:
            return ...  # ticker -> score (higher = better)

By default:
  - Selection = `top_decile_select` (top 10% by score, floor of `min_holdings`).
  - Weighting = `equal_weight`.

Override `select_and_weight` to customize.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def top_decile_select(
    scores: pd.Series,
    breakpoint: float = 0.10,
    min_holdings: int = 20,
    max_holdings: int = 100,
) -> pd.Series:
    """Return the top-`breakpoint` fraction of scores, floored/capped."""
    s = scores.dropna().sort_values(ascending=False)
    if s.empty:
        return s
    n_target = max(min_holdings, int(len(s) * breakpoint))
    n_target = min(n_target, max_holdings, len(s))
    return s.iloc[:n_target]


def equal_weight(selected: pd.Series) -> pd.Series:
    """Equal-weight across the selected tickers."""
    if selected.empty:
        return selected
    w = pd.Series(1.0 / len(selected), index=selected.index)
    return w


def inverse_vol_weight(
    selected: pd.Series,
    prices: pd.DataFrame,
    lookback_days: int = 252,
) -> pd.Series:
    """Weight proportional to 1 / realized volatility over `lookback_days`."""
    if selected.empty:
        return selected
    rets = prices[selected.index].pct_change().dropna().tail(lookback_days)
    vols = rets.std()
    inv = 1.0 / vols.replace(0, np.nan)
    inv = inv.dropna()
    if inv.empty:
        return equal_weight(selected)
    w = inv / inv.sum()
    return w


def mcap_weight(selected: pd.Series, mcaps: pd.Series) -> pd.Series:
    """Weight proportional to market cap."""
    if selected.empty:
        return selected
    m = mcaps.reindex(selected.index).dropna()
    if m.empty or m.sum() <= 0:
        return equal_weight(selected)
    return m / m.sum()


class CrossSectionalStrategy(ABC):
    """Abstract base for stock-picking strategies.

    The backtester calls:
        fit(train_panel, train_start, train_end)   — once per test window
        generate_weights(panel)                     — every rebalance

    Sub-classes implement `score(panel)`. Override `select_and_weight`
    only if defaults are unsuitable.
    """

    # Defaults — subclasses may override
    decile_breakpoint: float = 0.10
    min_holdings: int = 20
    max_holdings: int = 100
    weighting: str = "equal"  # "equal" | "inv_vol" | "mcap"

    @abstractmethod
    def score(self, panel: dict) -> pd.Series:
        """Score each active ticker (higher score → stronger buy signal).

        Must only use panel["prices"] (already < rebal_date) and PIT-safe
        fundamentals via `compute_fundamentals(facts, decision_date)`.
        """

    def fit(
        self,
        train_panel: dict,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> None:
        """Optional per-fold fit (e.g., ML training). Default: no-op."""

    def select_and_weight(self, scores: pd.Series, panel: dict) -> pd.Series:
        """Default: top decile, equal-weight."""
        selected = top_decile_select(
            scores,
            breakpoint=self.decile_breakpoint,
            min_holdings=self.min_holdings,
            max_holdings=self.max_holdings,
        )
        if self.weighting == "inv_vol":
            return inverse_vol_weight(selected, panel["prices"])
        if self.weighting == "mcap":
            return mcap_weight(selected, panel["mcaps"])
        return equal_weight(selected)

    def generate_weights(self, panel: dict) -> pd.Series:
        scores = self.score(panel)
        active = panel["active_tickers"]
        scores = scores[scores.index.isin(active)]
        return self.select_and_weight(scores, panel)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def params(self) -> dict:
        return {
            "decile_breakpoint": self.decile_breakpoint,
            "min_holdings": self.min_holdings,
            "max_holdings": self.max_holdings,
            "weighting": self.weighting,
        }


class EqualWeightBenchmark(CrossSectionalStrategy):
    """Equal-weighted exposure to the current universe (diagnostic baseline).

    Useful for isolating pure selection alpha from weighting / universe
    effects. NOT the Phase 1 benchmark (SPY is).
    """

    decile_breakpoint = 1.0
    min_holdings = 1
    max_holdings = 10_000

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        return pd.Series(1.0, index=list(active))


class BuyAndHoldETF(CrossSectionalStrategy):
    """Single-ticker benchmark (e.g., SPY buy-and-hold).

    Used as the confirmatory benchmark per CLAUDE.md principle #3.
    """

    decile_breakpoint = 1.0
    min_holdings = 1
    max_holdings = 1

    def __init__(self, ticker: str = "SPY"):
        self.ticker = ticker

    def score(self, panel: dict) -> pd.Series:
        return pd.Series({self.ticker: 1.0})

    def select_and_weight(self, scores: pd.Series, panel: dict) -> pd.Series:
        return pd.Series({self.ticker: 1.0})

    def generate_weights(self, panel: dict) -> pd.Series:
        # Bypass the active_tickers filter: SPY is the benchmark regardless
        return pd.Series({self.ticker: 1.0})

    @property
    def name(self) -> str:
        return f"buy_hold_{self.ticker}"

    @property
    def params(self) -> dict:
        return {"ticker": self.ticker}
