"""Concrete transaction cost model for ETF strategies.

Pre-specified costs per ETF category. Numbers are locked before
backtesting to prevent cost-model shopping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yaml


# Pre-specified cost schedule (bps, one-way)
# These are committed before seeing any backtest results.
COST_SCHEDULE: dict[str, dict[str, float]] = {
    "broad_us_equity": {"bid_ask_bps": 1.0, "slippage_bps": 0.0},
    "broad_intl_equity": {"bid_ask_bps": 2.0, "slippage_bps": 1.0},
    "broad_us_bond": {"bid_ask_bps": 1.0, "slippage_bps": 0.0},
    "sector_thematic": {"bid_ask_bps": 3.0, "slippage_bps": 1.0},
    "factor": {"bid_ask_bps": 2.0, "slippage_bps": 1.0},
    "cash_equivalent": {"bid_ask_bps": 1.0, "slippage_bps": 0.0},
    "default": {"bid_ask_bps": 3.0, "slippage_bps": 1.0},
}


@dataclass
class CostModel:
    """Transaction cost model for ETF strategies.

    Expense ratios are applied as continuous daily drag.
    Trading costs (bid-ask + slippage) are applied per rebalance.
    """

    # Per-ticker expense ratio (annual, as decimal e.g. 0.0003 for 3bps)
    expense_ratios: dict[str, float] = field(default_factory=dict)
    # Per-ticker category for cost schedule lookup
    ticker_categories: dict[str, str] = field(default_factory=dict)

    def trade_cost_bps(self, ticker: str) -> float:
        """One-way trading cost in basis points for a ticker."""
        category = self.ticker_categories.get(ticker, "default")
        schedule = COST_SCHEDULE.get(category, COST_SCHEDULE["default"])
        return schedule["bid_ask_bps"] + schedule["slippage_bps"]

    def trade_cost_decimal(self, ticker: str) -> float:
        """One-way trading cost as a decimal fraction."""
        return self.trade_cost_bps(ticker) / 10_000

    def rebalance_cost(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        portfolio_value: float,
    ) -> float:
        """Total dollar cost of a rebalancing event.

        Computes turnover (sum of absolute weight changes / 2) and
        applies per-ticker trading costs.
        """
        all_tickers = old_weights.index.union(new_weights.index)
        old = old_weights.reindex(all_tickers, fill_value=0.0)
        new = new_weights.reindex(all_tickers, fill_value=0.0)

        total_cost = 0.0
        for ticker in all_tickers:
            delta = abs(new[ticker] - old[ticker])
            trade_value = delta * portfolio_value
            cost_rate = self.trade_cost_decimal(ticker)
            total_cost += trade_value * cost_rate

        return total_cost

    def daily_expense_drag(self, weights: pd.Series) -> float:
        """Daily expense ratio drag as a decimal, weighted by current allocation."""
        daily_drag = 0.0
        for ticker, weight in weights.items():
            annual_er = self.expense_ratios.get(ticker, 0.0008)  # default 8bps
            daily_drag += weight * (annual_er / 252)
        return daily_drag

    def turnover(
        self, old_weights: pd.Series, new_weights: pd.Series
    ) -> float:
        """One-way turnover: sum of absolute weight changes / 2."""
        all_tickers = old_weights.index.union(new_weights.index)
        old = old_weights.reindex(all_tickers, fill_value=0.0)
        new = new_weights.reindex(all_tickers, fill_value=0.0)
        return float(np.abs(new - old).sum() / 2)

    @classmethod
    def from_universe(cls, universe: pd.DataFrame) -> CostModel:
        """Build cost model from universe DataFrame.

        Expects columns: ticker, expense_ratio, category
        """
        expense_ratios = dict(
            zip(universe["ticker"], universe["expense_ratio"])
        )
        ticker_categories = dict(
            zip(universe["ticker"], universe["category"])
        )
        return cls(
            expense_ratios=expense_ratios,
            ticker_categories=ticker_categories,
        )
