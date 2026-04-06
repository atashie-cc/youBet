"""Tests for transaction cost model."""

import numpy as np
import pandas as pd
import pytest

from youbet.etf.costs import CostModel, COST_SCHEDULE


class TestCostModel:
    def setup_method(self):
        self.model = CostModel(
            expense_ratios={"VTI": 0.0003, "VGSH": 0.0004, "VNQ": 0.0012},
            ticker_categories={
                "VTI": "broad_us_equity",
                "VGSH": "cash_equivalent",
                "VNQ": "sector_thematic",
            },
        )

    def test_trade_cost_by_category(self):
        """Different categories have different costs."""
        vti_cost = self.model.trade_cost_bps("VTI")
        vnq_cost = self.model.trade_cost_bps("VNQ")
        assert vti_cost < vnq_cost  # Broad cheaper than sector

    def test_trade_cost_decimal(self):
        """Decimal conversion correct."""
        bps = self.model.trade_cost_bps("VTI")
        decimal = self.model.trade_cost_decimal("VTI")
        assert abs(decimal - bps / 10_000) < 1e-10

    def test_unknown_ticker_uses_default(self):
        """Unknown tickers get default cost schedule."""
        cost = self.model.trade_cost_bps("UNKNOWN")
        default = COST_SCHEDULE["default"]
        assert cost == default["bid_ask_bps"] + default["slippage_bps"]

    def test_rebalance_cost_no_change(self):
        """No weight changes → zero cost."""
        weights = pd.Series({"VTI": 0.6, "VGSH": 0.4})
        cost = self.model.rebalance_cost(weights, weights, 100_000)
        assert cost == 0.0

    def test_rebalance_cost_positive(self):
        """Weight changes → positive cost."""
        old = pd.Series({"VTI": 0.6, "VGSH": 0.4})
        new = pd.Series({"VTI": 0.8, "VGSH": 0.2})
        cost = self.model.rebalance_cost(old, new, 100_000)
        assert cost > 0

    def test_turnover_calculation(self):
        """One-way turnover = sum(|delta|) / 2."""
        old = pd.Series({"VTI": 0.6, "VGSH": 0.4})
        new = pd.Series({"VTI": 0.8, "VGSH": 0.2})
        turnover = self.model.turnover(old, new)
        # |0.2| + |-0.2| = 0.4, / 2 = 0.2
        assert abs(turnover - 0.2) < 1e-10

    def test_daily_expense_drag(self):
        """Expense drag is weight-averaged annual ER / 252."""
        weights = pd.Series({"VTI": 0.6, "VGSH": 0.4})
        drag = self.model.daily_expense_drag(weights)
        expected = (0.6 * 0.0003 + 0.4 * 0.0004) / 252
        assert abs(drag - expected) < 1e-12

    def test_from_universe(self):
        """Build cost model from universe DataFrame."""
        universe = pd.DataFrame({
            "ticker": ["VTI", "VNQ"],
            "expense_ratio": [0.0003, 0.0012],
            "category": ["broad_us_equity", "sector_thematic"],
        })
        model = CostModel.from_universe(universe)
        assert model.expense_ratios["VTI"] == 0.0003
        assert model.ticker_categories["VNQ"] == "sector_thematic"
