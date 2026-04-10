"""Commodity-specific transaction cost schedule.

Pre-specified costs per commodity ETF category. Numbers are locked
before backtesting to prevent cost-model shopping.

Registration pattern: call register_commodity_costs() at import time
in workflow code to add these entries to the shared COST_SCHEDULE.
"""

from __future__ import annotations


# Pre-specified commodity cost schedule (bps, one-way).
# Committed before seeing any backtest results.
COMMODITY_COST_SCHEDULE: dict[str, dict[str, float]] = {
    "commodity_physical_metal": {"bid_ask_bps": 3.0, "slippage_bps": 0.0},
    "commodity_broad_futures": {"bid_ask_bps": 5.0, "slippage_bps": 2.0},
    "commodity_energy_futures": {"bid_ask_bps": 5.0, "slippage_bps": 2.0},
    "commodity_agriculture": {"bid_ask_bps": 15.0, "slippage_bps": 5.0},
    "commodity_industrial_metals": {"bid_ask_bps": 10.0, "slippage_bps": 3.0},
    "commodity_miner_equity": {"bid_ask_bps": 3.0, "slippage_bps": 1.0},
    "commodity_energy_equity": {"bid_ask_bps": 3.0, "slippage_bps": 1.0},
}


def register_commodity_costs() -> None:
    """Register commodity cost categories into the shared COST_SCHEDULE.

    Must be called before building a CostModel for commodity instruments.
    Safe to call multiple times (idempotent).
    """
    from youbet.etf.costs import COST_SCHEDULE

    COST_SCHEDULE.update(COMMODITY_COST_SCHEDULE)
