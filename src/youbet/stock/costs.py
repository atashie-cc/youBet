"""Per-stock transaction cost model with market-cap bucketing.

Unlike ETFs (flat bps per category), individual stocks have costs that
scale with liquidity. Large-caps trade at ~5bps one-way; micro-caps can
exceed 75bps plus wider spreads. This model buckets by market cap at
rebalance time and adds a per-share commission floor.

Cost components:
  - Bid-ask (one-way, bps of trade value)
  - Slippage (one-way, bps of trade value) — scales with position size /
    average daily volume; bucketed proxy here.
  - Commission floor ($0.005/share is Interactive Brokers' pro rate).

All schedules are loaded from `config.yaml` at construction time.
Numbers are LOCKED before any backtest (per CLAUDE.md principle #11).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default cost schedule — matches config.yaml. Override via from_config().
DEFAULT_MCAP_BUCKETS: dict[str, dict[str, float]] = {
    "mega":  {"min_usd_b": 200.0, "bid_ask_bps": 1.0,  "slippage_bps": 1.0},
    "large": {"min_usd_b":  10.0, "bid_ask_bps": 2.0,  "slippage_bps": 3.0},
    "mid":   {"min_usd_b":   2.0, "bid_ask_bps": 4.0,  "slippage_bps": 6.0},
    "small": {"min_usd_b":   0.3, "bid_ask_bps": 10.0, "slippage_bps": 15.0},
    "micro": {"min_usd_b":   0.0, "bid_ask_bps": 25.0, "slippage_bps": 50.0},
}

DEFAULT_COMMISSION_PER_SHARE = 0.005


def bucket_for_mcap(mcap_usd: float | None, buckets: dict[str, dict[str, float]]) -> str:
    """Pick the mcap bucket for a given market cap in USD.

    Missing/None mcap → "micro" (conservative: assume highest costs).
    """
    if mcap_usd is None or not np.isfinite(mcap_usd) or mcap_usd <= 0:
        return "micro"
    mcap_billions = mcap_usd / 1e9
    # Walk from largest to smallest: first bucket whose min is <= mcap wins
    order = sorted(buckets.items(), key=lambda kv: -kv[1]["min_usd_b"])
    for name, spec in order:
        if mcap_billions >= spec["min_usd_b"]:
            return name
    return "micro"


@dataclass
class StockCostModel:
    """Per-stock transaction cost model with mcap bucketing.

    Attributes:
        buckets: {bucket_name: {min_usd_b, bid_ask_bps, slippage_bps}}.
        commission_per_share: USD per share traded (floor).
        _latest_mcap: Cache of most recent market caps by ticker (set by
            the backtester on each rebalance).
    """

    buckets: dict[str, dict[str, float]] = field(
        default_factory=lambda: {**DEFAULT_MCAP_BUCKETS}
    )
    commission_per_share: float = DEFAULT_COMMISSION_PER_SHARE
    _latest_mcap: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> StockCostModel:
        """Build from workflow config.yaml `costs` section."""
        section = config.get("costs", {})
        buckets = section.get("mcap_buckets", DEFAULT_MCAP_BUCKETS)
        comm = float(
            section.get("commission_per_share_usd", DEFAULT_COMMISSION_PER_SHARE)
        )
        return cls(
            buckets=dict(buckets),
            commission_per_share=comm,
        )

    def update_mcaps(self, mcaps: pd.Series) -> None:
        """Store the rebalance-date market caps used by subsequent calls."""
        self._latest_mcap = mcaps.dropna().to_dict()

    def trade_cost_bps(self, ticker: str, mcap: float | None = None) -> float:
        """One-way trading cost in bps for a ticker (bid-ask + slippage)."""
        m = mcap if mcap is not None else self._latest_mcap.get(ticker)
        bucket = bucket_for_mcap(m, self.buckets)
        spec = self.buckets[bucket]
        return float(spec["bid_ask_bps"] + spec["slippage_bps"])

    def trade_cost_decimal(self, ticker: str, mcap: float | None = None) -> float:
        return self.trade_cost_bps(ticker, mcap) / 10_000.0

    def rebalance_cost(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        portfolio_value: float,
        prices: pd.Series | None = None,
    ) -> float:
        """Total dollar cost of a rebalance.

        cost_per_ticker = |Δweight| × portfolio_value × trade_cost_decimal
                        + commission_per_share × shares_traded

        Args:
            old_weights: Current target weights (pre-rebalance).
            new_weights: New target weights.
            portfolio_value: Portfolio NAV at rebalance.
            prices: Optional per-ticker prices (USD). Required to compute
                the per-share commission floor. Without prices, commission
                is skipped (conservative under-estimate).
        """
        all_tickers = old_weights.index.union(new_weights.index)
        old = old_weights.reindex(all_tickers, fill_value=0.0)
        new = new_weights.reindex(all_tickers, fill_value=0.0)
        dw = (new - old).abs()

        total_cost = 0.0
        for ticker in all_tickers:
            delta = float(dw[ticker])
            if delta == 0.0:
                continue
            trade_value = delta * portfolio_value
            bps_cost = trade_value * self.trade_cost_decimal(ticker)
            total_cost += bps_cost

            if prices is not None and ticker in prices.index:
                p = float(prices[ticker])
                if np.isfinite(p) and p > 0:
                    shares = trade_value / p
                    total_cost += shares * self.commission_per_share

        return total_cost

    def turnover(self, old_weights: pd.Series, new_weights: pd.Series) -> float:
        """One-way turnover: sum |Δweight| / 2."""
        all_tickers = old_weights.index.union(new_weights.index)
        old = old_weights.reindex(all_tickers, fill_value=0.0)
        new = new_weights.reindex(all_tickers, fill_value=0.0)
        return float(np.abs(new - old).sum() / 2)
