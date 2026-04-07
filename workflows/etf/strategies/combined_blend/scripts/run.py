"""Combined blend strategies: average weights from multiple sub-strategies.

No fitting — just averages the weight vectors from pre-existing strategies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(WORKFLOW_ROOT))

from youbet.etf.strategy import BaseStrategy


class CombinedBlend(BaseStrategy):
    """Blend multiple sub-strategies by averaging their weight vectors."""

    def __init__(
        self,
        members: list[BaseStrategy],
        blend_weights: list[float] | None = None,
        variant_name: str = "combined",
    ):
        if not members:
            raise ValueError("CombinedBlend requires at least one member strategy")
        weights = blend_weights or [1.0 / len(members)] * len(members)
        if len(weights) != len(members):
            raise ValueError(
                f"blend_weights length ({len(weights)}) must match "
                f"members length ({len(members)})"
            )
        if any(w < 0 for w in weights):
            raise ValueError("blend_weights must be non-negative")
        self._members = members
        self._blend_weights = weights
        self._variant_name = variant_name

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        for m in self._members:
            m.fit(prices, as_of_date)

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        all_weights = []
        for m, bw in zip(self._members, self._blend_weights):
            w = m.generate_weights(prices, as_of_date)
            all_weights.append(w * bw)

        # Combine: sum weighted contributions across all tickers
        combined = pd.concat(all_weights, axis=1).sum(axis=1)
        return combined

    @property
    def name(self) -> str:
        return f"blend_{self._variant_name}"

    @property
    def params(self) -> dict:
        return {
            "variant": self._variant_name,
            "members": [m.name for m in self._members],
            "blend_weights": self._blend_weights,
        }
