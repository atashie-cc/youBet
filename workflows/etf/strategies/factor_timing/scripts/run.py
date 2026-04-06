"""Factor timing: rotate between growth/value and large/small based on macro regime.

Recession signals → value + large cap + dividend (defensive)
Expansion signals → growth + small cap + tech (risk-on)
Neutral → equal weight between pairs

Literature: value outperforms in recoveries, growth in late-cycle expansions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.etf.pit import PITFeatureSeries
from youbet.etf.strategy import BaseStrategy
from youbet.etf.transforms import Normalizer
from youbet.utils.io import load_config

logger = logging.getLogger(__name__)

# Minimum trading days required to consider a ticker tradable
MIN_HISTORY_DAYS = 63


def _is_tradable(
    ticker: str, prices: pd.DataFrame, as_of_date: pd.Timestamp
) -> bool:
    """Check ticker has enough price history to be safely traded."""
    if ticker not in prices.columns:
        return False
    series = prices[ticker].loc[prices.index < as_of_date].dropna()
    return len(series) >= MIN_HISTORY_DAYS


class FactorTiming(BaseStrategy):
    """Macro-regime-driven factor pair rotation."""

    def __init__(
        self,
        pairs: dict | None = None,
        bond: str = "BND",
        intl: str = "VXUS",
        pair_alloc: float = 0.60,
        bond_alloc: float = 0.20,
        intl_alloc: float = 0.20,
    ):
        self.pairs = pairs or {
            "style": {"growth": "VUG", "value": "VTV"},
            "size": {"small": "VB", "large": "VV"},
            "quality": {"dividend": "VIG", "tech": "VGT"},
        }
        self.bond = bond
        self.intl = intl
        self.pair_alloc = pair_alloc
        self.bond_alloc = bond_alloc
        self.intl_alloc = intl_alloc
        self._macro_features: dict[str, PITFeatureSeries] = {}
        self._normalizers: dict[str, Normalizer] = {}

    def set_features(self, features: dict[str, PITFeatureSeries]) -> None:
        self._macro_features = features

    def fit(self, prices: pd.DataFrame, as_of_date: pd.Timestamp) -> None:
        self._normalizers = {}
        for feat_name, feat_series in self._macro_features.items():
            safe = feat_series.as_of(as_of_date)
            if len(safe) < 12:
                continue
            norm = Normalizer(method="zscore")
            norm.fit(safe.to_frame(name=feat_name))
            self._normalizers[feat_name] = norm

    def generate_weights(
        self, prices: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.Series:
        z = self._get_z_scores(as_of_date)

        # Log missing macro features (Codex fix 5)
        required = {"yield_curve", "credit_spread", "vix"}
        missing = required - set(z.keys())
        if missing:
            logger.warning(
                "factor_timing: missing macro z-scores %s at %s — "
                "defaulting to neutral regime for missing signals",
                missing, as_of_date.date(),
            )

        # Determine regime: recession vs expansion
        # Use None sentinel for missing signals instead of defaulting to 0
        recession_score = 0
        n_signals = 0
        yc = z.get("yield_curve")
        if yc is not None:
            n_signals += 1
            if yc < -0.5:
                recession_score += 1
        cs = z.get("credit_spread")
        if cs is not None:
            n_signals += 1
            if cs > 0.5:
                recession_score += 1
        vx = z.get("vix")
        if vx is not None:
            n_signals += 1
            if vx > 0.5:
                recession_score += 1

        # Need at least 2 signals available and 2 recession indicators to call recession
        is_recession = n_signals >= 2 and recession_score >= 2

        # Pair allocation per pair (equal split among 3 pairs)
        per_pair = self.pair_alloc / len(self.pairs)
        weights = {}

        for pair_name, pair_tickers in self.pairs.items():
            defensive = pair_tickers.get("value") or pair_tickers.get("large") or pair_tickers.get("dividend")
            offensive = pair_tickers.get("growth") or pair_tickers.get("small") or pair_tickers.get("tech")

            # Check tradability, not just column existence (Codex fix 3)
            def_tradable = defensive and _is_tradable(defensive, prices, as_of_date)
            off_tradable = offensive and _is_tradable(offensive, prices, as_of_date)

            if is_recession:
                # 70/30 tilt toward defensive
                if def_tradable:
                    weights[defensive] = weights.get(defensive, 0) + per_pair * 0.70
                if off_tradable:
                    weights[offensive] = weights.get(offensive, 0) + per_pair * 0.30
            else:
                # 70/30 tilt toward offensive
                if off_tradable:
                    weights[offensive] = weights.get(offensive, 0) + per_pair * 0.70
                if def_tradable:
                    weights[defensive] = weights.get(defensive, 0) + per_pair * 0.30

        # Add diversifiers — check tradability, use explicit fallbacks (Codex fix 4)
        bond_ticker = self._find_tradable(
            [self.bond, "BSV", "BIV"], prices, as_of_date
        )
        if bond_ticker:
            weights[bond_ticker] = self.bond_alloc

        intl_ticker = self._find_tradable(
            [self.intl, "VEA", "VWO"], prices, as_of_date
        )
        if intl_ticker:
            weights[intl_ticker] = self.intl_alloc

        # If no tradable assets found, hold VTI as fallback
        if not weights:
            if _is_tradable("VTI", prices, as_of_date):
                return pd.Series({"VTI": 1.0})
            return pd.Series(dtype=float)

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return pd.Series(weights)

    def _find_tradable(
        self,
        candidates: list[str],
        prices: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> str | None:
        """Return the first tradable ticker from a fallback list."""
        for ticker in candidates:
            if _is_tradable(ticker, prices, as_of_date):
                return ticker
        return None

    def _get_z_scores(self, as_of_date: pd.Timestamp) -> dict[str, float]:
        z_scores = {}
        for feat_name, feat_series in self._macro_features.items():
            if feat_name not in self._normalizers:
                continue
            safe = feat_series.as_of(as_of_date)
            if len(safe) == 0:
                continue
            df = safe.iloc[-1:].to_frame(name=feat_name)
            z = self._normalizers[feat_name].transform(df)
            val = float(z.iloc[0, 0])
            if np.isfinite(val):
                z_scores[feat_name] = val
        return z_scores

    @property
    def name(self) -> str:
        return "factor_timing"

    @property
    def params(self) -> dict:
        return {"n_pairs": len(self.pairs), "pair_alloc": self.pair_alloc}

    @classmethod
    def from_config(cls, config: dict) -> FactorTiming:
        sig = config.get("signal", {})
        return cls(
            pairs=sig.get("pairs"),
            bond=sig.get("diversifiers", {}).get("bond", "BND"),
            intl=sig.get("diversifiers", {}).get("intl", "VXUS"),
            pair_alloc=sig.get("pair_allocation", 0.60),
            bond_alloc=sig.get("bond_allocation", 0.20),
            intl_alloc=sig.get("intl_allocation", 0.20),
        )
