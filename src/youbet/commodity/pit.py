"""Commodity-specific publication lag metadata.

Extends the shared PUBLICATION_LAGS table with entries for
commodity macro signals (breakeven inflation, dollar index).

Registration pattern: call register_commodity_lags() at import time
in workflow code to add these entries to the shared PUBLICATION_LAGS.
"""

from __future__ import annotations


# Commodity-specific publication lags.
# These are real-time market signals with no publication delay.
COMMODITY_PUBLICATION_LAGS: dict[str, dict] = {
    "breakeven_inflation": {"lag_days": 0, "revision_risk": "none"},
    "dollar_index": {"lag_days": 0, "revision_risk": "none"},
}


def register_commodity_lags() -> None:
    """Register commodity publication lags into the shared PUBLICATION_LAGS.

    Must be called before creating PITFeatureSeries for commodity
    macro signals. Safe to call multiple times (idempotent).
    """
    from youbet.etf.pit import PUBLICATION_LAGS

    PUBLICATION_LAGS.update(COMMODITY_PUBLICATION_LAGS)
