"""Firm-characteristic feature computations for ML ranking strategies.

Pure functions from panel data → per-(ticker, date) feature DataFrame.
"""

from youbet.stock.features.gkx_chars import (
    GKX_MVP_FEATURE_NAMES,
    compute_chars_at_date,
)

__all__ = ["GKX_MVP_FEATURE_NAMES", "compute_chars_at_date"]
