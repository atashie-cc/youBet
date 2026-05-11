"""Phase -1 — Cost-model crisis sensitivity verification.

Round 1 review M7: ETF spreads spike in crises (March 2020 saw VEA bid-ask
hit 10-15 bps). The locked default of 3 bps one-way for broad_intl_equity
under-states realistic costs for any regime-conditional strategy that
trades during stress. The plan v1.1 commits to running every PROCEED
candidate at cost in {3, 5, 10} bps and requiring pass at 10 bps.

This Phase -1 script does NOT run any strategy; it verifies the override
mechanism works and documents the procedure. Phase 0/1/2 backtests will
re-call their evaluation code three times with the override applied.

Output:
  - artifacts/phase_minus_1_cost_override_verified.json (per-category bps demo)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from youbet.etf.costs import COST_SCHEDULE, CostModel

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = WORKFLOW_DIR / "artifacts"
UNIVERSE_PATH = WORKFLOW_DIR / "data" / "reference" / "international_universe.csv"


def build_cost_model_at_bps(universe: pd.DataFrame, intl_bps_one_way: float) -> CostModel:
    """Construct a CostModel with broad_intl_equity overridden to a target bps.

    Implementation: subclass CostModel and override trade_cost_bps for that
    single category. The default schedule for other categories is unchanged.
    """

    class OverrideCostModel(CostModel):
        def trade_cost_bps(self, ticker: str) -> float:
            category = self.ticker_categories.get(ticker, "default")
            if category == "broad_intl_equity":
                return float(intl_bps_one_way)
            schedule = COST_SCHEDULE.get(category, COST_SCHEDULE["default"])
            return schedule["bid_ask_bps"] + schedule["slippage_bps"]

    return OverrideCostModel.from_universe(universe)


def verify_override(universe: pd.DataFrame) -> dict:
    """Confirm broad_intl_equity bps move as expected; non-intl unchanged."""
    results = {}
    for bps in [3.0, 5.0, 10.0]:
        cm = build_cost_model_at_bps(universe, bps)
        # Spot-check a handful of tickers
        sample = {
            "VEA": cm.trade_cost_bps("VEA"),    # broad_intl_equity, expect bps
            "VWO": cm.trade_cost_bps("VWO"),    # broad_intl_equity
            "VTI": cm.trade_cost_bps("VTI"),    # broad_us_equity, expect 1.0
            "VGSH": cm.trade_cost_bps("VGSH"),  # cash_equivalent, expect 1.0
        }
        results[f"intl_{int(bps)}bps"] = sample

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(f"Universe not found: {UNIVERSE_PATH}")

    universe = pd.read_csv(UNIVERSE_PATH)
    logger.info("Loaded universe: %d tickers", len(universe))

    results = verify_override(universe)
    logger.info("Override verification: %s", results)

    # Sanity assertions
    assert results["intl_3bps"]["VEA"] == 3.0, f"intl 3bps override broken: {results['intl_3bps']}"
    assert results["intl_10bps"]["VEA"] == 10.0, f"intl 10bps override broken: {results['intl_10bps']}"
    assert results["intl_3bps"]["VTI"] == 1.0, f"VTI cost should be unchanged at 1bps"
    assert results["intl_3bps"]["VGSH"] == 1.0, f"VGSH cost should be unchanged at 1bps"
    logger.info("All assertions passed.")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "phase_minus_1_cost_override_verified.json"
    with out_path.open("w") as f:
        json.dump(
            {
                "verified_2026-05-01": True,
                "results": results,
                "procedure": (
                    "Phase 0/1/2 backtests must wrap their CostModel construction "
                    "with build_cost_model_at_bps(universe, intl_bps_one_way=BPS) "
                    "for BPS in {3, 5, 10}. Strict gate (PROCEED) requires pass at 10 bps."
                ),
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
