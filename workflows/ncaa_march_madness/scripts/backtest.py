"""Backtest model against historical tournaments (2003-2025)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    # TODO: Implement backtesting
    # 1. For each historical tournament (2003-2025):
    #    a. Train model on all data before that season
    #    b. Predict tournament game outcomes
    #    c. Evaluate: log loss, accuracy, Brier score
    # 2. Report aggregate metrics across all tournaments
    # 3. Report per-round metrics (R64, R32, S16, E8, F4, Championship)
    # 4. Generate calibration curve across all predictions
    # 5. Save results to output/reports/

    logger.info("Backtesting not yet implemented")


if __name__ == "__main__":
    main()
