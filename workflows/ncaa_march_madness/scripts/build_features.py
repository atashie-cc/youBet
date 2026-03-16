"""Build feature matrix with 18 stat differentials for NCAA March Madness."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    processed_dir = WORKFLOW_DIR / config["data"]["processed_dir"]
    ensure_dirs(processed_dir)

    # TODO: Implement feature building
    # 1. Load raw data from data/raw/
    # 2. Compute per-team season stats
    # 3. Build Elo ratings across all seasons
    # 4. Merge all features per team-season
    # 5. For each matchup, compute 18 differentials (team_a - team_b)
    # 6. Save to data/processed/features.parquet

    logger.info("Feature building not yet implemented")
    logger.info("Features to build: %s", config["features"]["differentials"])


if __name__ == "__main__":
    main()
