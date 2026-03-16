"""Generate bracket picks via Monte Carlo simulation."""

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

    brackets_dir = WORKFLOW_DIR / "output" / "brackets"
    ensure_dirs(brackets_dir)

    # TODO: Implement bracket generation
    # 1. Load matchup probability matrix from output/
    # 2. Run Monte Carlo simulation (10k runs per strategy)
    # 3. Generate brackets for each strategy:
    #    - Chalk: always pick higher probability team
    #    - Balanced: pick upsets proportional to model probability
    #    - Contrarian: slightly favor upsets for pool differentiation
    # 4. Save bracket JSONs to output/brackets/

    logger.info("Bracket generation not yet implemented")
    logger.info("Strategies: %s", [s["name"] for s in config["bracket"]["strategies"]])


if __name__ == "__main__":
    main()
