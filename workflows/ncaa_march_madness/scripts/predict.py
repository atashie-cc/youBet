"""Generate tournament matchup probabilities for NCAA March Madness."""

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

    output_dir = WORKFLOW_DIR / "output"
    ensure_dirs(output_dir)

    # TODO: Implement prediction
    # 1. Load trained model and calibrator from models/
    # 2. Load current season team features
    # 3. Generate all possible tournament matchup features
    # 4. Predict calibrated probabilities for each matchup
    # 5. Save matchup probability matrix to output/

    logger.info("Prediction not yet implemented")


if __name__ == "__main__":
    main()
