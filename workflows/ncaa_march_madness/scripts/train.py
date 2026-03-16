"""Train XGBoost model with isotonic calibration for NCAA March Madness."""

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

    models_dir = WORKFLOW_DIR / "models"
    ensure_dirs(models_dir)

    # TODO: Implement training
    # 1. Load features from data/processed/features.parquet
    # 2. Temporal split by season (60/20/20)
    # 3. Train XGBoost on training set with early stopping on val
    # 4. Fit isotonic calibrator on validation predictions
    # 5. Evaluate on test set (log loss primary)
    # 6. Save model and calibrator to models/
    # 7. Log results to research/log.md

    logger.info("Training not yet implemented")


if __name__ == "__main__":
    main()
