"""Collect raw data for NCAA March Madness prediction.

Sources: Kaggle March ML Mania (primary), KenPom, Bart Torvik, Sports Reference.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]


def collect_kaggle_data(config: dict) -> None:
    """Download Kaggle March ML Mania competition data."""
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    ensure_dirs(raw_dir)

    # TODO: Implement Kaggle API download
    # kaggle competitions download -c march-machine-learning-mania-2026 -p data/raw/
    logger.info("Kaggle data collection not yet implemented")
    logger.info("Manual: kaggle competitions download -c %s -p %s",
                config["data"]["sources"][0]["competition"], raw_dir)


def collect_kenpom_data(config: dict) -> None:
    """Scrape KenPom efficiency data (requires subscription)."""
    # TODO: Implement KenPom scraping
    logger.info("KenPom data collection not yet implemented")


def collect_torvik_data(config: dict) -> None:
    """Scrape Bart Torvik T-Rank data (free)."""
    # TODO: Implement T-Rank scraping
    logger.info("T-Rank data collection not yet implemented")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    logger.info("Collecting data for NCAA March Madness %s", config["workflow"]["year"])
    collect_kaggle_data(config)
    collect_kenpom_data(config)
    collect_torvik_data(config)
    logger.info("Data collection complete")


if __name__ == "__main__":
    main()
