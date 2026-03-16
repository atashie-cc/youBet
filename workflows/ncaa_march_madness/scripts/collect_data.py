"""Collect raw data for NCAA March Madness prediction.

Primary source: Kaggle March ML Mania competition CSVs.
Secondary: Bart Torvik T-Rank (free), KenPom (subscription).
"""

from __future__ import annotations

import logging
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.utils.io import ensure_dirs, load_config

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "MRegularSeasonDetailedResults.csv",
    "MNCAATourneyDetailedResults.csv",
    "MNCAATourneySeeds.csv",
    "MTeams.csv",
    "MSeasons.csv",
]

OPTIONAL_FILES = [
    "MMasseyOrdinals.csv",
    "MRegularSeasonCompactResults.csv",
    "MNCAATourneyCompactResults.csv",
]


def collect_kaggle_data(config: dict) -> bool:
    """Download Kaggle March ML Mania competition data.

    Uses the Kaggle CLI if available, otherwise provides manual instructions.
    Returns True if all required files are present after collection.
    """
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    ensure_dirs(raw_dir)

    competition = config["data"]["sources"][0]["competition"]

    # Check if required files already exist
    missing = [f for f in REQUIRED_FILES if not (raw_dir / f).exists()]
    if not missing:
        logger.info("All required Kaggle files already present in %s", raw_dir)
        return True

    logger.info("Missing files: %s", missing)

    # Try Kaggle Python API (more reliable than CLI which may not be on PATH)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        logger.info("Downloading from Kaggle competition: %s", competition)
        api.competition_download_files(competition, path=str(raw_dir), quiet=False)

        # Unzip if a zip file was downloaded
        for zip_file in raw_dir.glob("*.zip"):
            logger.info("Extracting %s", zip_file.name)
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(raw_dir)
            zip_file.unlink()
            logger.info("Extracted and removed %s", zip_file.name)

    except ImportError:
        logger.warning("Kaggle package not found. Install with: pip install kaggle")
        _print_manual_instructions(competition, raw_dir)
    except Exception as e:
        logger.warning("Kaggle download failed: %s", e)
        _print_manual_instructions(competition, raw_dir)

    return _check_required_files(raw_dir)


def _check_required_files(raw_dir: Path) -> bool:
    """Check if all required files exist and log their sizes."""
    all_present = True
    for f in REQUIRED_FILES:
        path = raw_dir / f
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("  OK: %s (%.2f MB)", f, size_mb)
        else:
            logger.error("  MISSING: %s", f)
            all_present = False

    # Check optional files
    for f in OPTIONAL_FILES:
        path = raw_dir / f
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("  OPTIONAL: %s (%.2f MB)", f, size_mb)
        else:
            # Also check for variations (Massey ordinals often have a different name)
            matches = list(raw_dir.glob("*Massey*")) if "Massey" in f else []
            if matches:
                actual = matches[0]
                size_mb = actual.stat().st_size / (1024 * 1024)
                logger.info("  OPTIONAL: %s (%.2f MB) [as %s]", f, size_mb, actual.name)
            else:
                logger.info("  OPTIONAL (not found): %s", f)

    return all_present


def _print_manual_instructions(competition: str, raw_dir: Path) -> None:
    """Print instructions for manual data download."""
    logger.info("")
    logger.info("=== Manual Download Instructions ===")
    logger.info("1. Go to: https://www.kaggle.com/competitions/%s/data", competition)
    logger.info("2. Accept competition rules if needed")
    logger.info("3. Download all CSV files to: %s", raw_dir)
    logger.info("   OR run: kaggle competitions download -c %s -p %s", competition, raw_dir)
    logger.info("Required files: %s", REQUIRED_FILES)
    logger.info("====================================")
    logger.info("")


def collect_torvik_data(config: dict) -> None:
    """Download Bart Torvik T-Rank data (free, no auth needed).

    T-Rank provides KenPom-like adjusted efficiency metrics for free.
    """
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    torvik_dir = raw_dir / "torvik"
    ensure_dirs(torvik_dir)

    # TODO: Phase 2+ — Implement T-Rank scraping
    # barttorvik.com provides team stats tables that can be scraped
    # Key metrics: AdjOE, AdjDE, Barthag, EFG%, TO%, ORB%, FTRate
    logger.info("T-Rank data collection: not yet implemented (Phase 2+)")
    logger.info("Will scrape adjusted efficiency data from barttorvik.com")


def collect_kenpom_data(config: dict) -> None:
    """Download KenPom efficiency data (requires $24.95/yr subscription)."""
    raw_dir = WORKFLOW_DIR / config["data"]["raw_dir"]
    kenpom_dir = raw_dir / "kenpom"
    ensure_dirs(kenpom_dir)

    # TODO: Phase 2+ — Implement KenPom scraping
    # Requires authenticated session (KENPOM_EMAIL / KENPOM_PASSWORD)
    logger.info("KenPom data collection: not yet implemented (Phase 2+)")
    logger.info("Requires subscription credentials in .env")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(WORKFLOW_DIR / "config.yaml")

    logger.info("Collecting data for NCAA March Madness %s", config["workflow"]["year"])
    logger.info("=" * 60)

    success = collect_kaggle_data(config)
    if success:
        logger.info("Kaggle data ready")
    else:
        logger.warning("Some required Kaggle files are missing — see instructions above")

    collect_torvik_data(config)
    collect_kenpom_data(config)

    logger.info("=" * 60)
    logger.info("Data collection complete")


if __name__ == "__main__":
    main()
