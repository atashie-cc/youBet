"""Phase 0: Download international football results dataset.

Primary source: martj42/international_results GitHub repo
(same dataset as Kaggle 'International football results 1872 to 2024',
maintained by the same author, no authentication required).

Files downloaded:
    results.csv       — ~49K match results 1872-2026 (date, teams, score, tournament, city, country, neutral)
    shootouts.csv     — penalty shootout outcomes (critical for Phase 3 knockout model tail)
    goalscorers.csv   — individual goal scorers
    former_names.csv  — team name mapping (e.g. West Germany → Germany)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import requests

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = WORKFLOW_DIR / "data" / "raw"

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv"]


def download(filename: str, dest_dir: Path, timeout: int = 60) -> Path:
    url = f"{BASE_URL}/{filename}"
    dest = dest_dir / filename
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Wrote %s (%d bytes)", dest, len(response.content))
    return dest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        try:
            download(filename, RAW_DIR)
        except Exception as exc:
            logger.error("Failed to download %s: %s", filename, exc)
            return 1

    # Quick validation
    results_path = RAW_DIR / "results.csv"
    with results_path.open("r", encoding="utf-8") as fh:
        rows = sum(1 for _ in fh) - 1
    logger.info("results.csv: %d data rows", rows)

    shootouts_path = RAW_DIR / "shootouts.csv"
    with shootouts_path.open("r", encoding="utf-8") as fh:
        shootouts = sum(1 for _ in fh) - 1
    logger.info("shootouts.csv: %d data rows", shootouts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
