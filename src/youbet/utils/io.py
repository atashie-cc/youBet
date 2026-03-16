"""Data loading and saving helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML config file."""
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", path)
    return config


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load CSV with logging."""
    df = pd.read_csv(path, **kwargs)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save DataFrame to CSV, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info("Saved %d rows to %s", len(df), path)


def save_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Save DataFrame to Parquet, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    logger.info("Saved %d rows to %s", len(df), path)


def ensure_dirs(*dirs: Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
