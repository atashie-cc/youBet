"""Ken French Data Library fetcher with snapshot caching.

Downloads factor return series (Mkt-RF, SMB, HML, RMW, CMA, UMD)
from the Ken French Data Library. Returns are daily and stored as
date-stamped snapshots for reproducibility.

Data source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
All factor returns are survivorship-bias-free by construction (built from CRSP).
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ken French Data Library URL patterns
# ---------------------------------------------------------------------------

_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"

# Dataset names → (url_suffix, columns_of_interest, frequency)
_DATASETS = {
    "ff5_daily": {
        "url": f"{_BASE_URL}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
        "factors": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    },
    "momentum_daily": {
        "url": f"{_BASE_URL}/F-F_Momentum_Factor_daily_CSV.zip",
        "factors": ["Mom"],  # labeled "Mom   " in the file
    },
    "ff5_monthly": {
        "url": f"{_BASE_URL}/F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "factors": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    },
    "momentum_monthly": {
        "url": f"{_BASE_URL}/F-F_Momentum_Factor_CSV.zip",
        "factors": ["Mom"],
    },
}

# Canonical factor names used throughout the codebase
FACTOR_NAMES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]

# Publication dates for decay measurement (Workflow 2)
PUBLICATION_DATES = {
    "Mkt-RF": "1964-01-01",  # CAPM (Sharpe 1964)
    "SMB": "1992-06-01",     # Fama-French 1992
    "HML": "1992-06-01",     # Fama-French 1992
    "RMW": "2013-09-01",     # Novy-Marx 2013 / Fama-French 2015
    "CMA": "2013-09-01",     # Fama-French 2015
    "UMD": "1993-03-01",     # Jegadeesh-Titman 1993
}


def _parse_french_csv(content: str, expected_cols: list[str]) -> pd.DataFrame:
    """Parse Ken French CSV format.

    The CSV files have a text header, then data starting after the first
    line that looks like a date. Returns are in PERCENTAGE terms (e.g., 0.03
    means 0.03%, not 3%). We convert to decimal.
    """
    lines = content.strip().split("\n")

    # Find the header row (first line where the second field is a factor name
    # or the line starts with a date-like number)
    data_start = None
    header_line = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # Check if line starts with a date (YYYYMMDD format)
        parts = stripped.split(",")
        first = parts[0].strip()
        if first.isdigit() and len(first) == 8:
            # This is data. Header is the previous non-empty line.
            if header_line is None:
                # Look backwards for header
                for j in range(i - 1, -1, -1):
                    if lines[j].strip():
                        header_line = j
                        break
            data_start = i
            break

    if data_start is None:
        raise ValueError("Could not find data start in French CSV")

    # Parse header
    if header_line is not None:
        header = [c.strip() for c in lines[header_line].split(",")]
    else:
        header = ["date"] + expected_cols

    # Parse data rows
    rows = []
    for line in lines[data_start:]:
        stripped = line.strip()
        if not stripped:
            break  # End of daily data (monthly files have annual data below)
        parts = stripped.split(",")
        first = parts[0].strip()
        # Daily data: 8 digits (YYYYMMDD)
        # Monthly data: 6 digits (YYYYMM)
        if not first.isdigit():
            break
        if len(first) < 6:
            break
        rows.append(parts)

    if not rows:
        raise ValueError("No data rows found in French CSV")

    # Build DataFrame
    df = pd.DataFrame(rows, columns=header[:len(rows[0])])
    date_col = df.columns[0]

    # Parse dates
    sample_date = df[date_col].iloc[0].strip()
    if len(sample_date) == 8:
        df["date"] = pd.to_datetime(df[date_col].str.strip(), format="%Y%m%d")
    else:
        # Monthly: YYYYMM → last day of month
        df["date"] = pd.to_datetime(df[date_col].str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)

    df = df.set_index("date").drop(columns=[date_col] if date_col != "date" else [])

    # Clean column names
    df.columns = [c.strip() for c in df.columns]

    # Convert from percentage to decimal
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    return df


def _download_french_dataset(dataset_key: str) -> pd.DataFrame:
    """Download and parse a single Ken French dataset."""
    info = _DATASETS[dataset_key]
    url = info["url"]
    expected = info["factors"]

    logger.info("Downloading %s from %s", dataset_key, url)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # There's usually one CSV file inside
        csv_names = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in {url}")
        content = zf.read(csv_names[0]).decode("utf-8", errors="replace")

    df = _parse_french_csv(content, expected)
    return df


def fetch_french_factors(
    frequency: str = "daily",
    snapshot_dir: Path | None = None,
    max_staleness_days: int = 30,
) -> pd.DataFrame:
    """Fetch all 6 Ken French factors (Mkt-RF, SMB, HML, RMW, CMA, UMD).

    Downloads from Ken French Data Library if no recent snapshot exists.
    Caches as date-stamped parquet for reproducibility.

    Args:
        frequency: "daily" or "monthly".
        snapshot_dir: Directory for cached snapshots. If None, downloads only.
        max_staleness_days: Warn if snapshot is older than this.

    Returns:
        DataFrame with DatetimeIndex and columns:
        [Mkt-RF, SMB, HML, RMW, CMA, UMD, RF]
        All values are daily simple returns as decimals (0.01 = 1%).
    """
    # Try loading from snapshot first
    if snapshot_dir is not None:
        snapshot_dir = Path(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        pattern = f"french_factors_{frequency}_*.parquet"
        existing = sorted(snapshot_dir.glob(pattern))

        if existing:
            latest = existing[-1]
            # Check staleness
            snap_date_str = latest.stem.split("_")[-1]
            try:
                snap_date = pd.to_datetime(snap_date_str)
                days_old = (pd.Timestamp.now() - snap_date).days
                if days_old > max_staleness_days:
                    logger.warning(
                        "Snapshot %s is %d days old (max %d). Consider refreshing.",
                        latest, days_old, max_staleness_days,
                    )
            except ValueError:
                pass
            logger.info("Loading cached factors from %s", latest)
            return pd.read_parquet(latest)

    # Download fresh data
    ff5_key = f"ff5_{frequency}"
    mom_key = f"momentum_{frequency}"

    ff5 = _download_french_dataset(ff5_key)
    mom = _download_french_dataset(mom_key)

    # Rename momentum column to standard name
    mom_col = [c for c in mom.columns if "mom" in c.lower() or "umd" in c.lower()]
    if mom_col:
        mom = mom.rename(columns={mom_col[0]: "UMD"})
    elif len(mom.columns) == 1:
        mom.columns = ["UMD"]

    # Keep only the UMD column from momentum file
    mom = mom[["UMD"]]

    # Join on common dates
    combined = ff5.join(mom, how="inner")

    # Ensure standard column order
    available = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD", "RF"] if c in combined.columns]
    combined = combined[available]

    # Drop rows with all NaN
    combined = combined.dropna(how="all")

    # Sort by date
    combined = combined.sort_index()

    logger.info(
        "Loaded %d %s observations from %s to %s (%d factors)",
        len(combined),
        frequency,
        combined.index[0].strftime("%Y-%m-%d"),
        combined.index[-1].strftime("%Y-%m-%d"),
        len(available) - (1 if "RF" in available else 0),
    )

    # Save snapshot
    if snapshot_dir is not None:
        today_str = date.today().strftime("%Y-%m-%d")
        path = snapshot_dir / f"french_factors_{frequency}_{today_str}.parquet"
        combined.to_parquet(path)
        logger.info("Saved snapshot to %s", path)

    return combined


def load_french_snapshot(
    snapshot_dir: Path,
    frequency: str = "daily",
) -> pd.DataFrame:
    """Load the most recent French factor snapshot without downloading.

    Raises FileNotFoundError if no snapshot exists.
    """
    snapshot_dir = Path(snapshot_dir)
    pattern = f"french_factors_{frequency}_*.parquet"
    existing = sorted(snapshot_dir.glob(pattern))

    if not existing:
        raise FileNotFoundError(
            f"No French factor snapshot found in {snapshot_dir}. "
            f"Run fetch_french_factors(snapshot_dir=...) first."
        )

    latest = existing[-1]
    logger.info("Loading French factors from %s", latest)
    return pd.read_parquet(latest)


def factor_cumulative_returns(
    factors: pd.DataFrame,
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute cumulative return series (growth of $1) for each factor.

    Args:
        factors: DataFrame from fetch_french_factors().
        factor_names: Which factors to include. Default: all 6.

    Returns:
        DataFrame with cumulative returns (starting at 1.0).
    """
    if factor_names is None:
        factor_names = [c for c in FACTOR_NAMES if c in factors.columns]

    cum = pd.DataFrame(index=factors.index)
    for name in factor_names:
        cum[name] = (1 + factors[name]).cumprod()

    return cum
