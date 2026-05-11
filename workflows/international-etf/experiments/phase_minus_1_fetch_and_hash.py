"""Phase -1 — Pre-fetch all required data and freeze SHA256 hashes.

Discipline inherited from real-world-test workflow: a strategy result is only
defensible if the underlying data is reproducible byte-for-byte. We fetch
every ticker and macro series ONCE, write them to a versioned snapshot, and
hash each parquet/CSV file. Phase 4 robustness re-fetches and re-hashes;
mismatch = vendor revision = audit before claim.

Outputs:
  - data/snapshots/<today>/prices.parquet     (all ETF tickers)
  - data/snapshots/<today>/macro/dxy.csv      (DXY)
  - data/snapshots/<today>/macro/dgs10.csv    (US 10y)
  - data/snapshots/<today>/macro/bund10y.csv  (German 10y, OECD via FRED)
  - data/snapshots/<today>/macro/tb3ms.csv    (3-month T-bill)
  - precommit/data_hashes.json                (SHA256 of every file)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
SNAPSHOTS_DIR = WORKFLOW_DIR / "data" / "snapshots"
PRECOMMIT_DIR = WORKFLOW_DIR / "precommit"
UNIVERSE_PATH = WORKFLOW_DIR / "data" / "reference" / "international_universe.csv"

START_DATE = "2001-01-01"   # buffer before EFA inception 2001-08-14
END_DATE = "2026-04-30"


def fetch_yfinance_prices(tickers: list[str], snap_dir: Path) -> Path:
    """Fetch adjusted-close prices for all tickers; write parquet."""
    logger.info("Fetching %d tickers from yfinance", len(tickers))
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,   # use adjusted close for total return
        progress=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"

    path = snap_dir / "prices.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    logger.info("Wrote %s (%d rows × %d cols)", path, len(prices), prices.shape[1])
    return path


def fetch_yfinance_macro(snap_dir: Path) -> list[Path]:
    """Fetch DXY (yfinance proxy) and DBC (commodity index)."""
    macro_dir = snap_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for ticker, label in [("DX-Y.NYB", "dxy"), ("DBC", "dbc")]:
        logger.info("Fetching %s from yfinance", ticker)
        df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if df.empty and label == "dxy":
            logger.warning("DXY empty; trying UUP ETF proxy")
            df = yf.download("UUP", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
        if df.empty:
            logger.warning("%s came back empty; skipping", ticker)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            series = df["Close"].iloc[:, 0]
        else:
            series = df["Close"]
        series.index = pd.to_datetime(series.index)
        series.index.name = "date"
        path = macro_dir / f"{label}.csv"
        series.to_frame(label).to_csv(path)
        logger.info("Wrote %s (%d rows)", path, len(series))
        written.append(path)

    return written


def fetch_fred_macro(snap_dir: Path) -> list[Path]:
    """Fetch FRED macro series via pandas-datareader (or fall back to yfinance)."""
    macro_dir = snap_dir / "macro"
    macro_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # FRED series we need
    series = {
        "dgs10": "DGS10",                       # US 10-year Treasury
        "bund10y": "IRLTLT01DEM156N",           # German 10y bund (monthly)
        "tb3ms": "TB3MS",                       # 3-month T-bill (monthly)
        "twi_broad_dollar": "DTWEXBGS",         # Broad dollar index (daily)
        "breakeven_10y": "T10YIE",              # 10y breakeven inflation
    }

    try:
        from pandas_datareader import data as pdr
        for name, code in series.items():
            logger.info("Fetching FRED %s", code)
            try:
                df = pdr.DataReader(code, "fred", START_DATE, END_DATE)
                df.index.name = "date"
                df.columns = [name]
                path = macro_dir / f"{name}.csv"
                df.to_csv(path)
                logger.info("Wrote %s (%d rows)", path, len(df))
                written.append(path)
            except Exception as e:
                logger.warning("FRED %s failed: %s", code, e)
    except ImportError:
        logger.warning(
            "pandas_datareader not installed; FRED series skipped. "
            "Install with: pip install pandas-datareader. "
            "Phase 0 can proceed without FRED but Phase 2 regime gates need it."
        )

    return written


def hash_file(path: Path) -> str:
    """SHA256 of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_precommit_hashes(paths: list[Path], snap_dir: Path) -> Path:
    """Hash every produced file; write precommit JSON."""
    PRECOMMIT_DIR.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in paths:
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue
        rel = path.relative_to(WORKFLOW_DIR).as_posix()
        hashes[rel] = {
            "sha256": hash_file(path),
            "bytes": path.stat().st_size,
            "mtime": int(path.stat().st_mtime),
        }
    precommit_path = PRECOMMIT_DIR / "data_hashes.json"
    with precommit_path.open("w") as f:
        json.dump(
            {
                "snapshot_date": snap_dir.name,
                "snapshot_dir": snap_dir.relative_to(WORKFLOW_DIR).as_posix(),
                "n_files": len(hashes),
                "hashes": hashes,
            },
            f,
            indent=2,
        )
    logger.info("Wrote %s with %d hashes", precommit_path, len(hashes))
    return precommit_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    universe = pd.read_csv(UNIVERSE_PATH)
    tickers = universe["ticker"].tolist()
    logger.info("Universe: %d tickers", len(tickers))

    today = date.today().isoformat()
    snap_dir = SNAPSHOTS_DIR / today
    snap_dir.mkdir(parents=True, exist_ok=True)

    written = []

    prices_path = fetch_yfinance_prices(tickers, snap_dir)
    written.append(prices_path)

    written.extend(fetch_yfinance_macro(snap_dir))
    written.extend(fetch_fred_macro(snap_dir))

    precommit = write_precommit_hashes(written, snap_dir)

    logger.info("Phase -1 fetch+hash complete. Snapshot dir: %s", snap_dir)
    logger.info("Precommit hashes: %s", precommit)


if __name__ == "__main__":
    main()
