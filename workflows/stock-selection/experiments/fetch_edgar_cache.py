"""Bulk-fetch SEC EDGAR XBRL company-facts for every CIK in the universe.

Caches one parquet file per CIK at
`workflows/stock-selection/data/snapshots/edgar/CIK{cik}.parquet`.

SEC rate limit: 10 req/s max per IP (policy:
https://www.sec.gov/about/developer.htm). The EDGAR client throttles at
8 req/s to leave headroom. A full S&P 500 history (~800 CIKs with data)
takes ~15-25 minutes of wall time including parse + parquet write.

Usage:
    # Default: fetch missing only, verbose progress
    python fetch_edgar_cache.py

    # Force re-fetch all (wipes cache)
    python fetch_edgar_cache.py --force-refresh

    # Limit to first N CIKs for smoke testing
    python fetch_edgar_cache.py --limit 20

Environment:
    STOCK_EDGAR_USER_AGENT — override the User-Agent header. SEC requires
    a real email. Default is a generic placeholder; set this to your
    actual contact before heavy use.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from youbet.stock.edgar import EdgarConfig, fetch_bulk  # noqa: E402
from youbet.stock.universe import Universe  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
logger = logging.getLogger(__name__)

UNIVERSE_DIR = WORKFLOW_ROOT / "universe"
CACHE_DIR = WORKFLOW_ROOT / "data" / "snapshots" / "edgar"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-refresh", action="store_true",
                        help="Ignore existing cache and refetch everything.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only fetch the first N CIKs (smoke test).")
    parser.add_argument("--user-agent", default=None,
                        help="Override User-Agent (defaults to env var "
                             "STOCK_EDGAR_USER_AGENT).")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    u = Universe.from_csv(
        UNIVERSE_DIR / "sp500_membership.csv",
        UNIVERSE_DIR / "delisting_returns.csv",
    )
    # Unique, non-empty CIKs from the membership
    ciks_raw = u.membership["cik"].astype(str).str.strip()
    ciks = sorted({c for c in ciks_raw if c and c != "nan" and c != "<NA>"})
    logger.info("Discovered %d unique CIKs in universe", len(ciks))

    if args.limit is not None:
        ciks = ciks[: args.limit]
        logger.info("Limiting to first %d CIKs (--limit)", len(ciks))

    cfg = EdgarConfig(
        user_agent=args.user_agent or os.environ.get(
            "STOCK_EDGAR_USER_AGENT", "youBet-research contact@example.com",
        ),
        cache_dir=CACHE_DIR,
    )

    start = time.monotonic()
    cached = set(p.stem.replace("CIK", "") for p in CACHE_DIR.glob("CIK*.parquet"))
    to_fetch = [c for c in ciks if c not in cached] if not args.force_refresh else ciks
    logger.info(
        "Cache: %d already cached, %d to fetch (force=%s)",
        len(cached), len(to_fetch), args.force_refresh,
    )

    results = fetch_bulk(
        ciks, config=cfg, skip_existing=not args.force_refresh,
    )

    elapsed = time.monotonic() - start
    logger.info(
        "Done. Fetched/loaded %d/%d CIKs in %.1fs (%.2fs/cik)",
        len(results), len(ciks), elapsed, elapsed / max(len(ciks), 1),
    )

    # Quick post-fetch summary: row counts per CIK
    cached_after = list(CACHE_DIR.glob("CIK*.parquet"))
    total_rows = 0
    for p in cached_after:
        try:
            df = pd.read_parquet(p, columns=["concept"])
            total_rows += len(df)
        except Exception:
            pass
    logger.info(
        "Cache now has %d parquet files, %d total fact rows",
        len(cached_after), total_rows,
    )


if __name__ == "__main__":
    main()
