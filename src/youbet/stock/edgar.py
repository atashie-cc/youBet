"""SEC EDGAR XBRL company-facts fetcher with filing-date preservation.

Point-in-time correctness depends on knowing WHEN each fundamental value
was publicly filed, not just the fiscal period it describes. Restatements
can revise a period's values years later; a backtest that uses the
restated value is leaking the future. This module persists every
reported value with its filing accession + filed date so downstream
code can answer "what did the public know about period X as of decision
date Y?" without leakage.

Endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json
- Requires a User-Agent header identifying the requester (SEC policy).
  Set env var STOCK_EDGAR_USER_AGENT or pass explicitly.
- Rate limit: SEC's documented ceiling is 10 requests/second. We throttle
  to 8 req/s to leave headroom.

One parquet per CIK, cached at `data/snapshots/edgar/CIK{cik}.parquet`.
Columns: concept, unit, start, end, val, filed, form, accn, fp, fy
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

EDGAR_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"
DEFAULT_USER_AGENT = (
    "youBet-research research@example.com"  # override via env for production use
)
RATE_LIMIT_SLEEP_SECONDS = 1.0 / 8  # 8 req/s → well under SEC's 10/s ceiling


@dataclass
class EdgarConfig:
    """SEC EDGAR fetch configuration.

    User-Agent is REQUIRED by the SEC. Override via:
      - EdgarConfig(user_agent="Your Name email@domain.com")
      - environment variable STOCK_EDGAR_USER_AGENT
    """

    user_agent: str = ""
    cache_dir: Path | None = None
    timeout_seconds: float = 30.0

    def __post_init__(self):
        if not self.user_agent:
            self.user_agent = os.environ.get(
                "STOCK_EDGAR_USER_AGENT", DEFAULT_USER_AGENT
            )


_LAST_REQUEST_TS: float = 0.0


def _throttle() -> None:
    """Sleep to respect SEC rate limit."""
    global _LAST_REQUEST_TS
    elapsed = time.monotonic() - _LAST_REQUEST_TS
    if elapsed < RATE_LIMIT_SLEEP_SECONDS:
        time.sleep(RATE_LIMIT_SLEEP_SECONDS - elapsed)
    _LAST_REQUEST_TS = time.monotonic()


def fetch_company_facts_raw(cik: str, config: EdgarConfig) -> dict:
    """Fetch raw XBRL company facts JSON from SEC EDGAR.

    Args:
        cik: Zero-padded 10-digit CIK (e.g., "0000320193").
        config: Fetch configuration including User-Agent.

    Returns:
        Parsed JSON dict. Keys: cik, entityName, facts.
        facts is keyed by taxonomy ("us-gaap", "dei", etc.), then concept.

    Raises:
        requests.HTTPError on non-200 response.
    """
    if len(cik) != 10 or not cik.isdigit():
        raise ValueError(f"CIK must be 10-digit zero-padded string, got {cik!r}")

    url = f"{EDGAR_BASE_URL}/CIK{cik}.json"
    headers = {"User-Agent": config.user_agent, "Accept": "application/json"}

    _throttle()
    logger.debug("GET %s", url)
    resp = requests.get(url, headers=headers, timeout=config.timeout_seconds)
    resp.raise_for_status()
    return resp.json()


def parse_company_facts(raw: dict) -> pd.DataFrame:
    """Flatten the nested company-facts JSON into a long DataFrame.

    Each row is one (concept, unit, fiscal-period, reported-value, filing).
    Restatements appear as multiple rows for the same (concept, end, unit)
    with different `filed` dates. The caller filters by `filed < decision_date`
    for PIT safety.

    Returns:
        DataFrame with columns:
            concept (str)  — XBRL concept name (e.g., "Assets", "NetIncomeLoss")
            taxonomy (str) — "us-gaap", "dei", etc.
            unit (str)     — "USD", "shares", "USD/shares"
            start (datetime64[ns], nullable) — fiscal period start
            end (datetime64[ns]) — fiscal period end
            val (float)    — reported value
            filed (datetime64[ns]) — filing date (THE POINT-IN-TIME KEY)
            form (str)     — "10-K", "10-Q", "8-K", etc.
            accn (str)     — accession number
            fy (int, nullable)
            fp (str, nullable) — "Q1"/"Q2"/"Q3"/"FY"
    """
    rows: list[dict] = []
    facts = raw.get("facts", {})
    for taxonomy, concepts in facts.items():
        for concept, payload in concepts.items():
            units = payload.get("units", {})
            for unit, entries in units.items():
                for e in entries:
                    rows.append({
                        "concept": concept,
                        "taxonomy": taxonomy,
                        "unit": unit,
                        "start": e.get("start"),
                        "end": e.get("end"),
                        "val": e.get("val"),
                        "filed": e.get("filed"),
                        "form": e.get("form"),
                        "accn": e.get("accn"),
                        "fy": e.get("fy"),
                        "fp": e.get("fp"),
                    })

    if not rows:
        return pd.DataFrame(
            columns=["concept", "taxonomy", "unit", "start", "end", "val",
                     "filed", "form", "accn", "fy", "fp"]
        )

    df = pd.DataFrame(rows)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    return df


def get_company_facts(
    cik: str,
    config: EdgarConfig | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch + cache + return parsed company facts for one CIK.

    Cache location: `{config.cache_dir}/CIK{cik}.parquet`. If cache exists
    and `force_refresh=False`, reads from disk without hitting SEC.
    """
    cfg = config or EdgarConfig()
    if cfg.cache_dir is None:
        raise ValueError(
            "EdgarConfig.cache_dir must be set. Typically "
            "workflows/stock-selection/data/snapshots/edgar/"
        )

    cache_path = cfg.cache_dir / f"CIK{cik}.parquet"
    if cache_path.exists() and not force_refresh:
        logger.debug("Loading cached facts for CIK %s", cik)
        return pd.read_parquet(cache_path)

    raw = fetch_company_facts_raw(cik, cfg)
    df = parse_company_facts(raw)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(
        "Cached %d XBRL facts for CIK %s to %s",
        len(df), cik, cache_path.name,
    )
    return df


class IndexedFacts:
    """Concept-indexed facts wrapper for O(1) concept lookup.

    A plain facts DataFrame is ~20k rows; `pit_concept_series` used to
    filter it by (taxonomy, concept, unit) on every call, which made
    Phase 1 backtest loops (~14 concept lookups × N tickers × M rebals)
    quadratically slow. `IndexedFacts` builds a {(taxonomy, concept,
    unit): slice} dict once per ticker; subsequent lookups are O(slice_size).

    Behaves enough like a DataFrame for the legacy call sites — the
    `.df`, `.empty`, `["filed"]`, and `.columns` attributes pass through
    — so existing code that does `facts.empty`, `(facts["filed"] < d).any()`,
    or `"filed" in facts.columns` continues to work.
    """

    __slots__ = ("df", "_by_key")

    def __init__(self, df: pd.DataFrame):
        self.df = df
        # groupby().indices is {(tax, concept, unit): ndarray_of_row_positions}
        if df.empty:
            self._by_key: dict[tuple[str, str, str], np.ndarray] = {}
        else:
            grouped = df.groupby(
                ["taxonomy", "concept", "unit"], sort=False, observed=True,
            )
            self._by_key = dict(grouped.indices)

    # Pandas-proxy niceties so call sites that passed around the raw df
    # keep working.
    @property
    def empty(self) -> bool:
        return self.df.empty

    @property
    def columns(self):
        return self.df.columns

    def __getitem__(self, key):
        return self.df[key]

    def slice(self, taxonomy: str, concept: str, unit: str) -> pd.DataFrame:
        """Return the pre-grouped subset for (taxonomy, concept, unit).

        Returns an empty DataFrame (with the right columns) if absent.
        """
        idx = self._by_key.get((taxonomy, concept, unit))
        if idx is None or len(idx) == 0:
            return self.df.iloc[0:0]
        # `.iloc[idx]` reuses the pre-filtered positions without rescanning.
        return self.df.iloc[idx]


def pit_concept_series(
    facts,
    concept: str,
    unit: str = "USD",
    decision_date: pd.Timestamp | None = None,
    prefer_taxonomy: str = "us-gaap",
) -> pd.DataFrame:
    """Extract a point-in-time series for one XBRL concept.

    For each fiscal period (`end` date), returns the MOST RECENT value
    that was filed strictly before `decision_date`. This is the as-known
    value, not the latest restatement.

    Args:
        facts: Long-format DataFrame from parse_company_facts, OR an
            `IndexedFacts` wrapper (faster — avoids O(n) concept scan).
        concept: XBRL concept name (e.g., "NetIncomeLoss", "Assets").
        unit: Unit filter (default "USD").
        decision_date: If given, filter to filings with filed < decision_date.
            If None, returns the full restatement history.
        prefer_taxonomy: Prefer this taxonomy (typically "us-gaap").

    Returns:
        DataFrame with columns [end, start, val, filed, form, fp, fy],
        one row per fiscal period (latest-filed as of decision_date).
    """
    if isinstance(facts, IndexedFacts):
        df = facts.slice(prefer_taxonomy, concept, unit)
    else:
        df = facts[
            (facts["concept"] == concept)
            & (facts["unit"] == unit)
            & (facts["taxonomy"] == prefer_taxonomy)
        ]

    if df.empty:
        return df.iloc[0:0][["end", "start", "val", "filed", "form", "fp", "fy"]]

    if decision_date is not None:
        df = df[df["filed"] < pd.Timestamp(decision_date)]
        if df.empty:
            return df[["end", "start", "val", "filed", "form", "fp", "fy"]]

    # Keep the latest-filed value per fiscal-period end
    df = df.sort_values(["end", "filed"]).drop_duplicates(
        subset=["end"], keep="last"
    )
    return df[["end", "start", "val", "filed", "form", "fp", "fy"]].reset_index(
        drop=True
    )


def fetch_bulk(
    ciks: list[str],
    config: EdgarConfig,
    skip_existing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch and cache facts for many CIKs with throttling.

    Args:
        ciks: List of 10-digit zero-padded CIK strings.
        config: Must have cache_dir set.
        skip_existing: If True, skip CIKs that already have a cached parquet.

    Returns:
        Dict of {cik: facts_df}. Failed CIKs are logged and absent from
        the returned dict.
    """
    if config.cache_dir is None:
        raise ValueError("EdgarConfig.cache_dir is required for bulk fetch")

    out: dict[str, pd.DataFrame] = {}
    for i, cik in enumerate(ciks):
        cache_path = config.cache_dir / f"CIK{cik}.parquet"
        if skip_existing and cache_path.exists():
            out[cik] = pd.read_parquet(cache_path)
            continue
        try:
            out[cik] = get_company_facts(cik, config)
        except Exception as exc:
            logger.warning("EDGAR fetch failed for CIK %s: %s", cik, exc)
            continue
        if (i + 1) % 25 == 0:
            logger.info("Fetched %d / %d", i + 1, len(ciks))

    return out
