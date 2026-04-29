"""Build the historical S&P 500 membership CSV and delisting_returns CSV.

Sources (all free):
  1. Wikipedia "List of S&P 500 companies" — current constituents
     (https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
     The page contains TWO tables:
       - Current constituents: Symbol, Security, GICS Sector,
         GICS Sub-Industry, Headquarters Location, Date added, CIK, Founded
       - Selected changes: Date | Added (Ticker, Security) | Removed
         (Ticker, Security) | Reason
  2. SEC company_tickers.json — ticker → CIK mapping for currently-active
     tickers (https://www.sec.gov/files/company_tickers.json)
  3. yfinance (optional, for delisting terminal-return estimation) — pulled
     on demand per delisted ticker.

Membership schema (matches `universe.py::REQUIRED_MEMBERSHIP_COLS`):
    ticker, name, gics_sector, gics_subindustry, start_date, end_date,
    cik, notes

Algorithm for membership intervals:
  - Seed with CURRENT constituents: start_date = Wikipedia "Date added"
    (or 1957-03-04 as a conservative fallback for tickers that predate
    the table), end_date = "" (still a member).
  - Walk the HISTORICAL CHANGES table chronologically. For each row:
      * If a ticker was ADDED on date D: append a new membership row for
        that ticker starting D with empty end_date.
      * If a ticker was REMOVED on date D: locate the most recent OPEN
        membership row (end_date == "") for that ticker and close it at
        D. If no open row exists, the ticker may have been added before
        the changes table's coverage — emit an unclosed record and skip.
  - A ticker may legitimately appear multiple times if it was re-added
    after removal (e.g., spin-offs). Each (ticker, start_date) tuple is
    one row.

Delisting schema:
    ticker, delist_date, delist_return, reason

We derive delistings from the removal rows in the changes table:
  - Parse the "Reason" column for acquisition / bankruptcy / index
    reshuffle keywords.
  - Assign a conservative terminal return: acquisitions → 0.0 (absent
    the actual deal premium), bankruptcies → -0.99, reshuffles → 0.0.
  - Optionally refine with yfinance by fetching final prices, but that
    is slow and off by default — enable with --use-yfinance.

Usage:
    python build_universe.py                 # scrape + write CSVs
    python build_universe.py --dry-run       # scrape, print summary only
    python build_universe.py --use-yfinance  # refine delisting returns
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_DIR = WORKFLOW_ROOT / "universe"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

DEFAULT_USER_AGENT = (
    "youBet-research contact@example.com "
    "(override via STOCK_EDGAR_USER_AGENT env)"
)

# Reasons we classify as "ticker died" rather than "index rotation".
# Acquisitions are ambiguous — the stock ceased trading but shareholders
# typically received cash/stock near market price (not a -100% loss).
# We flag them with a 0.0 terminal return for survivorship purposes.
BANKRUPTCY_KEYWORDS = (
    "bankrupt", "chapter 11", "liquidat", "delist", "ceased trading",
    "dissolved", "insolven",
)
ACQUISITION_KEYWORDS = (
    "acquir", "merg", "bought", "taken private", "taken over",
    "deal closed", "spin-off", "spinoff",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get(url: str, user_agent: str = DEFAULT_USER_AGENT) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


# ---------------------------------------------------------------------------
# Wikipedia parsing
# ---------------------------------------------------------------------------


def fetch_wiki_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_constituents, historical_changes) DataFrames."""
    html = _http_get(WIKI_URL)
    tables = pd.read_html(html)
    # Wikipedia's List of S&P 500 companies page has two key tables:
    # tables[0] = current constituents, tables[1] = selected changes.
    # If structure shifts, log and fail loudly — the caller can inspect.
    if len(tables) < 2:
        raise RuntimeError(
            f"Expected 2+ tables on Wikipedia S&P 500 page, got {len(tables)}"
        )
    current = tables[0]
    changes = tables[1]
    return current, changes


def _clean_ticker(t: str) -> str:
    """Normalize tickers: strip whitespace, replace dots with hyphens for
    yfinance compatibility (e.g., BRK.B -> BRK-B)."""
    if t is None or pd.isna(t):
        return ""
    s = str(t).strip().upper()
    # Wikipedia uses BRK.B; yfinance uses BRK-B.
    s = s.replace(".", "-")
    return s


def _clean_cik(c) -> str:
    if c is None or pd.isna(c):
        return ""
    s = str(c).strip()
    # Strip anything non-digit, then zero-pad
    digits = re.sub(r"\D", "", s)
    if not digits:
        return ""
    return digits.zfill(10)


def _parse_date(s) -> Optional[pd.Timestamp]:
    if s is None or pd.isna(s):
        return None
    try:
        return pd.to_datetime(str(s), errors="coerce")
    except Exception:
        return None


def parse_current_constituents(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the current-constituents table into membership-row form.

    Returns DataFrame with columns:
        ticker, name, gics_sector, gics_subindustry, start_date,
        end_date (empty), cik, notes.
    """
    # Column names vary slightly between Wikipedia revisions. Normalize.
    colmap = {}
    for col in df.columns:
        lc = str(col).lower().strip()
        if lc in ("symbol", "ticker"):
            colmap[col] = "ticker"
        elif "security" in lc:
            colmap[col] = "name"
        elif lc == "gics sector":
            colmap[col] = "gics_sector"
        elif "gics sub" in lc or "gics industry" in lc:
            colmap[col] = "gics_subindustry"
        elif "date added" in lc or "date first added" in lc:
            colmap[col] = "date_added"
        elif lc == "cik":
            colmap[col] = "cik"
    df = df.rename(columns=colmap)

    required = {"ticker", "name", "gics_sector", "gics_subindustry", "date_added", "cik"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Wikipedia current-constituents missing columns: {missing}")

    out = pd.DataFrame({
        "ticker": df["ticker"].map(_clean_ticker),
        "name": df["name"].astype(str).str.strip(),
        "gics_sector": df["gics_sector"].astype(str).str.strip(),
        "gics_subindustry": df["gics_subindustry"].astype(str).str.strip(),
        "start_date": df["date_added"].map(lambda s: _parse_date(s) or pd.Timestamp("1957-03-04")),
        "end_date": pd.NaT,
        "cik": df["cik"].map(_clean_cik),
        "notes": "current",
    })
    out = out[out["ticker"] != ""].reset_index(drop=True)
    return out


def parse_historical_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the multi-index 'Selected changes' table into one row per event.

    Output columns:
        event_date, added_ticker, added_name, removed_ticker,
        removed_name, reason.
    """
    # The changes table has a multi-index header: Date | Added(Ticker,
    # Security) | Removed(Ticker, Security) | Reason. `pandas.read_html`
    # returns it with flattened column names; normalize them.
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten: ('Added', 'Ticker') -> 'added_ticker'
        flat = []
        for top, bot in df.columns:
            t = str(top).strip().lower()
            b = str(bot).strip().lower()
            if t in ("", "nan") or t == b:
                flat.append(b)
            elif b in ("", "nan"):
                flat.append(t)
            else:
                flat.append(f"{t}_{b}")
        df = df.copy()
        df.columns = flat
    else:
        df = df.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if "date" in cl and "added" not in cl and "removed" not in cl:
            rename[c] = "event_date"
        elif "added" in cl and "ticker" in cl:
            rename[c] = "added_ticker"
        elif "added" in cl and ("security" in cl or "name" in cl):
            rename[c] = "added_name"
        elif "removed" in cl and "ticker" in cl:
            rename[c] = "removed_ticker"
        elif "removed" in cl and ("security" in cl or "name" in cl):
            rename[c] = "removed_name"
        elif "reason" in cl:
            rename[c] = "reason"
    df = df.rename(columns=rename)

    # Some Wikipedia revisions use "added" / "removed" as scalar string
    # instead of nested. Normalize to consistent output columns.
    for c in ("event_date", "added_ticker", "added_name", "removed_ticker",
              "removed_name", "reason"):
        if c not in df.columns:
            df[c] = None

    out = pd.DataFrame({
        "event_date": df["event_date"].map(_parse_date),
        "added_ticker": df["added_ticker"].map(_clean_ticker),
        "added_name": df["added_name"].astype(str).str.strip(),
        "removed_ticker": df["removed_ticker"].map(_clean_ticker),
        "removed_name": df["removed_name"].astype(str).str.strip(),
        "reason": df["reason"].astype(str).str.strip(),
    })
    # Drop rows with no event_date
    out = out[out["event_date"].notna()].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Membership reconstruction
# ---------------------------------------------------------------------------


def build_membership(
    current: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    """Merge current-constituents + historical changes into membership rows.

    Algorithm:
      1. Take all current constituents as OPEN intervals.
      2. Walk changes chronologically (oldest first).
         - Removal: close the most recent OPEN interval for that ticker.
                    If no open interval exists, record an orphan — the
                    ticker was removed before the current table's coverage.
         - Addition: open a new interval for that ticker.
      3. Final dataset: union of closed historical intervals + open intervals.

    Limitation: Wikipedia's changes table typically covers ~2000-present.
    Tickers added before Wikipedia's coverage and still current will show
    their current "Date added" as start_date (accurate).
    Tickers added before coverage and since removed will be absent
    entirely (documented orphan). These are minority cases.
    """
    current = current.copy()
    current["start_date"] = pd.to_datetime(current["start_date"])
    current["end_date"] = pd.NaT

    # Open intervals keyed by ticker (list because a ticker might have
    # multiple open/closed rows if re-added)
    membership_rows: list[dict] = []
    open_by_ticker: dict[str, dict] = {}

    for _, row in current.iterrows():
        rec = row.to_dict()
        rec["end_date"] = pd.NaT
        membership_rows.append(rec)
        open_by_ticker[rec["ticker"]] = rec

    # H-R2-1 fix. Wikipedia's "Date added" on the current-constituents
    # table and the changes table's addition events disagree in two
    # distinct ways:
    #   (1) SAME event reported twice ±1 day apart (ACN, ABNB, ACGL, ABBV)
    #       → consolidate to one interval.
    #   (2) DIFFERENT events: ticker had an earlier tenure in the index,
    #       went away pre-coverage or via a removal we DO track, then
    #       re-entered as the current seed (DELL 2024 is a re-IPO from
    #       its 2013 delisting). Closing the FUTURE-DATED seed at the
    #       EARLIER event is a negative interval.
    # The routine below handles both correctly: operations never modify a
    # seed whose start_date > event_date; such events are treated as
    # pre-coverage orphans and ignored for membership (the delistings
    # builder reads the changes table directly so they still land there).
    CONSOLIDATE_WINDOW_DAYS = 30
    orphans = 0
    changes_sorted = changes.sort_values("event_date").reset_index(drop=True)
    for _, ev in changes_sorted.iterrows():
        d = ev["event_date"]

        # --- REMOVAL ---
        rem = ev.get("removed_ticker")
        if rem:
            open_row = open_by_ticker.get(rem)
            if (
                open_row is not None
                and pd.isna(open_row["end_date"])
                and open_row["start_date"] < d
            ):
                # Genuine removal within an open interval we track.
                open_row["end_date"] = d
                open_by_ticker.pop(rem, None)
            else:
                # No open interval OR the seed starts AFTER this removal
                # (pre-coverage orphan). The delistings CSV still captures
                # this event via parse_historical_changes.
                orphans += 1

        # --- ADDITION ---
        add = ev.get("added_ticker")
        if not add:
            continue
        prior = open_by_ticker.get(add)

        if prior is not None and pd.isna(prior["end_date"]):
            gap = (d - prior["start_date"]).days
            if abs(gap) <= CONSOLIDATE_WINDOW_DAYS:
                # Case 1: same event described twice. Consolidate to the
                # earlier date; keep the seed's richer metadata.
                prior["start_date"] = min(prior["start_date"], d)
                continue
            if gap < 0:
                # Case 2: this ADD event predates the current seed by a
                # lot. The seed is a LATER re-addition; the earlier
                # instance had its own tenure which we can only honor if
                # a matching pre-seed removal also exists. Since we can't
                # recover the close date without a later REMOVAL event,
                # drop this pre-coverage addition (orphan) rather than
                # corrupt the seed.
                orphans += 1
                continue
            # Genuine later re-addition: close the prior at d, append new.
            prior["end_date"] = d

        rec = {
            "ticker": add,
            "name": (ev.get("added_name") or "").strip() or add,
            "gics_sector": "",
            "gics_subindustry": "",
            "start_date": d,
            "end_date": pd.NaT,
            "cik": "",
            "notes": "added_via_changes",
        }
        membership_rows.append(rec)
        open_by_ticker[add] = rec

    logger.info(
        "Built %d membership rows (%d current + %d historical events; %d orphans)",
        len(membership_rows), len(current), len(changes_sorted), orphans,
    )

    df = pd.DataFrame(membership_rows)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    # Sort by ticker, then start_date
    df = df.sort_values(["ticker", "start_date"]).reset_index(drop=True)

    # Guard: intervals for the same ticker must not overlap
    _validate_no_overlap(df)
    return df


def _validate_no_overlap(df: pd.DataFrame) -> None:
    """Assert membership rows are well-formed. H-R2-1 promoted to hard-fail.

    Hard errors:
      - end_date < start_date (negative-length interval)
      - end_date == start_date (zero-length interval)
      - overlapping intervals for the same ticker

    Any of these would corrupt `Universe.active_as_of(d)` for the affected
    ticker on the affected days; better to fail the build than ship.
    """
    errors: list[str] = []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("start_date")
        prev_end = None
        for _, row in grp.iterrows():
            s, e = row["start_date"], row["end_date"]
            if pd.notna(e) and e < s:
                errors.append(
                    f"{ticker}: negative interval {s.date()} → {e.date()}"
                )
            elif pd.notna(e) and e == s:
                errors.append(
                    f"{ticker}: zero-length interval at {s.date()}"
                )
            if prev_end is not None and s < prev_end:
                errors.append(
                    f"{ticker}: overlap prev end {prev_end.date()} vs "
                    f"next start {s.date()}"
                )
            if pd.notna(e):
                prev_end = e
    if errors:
        raise ValueError(
            f"Malformed membership intervals ({len(errors)} errors):\n  "
            + "\n  ".join(errors[:20])
            + ("\n  ..." if len(errors) > 20 else "")
        )


# ---------------------------------------------------------------------------
# CIK resolution
# ---------------------------------------------------------------------------


def load_sec_tickers(user_agent: str = DEFAULT_USER_AGENT) -> dict[str, str]:
    """Return {ticker: zero-padded 10-digit CIK} for currently-active tickers."""
    import json
    raw = _http_get(SEC_TICKERS_URL, user_agent=user_agent)
    payload = json.loads(raw)
    # Payload is {"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}
    out: dict[str, str] = {}
    for _, rec in payload.items():
        t = _clean_ticker(rec.get("ticker", ""))
        c = _clean_cik(rec.get("cik_str"))
        if t and c:
            out[t] = c
    logger.info("Loaded %d ticker → CIK mappings from SEC", len(out))
    return out


def resolve_ciks(membership: pd.DataFrame, sec_map: dict[str, str]) -> pd.DataFrame:
    """Fill missing CIK values from SEC map (active tickers only)."""
    df = membership.copy()
    # Rows where we already have a CIK (from Wikipedia current-constituents)
    mask_missing = df["cik"].astype(str).str.strip().eq("") | df["cik"].isna()
    df.loc[mask_missing, "cik"] = df.loc[mask_missing, "ticker"].map(
        lambda t: sec_map.get(_clean_ticker(t), "")
    )
    n_resolved = int((df["cik"].astype(str).str.strip() != "").sum())
    logger.info("CIK coverage: %d/%d rows (%d still missing)",
                n_resolved, len(df), len(df) - n_resolved)
    return df


# ---------------------------------------------------------------------------
# Delisting returns
# ---------------------------------------------------------------------------


def classify_reason(reason: str) -> tuple[float, str]:
    """Map a free-text reason to (terminal_return, category).

    Heuristics:
      - Bankruptcy / liquidation  → -0.99, category='bankruptcy'
      - Acquisition / merger      →  0.00, category='acquisition'
      - Other / index reshuffle   →  0.00, category='reshuffle'
    """
    if not reason:
        return 0.0, "reshuffle"
    r = reason.lower()
    for k in BANKRUPTCY_KEYWORDS:
        if k in r:
            return -0.99, "bankruptcy"
    for k in ACQUISITION_KEYWORDS:
        if k in r:
            return 0.0, "acquisition"
    return 0.0, "reshuffle"


def build_delistings(
    membership: pd.DataFrame,
    changes: pd.DataFrame,
    use_yfinance: bool = False,
) -> pd.DataFrame:
    """Generate delisting_returns.csv from removal events.

    One row per delisted ticker (last removal if a ticker was removed
    multiple times). The terminal return is an estimate — for rigorous
    backtests, override with CRSP or manual research on high-impact names.

    Columns: ticker, delist_date, delist_return, reason
    """
    # Start from changes: removal events (may not all correspond to
    # delisting — could be index rotation where the ticker keeps trading)
    rem = changes[changes["removed_ticker"].astype(str).str.strip() != ""].copy()
    rem["event_date"] = pd.to_datetime(rem["event_date"])
    rem = rem.sort_values("event_date").drop_duplicates(
        subset=["removed_ticker"], keep="last",
    )

    current_tickers = set(
        membership[membership["end_date"].isna()]["ticker"].unique()
    )

    rows = []
    for _, ev in rem.iterrows():
        t = _clean_ticker(ev["removed_ticker"])
        if not t:
            continue
        # Skip tickers that re-appear as current members (likely re-added)
        if t in current_tickers:
            continue
        dd = ev["event_date"]
        dr, cat = classify_reason(str(ev.get("reason", "")))
        rows.append({
            "ticker": t,
            "delist_date": dd.strftime("%Y-%m-%d"),
            "delist_return": dr,
            "reason": f"{cat}: {str(ev.get('reason', ''))[:200]}".strip(": "),
        })

    df = pd.DataFrame(rows)

    if use_yfinance and not df.empty:
        df = _refine_with_yfinance(df)

    logger.info("Built %d delisting rows", len(df))
    return df


def _refine_with_yfinance(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: pull the last few days of yfinance prices per ticker to
    estimate an actual pct-change on delist_date, falling back to the
    classify-based default on failure.

    This is SLOW (one yfinance call per ticker, rate-limited upstream)
    and often fails on delisted symbols. Off by default.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed; skipping refinement")
        return df

    out = df.copy()
    for i, row in out.iterrows():
        t = row["ticker"]
        dd = pd.Timestamp(row["delist_date"])
        start = (dd - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        end = (dd + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        try:
            df_px = yf.download(t, start=start, end=end,
                                auto_adjust=True, progress=False, threads=False)
            if df_px.empty:
                continue
            # Find the last trading day <= delist_date
            px = df_px.get("Close")
            if px is None:
                continue
            px_series = px[t] if isinstance(px, pd.DataFrame) and t in px.columns else px
            px_series = pd.Series(px_series).dropna()
            if len(px_series) < 2:
                continue
            before = px_series[px_series.index < dd]
            on = px_series[px_series.index <= dd]
            if len(before) >= 1 and len(on) >= 1:
                p_before = float(before.iloc[-1])
                p_on = float(on.iloc[-1])
                if p_before > 0 and p_on > 0:
                    out.at[i, "delist_return"] = round((p_on - p_before) / p_before, 4)
                    out.at[i, "reason"] = f"yfinance-derived | {out.at[i, 'reason']}"
        except Exception as exc:
            logger.debug("yfinance fetch failed for %s: %s", t, exc)
            continue
        time.sleep(0.2)  # throttle
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Scrape + print summary; don't write CSVs.")
    parser.add_argument("--use-yfinance", action="store_true",
                        help="Refine delisting returns with yfinance (slow).")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT,
                        help="User-Agent header for Wikipedia + SEC requests.")
    args = parser.parse_args()

    logger.info("Fetching Wikipedia S&P 500 tables...")
    current_raw, changes_raw = fetch_wiki_tables()
    logger.info("Current table: %d rows; Changes table: %d rows",
                len(current_raw), len(changes_raw))

    current = parse_current_constituents(current_raw)
    changes = parse_historical_changes(changes_raw)
    logger.info("Parsed: %d current constituents, %d change events",
                len(current), len(changes))

    membership = build_membership(current, changes)

    logger.info("Resolving CIKs via SEC company_tickers.json...")
    sec_map = load_sec_tickers(user_agent=args.user_agent)
    membership = resolve_ciks(membership, sec_map)

    delistings = build_delistings(
        membership, changes, use_yfinance=args.use_yfinance,
    )

    # --- summary ---
    print("\nSummary:")
    print(f"  Membership rows:        {len(membership)}")
    print(f"    Unique tickers:       {membership['ticker'].nunique()}")
    print(f"    Currently-active:     {int(membership['end_date'].isna().sum())}")
    print(f"    Historical (closed):  {int(membership['end_date'].notna().sum())}")
    print(f"  Delisting rows:         {len(delistings)}")
    print(f"    Bankruptcies:         "
          f"{int((delistings.get('reason', pd.Series(dtype=str)).str.startswith('bankruptcy')).sum())}")
    print(f"    Acquisitions:         "
          f"{int((delistings.get('reason', pd.Series(dtype=str)).str.startswith('acquisition')).sum())}")
    print(f"    Reshuffles:           "
          f"{int((delistings.get('reason', pd.Series(dtype=str)).str.startswith('reshuffle')).sum())}")

    if args.dry_run:
        logger.info("--dry-run: not writing CSVs")
        return

    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    membership_out = UNIVERSE_DIR / "sp500_membership.csv"
    delistings_out = UNIVERSE_DIR / "delisting_returns.csv"

    # Format dates as YYYY-MM-DD, leaving end_date empty for open intervals.
    m_out = membership.copy()
    m_out["start_date"] = m_out["start_date"].dt.strftime("%Y-%m-%d")
    m_out["end_date"] = m_out["end_date"].dt.strftime("%Y-%m-%d").fillna("")
    m_out = m_out[[
        "ticker", "name", "gics_sector", "gics_subindustry",
        "start_date", "end_date", "cik", "notes",
    ]]
    m_out.to_csv(membership_out, index=False)
    logger.info("Wrote %d membership rows to %s", len(m_out), membership_out)

    if not delistings.empty:
        delistings.to_csv(delistings_out, index=False)
        logger.info("Wrote %d delisting rows to %s", len(delistings), delistings_out)


if __name__ == "__main__":
    main()
