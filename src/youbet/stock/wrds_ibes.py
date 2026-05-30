"""WRDS / I/B/E/S ingestion + point-in-time analyst-signal panels.

This is the ONLY honest path to a confirmatory test of forward-analyst signals
(estimate revisions, consensus recommendations, price targets) on S&P 500
stocks: I/B/E/S Summary History is the genuine PIT consensus archive (US since
1976, survivorship-free incl. delisted names), and its `statpers` field is the
archived as-of snapshot date — so the consensus AS KNOWN on any past month-end
is reconstructable without look-ahead. No free source provides this (see
`workflows/individual-stocks-snp500/research/data-availability.md`).

Access is institutional (WRDS PostgreSQL). This module is **dual-mode**:
  (1) LIVE  — query WRDS via the `wrds` Python package (the user's account/creds
      are supplied by the `wrds` package's own auth; this module never handles
      passwords).
  (2) EXTRACT — read a local parquet/CSV extract the user downloaded from the
      WRDS web query (same column names).

PIT contract (enforced):
  At decision date d, the consensus for a ticker is the row with the LATEST
  `statpers` strictly < d. `IbesPanel.as_of` and the signal-panel builders all
  honor this. statpers is the IBES statistical-period date (~3rd Thursday
  monthly) — the archived snapshot, NOT a restated value.

Linking:
  IBES is keyed by its own 6-char `ticker` (and historical 8-char `cusip` /
  `oftic` official ticker). Our universe is keyed by exchange ticker + CIK. We
  link via `oftic`/`cusip` against the universe membership (with a documented
  fallback crosswalk), retaining delisted names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from youbet.etf.pit import PITViolation
from youbet.stock.strategies.base import CrossSectionalStrategy
from youbet.stock.universe import Universe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + ingestion (dual-mode)
# ---------------------------------------------------------------------------

@dataclass
class IbesConfig:
    """Where to read IBES from. Provide ONE of: a local extract dir, or
    use_live_wrds=True (the `wrds` package handles auth from the user's env).
    """
    extract_dir: Path | None = None          # local WRDS extract (parquet/csv)
    use_live_wrds: bool = False
    wrds_username: str | None = None          # passed to wrds.Connection
    start: str = "2004-01-01"

    def __post_init__(self):
        if self.extract_dir is not None:
            self.extract_dir = Path(self.extract_dir)
        if not self.use_live_wrds and self.extract_dir is None:
            raise ValueError(
                "IbesConfig requires either extract_dir (local WRDS extract) "
                "or use_live_wrds=True.")


# The exact WRDS queries (run when use_live_wrds=True). Table/column names are
# the standard IBES-on-WRDS schema; minor vintage differences may require
# tweaks. statpers is the PIT snapshot date.
_WRDS_QUERIES = {
    "summary": """
        select ticker, cusip, oftic, cname, statpers, fpedats, fpi, measure,
               meanest, medest, numest, stdev, actual
        from ibes.statsum_epsus
        where fpi in ('0','1','2') and statpers >= '{start}'
    """,
    "recommendations": """
        select ticker, cusip, statpers, meanrec, numrec
        from ibes.recdsum
        where statpers >= '{start}'
    """,
    "targets": """
        select ticker, cusip, statpers, meanptg, numptg, horizon
        from ibes.ptgsum
        where statpers >= '{start}'
    """,
    "id": "select ticker, cusip, oftic, cname, sdates from ibes.id",
}

_EXTRACT_FILES = {
    "summary": "ibes_statsum_epsus",
    "recommendations": "ibes_recdsum",
    "targets": "ibes_ptgsum",
    "id": "ibes_id",
}


def _read_extract(extract_dir: Path, key: str) -> pd.DataFrame:
    base = extract_dir / _EXTRACT_FILES[key]
    for ext in (".parquet", ".csv", ".csv.gz"):
        p = base.with_suffix(ext) if ext == ".parquet" else Path(str(base) + ext)
        if p.exists():
            df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            df.columns = [c.lower() for c in df.columns]
            return df
    raise FileNotFoundError(
        f"IBES extract for '{key}' not found at {base}.[parquet|csv]. "
        f"Download {_WRDS_QUERIES.get(key, '').strip()[:60]}... from WRDS.")


def load_ibes_table(config: IbesConfig, key: str) -> pd.DataFrame:
    """Load one IBES table (summary | recommendations | targets | id),
    live from WRDS or from a local extract. Lower-cases columns; parses
    statpers/fpedats to datetime."""
    if config.use_live_wrds:
        import wrds  # noqa: F401 — optional dep; only needed in live mode
        conn = wrds.Connection(wrds_username=config.wrds_username)
        try:
            sql = _WRDS_QUERIES[key].format(start=config.start)
            df = conn.raw_sql(sql)
        finally:
            conn.close()
        df.columns = [c.lower() for c in df.columns]
    else:
        df = _read_extract(config.extract_dir, key)
    for dcol in ("statpers", "fpedats", "sdates", "anndats"):
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Linking IBES -> our universe (exchange ticker), survivorship-safe
# ---------------------------------------------------------------------------

def link_to_universe(
    ibes_df: pd.DataFrame,
    universe: Universe,
    crosswalk: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add an `uticker` column (our exchange ticker) to an IBES frame.

    Strategy: (1) explicit user crosswalk {ibes_ticker: uticker} wins;
    (2) else match IBES `oftic` (official ticker) to a universe member ticker;
    (3) else leave NaN (logged). Delisted universe members ARE matched (IBES
    retains them), so the link is survivorship-safe to the extent oftic/cusip
    resolve. Reports the match rate.
    """
    members = set(universe.all_tickers_ever())
    out = ibes_df.copy()
    cw = {k.upper(): v for k, v in (crosswalk or {}).items()}

    def _resolve(row) -> str | None:
        it = str(row.get("ticker", "")).upper()
        if it in cw and cw[it] in members:
            return cw[it]
        oft = str(row.get("oftic", "") or "").upper()
        if oft in members:
            return oft
        return None

    out["uticker"] = out.apply(_resolve, axis=1)
    n_total = out["ticker"].nunique() if "ticker" in out.columns else len(out)
    n_linked = out.loc[out["uticker"].notna(), "ticker"].nunique() \
        if "ticker" in out.columns else out["uticker"].notna().sum()
    logger.info("IBES->universe link: %d of %d IBES tickers resolved (%.0f%%)",
                n_linked, n_total, 100 * n_linked / max(n_total, 1))
    return out


# ---------------------------------------------------------------------------
# PIT panel + signal builders
# ---------------------------------------------------------------------------

class IbesPanel:
    """PIT reader over an IBES summary frame (linked, with `uticker`,
    `statpers`, and a value column). `as_of(d)` returns, per uticker, the row
    with the LATEST statpers strictly < d."""

    def __init__(self, df: pd.DataFrame, value_cols: list[str]):
        req = {"uticker", "statpers"} | set(value_cols)
        missing = req - set(df.columns)
        if missing:
            raise ValueError(f"IbesPanel missing columns: {missing}")
        self.df = df.dropna(subset=["uticker", "statpers"]).sort_values("statpers")
        self.value_cols = value_cols

    def as_of(self, decision_date: pd.Timestamp) -> pd.DataFrame:
        d = pd.Timestamp(decision_date)
        sub = self.df[self.df["statpers"] < d]            # strict PIT
        if sub.empty:
            return sub.iloc[0:0]
        # latest statpers per uticker
        latest = sub.sort_values("statpers").groupby("uticker", as_index=False).last()
        return latest


def _signal_lut(df: pd.DataFrame, statpers_col: str, ticker_col: str,
                value: pd.Series) -> pd.DataFrame:
    """Pivot a (statpers, ticker, value) long frame to a statpers-indexed
    wide LUT (index=statpers dates, columns=ticker). The strategy looks up the
    latest statpers < decision_date."""
    tmp = pd.DataFrame({"statpers": df[statpers_col].values,
                        "uticker": df[ticker_col].values, "v": value.values})
    tmp = tmp.dropna(subset=["statpers", "uticker"])
    return tmp.pivot_table(index="statpers", columns="uticker", values="v",
                           aggfunc="last").sort_index()


def build_revision_panel(summary_fy1: pd.DataFrame, window_months: int = 3,
                         scale: str = "prior") -> pd.DataFrame:
    """Estimate-revision momentum LUT (statpers x uticker).

    Signal = (current FY1 mean consensus EPS - consensus `window_months` ago)
    scaled by |prior consensus| (scale='prior'). Higher = upward revision =
    stronger buy (So 2013; Chan-Jegadeesh-Lakonishok 1996). Uses ONLY the FY1
    summary rows; the change is computed per uticker across statpers, so it is
    inherently PIT (each statpers is an archived snapshot).
    """
    df = summary_fy1.dropna(subset=["uticker", "statpers", "meanest"]).copy()
    df = df.sort_values(["uticker", "statpers"])
    # approximate window in monthly statpers steps
    steps = max(1, window_months)
    df["prior"] = df.groupby("uticker")["meanest"].shift(steps)
    if scale == "prior":
        denom = df["prior"].abs().replace(0, np.nan)
        df["rev"] = (df["meanest"] - df["prior"]) / denom
    else:
        df["rev"] = df["meanest"] - df["prior"]
    return _signal_lut(df, "statpers", "uticker", df["rev"])


def build_consensus_rec_panel(rec: pd.DataFrame) -> pd.DataFrame:
    """Recommendation-consensus LUT. IBES meanrec is 1=strong buy..5=sell, so
    the buy signal is NEGATIVE meanrec (higher = more buy). (Barber-Lehavy-
    McNichols-Trueman 2001; expected weak/DOA but pre-registered.)"""
    df = rec.dropna(subset=["uticker", "statpers", "meanrec"]).copy()
    return _signal_lut(df, "statpers", "uticker", -df["meanrec"])


def build_target_upside_panel(tgt: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Price-target implied-return LUT = mean target / price(at statpers) - 1.
    Higher = more implied upside = buy (Da-Schaumburg 2011 — note ABSOLUTE
    implied return is optimism-biased; within-industry ranking is the stronger
    form, handled at scoring via sector-neutralization if desired)."""
    df = tgt.dropna(subset=["uticker", "statpers", "meanptg"]).copy()
    px = prices.copy()
    px.index = pd.to_datetime(px.index)

    def _px_at(row):
        tk = row["uticker"]
        if tk not in px.columns:
            return np.nan
        s = px[tk].dropna()
        s = s[s.index <= row["statpers"]]
        return float(s.iloc[-1]) if len(s) else np.nan

    df["px"] = df.apply(_px_at, axis=1)
    df["upside"] = df["meanptg"] / df["px"] - 1.0
    df.loc[~np.isfinite(df["upside"]), "upside"] = np.nan
    return _signal_lut(df, "statpers", "uticker", df["upside"])


def build_dispersion_panel(summary_fy1: pd.DataFrame) -> pd.DataFrame:
    """Forecast-dispersion LUT = stdev / |meanest|. Diether-Malloy-Scherbina
    (2002): HIGH dispersion -> LOW returns, so the buy signal is NEGATIVE
    dispersion (avoid high-dispersion). Descriptive/short-side-tilted."""
    df = summary_fy1.dropna(subset=["uticker", "statpers", "stdev", "meanest"]).copy()
    denom = df["meanest"].abs().replace(0, np.nan)
    return _signal_lut(df, "statpers", "uticker", -(df["stdev"] / denom))


# ---------------------------------------------------------------------------
# Strategy: PIT lookup of a statpers-indexed signal LUT
# ---------------------------------------------------------------------------

class IbesSignalStrategy(CrossSectionalStrategy):
    """Score active tickers by the latest IBES signal with statpers < as_of
    (PIT). Higher score = stronger buy (build the LUT so this holds). Top
    decile, equal-weight by default — same selection as the other confirmatory
    strategies. `max_staleness_days` refuses signals older than N days (IBES
    statpers is monthly; default 100 = ~3 monthly snapshots)."""

    def __init__(self, lut: pd.DataFrame, name: str,
                 max_staleness_days: int = 100, decile_breakpoint: float = 0.10,
                 min_holdings: int = 20, max_holdings: int = 100):
        self._lut = lut.sort_index()
        self._name = name
        self.max_staleness_days = max_staleness_days
        self.decile_breakpoint = decile_breakpoint
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        d = pd.Timestamp(panel["as_of_date"])
        idx = self._lut.index
        prior = idx[idx < d]                       # strict PIT
        if len(prior) == 0 or (d - prior[-1]).days > self.max_staleness_days:
            return pd.Series(dtype=float)
        row = self._lut.loc[prior[-1]].dropna()
        active = panel["active_tickers"]
        return row[row.index.isin(active)].astype(float)

    @property
    def name(self) -> str:
        return self._name


def validate_ibes_pit(panel: IbesPanel, decision_date: pd.Timestamp) -> None:
    """Assert no statpers >= decision_date leaks into the as-of view."""
    av = panel.as_of(decision_date)
    if not av.empty and (av["statpers"] >= pd.Timestamp(decision_date)).any():
        raise PITViolation(
            f"IBES leak: statpers >= decision_date {decision_date} in as_of view.")
