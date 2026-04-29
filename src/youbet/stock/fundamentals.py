"""Standardized fundamental metrics from SEC EDGAR XBRL facts.

All metrics are computed as AS-OF a decision date: the result reflects
only filings whose `filed` date was strictly before `decision_date`.
Callers must never pass a decision_date later than the portfolio rebalance
date — doing so leaks restatements.

Supported concepts (XBRL us-gaap names):
  - Income:     NetIncomeLoss, Revenues (or RevenueFromContractWithCustomerExcludingAssessedTax),
                GrossProfit, OperatingIncomeLoss, CostOfRevenue (or CostOfGoodsAndServicesSold)
  - Balance:    Assets, Liabilities, StockholdersEquity, CashAndCashEquivalentsAtCarryingValue,
                LongTermDebt, LongTermDebtNoncurrent, ShortTermDebt, DebtCurrent,
                AssetsCurrent, LiabilitiesCurrent
  - Cash flow:  NetCashProvidedByUsedInOperatingActivities
  - Shares:     CommonStockSharesOutstanding (dei: EntityCommonStockSharesOutstanding)

Each "flow" (income statement / cash flow) metric supports trailing-twelve-month
aggregation. "Stock" (balance sheet) metrics use the latest available filing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from youbet.etf.pit import PITFeatureSeries
from youbet.stock.edgar import pit_concept_series

logger = logging.getLogger(__name__)


# Cache for compute_fundamentals: keyed by (ticker, decision_date_ns).
# Populated on first call; each backtest run processes tens of thousands
# of (ticker, rebal_date) pairs with many repeats (Value + Quality both
# query the same pair). 500k-entry cap protects against unbounded growth
# while comfortably fitting a full-universe Piotroski run (~200k entries
# for 500 tickers × 17 folds × 12 rebals × 2 dates; R4 H-bug: prior 50k
# cap silently saturated on full-universe runs). Clear via _clear_caches()
# before running a new experiment if needed.
_FUNDAMENTALS_CACHE: dict[tuple, dict] = {}
_PIOTROSKI_CACHE: dict[tuple, tuple] = {}
_CACHE_MAX_SIZE = 500_000
_CACHE_WARN_PCT = 0.80
_cache_warned: bool = False


def _maybe_warn_cache_saturation() -> None:
    """Emit a single warning when the cache crosses 80% capacity.

    Silent cache-miss under saturation is the insidious failure — the
    workflow keeps running but its hot path silently reverts to the
    slow per-call union (~550ms/call). This warning gives a single
    breadcrumb at 80% so operators can raise the cap before 100% hit.
    """
    global _cache_warned
    if _cache_warned:
        return
    if len(_FUNDAMENTALS_CACHE) >= int(_CACHE_MAX_SIZE * _CACHE_WARN_PCT):
        logger.warning(
            "compute_fundamentals cache at %.0f%% of %d-entry cap "
            "(%d entries); further inserts will silently no-op and "
            "downstream calls revert to cold path. Consider raising "
            "_CACHE_MAX_SIZE before next run.",
            100 * len(_FUNDAMENTALS_CACHE) / _CACHE_MAX_SIZE,
            _CACHE_MAX_SIZE, len(_FUNDAMENTALS_CACHE),
        )
        _cache_warned = True


def _clear_caches() -> None:
    """Drop memoized fundamentals + Piotroski results. Use between
    independent experiments to avoid stale-facts collisions when facts
    objects are rebuilt."""
    global _cache_warned
    _FUNDAMENTALS_CACHE.clear()
    _PIOTROSKI_CACHE.clear()
    _cache_warned = False


# ---------------------------------------------------------------------------
# Precomputed per-ticker panel — the hot-path speedup.
# ---------------------------------------------------------------------------


@dataclass
class TickerFundamentalsPanel:
    """Per-ticker pre-unioned fact panels, built once at load time.

    For each fundamentals alias, `alias_frames[alias]` is the union of
    all concept-variants' filings (e.g., Revenues ∪
    RevenueFromContractWithCustomer... ∪ SalesRevenueNet for "revenue"),
    sorted by (end, filed). This removes the O(alias × decision_date)
    work that `_pick_first_available` did at every compute_fundamentals
    call — union+sort runs ONCE per ticker.

    At query time, compute_fundamentals_from_panel filters each alias's
    frame by `filed < decision_date`, dedupes per-end (keep latest filed),
    and runs _quarterize on the small result. Each call is a few
    numpy-vectorized filters plus the quarterize call, ~5-10ms vs the
    pre-panel ~220ms.

    The panel is keyed by ticker for stable cache semantics (safer than
    id(facts) across GC pressure).
    """

    ticker: str
    alias_frames: dict[str, pd.DataFrame]  # unioned, sorted by (end, filed)

    @classmethod
    def build(cls, ticker: str, facts) -> "TickerFundamentalsPanel":
        """Build a panel from raw facts. Stores ALL filings (including
        restatement history) so PIT semantics are preserved at query
        time. Pass `facts` as IndexedFacts or a plain DataFrame; the
        former is significantly faster."""
        frames: dict[str, pd.DataFrame] = {}
        for alias in CONCEPT_ALIASES:
            union = _union_alias_frames_all(facts, alias)
            if not union.empty:
                frames[alias] = union
        return cls(ticker=ticker, alias_frames=frames)


def _union_alias_frames_all(facts, alias: str) -> pd.DataFrame:
    """Union of all concept-variant filings for `alias`, WITHOUT per-end
    dedupe — preserves every filing (including restatements) so PIT
    dedupe at query time is correct under any decision_date."""
    frames: list[pd.DataFrame] = []
    for concept in CONCEPT_ALIASES[alias]:
        if alias == "shares_outstanding":
            ser = pit_concept_series_all(
                facts, concept, unit="shares", prefer_taxonomy="dei",
            )
            if ser.empty:
                ser = pit_concept_series_all(
                    facts, concept, unit="shares", prefer_taxonomy="us-gaap",
                )
        else:
            ser = pit_concept_series_all(
                facts, concept, unit="USD", prefer_taxonomy="us-gaap",
            )
        if not ser.empty:
            frames.append(ser)

    if not frames:
        return pd.DataFrame(
            columns=["end", "start", "val", "filed", "form", "fp", "fy"]
        )
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.sort_values(["end", "filed"]).reset_index(drop=True)
    return all_rows


def pit_concept_series_all(
    facts,
    concept: str,
    unit: str = "USD",
    prefer_taxonomy: str = "us-gaap",
) -> pd.DataFrame:
    """Like `edgar.pit_concept_series` but returns ALL filings for the
    concept (no decision-date filter, no per-end dedupe). Used by
    TickerFundamentalsPanel build."""
    from youbet.stock.edgar import IndexedFacts
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
    return df[["end", "start", "val", "filed", "form", "fp", "fy"]].reset_index(drop=True)


def _union_alias_frames(
    facts,
    alias: str,
    decision_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """Inner union routine. Shared between `_pick_first_available`
    (decision-date-filtered) and `TickerFundamentalsPanel.build`
    (unfiltered — build once, filter at query time)."""
    frames: list[pd.DataFrame] = []
    for concept in CONCEPT_ALIASES[alias]:
        if alias == "shares_outstanding":
            ser = pit_concept_series(
                facts, concept, unit="shares",
                decision_date=decision_date, prefer_taxonomy="dei",
            )
            if ser.empty:
                ser = pit_concept_series(
                    facts, concept, unit="shares",
                    decision_date=decision_date, prefer_taxonomy="us-gaap",
                )
        else:
            ser = pit_concept_series(
                facts, concept, unit="USD", decision_date=decision_date,
            )
        if not ser.empty:
            frames.append(ser)

    if not frames:
        return pd.DataFrame(
            columns=["end", "start", "val", "filed", "form", "fp", "fy"]
        )
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.sort_values(["end", "filed"]).drop_duplicates(
        subset=["end"], keep="last"
    ).reset_index(drop=True)
    return all_rows


def _panel_alias_as_of(
    panel: TickerFundamentalsPanel,
    alias: str,
    decision_date: pd.Timestamp,
) -> pd.DataFrame:
    """Query the panel at `decision_date`: filter filings with
    filed < decision_date, then dedupe per `end` (keep latest filed).
    This preserves PIT semantics including restatements — the panel
    stores ALL filings via `_union_alias_frames_all` so the dedupe here
    picks the as-known value at `decision_date`, not the latest-ever
    restatement."""
    frame = panel.alias_frames.get(alias)
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["end", "start", "val", "filed", "form", "fp", "fy"]
        )
    filed_vals = frame["filed"].values
    d64 = np.datetime64(decision_date.to_datetime64(), "ns")
    mask = filed_vals < d64
    if not mask.any():
        return frame.iloc[0:0]
    sub = frame[mask]
    # Per-end dedup: keep latest-filed. Frame is sorted (end, filed), so
    # `drop_duplicates keep="last"` leaves the latest per end within the
    # filtered subset.
    return sub.drop_duplicates(subset=["end"], keep="last")


# XBRL concept aliases: try each in order; first non-empty wins.
# Companies report the same economic quantity under different concept names
# depending on taxonomy version / industry.
CONCEPT_ALIASES: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "cogs": [
        "CostOfRevenue",
        "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "Cash",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
    ],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
}


def _pick_first_available(
    facts: pd.DataFrame,
    alias: str,
    decision_date: pd.Timestamp | None,
) -> pd.DataFrame:
    """Union of all alias concepts, deduped to one row per fiscal period end.

    Companies rotate XBRL concepts as the taxonomy evolves: e.g., AAPL
    reported revenue under `SalesRevenueNet` (2007-2018), `Revenues`
    (2018 only), and `RevenueFromContractWithCustomerExcludingAssessedTax`
    (2019-present). A first-match shortcircuit would silently lock onto
    an obsolete concept and miss all later filings. Instead, concatenate
    every alias's PIT series, sort by (end, filed), and keep the most
    recent filing per fiscal period end — same PIT semantics as
    `edgar.pit_concept_series` but across aliases.
    """
    frames: list[pd.DataFrame] = []
    for concept in CONCEPT_ALIASES[alias]:
        if alias == "shares_outstanding":
            ser = pit_concept_series(
                facts, concept, unit="shares",
                decision_date=decision_date, prefer_taxonomy="dei",
            )
            if ser.empty:
                ser = pit_concept_series(
                    facts, concept, unit="shares",
                    decision_date=decision_date, prefer_taxonomy="us-gaap",
                )
        else:
            ser = pit_concept_series(
                facts, concept, unit="USD", decision_date=decision_date,
            )
        if not ser.empty:
            frames.append(ser)

    if not frames:
        return pd.DataFrame(
            columns=["end", "start", "val", "filed", "form", "fp", "fy"]
        )
    all_rows = pd.concat(frames, ignore_index=True)
    # Keep most-recent-filed value per fiscal period end (PIT safe).
    all_rows = all_rows.sort_values(["end", "filed"]).drop_duplicates(
        subset=["end"], keep="last"
    ).reset_index(drop=True)
    return all_rows


def _quarterize(series: pd.DataFrame) -> pd.DataFrame:
    """Convert filings to pure-quarterly, handling three reporting patterns.

    Key observation: a filer's Q1 10-Q always has start == fiscal-year
    start and period_days ~90. This Q1 row has identical shape for both
    AAPL-style filers and YTD-only legacy filers. The pattern diverges
    starting at Q2:

    **AAPL pattern**: Q2/Q3 10-Qs have start == prior-quarter-end (not
    fiscal-year start), period_days ~90. Q4 isn't reported standalone;
    the 10-K gives the annual. 3 pure-quarterly rows + 1 annual → derive
    Q4 residual.

    **YTD-only pattern** (H-R2-2 fix): Q2/Q3 10-Qs have start ==
    fiscal-year start (cumulative from beginning), period_days ~180/~270.
    Derive each quarter by differencing consecutive cumulative values
    plus the annual.

    **All-four-pure pattern**: all quarters as pure 10-Qs (rare; usually
    there's still a 10-K annual too).

    Mixed or incomplete patterns are dropped — conservative, with `None`
    TTM for affected decision dates.

    Perf: numpy-vectorized to avoid pandas' per-cell overhead in the
    annual loop. The previous pandas-native version cost ~82ms per
    _quarterize call due to ~70 boolean-filter DataFrame constructions;
    this version is 10-20× faster.
    """
    if series.empty:
        return series

    # Materialize numpy arrays ONCE. The hot path reads these, never touches
    # pandas row-indexing (which was ~80% of the profile before this pass).
    df = series.sort_values(["end", "filed"]).reset_index(drop=True)
    ends = df["end"].values.astype("datetime64[D]")
    starts = df["start"].values.astype("datetime64[D]")
    vals = df["val"].values.astype("float64")
    period_days = (ends - starts).astype("int64")

    # Snapshot non-numeric columns as python lists (cheaper to index than df.iloc).
    filed = df["filed"].values
    forms = df["form"].tolist()
    fps = df.get("fp", pd.Series([None] * len(df))).tolist()
    fys = df.get("fy", pd.Series([None] * len(df))).tolist()

    is_annual = period_days > 300
    is_pure = (period_days >= 60) & (period_days <= 100)
    is_cum_mid = (period_days >= 101) & (period_days <= 300)

    annual_idx = np.flatnonzero(is_annual)

    # Accumulate output as parallel arrays (cheaper than list-of-dicts).
    out_end: list = []
    out_start: list = []
    out_val: list = []
    out_filed: list = []
    out_form: list = []
    out_fp: list = []
    out_fy: list = []
    claimed: set[int] = set()

    def _emit(i: int) -> None:
        out_end.append(ends[i])
        out_start.append(starts[i])
        out_val.append(vals[i])
        out_filed.append(filed[i])
        out_form.append(forms[i])
        out_fp.append(fps[i])
        out_fy.append(fys[i])

    def _emit_synthetic(end, start, val, prov_i: int) -> None:
        out_end.append(end)
        out_start.append(start)
        out_val.append(val)
        out_filed.append(filed[prov_i])
        out_form.append(forms[prov_i])
        out_fp.append(fps[prov_i])
        out_fy.append(fys[prov_i])

    for a_i in annual_idx:
        a_start = starts[a_i]
        a_end = ends[a_i]
        if a_start != a_start or a_end != a_end:  # NaT check
            continue

        start_at_astart = (starts == a_start)
        start_after_astart = (starts > a_start)

        q1_mask = is_pure & start_at_astart & (ends < a_end)
        q1_indices = np.flatnonzero(q1_mask)
        pure_mask = is_pure & start_after_astart & (ends <= a_end)
        pure_indices = np.flatnonzero(pure_mask)
        cum_mask = is_cum_mid & start_at_astart & (ends < a_end)
        cum_indices = np.flatnonzero(cum_mask)

        if len(q1_indices) != 1:
            continue
        q1 = int(q1_indices[0])
        npp = len(pure_indices)
        ncum = len(cum_indices)

        if npp == 2 and ncum == 0:
            sorted_pure = pure_indices[np.argsort(ends[pure_indices])]
            _emit(q1); claimed.add(q1)
            for j in sorted_pure:
                _emit(int(j)); claimed.add(int(j))
            sum3 = vals[q1] + vals[sorted_pure].sum()
            q4_val = vals[a_i] - sum3
            q4_start = ends[sorted_pure].max()
            _emit_synthetic(a_end, q4_start, q4_val, int(a_i))
        elif npp == 0 and ncum == 2:
            sorted_cum = cum_indices[np.argsort(ends[cum_indices])]
            h1, nine = int(sorted_cum[0]), int(sorted_cum[1])
            _emit(q1); claimed.add(q1)
            _emit_synthetic(ends[h1], ends[q1], vals[h1] - vals[q1], h1)
            _emit_synthetic(ends[nine], ends[h1], vals[nine] - vals[h1], nine)
            _emit_synthetic(a_end, ends[nine], vals[a_i] - vals[nine], int(a_i))
        elif npp == 3 and ncum == 0:
            sorted_pure = pure_indices[np.argsort(ends[pure_indices])]
            _emit(q1); claimed.add(q1)
            for j in sorted_pure:
                _emit(int(j)); claimed.add(int(j))

    # Orphan pure-quarterly AFTER the latest annual.
    if annual_idx.size > 0:
        latest_annual_end = ends[annual_idx].max()
        orphan_mask = is_pure & (ends > latest_annual_end)
    else:
        orphan_mask = is_pure
    for j in np.flatnonzero(orphan_mask):
        ji = int(j)
        if ji in claimed:
            continue
        _emit(ji)

    if not out_end:
        return df.iloc[0:0][["end", "start", "val", "filed", "form", "fp", "fy"]]

    out = pd.DataFrame({
        "end": out_end, "start": out_start, "val": out_val,
        "filed": out_filed, "form": out_form, "fp": out_fp, "fy": out_fy,
    })
    out = out.sort_values(["end", "filed"]).drop_duplicates(
        subset=["end"], keep="last"
    ).reset_index(drop=True)
    return out


def _row_from_arrays(df: pd.DataFrame, i: int) -> dict:
    row = df.iloc[i]
    return {
        "end": row["end"], "start": row["start"], "val": row["val"],
        "filed": row["filed"], "form": row["form"],
        "fp": row.get("fp"), "fy": row.get("fy"),
    }


def _synthetic_row(end, start, val, df: pd.DataFrame, provenance_i: int) -> dict:
    """Emit a derived quarter row carrying the filing metadata of its
    source (annual for residual Q4, YTD cumulative for derived quarters)."""
    prov = df.iloc[provenance_i]
    return {
        "end": pd.Timestamp(end), "start": pd.Timestamp(start),
        "val": float(val),
        "filed": prov["filed"], "form": prov["form"],
        "fp": prov.get("fp"), "fy": prov.get("fy"),
    }


def ttm_sum(
    facts: pd.DataFrame,
    alias: str,
    decision_date: pd.Timestamp,
    min_quarters: int = 4,
) -> float | None:
    """Trailing-twelve-month sum of a flow metric, as known at decision_date.

    Returns None if fewer than `min_quarters` of quarterly data are available.
    """
    d = pd.Timestamp(decision_date)
    raw = _pick_first_available(facts, alias, d)
    if raw.empty:
        return None

    quarterly = _quarterize(raw)
    if quarterly.empty:
        return None

    # Take the 4 most recent quarters (by end date, filed already < d)
    recent = quarterly.sort_values("end").tail(min_quarters)
    if len(recent) < min_quarters:
        return None
    return float(recent["val"].sum())


def build_pit_shares_series_from_panel(
    panel: TickerFundamentalsPanel,
) -> pd.Series:
    """Build a `filed_date`-indexed series of shares outstanding.

    R9-HIGH-2 helper: Phase 4b's `turn_22d` feature requires PIT shares
    indexed by AVAILABILITY date (i.e. when the value was filed), not by
    fiscal period end. Returns a Series sorted ascending by filed.

    Used by orchestrators that pass `shares_outstanding_by_ticker={ticker:
    series}` to StockBacktester. The downstream `_compute_mcaps` and
    `_volume_features_for_ticker` both filter `series.index < decision_date`
    and take the latest, so the index MUST be filed_date for PIT safety.
    """
    frames = panel.alias_frames.get("shares_outstanding")
    if frames is None or frames.empty:
        return pd.Series(dtype=float, name="shares_outstanding")
    df = frames.copy()
    df["filed"] = pd.to_datetime(df["filed"])
    df = df.sort_values("filed").drop_duplicates(subset=["filed"], keep="last")
    return pd.Series(
        df["val"].values,
        index=pd.DatetimeIndex(df["filed"], name="filed"),
        name="shares_outstanding",
    )


def latest_stock(
    facts: pd.DataFrame,
    alias: str,
    decision_date: pd.Timestamp,
) -> float | None:
    """Latest balance-sheet value as known at decision_date.

    Returns None if no filing has reported this concept yet.
    """
    d = pd.Timestamp(decision_date)
    ser = _pick_first_available(facts, alias, d)
    if ser.empty:
        return None
    return float(ser.sort_values("end").iloc[-1]["val"])


def latest_filed_date(
    facts: pd.DataFrame,
    alias: str,
    decision_date: pd.Timestamp,
) -> pd.Timestamp | None:
    """Most recent filing date for this concept, <= decision_date."""
    d = pd.Timestamp(decision_date)
    ser = _pick_first_available(facts, alias, d)
    if ser.empty:
        return None
    return pd.Timestamp(ser.sort_values("filed").iloc[-1]["filed"])


def compute_fundamentals_from_panel(
    panel: TickerFundamentalsPanel,
    decision_date: pd.Timestamp,
) -> dict[str, float | None]:
    """Fast path: same output as `compute_fundamentals`, but uses
    pre-unioned per-alias frames. ~5-20ms per call vs ~220ms unpanelized.

    Memoizes by (panel.ticker, decision_date_ns) — safer than id(facts)
    under GC pressure.
    """
    d = pd.Timestamp(decision_date)
    cache_key = (panel.ticker, d.value, "panel")
    cached = _FUNDAMENTALS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    none_dict = {k: None for k in (
        "ttm_net_income", "ttm_revenue", "ttm_gross_profit",
        "ttm_operating_income", "ttm_operating_cash_flow",
        "total_assets", "total_liabilities", "stockholders_equity",
        "cash", "long_term_debt", "short_term_debt",
        "current_assets", "current_liabilities", "shares_outstanding",
        "gross_margin_ttm", "net_margin_ttm", "operating_margin_ttm",
        "roe_ttm", "roa_ttm", "debt_to_equity", "current_ratio",
        "earnings_yield_on_assets", "accruals_ratio",
        "latest_filing_date",
    )}

    if not panel.alias_frames:
        if len(_FUNDAMENTALS_CACHE) < _CACHE_MAX_SIZE:
            _FUNDAMENTALS_CACHE[cache_key] = none_dict
        return none_dict

    out: dict[str, float | None] = {}

    def _ttm(alias: str) -> float | None:
        sub = _panel_alias_as_of(panel, alias, d)
        if sub.empty:
            return None
        q = _quarterize(sub)
        if len(q) < 4:
            return None
        return float(q["val"].tail(4).sum())

    def _latest(alias: str) -> float | None:
        sub = _panel_alias_as_of(panel, alias, d)
        if sub.empty:
            return None
        return float(sub.iloc[-1]["val"])

    out["ttm_net_income"] = _ttm("net_income")
    out["ttm_revenue"] = _ttm("revenue")
    out["ttm_gross_profit"] = _ttm("gross_profit")
    out["ttm_operating_income"] = _ttm("operating_income")
    out["ttm_operating_cash_flow"] = _ttm("operating_cash_flow")

    out["total_assets"] = _latest("assets")
    out["total_liabilities"] = _latest("liabilities")
    out["stockholders_equity"] = _latest("equity")
    out["cash"] = _latest("cash")
    out["long_term_debt"] = _latest("long_term_debt")
    out["short_term_debt"] = _latest("short_term_debt")
    out["current_assets"] = _latest("assets_current")
    out["current_liabilities"] = _latest("liabilities_current")
    out["shares_outstanding"] = _latest("shares_outstanding")

    # Derived ratios (same as the non-panel path)
    out["gross_margin_ttm"] = _safe_ratio(out["ttm_gross_profit"], out["ttm_revenue"])
    out["net_margin_ttm"] = _safe_ratio(out["ttm_net_income"], out["ttm_revenue"])
    out["operating_margin_ttm"] = _safe_ratio(out["ttm_operating_income"], out["ttm_revenue"])
    out["roe_ttm"] = _safe_ratio(out["ttm_net_income"], out["stockholders_equity"])
    out["roa_ttm"] = _safe_ratio(out["ttm_net_income"], out["total_assets"])
    total_debt = _safe_sum(out["long_term_debt"], out["short_term_debt"])
    out["debt_to_equity"] = _safe_ratio(total_debt, out["stockholders_equity"])
    out["current_ratio"] = _safe_ratio(out["current_assets"], out["current_liabilities"])
    out["earnings_yield_on_assets"] = _safe_ratio(
        out["ttm_operating_income"], out["total_assets"]
    )
    out["accruals_ratio"] = _safe_ratio(
        _safe_diff(out["ttm_net_income"], out["ttm_operating_cash_flow"]),
        out["total_assets"],
    )

    ni_slice = _panel_alias_as_of(panel, "net_income", d)
    out["latest_filing_date"] = (
        pd.Timestamp(ni_slice.iloc[-1]["filed"]) if not ni_slice.empty else None
    )

    _maybe_warn_cache_saturation()
    if len(_FUNDAMENTALS_CACHE) < _CACHE_MAX_SIZE:
        _FUNDAMENTALS_CACHE[cache_key] = out
    return out


def compute_fundamentals(
    facts,
    decision_date: pd.Timestamp,
) -> dict[str, float | None]:
    """Compute all standardized fundamental metrics for one ticker × decision_date.

    Returns dict with None for missing metrics (never silently zero).

    Returned keys:
        ttm_net_income, ttm_revenue, ttm_gross_profit, ttm_operating_income,
        ttm_operating_cash_flow,
        total_assets, total_liabilities, stockholders_equity, cash,
        long_term_debt, short_term_debt, current_assets, current_liabilities,
        shares_outstanding,
        gross_margin_ttm, net_margin_ttm, operating_margin_ttm,
        roe_ttm, roa_ttm, debt_to_equity, current_ratio,
        earnings_yield_on_assets, accruals_ratio,
        latest_filing_date (datetime or None)
    """
    # Fast-path dispatch: if the caller passed a panel, use the
    # pre-computed path (~10× faster). Otherwise fall back to the
    # per-call union path.
    if isinstance(facts, TickerFundamentalsPanel):
        return compute_fundamentals_from_panel(facts, decision_date)

    d = pd.Timestamp(decision_date)

    # Memoize by (facts identity, decision_date_ns). A Phase 1 run
    # computes the same (ticker, rebal_date) many times (Value + Quality
    # + Piotroski all touch it), and backtester rebal-grids share dates
    # across strategies. Caching drops the per-call cost from ~330ms to
    # a hash lookup after the first touch.
    cache_key = (id(facts), d.value)
    cached = _FUNDAMENTALS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Early exit: no filings before decision_date means every metric will be
    # None. Saves ~14 concept-alias scans per ticker per rebalance during
    # pre-XBRL decision dates. `facts` may be a plain DataFrame or an
    # IndexedFacts wrapper — the filed-column accessor handles both.
    filed_series = facts["filed"]
    if facts.empty or not (filed_series < d).any():
        result = {k: None for k in (
            "ttm_net_income", "ttm_revenue", "ttm_gross_profit",
            "ttm_operating_income", "ttm_operating_cash_flow",
            "total_assets", "total_liabilities", "stockholders_equity",
            "cash", "long_term_debt", "short_term_debt",
            "current_assets", "current_liabilities", "shares_outstanding",
            "gross_margin_ttm", "net_margin_ttm", "operating_margin_ttm",
            "roe_ttm", "roa_ttm", "debt_to_equity", "current_ratio",
            "earnings_yield_on_assets", "accruals_ratio",
            "latest_filing_date",
        )}
        if len(_FUNDAMENTALS_CACHE) < _CACHE_MAX_SIZE:
            _FUNDAMENTALS_CACHE[cache_key] = result
        return result

    out: dict[str, float | None] = {}
    out["ttm_net_income"] = ttm_sum(facts, "net_income", d)
    out["ttm_revenue"] = ttm_sum(facts, "revenue", d)
    out["ttm_gross_profit"] = ttm_sum(facts, "gross_profit", d)
    out["ttm_operating_income"] = ttm_sum(facts, "operating_income", d)
    out["ttm_operating_cash_flow"] = ttm_sum(facts, "operating_cash_flow", d)

    out["total_assets"] = latest_stock(facts, "assets", d)
    out["total_liabilities"] = latest_stock(facts, "liabilities", d)
    out["stockholders_equity"] = latest_stock(facts, "equity", d)
    out["cash"] = latest_stock(facts, "cash", d)
    out["long_term_debt"] = latest_stock(facts, "long_term_debt", d)
    out["short_term_debt"] = latest_stock(facts, "short_term_debt", d)
    out["current_assets"] = latest_stock(facts, "assets_current", d)
    out["current_liabilities"] = latest_stock(facts, "liabilities_current", d)
    out["shares_outstanding"] = latest_stock(facts, "shares_outstanding", d)

    # Derived ratios
    out["gross_margin_ttm"] = _safe_ratio(out["ttm_gross_profit"], out["ttm_revenue"])
    out["net_margin_ttm"] = _safe_ratio(out["ttm_net_income"], out["ttm_revenue"])
    out["operating_margin_ttm"] = _safe_ratio(out["ttm_operating_income"], out["ttm_revenue"])
    out["roe_ttm"] = _safe_ratio(out["ttm_net_income"], out["stockholders_equity"])
    out["roa_ttm"] = _safe_ratio(out["ttm_net_income"], out["total_assets"])
    total_debt = _safe_sum(out["long_term_debt"], out["short_term_debt"])
    out["debt_to_equity"] = _safe_ratio(total_debt, out["stockholders_equity"])
    out["current_ratio"] = _safe_ratio(out["current_assets"], out["current_liabilities"])
    out["earnings_yield_on_assets"] = _safe_ratio(
        out["ttm_operating_income"], out["total_assets"]
    )
    out["accruals_ratio"] = _safe_ratio(
        _safe_diff(out["ttm_net_income"], out["ttm_operating_cash_flow"]),
        out["total_assets"],
    )

    out["latest_filing_date"] = latest_filed_date(facts, "net_income", d)

    if len(_FUNDAMENTALS_CACHE) < _CACHE_MAX_SIZE:
        _FUNDAMENTALS_CACHE[cache_key] = out
    return out


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    if den == 0 or not np.isfinite(den):
        return None
    return float(num) / float(den)


def _safe_sum(*vals: float | None) -> float | None:
    nonnull = [v for v in vals if v is not None]
    if not nonnull:
        return None
    return float(sum(nonnull))


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def piotroski_f_score(
    facts: pd.DataFrame,
    decision_date: pd.Timestamp,
) -> tuple[int | None, dict[str, int | None]]:
    """9-point Piotroski F-Score (Piotroski 2000).

    Each criterion contributes 0 or 1. Higher is better. Requires a
    preceding-year comparison: "is ROA higher than last year?" etc.

    Returns (total_score, component_dict). total_score is None if any
    critical inputs are unavailable.
    """
    d = pd.Timestamp(decision_date)

    # Current TTM
    cur = compute_fundamentals(facts, d)

    # Prior year: decision_date - 365 days (close enough for the Piotroski
    # trend tests — precision better than annual is irrelevant here since
    # trends are coded 0/1).
    prior_date = d - pd.Timedelta(days=365)
    prior = compute_fundamentals(facts, prior_date)

    components: dict[str, int | None] = {}

    # Profitability
    components["p1_roa_positive"] = _bin(cur["roa_ttm"], lambda x: x > 0)
    components["p2_ocf_positive"] = _bin(cur["ttm_operating_cash_flow"], lambda x: x > 0)
    components["p3_roa_increasing"] = _bin_pair(
        cur["roa_ttm"], prior["roa_ttm"], lambda c, p: c > p
    )
    components["p4_accruals"] = _bin_pair(
        cur["ttm_operating_cash_flow"], cur["ttm_net_income"],
        lambda cf, ni: cf > ni,
    )

    # Leverage / Liquidity
    components["p5_leverage_decreased"] = _bin_pair(
        cur["debt_to_equity"], prior["debt_to_equity"],
        lambda c, p: c < p,
    )
    components["p6_current_ratio_increased"] = _bin_pair(
        cur["current_ratio"], prior["current_ratio"],
        lambda c, p: c > p,
    )
    cur_shares = cur.get("shares_outstanding")
    prior_shares = prior.get("shares_outstanding")
    components["p7_no_dilution"] = _bin_pair(
        cur_shares, prior_shares, lambda c, p: c <= p,
    )

    # Operating efficiency
    components["p8_gross_margin_increased"] = _bin_pair(
        cur["gross_margin_ttm"], prior["gross_margin_ttm"],
        lambda c, p: c > p,
    )
    # Asset turnover: revenue / avg total assets (using current proxy)
    cur_turnover = _safe_ratio(cur["ttm_revenue"], cur["total_assets"])
    prior_turnover = _safe_ratio(prior["ttm_revenue"], prior["total_assets"])
    components["p9_asset_turnover_increased"] = _bin_pair(
        cur_turnover, prior_turnover, lambda c, p: c > p,
    )

    # Any None component disqualifies the score
    if any(v is None for v in components.values()):
        return None, components
    return int(sum(components.values())), components


def _bin(val: float | None, cond) -> int | None:
    if val is None:
        return None
    return int(bool(cond(val)))


def _bin_pair(a: float | None, b: float | None, cond) -> int | None:
    if a is None or b is None:
        return None
    return int(bool(cond(a, b)))


def build_fundamentals_panel(
    facts_by_ticker: dict[str, pd.DataFrame],
    decision_date: pd.Timestamp,
) -> pd.DataFrame:
    """Cross-sectional fundamentals at one decision date for many tickers.

    Each row is one ticker; columns are the metrics from compute_fundamentals
    plus the Piotroski score.
    """
    d = pd.Timestamp(decision_date)
    rows = []
    for ticker, facts in facts_by_ticker.items():
        f = compute_fundamentals(facts, d)
        score, components = piotroski_f_score(facts, d)
        f["ticker"] = ticker
        f["piotroski_f"] = score
        for k, v in components.items():
            f[f"piotroski_{k}"] = v
        rows.append(f)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("ticker")
    return df
