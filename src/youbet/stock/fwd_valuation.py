"""PIT-safe trailing/derived enterprise-multiple valuation for individual
stocks. (workflows/individual-stocks-snp500 Phase 1.)

All metrics are reconstructed point-in-time from SEC EDGAR XBRL facts
(`fundamentals.compute_fundamentals`, which asserts `filed < decision_date`
in its panel layer) plus the PIT market cap supplied by the backtester's
`panel["mcaps"]`. **No analyst / forward-estimate inputs** — every input is
an as-reported actual, so these signals are confirmatory-eligible (unlike
forward P/E or consensus revisions, which are not free-PIT-backtestable; see
`workflows/individual-stocks-snp500/research/data-availability.md`).

Enterprise value:
    net_debt = long_term_debt + short_term_debt - cash
    EV       = market_cap + net_debt
EBITDA (the enterprise-multiple numerator, Loughran-Wellman 2011):
    EBITDA   = ttm_operating_income + ttm_dep_amort   (D&A added back)
Yields (higher = cheaper = stronger buy, so they sort like a value signal):
    ebitda_yield = EBITDA / EV
    ebit_yield   = ttm_operating_income / EV   (D&A-free fallback)
    earnings_yield = ttm_net_income / market_cap

The headline core strategy is `EVEBITDAYield` — the one enterprise-multiple
construction the `stock-selection` workflow did NOT run. `EarningsYieldV2`
is the admitted ~95% clone of stock-selection's `value_EY` (+0.351, failed
Holm) and is DESCRIPTIVE-only per the design (it cannot independently
confirm anything the prior already tested).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from youbet.stock.fundamentals import compute_fundamentals
from youbet.stock.strategies.base import CrossSectionalStrategy

logger = logging.getLogger(__name__)


def _finite_pos(x: float | None) -> float | None:
    """Return float(x) if finite, else None. Does NOT enforce sign."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if np.isfinite(xf) else None


def net_debt_pit(f: dict) -> float | None:
    """Net debt = long_term_debt + short_term_debt - cash, PIT.

    Missing debt legs are treated as 0 (a firm that never reports
    short-term debt has none); missing cash is also treated as 0. Returns
    None only if ALL three components are missing (no balance sheet).
    """
    ltd = _finite_pos(f.get("long_term_debt"))
    std = _finite_pos(f.get("short_term_debt"))
    cash = _finite_pos(f.get("cash"))
    if ltd is None and std is None and cash is None:
        return None
    return (ltd or 0.0) + (std or 0.0) - (cash or 0.0)


def enterprise_value_pit(f: dict, market_cap: float | None) -> float | None:
    """EV = market_cap + net_debt, PIT. None if mcap missing/non-positive."""
    mcap = _finite_pos(market_cap)
    if mcap is None or mcap <= 0:
        return None
    nd = net_debt_pit(f)
    if nd is None:
        return None
    ev = mcap + nd
    return ev if ev > 0 else None


def ebitda_pit(f: dict) -> float | None:
    """EBITDA = ttm_operating_income + ttm_dep_amort, PIT.

    Requires BOTH operating income AND D&A — returns None if either is
    missing. This keeps "EBITDA" honest: a name with no D&A filing is NOT
    silently treated as EBITDA≈EBIT here; instead `EVEBITDAYield.score`
    routes it through the explicit `ebit_yield` fallback so the EBITDA-vs-
    EBIT distinction is visible in the coverage accounting.
    """
    ebit = _finite_pos(f.get("ttm_operating_income"))
    da = _finite_pos(f.get("ttm_dep_amort"))
    if ebit is None or da is None:
        return None
    return ebit + da


def ebitda_yield_pit(f: dict, market_cap: float | None) -> float | None:
    """EBITDA / EV. Higher = cheaper. None if either input unavailable."""
    ev = enterprise_value_pit(f, market_cap)
    ebitda = ebitda_pit(f)
    if ev is None or ebitda is None:
        return None
    return ebitda / ev


def ebit_yield_pit(f: dict, market_cap: float | None) -> float | None:
    """OperatingIncome / EV (D&A-free fallback). Higher = cheaper."""
    ev = enterprise_value_pit(f, market_cap)
    ebit = _finite_pos(f.get("ttm_operating_income"))
    if ev is None or ebit is None:
        return None
    return ebit / ev


def earnings_yield_pit(f: dict, market_cap: float | None) -> float | None:
    """ttm_net_income / market_cap (the value_EY estimand). Higher = cheaper."""
    mcap = _finite_pos(market_cap)
    ni = _finite_pos(f.get("ttm_net_income"))
    if mcap is None or mcap <= 0 or ni is None:
        return None
    return ni / mcap


def trailing_pe_pit(f: dict, market_cap: float | None) -> float | None:
    """Trailing P/E = market_cap / ttm_net_income. Only meaningful for
    positive earnings (negative P/E is not rankable as 'cheap'); returns
    None for non-positive earnings so the caller drops loss-makers rather
    than ranking them as expensive."""
    mcap = _finite_pos(market_cap)
    ni = _finite_pos(f.get("ttm_net_income"))
    if mcap is None or mcap <= 0 or ni is None or ni <= 0:
        return None
    return mcap / ni


class EVEBITDAYield(CrossSectionalStrategy):
    """CORE pre-registered strategy: rank active S&P 500 names by PIT EBITDA
    yield (EBITDA/EV); top decile, equal-weight, monthly rebalance.

    When D&A is unavailable for a name, falls back to EBIT yield
    (OperatingIncome/EV) so the name is still ranked on a comparable
    enterprise-multiple basis rather than dropped — `use_ebit_fallback`
    controls this. Names with neither EV nor operating income score NaN and
    are excluded by `top_decile_select`.

    Expected band (in-universe sibling anchor, NOT re-halved literature):
    excess Sharpe +0.05 to +0.35 — `value_EY` realized +0.351 here and
    failed Holm; `quality_value_zsum` realized +0.068. Pre-registered
    expectation: fails the gate.
    """

    def __init__(self, use_ebit_fallback: bool = True):
        self.use_ebit_fallback = use_ebit_fallback

    def score(self, panel: dict) -> pd.Series:
        as_of = panel["as_of_date"]
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker", {})

        scores: dict[str, float] = {}
        n_ebitda = n_ebit_fallback = 0
        for ticker in active:
            facts = facts_by_ticker.get(ticker)
            if facts is None:
                continue
            try:
                f = compute_fundamentals(facts, as_of)
            except Exception as exc:  # noqa: BLE001
                logger.debug("compute_fundamentals failed %s @ %s: %s",
                             ticker, as_of, exc)
                continue
            mcap = mcap_from_panel(panel, ticker, f)
            y = ebitda_yield_pit(f, mcap)
            if y is not None:
                n_ebitda += 1
            elif self.use_ebit_fallback:
                y = ebit_yield_pit(f, mcap)
                if y is not None:
                    n_ebitda += 0
                    n_ebit_fallback += 1
            if y is not None:
                scores[ticker] = y
        logger.info(
            "EVEBITDAYield @ %s: %d scored (%d EBITDA, %d EBIT-fallback) of %d active",
            pd.Timestamp(as_of).date(), len(scores), n_ebitda,
            n_ebit_fallback, len(active),
        )
        return pd.Series(scores, dtype=float)

    @property
    def name(self) -> str:
        return "evebitda_yield"

    @property
    def params(self) -> dict:
        return {**super().params, "use_ebit_fallback": self.use_ebit_fallback}


class EarningsYieldV2(CrossSectionalStrategy):
    """DESCRIPTIVE-only companion: ttm_net_income / market_cap. This is the
    ~95% clone of stock-selection's `value_EY` (which realized +0.351 and
    failed Holm); included only to report its point estimate + CI alongside
    the novel EV/EBITDA construction. NOT a gate-family member (it cannot
    independently confirm what the prior workflow already tested)."""

    def score(self, panel: dict) -> pd.Series:
        as_of = panel["as_of_date"]
        facts_by_ticker = panel.get("facts_by_ticker", {})
        scores: dict[str, float] = {}
        for ticker in panel["active_tickers"]:
            facts = facts_by_ticker.get(ticker)
            if facts is None:
                continue
            try:
                f = compute_fundamentals(facts, as_of)
            except Exception:  # noqa: BLE001
                continue
            y = earnings_yield_pit(f, mcap_from_panel(panel, ticker, f))
            if y is not None:
                scores[ticker] = y
        return pd.Series(scores, dtype=float)

    @property
    def name(self) -> str:
        return "earnings_yield_v2"


def mcap_from_panel(panel: dict, ticker: str, f: dict) -> float | None:
    """PIT market cap = last available price (strictly before as_of) x PIT
    shares outstanding (from EDGAR fundamentals). Falls back to panel['mcaps']
    only if a real shares-based mcap can't be formed.

    This is the CORRECT market cap for EV/EBITDA and earnings yield. Relying
    on panel['mcaps'] alone is unsafe: when the backtester is constructed
    WITHOUT shares_outstanding_by_ticker, `_compute_mcaps` returns price as a
    proxy (not price x shares), which would make EV ~= net_debt and corrupt
    the enterprise multiple.
    """
    shares = _finite_pos(f.get("shares_outstanding"))
    prices = panel.get("prices")
    if shares is not None and shares > 0 and prices is not None \
            and ticker in getattr(prices, "columns", []):
        ser = prices[ticker].dropna()
        if not ser.empty:
            lp = float(ser.iloc[-1])
            if np.isfinite(lp) and lp > 0:
                return lp * shares
    mcaps = panel.get("mcaps")
    if mcaps is not None and ticker in getattr(mcaps, "index", []):
        return _finite_pos(mcaps.get(ticker))
    return None


class PrecomputedScoreStrategy(CrossSectionalStrategy):
    """Fast backtest path: score() is an O(1) lookup into a precomputed
    (date x ticker -> signal) panel, so StockBacktester handles T+1 / costs /
    T-bill / delisting at full speed without per-rebal compute_fundamentals.
    The signal panel must be computed with the SAME PIT logic (see
    experiments/precompute_signals.py)."""

    def __init__(self, lut: pd.DataFrame, name: str,
                 decile_breakpoint: float = 0.10, min_holdings: int = 20,
                 max_holdings: int = 100):
        self._lut = lut                 # index=date, columns=ticker, values=signal
        self._name = name
        self.decile_breakpoint = decile_breakpoint
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        d = pd.Timestamp(panel["as_of_date"])
        idx = self._lut.index
        if d in idx:
            row = self._lut.loc[d]
        else:
            # R1 fix: the precompute keys on month-start trading days, but the
            # backtester's fold-boundary rebal is the fold's test_start (often
            # 1 trading day later) -> exact-match miss -> 100% cash. Snap to the
            # latest panel date <= d within 7 calendar days. PIT-safe: the
            # panel date is strictly before d, so the signal used strictly less
            # data (no leak). Beyond 7 days, refuse (genuine coverage gap).
            prior = idx[idx <= d]
            if len(prior) == 0 or (d - prior[-1]).days > 7:
                return pd.Series(dtype=float)
            row = self._lut.loc[prior[-1]]
        row = row.dropna()
        active = panel["active_tickers"]
        return row[row.index.isin(active)].astype(float)

    @property
    def name(self) -> str:
        return self._name
