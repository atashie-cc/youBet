"""Composite factor strategies (Phase 3).

Three Phase-3 confirmatory strategies:

- **PiotroskiF**: score each ticker by its 9-point Piotroski F-score
  (Piotroski 2000). Top decile buys high-F names; the score blends
  profitability, leverage/liquidity, and operating-efficiency signals.

- **MagicFormula**: Greenblatt's two-rank composite — rank by earnings
  yield (EY = TTM operating income / enterprise value proxy ≈ total
  assets) AND return on invested capital (ROIC ≈ TTM operating income
  / (total assets − cash)). Sum the two ranks; top rank = best.

- **QualityValue**: a simple z-score composite of (ROE, gross margin,
  earnings yield). Equal-weighted across standardised signals per
  rebalance date (stateless per-date standardisation — no fit window).

All three consume fundamentals via `compute_fundamentals` (which accepts
either `IndexedFacts` or a `TickerFundamentalsPanel` — strategies are
agnostic; Phase 1's orchestrator feeds panels for perf).

Note on universe and coverage: tickers whose fundamentals are `None`
(pre-XBRL, insufficient history, concept-coverage gaps) drop out of the
cross-sectional ranking naturally. Strategies do not fabricate scores
from incomplete data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from youbet.stock.fundamentals import compute_fundamentals, piotroski_f_score
from youbet.stock.strategies.base import CrossSectionalStrategy

logger = logging.getLogger(__name__)


def _earnings_yield_from_mcap(
    ttm_ni: float | None,
    ticker: str,
    panel_mcaps,
    last_prices,
    shares: float | None,
    _warn: dict,
) -> float | None:
    """TTM net income / market cap, using the backtester's centrally-computed
    panel['mcaps'] (raw split-unadjusted price basis when raw_prices were
    supplied). Falls back to adjusted price × shares with a one-time warning when
    mcaps is unavailable. Returns None if no valid mcap or earnings.

    Market-cap split-adjust fix (2026-05-30): pairing adjusted price with EDGAR
    as-reported shares understates mcap by the split factor and inflates the
    earnings-yield (value) leg on high-split names.
    """
    if ttm_ni is None or not np.isfinite(ttm_ni):
        return None
    mcap = None
    if panel_mcaps is not None and ticker in getattr(panel_mcaps, "index", ()):
        m = panel_mcaps.get(ticker)
        if m is not None and np.isfinite(m) and m > 0:
            mcap = float(m)
    if mcap is None:
        price = last_prices.get(ticker) if last_prices is not None else None
        if shares is None or shares <= 0 or price is None \
                or not np.isfinite(price) or price <= 0:
            return None
        if not _warn.get("done"):
            logger.warning(
                "composite value leg: panel['mcaps'] missing %s — falling back "
                "to ADJUSTED price × shares (contaminated). Supply raw_prices to "
                "the backtester.", ticker)
            _warn["done"] = True
        mcap = float(price) * float(shares)
    return float(ttm_ni) / mcap


class PiotroskiF(CrossSectionalStrategy):
    """Top-F-score long-only strategy.

    Score range: integer 0-9, higher is better. `min_f` threshold (default
    7) is applied before decile selection — tickers below threshold get
    NaN scores and drop out.
    """

    def __init__(
        self,
        min_f: int = 7,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.min_f = int(min_f)
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]

        scores: dict[str, float] = {}
        for ticker in active:
            facts = facts_by_ticker.get(ticker)
            if facts is None:
                continue
            f_score, _ = piotroski_f_score(facts, decision_date)
            if f_score is None:
                continue
            if f_score < self.min_f:
                continue
            scores[ticker] = float(f_score)
        return pd.Series(scores)

    @property
    def name(self) -> str:
        return f"piotroski_f_min{self.min_f}"

    @property
    def params(self) -> dict:
        return {**super().params, "min_f": self.min_f}


class MagicFormula(CrossSectionalStrategy):
    """Greenblatt Magic Formula.

    Signals:
      - Earnings yield (EY) = TTM operating income / total assets
        (approximation of EBIT/EV; uses assets as EV proxy).
      - Return on invested capital (ROIC) = TTM operating income /
        (total assets − cash).

    Each ticker gets two ranks (1 = best); the strategy selects the
    lowest rank-sum (equal weight).
    """

    def __init__(
        self,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]

        ey_by_ticker: dict[str, float] = {}
        roic_by_ticker: dict[str, float] = {}
        for ticker in active:
            facts = facts_by_ticker.get(ticker)
            if facts is None:
                continue
            f = compute_fundamentals(facts, decision_date)
            op_income = f.get("ttm_operating_income")
            assets = f.get("total_assets")
            cash = f.get("cash")
            if op_income is None or assets is None or assets <= 0:
                continue
            invested = assets - (cash if cash is not None else 0.0)
            if invested <= 0:
                continue
            ey_by_ticker[ticker] = float(op_income) / float(assets)
            roic_by_ticker[ticker] = float(op_income) / float(invested)

        if not ey_by_ticker:
            return pd.Series(dtype=float)

        ey = pd.Series(ey_by_ticker)
        roic = pd.Series(roic_by_ticker)
        # Rank descending (best = rank 1). Use 'min' ties so equal values
        # share the same rank; subsequent tickers skip ahead.
        ey_rank = ey.rank(ascending=False, method="min")
        roic_rank = roic.rank(ascending=False, method="min")
        common = ey_rank.index.intersection(roic_rank.index)
        rank_sum = ey_rank.loc[common] + roic_rank.loc[common]
        # Return negated rank-sum so "higher score = better" convention holds
        # and top-decile selection picks the lowest rank-sums.
        return -rank_sum

    @property
    def name(self) -> str:
        return "magic_formula"


class ValueProfitability(CrossSectionalStrategy):
    """AQR QMJ-lite per Codex R8: value × Novy-Marx profitability composite.

    Equal-weighted z-score sum of:
      - Earnings yield (TTM NI / mcap) — value
      - Gross profitability (TTM gross profit / total assets) — Novy-Marx 2013

    Phase 7 step 1 composite. Per-date cross-sectional standardization.
    """

    def __init__(
        self,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]
        prices = panel["prices"]
        if prices.empty:
            return pd.Series(dtype=float)
        last_prices = prices.ffill().iloc[-1]
        mcaps = panel.get("mcaps")
        _warn: dict = {}

        ey_by: dict[str, float] = {}
        gpa_by: dict[str, float] = {}
        for ticker in active:
            facts = facts_by_ticker.get(ticker)
            if facts is None:
                continue
            f = compute_fundamentals(facts, decision_date)
            ttm_ni = f.get("ttm_net_income")
            shares = f.get("shares_outstanding")
            gp = f.get("ttm_gross_profit")
            assets = f.get("total_assets")
            ey = _earnings_yield_from_mcap(
                ttm_ni, ticker, mcaps, last_prices, shares, _warn)
            if ey is not None:
                ey_by[ticker] = ey
            if (
                gp is not None and assets is not None
                and np.isfinite(gp) and np.isfinite(assets) and assets > 0
            ):
                gpa_by[ticker] = float(gp) / float(assets)

        if not ey_by and not gpa_by:
            return pd.Series(dtype=float)

        def _zscore(d: dict[str, float]) -> pd.Series:
            s = pd.Series(d)
            mu = s.mean()
            sd = s.std()
            if sd <= 0 or not np.isfinite(sd):
                return pd.Series(0.0, index=s.index)
            return (s - mu) / sd

        z_ey = _zscore(ey_by)
        z_gpa = _zscore(gpa_by)

        # R9-MED-1: enforce 2-of-2 signal requirement per phase7 precommit.
        # The precommit specifies a 2-signal composite; prior 1-of-2 fallback
        # silently turned ValueProfitability into a univariate strategy on
        # whichever signal was present, which is not the committed strategy.
        common = z_ey.index.intersection(z_gpa.index)
        rows = []
        for t in common:
            ey_val = z_ey.get(t)
            gpa_val = z_gpa.get(t)
            if ey_val is None or gpa_val is None:
                continue
            rows.append((t, float(ey_val + gpa_val) / 2))

        if not rows:
            return pd.Series(dtype=float)
        return pd.Series({t: v for t, v in rows})

    @property
    def name(self) -> str:
        return "value_profitability"


class QualityValue(CrossSectionalStrategy):
    """Multi-factor z-score composite.

    Combines three signals into an equal-weighted z-score sum:
      - ROE TTM (quality)
      - Gross margin TTM (quality)
      - Earnings yield (= TTM net income / (price × shares_out)) (value)

    Each signal is standardised cross-sectionally on the active-ticker
    set at each rebalance (per-date, no fit window). Missing values drop
    the ticker from the composite.
    """

    def __init__(
        self,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]
        prices = panel["prices"]
        if prices.empty:
            return pd.Series(dtype=float)
        last_prices = prices.ffill().iloc[-1]
        mcaps = panel.get("mcaps")
        _warn: dict = {}

        roe_by: dict[str, float] = {}
        gm_by: dict[str, float] = {}
        ey_by: dict[str, float] = {}
        for ticker in active:
            facts = facts_by_ticker.get(ticker)
            if facts is None:
                continue
            f = compute_fundamentals(facts, decision_date)
            roe = f.get("roe_ttm")
            gm = f.get("gross_margin_ttm")
            ttm_ni = f.get("ttm_net_income")
            shares = f.get("shares_outstanding")
            if roe is not None and np.isfinite(roe):
                roe_by[ticker] = float(roe)
            if gm is not None and np.isfinite(gm):
                gm_by[ticker] = float(gm)
            ey = _earnings_yield_from_mcap(
                ttm_ni, ticker, mcaps, last_prices, shares, _warn)
            if ey is not None:
                ey_by[ticker] = ey

        if not roe_by and not gm_by and not ey_by:
            return pd.Series(dtype=float)

        def _zscore(d: dict[str, float]) -> pd.Series:
            s = pd.Series(d)
            mu = s.mean()
            sd = s.std()
            if sd <= 0 or not np.isfinite(sd):
                return pd.Series(0.0, index=s.index)
            return (s - mu) / sd

        z_roe = _zscore(roe_by)
        z_gm = _zscore(gm_by)
        z_ey = _zscore(ey_by)

        # Outer-align then sum, requiring at least 2 of 3 signals per ticker.
        all_tickers = z_roe.index.union(z_gm.index).union(z_ey.index)
        rows = []
        for t in all_tickers:
            parts = [
                z_roe.get(t),
                z_gm.get(t),
                z_ey.get(t),
            ]
            present = [p for p in parts if p is not None]
            if len(present) < 2:
                continue
            rows.append((t, float(sum(present)) / len(present)))

        if not rows:
            return pd.Series(dtype=float)
        return pd.Series({t: v for t, v in rows})

    @property
    def name(self) -> str:
        return "quality_value_zsum"


class QualityComposite(CrossSectionalStrategy):
    """Phase 8 — MCAP-FREE quality composite (post-contamination directional bet).

    Equal-weighted cross-sectional z-score sum of three quality signals, NONE of
    which is price/mcap-denominated, so this strategy is immune to the market-cap
    split-adjust bug (`contamination_rerun_2026-05-30.md`):

      - ROE_ttm        = TTM net income / stockholders equity        (clean)
      - GP/A           = TTM gross profit / total assets (Novy-Marx) (clean)
      - magic_rank     = -(rank(EBIT/assets) + rank(EBIT/(assets-cash)))
                         (Greenblatt magic-formula combined rank; assets-EV proxy)

    Rationale: after the 2026-05-30 correction, pure value is dead on free large-cap
    PIT data; the only confirmed directional survivors are the clean quality signals
    quality_roe_ttm (+0.242) and magic_formula (+0.093). This recombines exactly
    those plus GP/A (the Novy-Marx quality anchor, low-correlation to ROE). Each
    signal is standardised cross-sectionally per rebal date; a ticker needs >=2 of 3
    signals present. Top decile, equal weight. EXPLORATORY (recombines tested
    signals, motivated post-hoc) — not a confirmatory claim.
    """

    def __init__(
        self,
        decile_breakpoint: float = 0.10,
        weighting: str = "equal",
        min_holdings: int = 20,
        max_holdings: int = 100,
    ):
        self.decile_breakpoint = decile_breakpoint
        self.weighting = weighting
        self.min_holdings = min_holdings
        self.max_holdings = max_holdings

    def score(self, panel: dict) -> pd.Series:
        active = panel["active_tickers"]
        facts_by_ticker = panel.get("facts_by_ticker") or {}
        decision_date = panel["as_of_date"]

        roe_by: dict[str, float] = {}
        gpa_by: dict[str, float] = {}
        mey_by: dict[str, float] = {}   # magic earnings yield = EBIT/assets
        mroic_by: dict[str, float] = {}  # magic ROIC = EBIT/(assets-cash)
        for ticker in active:
            facts = facts_by_ticker.get(ticker)
            if facts is None:
                continue
            f = compute_fundamentals(facts, decision_date)
            roe = f.get("roe_ttm")
            if roe is not None and np.isfinite(roe):
                roe_by[ticker] = float(roe)
            gp = f.get("ttm_gross_profit")
            assets = f.get("total_assets")
            if (gp is not None and assets is not None
                    and np.isfinite(gp) and np.isfinite(assets) and assets > 0):
                gpa_by[ticker] = float(gp) / float(assets)
            op_income = f.get("ttm_operating_income")
            cash = f.get("cash")
            if (op_income is not None and assets is not None
                    and np.isfinite(op_income) and np.isfinite(assets) and assets > 0):
                invested = float(assets) - (float(cash) if cash is not None and np.isfinite(cash) else 0.0)
                mey_by[ticker] = float(op_income) / float(assets)
                if invested > 0:
                    mroic_by[ticker] = float(op_income) / invested

        if not roe_by and not gpa_by and not mey_by:
            return pd.Series(dtype=float)

        def _zscore(d: dict[str, float]) -> pd.Series:
            s = pd.Series(d)
            mu, sd = s.mean(), s.std()
            if sd <= 0 or not np.isfinite(sd):
                return pd.Series(0.0, index=s.index)
            return (s - mu) / sd

        z_roe = _zscore(roe_by)
        z_gpa = _zscore(gpa_by)

        # Magic-formula combined rank -> a single z-scored signal. Rank EY and
        # ROIC descending (best = rank 1); negate the rank-sum so higher = better,
        # then z-score so it composes on the same scale as ROE/GP-A.
        z_magic = pd.Series(dtype=float)
        ey = pd.Series(mey_by)
        roic = pd.Series(mroic_by)
        common = ey.index.intersection(roic.index)
        if len(common) > 0:
            ey_rank = ey.loc[common].rank(ascending=False, method="min")
            roic_rank = roic.loc[common].rank(ascending=False, method="min")
            magic_score = -(ey_rank + roic_rank)  # higher = better
            z_magic = _zscore(magic_score.to_dict())

        all_tickers = z_roe.index.union(z_gpa.index).union(z_magic.index)
        rows = []
        for t in all_tickers:
            parts = [z_roe.get(t), z_gpa.get(t), z_magic.get(t)]
            present = [p for p in parts if p is not None and np.isfinite(p)]
            if len(present) < 2:   # require >=2 of 3 signals
                continue
            rows.append((t, float(sum(present)) / len(present)))

        if not rows:
            return pd.Series(dtype=float)
        return pd.Series({t: v for t, v in rows})

    @property
    def name(self) -> str:
        return "quality_composite_v1"
