# individual-stocks-snp500 — Final Report (Phases 0, 1, 4; +WRDS Phase 2 built)

**Date:** 2026-05-29 (v2, post adversarial review). Tier: DESCRIPTIVE/EXPLORATORY. Data: FREE + PIT (yfinance + SEC EDGAR XBRL + FRED + split history), survivorship-free S&P 500 membership reused from `stock-selection`.

## ⚠️ Correction notice (read first)

The v1 report headlined **`evebitda_yield` Sharpe-of-excess +0.367** and called it a "near-twin of the prior `value_EY`." An adversarial review (4 reviewers, `codex_review_round1.md` → `codex_review_round2.md`) found **two real bugs in the market-cap path** that v1 missed:

1. **Split-adjustment mismatch (dominant, inflating):** yfinance prices are split-adjusted but EDGAR shares are as-reported, so `mcap = adjusted_price × shares` understated market cap by the split factor (~28× for AAPL) → EV ≈ net_debt → impossible EBITDA/EV yields >1.0 that sorted to the top of the "cheap" ranking and got bought. yfinance never returns truly raw prices (even `auto_adjust=False` is split-adjusted), so the fix required reconstructing raw price = split-adjusted close × cumulative split factor (from yfinance split history), then `mcap = raw_price × as-reported shares`.
2. **Panel-lookup date mismatch (deflating):** the precompute keyed on month-start trading days, but the backtester's fold-boundary rebalances are the fold's `test_start` (often 1 trading day later) → 9 exact-match misses → 100% cash for ~1 month each. (The backtester logged these warnings; v1 never investigated them.) Fixed by snapping the lookup to the latest panel date ≤ decision date within 7 days (PIT-safe).

**Corrected result:** `evebitda_yield` excess Sharpe = **−0.057** (was +0.367 — the split bug added ≈ +0.42). The strategy *slightly underperforms* SPY. The verdict is unchanged (**0/2**) but the narrative is now a **clean fail, not a near-miss**, and the v1 "+0.367 / value re-skin" interpretation is **retracted**. The corrected signal panel stores raw components (mcap, net_debt, EBITDA, EBIT, ttm_NI, shares) so the market-cap basis can never silently corrupt the signal again. *Caveat: the prior `stock-selection` `value_EY` (+0.351) may share the same mcap-basis issue and should not be treated as a clean comparison.*

## Headline

**0/2 pre-registered core strategies pass the gate** (Holm on Sharpe-of-excess; pre-registered expectation 0/2). All tested free-PIT signals are **negative or null** — none beats SPY. The macro×sector descriptive cross is **indistinguishable from data-mining**. This reproduces and strengthens the `stock-selection` 0/11 outcome.

## Gate results (estimand = Sharpe of daily excess returns, Sharpe(strat − SPY))

| Phase | Strategy | Role | Sharpe-of-excess | own Sharpe vs SPY | raw p | Holm/adj p | 90% CI (excess) | Gate |
|---|---|---|---:|---|---:|---:|---|:--:|
| 1 | `evebitda_yield` | CORE | **−0.057** | 0.567 vs 0.766 (underperforms) | 0.601 | 1.000 | [−0.449, +0.323] | **FAIL** |
| 1 | `earnings_yield_v2` | descriptive | −0.100 | 0.531 vs 0.766 | 0.672 | — | [−0.494, +0.280] | FAIL |
| 4 | `vrp_defensive_tilt` | CORE | **−0.244** | 0.635 vs 0.646 | 0.892 | 1.000 | [−0.572, +0.075] | **FAIL** |

**Joint pre-registered family {`evebitda_yield`, `vrp_defensive_tilt`} (N=2), Holm on Sharpe-of-excess (authoritative): 0/2 PASS.** Both Holm-adjusted p = 1.000. (Note: the diff-based `simultaneous_sharpe_diff_ci` is a *different estimand* — Sharpe-diff, not Sharpe-of-excess — so Holm-on-excess is the estimand-consistent control here, not the Romano-Wolf diff CI mentioned in early config notes.)

Both estimands agree the strategies don't work: Sharpe-of-excess is negative AND the strategies' own Sharpes are below SPY's.

## Phase 0 — Infrastructure & power (GREEN)

7/7 checks pass. Power analysis on the ~2007–2024 effective sample: power @ excess Sharpe +0.20 = 0.13, +0.50 = 0.78 → **MDE > +0.50** → point-estimate-only tier; expectation 0 passes. Validated 3 repo fixes (`sector_as_of` NaN guard, ALFRED first-release CPI at 45-day release lag, BAA10Y deep credit spread) + caught a 4th truncated series (IG OAS `BAMLC0A0CM` → dropped). *(The power sim counts only the significance leg, so true MDE is even higher — the framework is if anything more underpowered, strengthening the descriptive tier.)*

## Phase 1 — PIT EV/EBITDA valuation (full S&P 500, 2010–2026, 17 walk-forward folds)

- 196/196 test-month rebalances covered; ~431 tickers with valid PIT fundamentals + price. Monthly top-decile, equal-weight, T+1, mcap-bucketed costs, T-bill cash. Turnover ≈ 41.
- **Market cap = raw (split-corrected) price × PIT as-reported shares** — the corrected basis. EV/EBITDA percentiles are now economically sane (median yield ≈ 0.07 ⇒ EV/EBITDA ≈ 14×; 99th ≈ 0.33 ⇒ ≈ 3×). A 1.5 plausibility backstop drops residual share/price-vintage glitches (20/71022 rows ≈ 0.03%).
- `evebitda_yield` excess Sharpe **−0.057** → FAIL; `earnings_yield_v2` (the value_EY clone) **−0.100** → FAIL. Both *underperform* SPY.
- **Interpretation:** the enterprise multiple / earnings yield on free PIT data does **not** beat SPY on this large-cap universe — mildly negative. Consistent with the decayed post-Reg-FD value premium. There is no marginal alpha here.

## Phase 4 — Macro × sector (survivorship-clean SPDR level)

**Core `vrp_defensive_tilt`: FAIL, slightly underperforms SPY** (excess Sharpe −0.244; strat Sharpe 0.635 vs SPY 0.646). Overweighting defensives vs cyclicals when the variance risk premium is elevated does not beat the market. *(Unaffected by the Phase 1 mcap bug — separate sector-ETF code path.)*

**Descriptive cross (5 macro × 9 sector × 3 threshold = 135 cells, gross, max-statistic null):**
- Observed best-cell excess Sharpe **+0.282**; random-regime permutation max-null mean **+0.361**, 95th **+0.500** → **max-statistic p = 0.835 → indistinguishable from data-mining.** The best of 135 hand-searched cells is *worse* than chance produces with 135 tries. Direct empirical confirmation of Molchanov-Stangl / Welch-Goyal on our own data; youBet's 3rd failed regime-conditioning experiment (after factor-timing + macro-exploratory). *(The pre-committed grid in config was 11 sectors = 165 cells; XLRE/XLC were dropped for lacking 2016/2018-reclassification-invariant history → executed 135. Conclusion robust either way.)*

## Phase 2 — WRDS / I/B/E/S confirmatory analyst path (BUILT, awaiting data)

The free-data ceiling made a confirmatory test of forward-analyst signals impossible. **WRDS I/B/E/S** is the genuine fix: `statpers` is the archived as-of snapshot (true PIT) and coverage is survivorship-free incl. delisted names. Built and unit-tested (synthetic IBES, 4/4) but **not run** — I/B/E/S is institutional, not free.

- `src/youbet/stock/wrds_ibes.py` — dual-mode ingestion (live `wrds` SQL OR local extract), `statpers`-keyed PIT contract, universe linking (delisted-inclusive), and 4 signal-panel builders.
- `experiments/phase2_analyst_wrds.py` — confirmatory runner (gracefully prints the data handoff when no IBES is present).
- `precommit/phase2_analyst.json` (N=3 confirmatory: `est_revision_momentum` [the one credible shot], `recommendation_consensus`, `price_target_upside`; + `forecast_dispersion_avoid` descriptive), `research/wrds_ibes_path.md` (exact queries + handoff), config `wrds_ibes` block.
- **To run:** provide a local WRDS extract (4 `ibes_*` files) or set `use_live_wrds: true` (the `wrds` package handles auth — this workflow never sees credentials). Expectation: most likely 0/3, but `est_revision_momentum` is the one signal with a credible path; if it passes the raw gate, Phase 5 + a turnover/capacity + small-cap-concentration audit decide whether it's real.

## What this establishes

1. **PIT-backtestable free valuation (EV/EBITDA, earnings yield) is mildly NEGATIVE vs SPY** on large-cap, 2010–2026. No free alpha; the corrected number is unambiguous (the +0.367 was a market-cap bug).
2. **Macro×sector fails twice over** — the pre-committed tilt underperforms, and the 135-cell cross can't beat its own data-mining null.
3. **The headline-novel signals (analyst estimates/ratings/revisions) are not free-PIT-testable** and remain WRDS-only — Phase 2 is built and pre-registered for when IBES is available.
4. **Net:** S&P 500 individual-stock selection on free PIT data does not beat SPY — 0/2 here on top of 0/11 in stock-selection. VTI/SPY hold remains unbeaten with significance on the data available to us.
5. **Methodological:** the adversarial review caught a result-inverting market-cap bug that v1 missed. Faithful reporting required the full split-corrected re-run; the v1 +0.367 is retracted.

## Caveats / limits

- Sharpe-of-excess is the gate estimand; both it and the strategies' own Sharpes are below SPY.
- 2010–2026 misses 2008 GFC (EDGAR XBRL coverage); power-limited (MDE > +0.50).
- ~431/646 tickers carry usable PIT fundamentals (coverage bias toward data-available names; 9 delisted names with data were dropped — a minor disclosed omission). Early folds (2010–2012) are thin (some months score <20 names; the min-holdings floor binds → wider-than-decile basket).
- Phase 4 cross is GROSS (apples-to-apples vs a gross max-null); the core tilt is net.
- The N=2 joint family members are tested on different windows (evebitda 2010–2026, vrp 2006–2026) — Holm is valid but it is not a common-window joint test.
- Phase 5 source-period-bias battery not triggered (mandatory only for a positive gate-surviving estimate; none survived).

## Artifacts

`artifacts/signal_panel*.parquet` — corrected PIT panel (raw-price mcap + stored components), 196 test dates × ~431 tickers. `unadj_close.parquet` — split-corrected raw prices. `artifacts/phase1_returns.parquet`, `phase4_returns.parquet`. `precommit/{phase1,phase4,phase2_analyst,prospective_revision_momentum}.json`. Engine: `src/youbet/stock/{fwd_valuation,macro_sector,wrds_ibes}.py` + the repo fixes.
