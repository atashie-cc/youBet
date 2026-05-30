# individual-stocks-snp500 — Forward/Analyst/Macro-Sector S&P 500 Strategy Evaluation

## What this workflow is

Tests whether **forward-looking valuation, analyst signals, or macro×sector-conditioned** long-only S&P 500 strategies beat SPY — the three data families the prior `stock-selection` workflow never touched (it exhausted *pure quantitative* selection: 0/11 beat SPY). Built on the same `src/youbet/stock/` engine.

**Status: Phases 0, 1, 4 COMPLETE + adversarial-review-corrected (2026-05-29). 0/2 core pass the gate.** Round-2 review caught a market-cap split-adjustment bug that had INFLATED `evebitda_yield` to +0.367; corrected (raw split-adjusted price × PIT shares) it is **−0.057 (FAIL, p 0.601, Holm 1.000)** — a CLEAN fail, slightly underperforming SPY. `earnings_yield_v2` −0.100; `vrp_defensive_tilt` −0.244; macro×sector 135-cell cross indistinguishable from data-mining (max-stat p=0.835). Joint N=2 Holm 0/2. Phase 5 not triggered (no positive). **WRDS/IBES confirmatory Phase 2 BUILT** (per user "proceed with WRDS") — `wrds_ibes.py` + `phase2_analyst_wrds.py` + `precommit/phase2_analyst.json` + `research/wrds_ibes_path.md`; awaiting institutional IBES data (local extract or live `wrds`). See `research/final-report.md` (v2) + `codex_review_round2.md`. Not committed to git.

**Engineering lessons (Session 1d):** investigate logged warnings + sanity-check signal distributions before reporting. yfinance never returns raw prices (auto_adjust=False is still split-adjusted) → mcap needs raw_price = split-adj close × cumulative split factor. Signal panel now stores raw components (mcap/net_debt/EBITDA/EBIT/ttm_NI/shares) so the mcap basis can't silently corrupt the signal.

## The honest headline (read before doing anything)

1. **Pre-registered expectation = 0 passes.** Framework MDE > +0.50 excess Sharpe under multiplicity control (inherited from stock-selection Phase 0); both academic-evidence scouts rate every new signal DOA or at best landing in the Holm-killed +0.20–0.40 band. This is **DESCRIPTIVE/EXPLORATORY** in aggregate with a **2-strategy pre-registered core** (`evebitda_yield`, `vrp_defensive_tilt`).
2. **The headline signals are NOT free-PIT-backtestable.** Forward estimates, estimate revisions, and analyst ratings are *current-snapshot-only* on every free source and *survivorship-leaking* (current-listed names only). A true 2005+ PIT confirmatory analyst test requires WRDS I/B/E/S (paid). See `research/data-availability.md`.
3. **Macro×sector is the most data-feasible but least likely to pay** (Welch-Goyal; Molchanov-Stangl; youBet already failed regime conditioning twice). Contained to ONE pre-registered tilt + a max-statistic-null descriptive cross.

## Three mandatory repo fixes (verified against the codebase; do BEFORE any results)

1. **`universe.sector_as_of` returns the literal string `'nan'`** for the 146 blank/delisted membership rows → survivorship-correlated label corruption. Patch to return `None` on NaN; normalize `{None,'nan','',NaN}`→MISSING; Phase-0 assertion. Prefer **SPDR proxies** for category conditioning.
2. **CPI/INDPRO leak via `fredapi.get_series`** (latest-revised). `revision_risk` lag-table tags are *inert* (`PITFeatureSeries` reads only `lag_days`). Add a real **ALFRED** path (`get_series_all_releases` + select latest obs with `realtime_start < decision_date`); `release_date` from `realtime_start`, not `index + fixed_lag`. Prefer CPIAUCNS.
3. **`etf/macro/fetchers.py::fetch_credit_spread` is repo-broken** — `BAMLH0A0HYM2` is now FRED-truncated to a rolling 3yr window (returns only 2023+). **Swap to `BAA10Y`** (public-domain, daily 1986+).

## Locked thresholds (inherited from `workflows/stock-selection/config.yaml`; do not change post-hoc)

- **Gate:** `Sharpe(strat − SPY) > 0.20` AND simultaneous/Holm-adjusted significance AND 90% block-bootstrap `ci_lower > 0`. Estimand = Sharpe-of-excess (NOT diff-of-Sharpes — the vti-as-challenger v1 bug).
- **Benchmark:** SPY (single, no shopping).
- **Bootstrap:** stationary Politis-Romano, 22-day blocks, 10,000 reps, seed 42.
- **Walk-forward:** 60mo train / 24mo test / 12mo step / monthly rebalance, T+1 (strict `<`).
- **Costs always on:** mcap-bucketed bps + $0.005/share. Cash earns 3mo T-bill.
- **Multiplicity:** Romano-Wolf `simultaneous_sharpe_diff_ci` is the **headline** control over the N=2 family (handles correlation); Holm(N=2) is the conservative cross-check. *(Holm-on-p assumes independence — wrong when family members correlate.)*
- **Power kill-gate:** MDE > 0.30 → headline point-estimate-only (expected branch).
- **Source-period-bias battery (Phase 5, mandatory for any positive):** characteristic-shuffle/random-decile placebo + long-leg mean-equalization + decile-breakpoint sweep {5,10,20,30%} + 3 sub-periods + GFC/COVID exclusion + cost sweep. *(Cross-sectional analogues — the sleeve-style checks from international-etf/real-world-test are ill-posed for decile baskets.)*

## Architecture

- Reuses `src/youbet/stock/` (universe, edgar, fundamentals, data, backtester, regime, pit, features, costs) and `etf/stats.py` UNCHANGED.
- New modules (to build): `fwd_valuation.py` (EDGAR-reconstructed PIT EV/EBITDA — core), `macro_sector.py` (new FRED series + ALFRED path + BAA10Y swap + SPDR map), `estimates.py` + `analyst.py` (optional, harness-gated), `sic_backfill.py` (optional, deferred).
- `precommit/phase{0,1,4}.json` + `prospective_revision_momentum.json` (exploratory-permanent) frozen before each phase.
- Forward-collection harness (`scripts/snapshot_forward.py`) is **optional** — single source (FMP only), survivorship-contaminated-by-construction, exploratory-permanent.

## Conventions
Inherits all `stock-selection` conventions (PIT fundamentals via `validate_fundamentals_pit`; survivorship-free membership + delisting returns; factor-zoo discipline — pre-commit JSON before each phase; audit-before-celebrating; paper-vs-net; expect 3-5 adversarial review rounds). Config-driven; research log read at session start.

## Open decisions for the user (recommended defaults bolded; full framing in `research/design.md` §8)
1. Free-only **(default)** vs paid (WRDS I/B/E/S flips the analyst phase to confirmatory — highest-value upgrade if affiliated).
2. Forward-collection harness: **do the historical core regardless; stand up the harness only if free AND committed to ~6yr** (else defer).
3. Macro×sector: **N=1 VRP/defensive tilt + hierarchical + max-statistic-null cross**.
4. Delisted-tail GICS: **SPDR proxies; drop `sic_backfill.py` from v1**.
5. Prospective revision eval date: **2032-06-01**, labeled exploratory-permanent.

## Key documents
- `research/design.md` — full experiment ladder (v1.1, post-review).
- `research/data-availability.md` — the data verdict matrix + citations.
- `research/codex_review_round1.md` — 3 adversarial review lenses + resolutions.
- `research/log.md` — session-by-session log.
