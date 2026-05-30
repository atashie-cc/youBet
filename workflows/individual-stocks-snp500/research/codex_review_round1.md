# Adversarial Review — Round 1 (design-stage)

**Date:** 2026-05-29. Three independent review lenses ran against design v1.0 (the synthesizer's first draft). Each verified load-bearing claims against the actual repo. design.md is now **v1.1** with every HIGH/MEDIUM resolved (flagged **[R1]** inline). Per `feedback_codex_review`, "Codex review" = adversarial general-purpose subagents.

## Lens A — PIT / survivorship / data-leakage

**Overall:** philosophy sound (the rare youBet design that does not treat snapshot-only analyst data as historical). 3 concrete leaks + 1 inert safeguard found.

| Sev | Issue | Resolution in v1.1 |
|---|---|---|
| HIGH | `universe.sector_as_of` returns literal `'nan'` (not None) for the 146 blank/delisted rows → survivorship-correlated label corruption inside the macro tilt's bucketing | Mandatory fix #1. Patch to return None; normalize MISSING; Phase-0 assertion; **confirmatory tilt uses SPDR proxies**, per-stock GICS → descriptive only (§6.1, Phase 0, Phase 4) |
| HIGH | ALFRED "fix" was inert metadata (`revision_risk` unread) + wrong primitive (`first_release` ≠ `realtime_start<d`) + monthly keying under-lags CPI ~2-4wk | Mandatory fix #2. Real ALFRED code path: `get_series_all_releases` + select latest obs with `realtime_start < d`; `release_date` from `realtime_start`; Phase-0 CPI-revision plant-test (§3.3, §6.2, Phase 0) |
| MED | Forward-collection PIT contract leak vectors: day-granularity `snapshot_date < decision_date`; FMP backfill / partial-write | `snapshot_timestamp < d − 1 trading day`; sha256 write-once quarantine; never read past-fiscal-period FMP rows (§3.4, §6.4) |
| MED | `.upgrades_downgrades` as Phase-6 ML feature re-imports survivorship (missing-rating ⇒ delisting) | Exclude, or restrict ≥2013 + coverage-invalid⇒exclude (never impute) + missingness-vs-delisting diagnostic (Phase 6) |
| MED | AAII single-compilation / VVIX / MOVE / CNN reconstruction trust under-guarded | Config assertion: confirmatory macro inputs ⊆ {market-derived, not-revised}; AAII/CNN/VVIX/MOVE descriptive-only; AAII re-fetch diff check (§6.8) |
| LOW | GICS 2016/2018 reclassification anachronism even for current members | Coarse defensive/cyclical buckets invariant across breaks, or SPDR-proxy historical constituents (§6.9, Phase 4) |

## Lens B — Statistical power / multiplicity / source-period bias

**Overall:** structurally honest about its own futility, but one logical incoherence had to be fixed before commit.

| Sev | Issue | Resolution in v1.1 |
|---|---|---|
| HIGH | Pre-registered effect bands (+0.05–0.30) sit BELOW the +0.20 gate → a "confirmatory family" pre-committed to test sub-threshold, undetectable hypotheses. N=4 Holm = theater. | Tier is **DESCRIPTIVE in aggregate**; honest pre-registered core = **N=2** with explicit expectation 0/2; descriptive companions report point estimate + CI only, no gate language (§1, §4, §5) |
| HIGH | 3 of 4 "confirmatory" strategies are ~90%+-correlated value re-skins of failed `value_EY`/`quality_value_zsum` → Holm independence violated, info content ~N=1 | Value cluster collapsed to the ONE novel member (`evebitda_yield`); `earnings_yield_v2` + `pit_value_quality_evebitda_zsum` → descriptive; **Romano-Wolf simultaneous CI is headline**, Holm cross-check; report candidate correlation matrix (§4 Phase 1/3, §5) |
| MED | Phase-5 source-period battery specified for single-asset sleeves, ill-posed for decile baskets | Cross-sectional analogues: characteristic-shuffle/random-decile placebo + long-leg mean-equalization + decile-breakpoint sweep {5,10,20,30%}; sleeve sweep applies only to `vrp_defensive_tilt` (Phase 5) |
| MED | Hierarchical gate protects confirmatory p-values but the descriptive macro×sector cross runs ungated with full researcher DoF (Molchanov-Stangl data-snooping) | Pre-register full 5×11×3 grid; report whole-grid distribution vs random-permutation null; no cell is a "finding" without a max-statistic (Romano-Wolf/White) permutation test (§4 Phase 4, §5) |
| MED | EV/EBITDA band double-counts decay; revision band ignores turnover cost | Anchor EV/EBITDA on in-universe sibling (`value_EY` +0.351), band ≈ +0.05–0.35; freeze revision band NET of computed monthly turnover (§4 Phase 1/2) |
| LOW | Single scalar MDE obscures per-strategy hurdle | Report per-strategy MDE under Romano-Wolf (effective-N), both at Holm(N=2) and effective-N (Phase 0, §5) |

## Lens C — Data-feasibility realism

**Overall:** APPROVE the data scoping; all load-bearing repo claims verified (credit-spread hard-codes BAMLH0A0HYM2; zero ALFRED usage in src; GICS gap exactly 503/146 with 61 CIK). The central verdict — analyst/forward not free-PIT-backtestable; only forward-collect or WRDS — is correct.

| Sev | Issue | Resolution in v1.1 |
|---|---|---|
| HIGH | Forward-collection harness can NEVER cover the delisted tail (free sources = current-listed only) → survivorship-contaminated by construction; a 2032 "confirmatory" test still fails the survivorship bar | Prospective revision test downgraded to **EXPLORATORY-PERMANENT**; harness re-resolves membership weekly + logs delistings; only WRDS/Benzinga could ever make it confirmatory (§3.4, §6.7, §8.2) |
| HIGH | FMP `/analyst-estimates` mis-scoped as clean consensus — it is algorithmic-aggregated, overwritten per fiscal period, backfills realized | Pre-register signal degradation (proxy of a proxy); guard `consensus_as_of` from reading past-fiscal-period rows; band halved again for the algorithmic gap (§3.4) |
| MED | Harness effort/payoff poor (4-source weekly cron for a near-certain future null on degraded data) | Trim to **single source (FMP only)**; drop yfinance/Finnhub from standing harness; make harness **conditional on Open-Decision 1/2 + 6yr commitment** (§3.4, §8.2) |
| MED | VVIX/MOVE/put-call can't reach 2005 → macro panel effectively 2007+ | Annotate per-series start floors; macro×sector effective sample ~2007-2024; recompute Phase-0 MDE on shorter sample; keep VVIX/MOVE out of confirmatory tilt (§3.5, config) |
| LOW | SIC backfill is a non-PIT, partial (61/146) artifact spent on a path SPDR proxies make unnecessary | `build_sic_backfill: false` by default; dropped from v1 + from the Phase-0 mandatory gate (§8.4) |

## Net
Design is sound and buildable after v1.1. The single most important honest conclusion the workflow commits to: **a true 2005-present confirmatory test of forward-valuation/analyst signals is not possible on free data** — so v1.1 tests only the PIT-safe trailing/derived-valuation + one fear tilt now, and forward-collects the rest (optionally, exploratory-permanent) rather than pretending. Expectation remains **0 passes**, consistent with the 0/11 prior and the underpowered-framework thesis.
