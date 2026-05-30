# Adversarial Review — Round 2 (post-execution, Phases 0/1/4)

**Date:** 2026-05-29. Four hostile reviewers (PIT/survivorship, statistics, implementation-integrity, interpretation) verified the executed Phase 0/1/4 code + artifacts with Read/Grep/Bash, explicitly flagging anything that could flip the 0/2 verdict. Per `feedback_codex_review`, reviewers = adversarial general-purpose subagents. **The review changed the headline result.**

## Verdict-critical findings (both fixed; both inverted/corrected the Phase 1 number)

| Sev | Finding | Evidence | Resolution |
|---|---|---|---|
| HIGH | **Market-cap split-adjustment mismatch (INFLATING).** Prices split+div-adjusted (yfinance) but EDGAR shares as-reported → `mcap = adj_price × shares` understates mcap by the split factor (~28× AAPL) → EV≈net_debt → impossible EBITDA/EV yields >1.0 sorting to the top of the "cheap" basket. Decisive test: capping implausible yields drops evebitda +0.367 → +0.012 (p 0.48). | 2 reviewers independently: 750/55268 rows EBITDA/EV>1.0; offenders BKNG/CMG/AAPL/AMZN/NVDA/GOOG; cap-test +0.174/+0.012/−0.141 at caps 1.0/0.5/0.3 | **FIXED:** yfinance never returns raw prices (even auto_adjust=False is split-adjusted), so reconstruct raw price = split-adj close × cumulative split factor (yfinance split history); `mcap = raw_price × shares`. Precompute now stores raw components + a 1.5 plausibility backstop. **Corrected evebitda = −0.057.** |
| HIGH | **Panel-lookup date mismatch (DEFLATING).** Precompute keyed on month-start trading days; backtester fold-boundary rebal = `test_start` (1 trading day later) → 9 exact-match misses → 100% cash ~1 month each (logged but uninvestigated in v1). | phase1_panel_run.log:76-97 (9 "100% cash" warnings); reconstructed 9/205 misses, all fold-first-rebals | **FIXED:** `PrecomputedScoreStrategy.score` snaps to latest panel date ≤ d within 7 days (PIT-safe). Corrected run: **0 cash-hold warnings.** |
| HIGH | **Overclaim** — v1 said evebitda "fails on significance AND CI-lower" / "clears only the point-estimate leg"; the contaminating cash warnings were never disclosed. | final-report.md v1:20,23,33 | **RESOLVED:** v2 report retracts the +0.367 / "value re-skin" narrative, adds a prominent Correction notice, reports corrected −0.057, and discloses both bugs. |

## Minor findings (addressed in v2 docs/config)

- **MED (stat):** Joint N=2 Holm combines p-values from different windows (evebitda 2010-26, vrp 2006-26); config mislabeled Romano-Wolf as the "headline" but `simultaneous_sharpe_diff_ci` is a *different estimand* (Sharpe-diff). → v2 states Holm-on-Sharpe-of-excess is authoritative + discloses the different-window caveat.
- **MED (impl):** thin early-fold coverage (2010-2012 scores <20 names; decile floor binds → wider-than-decile basket). → disclosed in Caveats.
- **LOW:** diff-of-Sharpes column in v1 (0.804/0.766) came from backtester net-of-tbill metrics, inconsistent with the gate's raw Sharpe. → v2 reports Sharpe-of-excess + own-Sharpe-vs-SPY consistently.
- **LOW:** config descriptive grid 165 (11 sectors) vs executed 135 (9 SPDRs; XLRE/XLC dropped). → disclosed in report; conclusion robust.
- **LOW:** 9 delisted names with data dropped from panel (coverage, not systematic survivorship — 47 delisted names ARE present). → disclosed.
- **LOW:** power sim counts only the significance leg (true MDE even higher). → noted; strengthens descriptive tier.
- **LOW (cosmetic):** `fetch_credit_spread` coverage guard is bypassed on a cached truncated series (>100 rows). Not exercised (Phase 4 uses BAA10Y). → noted; guard protects fresh fetches only.

## Confirmed-clean (no action)

EV/EBITDA pure-function math; Phase 4 `simulate` T+1 + burn-in mask + weight-sums; VRP strict-< causality; ALFRED first-release (45-day lag, replaces leaky index+14d); `sector_as_of` NaN fix; gate estimand consistently Sharpe-of-excess (no vti-as-challenger diff bug); 109/109 stock tests pass; the gross descriptive cross vs gross max-null is apples-to-apples; the macro×sector −0.244 / data-mining-null conclusions are unaffected by the mcap bug.

## Net

The review did exactly its job: it caught a **result-inverting** market-cap bug v1 missed (the only material defect; it made the strategy look better than it is). The 0/2 verdict survives and is now **cleaner** (all signals negative/null, no near-miss). The corrected, raw-component-storing panel is the durable artifact. Lesson reinforced: **investigate logged warnings ("100% cash") and sanity-check signal distributions (impossible >1.0 yields) before reporting.**
