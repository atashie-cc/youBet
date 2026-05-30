# individual-stocks-snp500 — Research Log

Read at the start of every session.

---

## Session 1 — 2026-05-29 — Design + data-availability assessment (DESIGN stage)

**Mandate (user):** Develop a new individual-stock S&P 500 strategy to assess rigorously. Assess what data are available (quantitative: PE/EBITDA/forward-PE; qualitative: Yahoo/FT analyst buy/sell/target ratings; general factors: fear/inflation × stock category tech/financial/mining). Think through a series of experiments. Context: youBet has shown across ~11 workflows that we lack data to confirm with significance that any strategy beats "VTI/SPY and hold."

**Approach:** reviewed `strategy-dashboard` (most recent analysis — stock-selection composites cluster high but descriptive-only, 0/11 passed strict gates) and the `src/youbet/stock/` engine (PIT EDGAR fundamentals, survivorship-free universe WITH GICS sector, walk-forward backtester, block-bootstrap+Holm, GKX ML). Ran a 10-agent orchestration: 4 data-availability scouts + 2 academic-evidence scouts (web research) → 1 design synthesizer → 3 adversarial reviewers (~916K tokens, 214 tool uses).

### Key findings

**Engine reuse:** the genuinely *new* territory vs `stock-selection` (which exhausted pure-quantitative selection) is forward-looking valuation, analyst ratings, and macro×sector conditioning — none used before. The backtester plugs in via `score(panel)`; new data = PIT-gated panel keys + new `score()` methods. The universe already carries `gics_sector`.

**Data verdict (full matrix in `data-availability.md`):**
- **Trailing/derived valuation (EV/EBITDA, trailing P/E):** FREE + PIT via EDGAR (already in repo). CORE-eligible. The one value construction stock-selection did NOT run is EBITDA-yield.
- **Forward estimates / revisions / analyst ratings:** NOT free-PIT-backtestable. Every free source is current-snapshot / 90-day-rolling / fiscal-period-keyed, overwritten, and *current-listed-only* (survivorship). `yfinance .upgrades_downgrades` is the only multi-year free dated stream (~2011-12+) but revisable + survivorship-contaminated → exploratory feature only. True PIT consensus archive = I/B/E/S Summary History (paid/WRDS).
- **Macro/fear/inflation:** FREE + PIT + deep (FRED + yfinance) — the most feasible family — but academic evidence (Welch-Goyal, Molchanov-Stangl) + youBet's own two prior failures say macro×sector timing is the *least* likely to pay and the worst multiplicity trap (165-cell cross).
- **Sector classification:** SPDR proxies are survivorship-clean 2004+ (9 sectors); per-stock repo GICS labels are blank on the 146 delisted/added rows.

**Academic evidence (both scouts skeptical):** of 7 analyst/forward signals, at most 2 (estimate-revision momentum, forward valuation) merit a slot and both expected in the Holm-killed +0.20–0.40 band; the rest DOA (short-side / small-illiquid / decayed-in-large-cap; Engelberg-McLean-Pontiff 2020: analyst levels bet *against* anomalies). Macro×sector DOA on power + data-mining grounds. Expectation: **0 passes**, reproducing the 0/11 prior.

### Three REAL repo bugs verified by reviewers (now mandatory pre-work)
1. `universe.sector_as_of` returns literal `'nan'` string for blank rows → survivorship-correlated label corruption.
2. CPI/INDPRO leak via `fredapi.get_series` (latest-revised); the `revision_risk` lag-tag is inert → need a real ALFRED `get_series_all_releases` + `realtime_start < d` path.
3. `etf/macro/fetchers.py::fetch_credit_spread` hard-codes `BAMLH0A0HYM2`, which FRED truncated to a rolling 3yr window (returns only 2023+) → swap to `BAA10Y`.

### Adversarial review (Round 1, `codex_review_round1.md`) → design v1.1
3 lenses, all HIGH/MEDIUM resolved. Most consequential corrections: (a) tier is DESCRIPTIVE in aggregate — the v1.0 "confirmatory family" had effect bands *below* its own gate; honest pre-registered core = **N=2** (`evebitda_yield`, `vrp_defensive_tilt`), expectation 0/2; (b) collapse the 3 near-duplicate value strategies to the one novel member, Romano-Wolf simultaneous CI as headline multiplicity control; (c) Phase-5 battery re-specified with cross-sectional analogues (characteristic-shuffle placebo, decile-breakpoint sweep); (d) forward-collection harness → exploratory-permanent, single-source, opt-in (survivorship ceiling).

### User decisions (2026-05-29)
Free-only; build Phase 0 + the 3 repo fixes; defer the forward-collection harness. Adopted defaults: macro×sector N=1 hierarchical; SPDR proxies for category conditioning.

---

## Session 1b — 2026-05-29 — Phase 0 build + run (3 repo fixes + modules + GREEN)

**Repo fixes applied (all tested):**
1. `stock/universe.py::sector_as_of` — returns `None` (not literal `'nan'`) for blank/delisted rows. Verified on the real 649-row universe (146 blank rows). Test: `tests/stock/test_individual_stocks_macro.py`.
2. New `stock/macro_sector.py` — real **ALFRED** path (`get_series_all_releases` + `realtime_start`) for CPI/INDPRO (release_date from realtime_start, not leaky `index+lag`); BAA10Y deep credit spread; new market series; VRP proxy; survivorship-safe sector normalization + SPDR map + coarse defensive/cyclical partition; `register_individual_stock_lags()`.
3. `etf/macro/fetchers.py::fetch_credit_spread` — added `series_id` param (default BAML, backward-compat) + per-series cache + **coverage guard** (loud error instead of silent NaN-pad on the ICE 3yr-rolling truncation).

**New modules:** `stock/fwd_valuation.py` (EV/EBITDA PIT — `EVEBITDAYield` core + `EarningsYieldV2` descriptive); D&A added to `fundamentals.py` (additive `dep_amort` alias + `ttm_dep_amort`). Workflow `experiments/{_shared.py, phase0_infrastructure.py}` (reuses stock-selection's data/EDGAR cache). Precommit: `phase1.json`, `phase4.json`, `prospective_revision_momentum.json` (frozen).

**Tests:** 31/31 pass (new EV/EBITDA + macro/sector + sector-fix, plus existing fundamentals/universe unaffected by the additive edits).

**Phase 0 — GREEN (all 7 checks pass):**
- **Power analysis (the headline):** power @ ExSharpe +0.20=0.13, +0.30=0.30, +0.40=0.43, +0.50=0.78 → **MDE > +0.50** on the ~2007-2024 (18yr) effective sample. Tier branch = **point-estimate-only**. Empirically reproduces the stock-selection underpowered finding on THIS workflow's sample → pre-registered expectation 0 passes stands.
- `sector_nan_fix`, `pit_fundamentals_plant`, `fwd_valuation_math` PASS.
- `alfred_cpi_pit` PASS — first-release lag median **45 days** (correctly captures CPI's true post-month-end release; confirms the `index+14d` would have under-lagged).
- `baa10y_coverage` PASS — deep history past 2005.
- **`macro_coverage_report` found a SECOND truncated series:** IG OAS (BAMLC0A0CM) is *also* ICE-truncated to 2023-05-30 (785 obs). **Dropped from `new_fred_series`** (config updated); credit conditioning uses BAA10Y. fed_funds/3m-10y/real-rate/5y5y-breakeven all deep to 2003.

### Status / next
Phase 0 GREEN; 3 repo fixes shipped; EV/EBITDA core + VRP-tilt modules built + tested; precommits frozen. Per user, **paused before any strategy results** (Phase 1/4 backtests). Harness deferred. Resume = run Phase 1/4.

---

## Session 1c — 2026-05-29 — Phase 1 + Phase 4 EXECUTED (0/2 gate, as pre-registered)

**Compute-infra fight (logged for posterity):** the full 646-ticker monthly walk-forward exceeds the 10-min Bash ceiling (compute_fundamentals ~14-45s/date depending on filing-history depth + CPU contention). Solution: a **resumable, incrementally-saved precompute** of the (date×ticker) EV/EBITDA + earnings-yield signal panel (`precompute_signals.py`, optional date-sharding), then a **fast backtest** (`phase1_run_from_panel.py`) where `PrecomputedScoreStrategy.score` is an O(1) lookup so StockBacktester handles T+1/costs/T-bill/delisting at full speed. Panel built over ~8 calls (solo + 2/3-way shards); 202 monthly dates × ~428 tickers. **Lessons:** (1) `python -u` mandatory — buffered stdout is lost on timeout-kill; (2) incremental save mandatory — load-time + loop can exceed the wall timeout; (3) **2 shards beat 3** on this box (3-way contention → 45s/date vs 14s solo). **Real correctness bug caught:** strategies must compute mcap = last_price × PIT shares; without it `panel["mcaps"]` falls back to price-proxy → EV≈net_debt → corrupt signal. Fixed in `fwd_valuation.mcap_from_panel`.

**RESULTS (gate estimand = Sharpe of excess, Sharpe(strat−SPY)):**
| Phase | Strategy | role | ExSharpe | raw p | Holm/adj | 90% CI | gate |
|---|---|---|---:|---:|---:|---|:--:|
| 1 | evebitda_yield | CORE | **+0.367** | 0.0575 | 0.115 | [−0.019,+0.748] | FAIL |
| 1 | earnings_yield_v2 | descr | +0.246 | 0.139 | — | [−0.123,+0.626] | FAIL |
| 4 | vrp_defensive_tilt | CORE | **−0.244** | 0.892 | 0.892 | [−0.572,+0.075] | FAIL |

**Joint N=2 Holm on Sharpe-of-excess: 0/2 PASS** (pre-registered expectation 0/2). evebitda_yield clears only the +0.20 point leg (raw p just misses 0.05, CI lower just below 0) — a near-exact twin of the prior `value_EY` (+0.351): EV/EBITDA is the value premium re-skinned, no marginal edge. vrp_defensive_tilt slightly underperforms SPY.

**Phase 4 descriptive cross (135 cells, gross, max-statistic null): observed best cell +0.282 vs random-regime permutation max-null mean +0.361 (95th +0.500) → max-stat p=0.835 → INDISTINGUISHABLE FROM DATA-MINING.** Best of 135 hand-searched cells is worse than chance with 135 tries. Empirically confirms Molchanov-Stangl / Welch-Goyal on our own data; youBet's 3rd failed regime-conditioning experiment.

**Phase 5 NOT triggered** (mandatory only for a positive gate-surviving estimate; none survived). See `final-report.md`. Recommendation stands: VTI/SPY hold unbeaten with significance. Not committed to git (awaiting user). **[SUPERSEDED by Session 1d — the v1 +0.367 was a market-cap bug.]**

---

## Session 1d — 2026-05-29 — Adversarial review (Round 2) + CORRECTION + WRDS Phase 2 built

**Review (`codex_review_round2.md`, 4 hostile reviewers) caught 2 verdict-relevant bugs v1 missed:**
1. **Market-cap split-adjustment mismatch (INFLATING).** yfinance prices split-adjusted, EDGAR shares as-reported → `mcap = adj_price × shares` understated by the split factor (~28× AAPL) → EV≈net_debt → impossible EBITDA/EV yields >1.0 bought into the basket. **This alone inflated evebitda from −0.057 to +0.367.** yfinance never returns raw prices (even auto_adjust=False is split-adjusted) → fix = raw_price = split-adj close × cumulative split factor (yfinance split history), mcap = raw_price × shares. Precompute now stores raw components + 1.5 plausibility backstop.
2. **Panel-lookup date mismatch (DEFLATING).** Precompute keyed month-start; backtester asks fold test_start (1 trading day later) → 9 exact-match misses → 100% cash ~1mo each (logged, uninvestigated in v1). Fix = `PrecomputedScoreStrategy.score` snaps to latest panel date ≤ d within 7 days (PIT-safe). Corrected run: 0 cash warnings.

**Full split-corrected re-grind** (raw-price mcap + stored components, ~8 sharded batches) → re-ran Phase 1 + joint.

**CORRECTED RESULTS:** evebitda_yield **−0.057** (was +0.367; FAIL, p 0.601, Holm 1.000, CI [−0.449,+0.323], strat Sharpe 0.567 < SPY 0.766). earnings_yield_v2 **−0.100** (FAIL descr). vrp_defensive_tilt −0.244 (unchanged — separate code path). **Joint N=2 Holm: 0/2.** All signals NEGATIVE/null — a CLEAN fail, not a near-miss. v1 "+0.367 near-twin of value_EY / value re-skin" RETRACTED. (Caveat: prior stock-selection value_EY +0.351 may share the same mcap bug.) Phase 5 not triggered (no positive). Descriptive cross unchanged (best +0.282 vs max-null +0.361, p 0.835).

**WRDS / I/B/E/S confirmatory Phase 2 BUILT** (the user's "proceed with WRDS"): `src/youbet/stock/wrds_ibes.py` (dual-mode live-wrds/local-extract ingestion, statpers-PIT contract, delisted-inclusive universe linking, 4 signal-panel builders); `experiments/phase2_analyst_wrds.py` (confirmatory runner, graceful no-data handoff); `precommit/phase2_analyst.json` (N=3: est_revision_momentum [credible shot] + recommendation_consensus + price_target_upside; + dispersion descriptive); `research/wrds_ibes_path.md` (exact WRDS queries + handoff); config `wrds_ibes` block; `tests/stock/test_wrds_ibes.py` 4/4 synthetic. **Not run** (IBES is institutional). To run: provide local WRDS extract OR `use_live_wrds: true` (workflow never handles credentials).

**Lesson:** investigate logged warnings ("100% cash") and sanity-check signal distributions (impossible >1.0 EV/EBITDA yields) BEFORE reporting. The review's value: it inverted a contaminated headline. Final verdict 0/2 (cleaner). Not committed to git.
