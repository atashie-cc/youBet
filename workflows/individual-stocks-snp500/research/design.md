# individual-stocks-snp500 — Workflow Design Document (v1.1, post-review)

**Status:** DESIGN (pre-Phase-0). Not yet committed for execution.
**Date:** 2026-05-29
**Predecessor:** `workflows/stock-selection/` (0/11 confirmatory pass; MDE > +0.50 excess Sharpe).
**Review:** v1.1 incorporates 3 adversarial review lenses (PIT/survivorship, power/multiplicity, data-feasibility) — see `codex_review_round1.md`. Changes from v1.0 are flagged **[R1]** inline.

**Falsifiable question:** *Does any pre-committed forward-valuation, analyst-signal, or macro×sector-conditioned long-only S&P 500 strategy produce a net-of-cost Sharpe-of-(strat−SPY) > +0.20 with a 90% block-bootstrap CI lower bound > 0 (Romano-Wolf-simultaneous over the pre-registered family) — using only FREE, point-in-time data?*

**Pre-registered expectation: 0 passes.** Given (a) the 0/11 prior, (b) MDE > +0.50, and (c) both academic-evidence scouts rating every new signal DOA or at best landing in the Holm-killed +0.20–0.40 band, the honest expectation is zero gate-clears. The value of running it: (1) convert "we never tested forward/analyst/macro signals" into a *demonstrated* null on this universe; (2) ship the durable PIT-safe trailing-EV/EBITDA construction the prior workflow lacked; (3) optionally accrue a prospective forward-collection dataset for a future, better-powered re-test; (4) deliver three real repo bug-fixes uncovered during design.

---

## §0 — What changed in v1.1 (post-review summary)

The three reviewers agreed the design's *philosophy* is sound (it is the rare youBet design that does not pretend snapshot-only analyst data is historical) but flagged structural fixes that materially change the ladder:

1. **The "confirmatory family" was incoherent.** v1.0 pre-registered effect bands (+0.05 to +0.30) that sit *below* the +0.20 gate, then called it a hypothesis test. **[R1]** v1.1 makes the tier honestly **DESCRIPTIVE in aggregate**, with a **2-strategy pre-registered core** whose expectation is explicitly 0/2 and which is reported with Romano-Wolf simultaneous CIs, not as a "powered test."
2. **3 of 4 v1.0 "confirmatory" strategies were ~90%+-correlated value re-skins** of the already-failed `value_EY` (+0.351) and `quality_value_zsum` (+0.068). Holm assumes independence; three clones test one hypothesis three times. **[R1]** v1.1 collapses the value cluster to the **single genuinely-novel** member (`evebitda_yield`) in the core; the other value variants run **descriptive-only**.
3. **Phase 5 source-period-bias battery was specified for single-asset sleeve tilts**, not cross-sectional decile baskets. **[R1]** v1.1 substitutes the correct cross-sectional analogues (characteristic-shuffle/random-decile placebo, decile-breakpoint monotonicity sweep, long-leg mean-equalization).
4. **The forward-collection harness cannot escape survivorship contamination** (every free source serves current-listed tickers only). **[R1]** the prospective revision test is downgraded to **EXPLORATORY-PERMANENT** on free data, trimmed to a single lean source, and made **conditional on user opt-in** (Open-Decision 1/2).
5. **Three real repo bugs were verified against the codebase** and become mandatory pre-work — see §6.

---

## §1 — Framing & Tier

### Tier: **DESCRIPTIVE / EXPLORATORY** in aggregate. Pre-registered core = **2 strategies**, expectation **0/2**.

Mirrors `international-etf` (descriptive after a power analysis) and `macro-exploratory`. Justification in order of decisiveness:

1. **Framework is structurally underpowered.** `stock-selection` Phase 0 established MDE > +0.50 excess Sharpe at 20yr daily data under multiplicity control. The best prior signal (`value_EY` +0.351, raw p=0.064) was killed. No new signal family has a credible *net-of-cost, value-weighted, large-cap, post-decay* excess Sharpe near +0.50:
   - `evidence-analyst-fwd`: at most **2 of 7** signals (estimate-revision momentum, forward valuation) merit a slot; both expected in the **+0.20–0.40 band** (positive point estimate, killed by multiplicity). The other 5 (rec levels, rec changes, dispersion, PEAD, target-price) are **DOA** — their tradable alpha is short-side / small-illiquid / decayed-in-large-cap. Engelberg-McLean-Pontiff (2020): analyst *levels* bet *against* established anomalies.
   - `evidence-macro-sector`: **DOA on power grounds for almost the entire category.** Molchanov-Stangl (2024) — sector rotation is a data-mining artifact; Welch-Goyal / Goyal-Welch-Zafirov — macro predictors fail OOS; youBet has *already* run regime conditioning twice (factor-timing "diagnostic not strategic"; macro-exploratory breadth-gated VTI −85% MaxDD). A naive 5 macro × 11 sector × 3 threshold cross = 165 hypotheses → guaranteed 0 survivors + inflated false positives.

2. **The headline signals are not PIT-backtestable free.** Consensus forward EPS, estimate *revisions*, ratings *changes*, price targets — none reconstructable as-of a historical decision date on free data (§2, `data-availability.md`). A gate cannot be honestly claimed on data that cannot be reconstructed as-of the decision date.

3. **[R1] The pre-registered core is honestly N=2, not N=4.** The reviewers showed a "family" whose own effect bands lie below its pass threshold is theater. v1.1 pre-registers exactly two *distinct-hypothesis* tests and is explicit that the expectation is 0 passes:
   - `evebitda_yield` — the one value construction the prior workflow did *not* run (EDGAR-reconstructed EBITDA/EV decile). Genuinely novel; everything else in the value space is a near-clone of `value_EY` and runs descriptive-only.
   - `vrp_defensive_tilt` — a single theory-anchored macro tilt, uncorrelated with the value signal (distinct hypothesis), so it is a legitimate second family member.

### Tier decision rule (pre-committed, mirrors international-etf power analysis)
- **Phase 0** recomputes MDE under realistic per-strategy TE. **[R1]** report MDE *per-strategy under the family it belongs to* using the Romano-Wolf simultaneous framework (empirical correlation), not a single Holm(N=4) scalar. If MDE > `config.power_analysis.kill_gate` (0.30) — the expected branch — the headline (analyst/forward/macro) findings are **point-estimate + CI only, no gate claims**.
- The 2-strategy core is still gate-tested with an explicit **pre-registered expectation of 0/2**, reported via Romano-Wolf simultaneous CIs (headline) + Holm (conservative cross-check). **No cross-phase confirmatory claim** is made for any signal needing forward-collected or snapshot-only data.

---

## §2 — Data-Availability Verdict (summary; full matrix + citations in `data-availability.md`)

**Binding constraint:** true point-in-time analyst-estimate / ratings / price-target history back to ~2005 is **NOT obtainable free**. Every free source returns a *current snapshot* (or shallow ~90-day rolling window) keyed to the live date or a fiscal period, with no archived as-of consensus, covering *currently-listed names only* (survivorship leak). The only genuine PIT consensus archive (I/B/E/S Summary History, 1976+) is paid/WRDS-only.

| Data family | FREE + PIT to ~2005? | Verdict |
|---|---|---|
| **Trailing/derived valuation** (EV/EBITDA, trailing P/E, EBITDA, net debt) | **YES** — reconstruct PIT from EDGAR XBRL (in repo) + PIT price | **CORE-ELIGIBLE** |
| Forward P/E, PEG, consensus fwd EPS/revenue, **estimate revisions** | **NO** — current-snapshot / fiscal-period-keyed / overwritten / survivorship-leaking | **FORWARD-COLLECT-ONLY → EXPLORATORY-PERMANENT [R1]** |
| I/B/E/S Summary History (true PIT 2005+) | only if user has **WRDS** | REJECTED (free) / USE-IF-WRDS |
| Analyst ratings — `yfinance .upgrades_downgrades` (dated, ~2011-12+) | **PARTIAL** — revisable rolling store, **severe survivorship**, empty pre-2012 | **EXPLORATORY-ONLY, never gate** |
| Analyst ratings — consensus counts, price targets, Zacks/Benzinga/TipRanks | **NO** (free) / paid / institution-walled / ToS-violating | FORWARD-COLLECT or REJECTED |
| **Macro / fear / inflation** — VIX, HY+IG OAS, yield-curve (T10Y2Y/T10Y3M/DFF/DFII10), breakevens (T10YIE/T5YIFR), USD, AAII, CBOE P/C | **YES** — market-derived, not revised, deep, free | **USE** |
| **CPI (CPIAUCSL) & PMI-proxy (INDPRO)** | **YES only via ALFRED real-time vintages** | USE-WITH-MANDATORY-FIX (§6) |
| CNN Fear&Greed | 1-yr rolling overwrite + methodology drift + reconstructed backfills | **REJECTED** → reconstruct from VIX/PC/HY-OAS primitives |
| VVIX (2007+), MOVE (gappy) | yfinance only, fragile, **cannot reach 2005** | EXPLORATORY-ONLY |
| Per-stock GICS sector (repo `universe.py`) | **PARTIAL** — labeled for 503 current members; **BLANK for 146 added/delisted rows** | USE-FOR-CURRENT-MEMBERS; prefer SPDR proxies |
| Sector SPDR ETFs (XLK…XLY) as category proxies | **YES** — 9 SPDRs 2004+; XLRE 2015+, XLC 2018+ | **USE** (survivorship-clean) |

**Bottom line:** macro×sector data is the *most* feasible (all free/PIT/deep) but the *least* likely to produce alpha. Forward/analyst data is the *most* promising on paper but the *least* PIT-feasible free. The honest design therefore (i) pre-registers only the PIT-safe trailing/derived-valuation and one fear tilt; (ii) treats forward/analyst signals as exploratory-permanent with an *optional* prospective harness; (iii) treats macro×sector as strictly descriptive with a max-statistic null.

---

## §3 — Data-Acquisition Plan

### 3.1 Reuse `src/youbet/stock/` UNCHANGED
`universe.py` (membership, `active_as_of`, `sector_as_of`, `cik_for`, `delisting_for`), `edgar.py`+`fundamentals.py` (PIT XBRL, filed-date keyed), `data.py` (yfinance close+OHLCV, spurious-price filter, mcap), `backtester.py` (walk-forward, T+1 strict `<`, mcap-bucketed costs, T-bill cash, `first_test_start_min` floor), `regime.py` (masks), `pit.py` (`PITFeatureSeries`, `validate_*_pit`, `apply_delisting_returns`), `etf/stats.py` (`block_bootstrap_test`, `excess_sharpe_ci`, `holm_bonferroni`, `simultaneous_sharpe_diff_ci`).

### 3.2 New modules under `src/youbet/stock/`
| Module | Responsibility |
|---|---|
| `fwd_valuation.py` | EDGAR-reconstructed **PIT trailing/derived valuation**: `ev_ebitda_pit` (`net_debt = LongTermDebt + ShortTermDebt − Cash`; `EV = mcap + net_debt`; `EBITDA = OperatingIncome + D&A`), `ebitda_yield_pit` (= EBITDA/EV), `trailing_pe_pit`. Pure functions over `fundamentals.py` facts + PIT price. **No analyst inputs → core-eligible.** |
| `macro_sector.py` | Wrapper over `etf/macro/fetchers.py` adding new FRED series (IG OAS, T10Y3M, DFF, DFII10, T5YIFR), AAII xls loader, CBOE put/call loader, the **`BAA10Y` credit-spread swap**, and a **real ALFRED code path** for CPI/INDPRO (§6.2). Emits a **per-series coverage report** (start-date floors). Maps normalized `sector_as_of` label → SPDR proxy. |
| `estimates.py` *(optional, harness-gated)* | Forward-collected estimate panel reader/writer. `consensus_as_of(ticker, decision_date)` returns the latest snapshot with `snapshot_timestamp < decision_date − 1 trading day` **[R1]**. **Never** reads FMP rows whose fiscal-period date precedes `snapshot_date` (avoids FMP realized-number backfill) **[R1]**. Raises if asked before the first snapshot. |
| `analyst.py` *(optional, exploratory)* | `upgrades_downgrades_events(ticker)` → numeric 1–5 path, **flagged exploratory + survivorship-contaminated**; forward-collected `recommendation_consensus_as_of`, `price_target_upside_as_of`. |
| `sic_backfill.py` *(optional, deferred — §8.4)* | EDGAR `submissions/CIK{cik}.json` → `sicDescription`; SIC→GICS crosswalk; only built if Open-Decision-4 chooses per-stock conditioning over SPDR proxies. |

### 3.3 PIT publication-lag entries — register via idempotent `register_individual_stock_lags()` (mirrors `commodity/pit.py`)
Market-derived (not revised), lag 0–1: `ig_oas`, `yield_3m10y` (T10Y3M), `real_rate_10y` (DFII10), `fed_funds` (DFF), `breakeven_5y5y` (T5YIFR), `putcall` (CBOE), `vvix`/`move` (exploratory). `aaii_sentiment` (7, none) already present.
**[R1] CPI/INDPRO are NOT fixed by a lag-table label.** `PITFeatureSeries` consumes only `lag_days`; a `revision_risk:"first_release"` tag is **inert**. The fix is a real ALFRED fetcher path (§6.2), with `release_date` taken from ALFRED `realtime_start`, *not* `period_index + fixed_lag` (which under-lags CPI by ~2–4 weeks).

### 3.4 Forward-collection harness — **OPTIONAL, single-source, exploratory-permanent [R1]**
`scripts/snapshot_forward.py`, weekly, write-once dated parquet. **Stood up only if Open-Decision-1 = "stay free" AND Open-Decision-2 = "do forward-collection" AND the user commits to a ~6-yr horizon.** Otherwise deferred.
- **Single lean source [R1]:** FMP `/stable/analyst-estimates` only (broadest free forward-estimate feed; 250 calls/day → batch ~500 active names over ~2 days). yfinance/Finnhub snapshotting **dropped from the standing harness** (snapshot-only, redundant, brittle scrapes).
- **Re-resolve membership weekly [R1]:** snapshot the FULL active S&P 500 membership *as-of each snapshot date* and log delisting events, so names are captured while still listed. **Even so, the delisted tail can never be back-snapshotted** → the accrued panel is **survivorship-contaminated by construction**.
- **Signal degradation, pre-registered [R1]:** FMP estimates are *algorithmic-aggregated*, not true sell-side consensus, and overwrite per fiscal period — so the accrued `revision_score` is a *proxy of a proxy*. Pre-commit the expected effect at HALF of already-decayed literature *and* discounted again for the algorithmic-vs-consensus gap and net of computed monthly turnover cost.
- **Storage / integrity:** `data/forward_snapshots/{YYYY-MM-DD}/fmp/{ticker}.parquet`, write-once; manifest logs `snapshot_date, source, ticker, n_fields, sha256`. **[R1]** re-fetch of an existing `(snapshot_date, source, ticker)` is a hard error; any file whose recomputed sha256 ≠ manifest is quarantined and never read.
- **Status:** the prospective revision test is **EXPLORATORY-PERMANENT** on free data (survivorship bar unreachable). Only WRDS/Benzinga (delisted-inclusive) could ever make it confirmatory.

### 3.5 Macro/sector caching (PIT-safe, no forward-collection)
FRED via `fredapi` (`FRED_API_KEY` present). **[R1] Market-derived series** use `get_series` (not revised). **CPI/INDPRO** use the real ALFRED path (§6.2). Annotate start-date floors: VVIX 2007-01, MOVE fragile, CBOE P/C 2006-11, DTWEXBGS 2006-01, DFII10 2003-01 — **so any macro panel mixing these starts ~2007, not 2005**; recompute Phase-0 MDE on the shorter effective sample. AAII xls + CBOE CSV downloaded once, snapshot-cached with a fetch-date stamp and a periodic re-fetch diff check (detect silent old-week revisions). Sector SPDRs via existing `data.fetch_stock_prices`.

### 3.6 New config keys — additive; locked thresholds inherited from stock-selection (see `config.yaml`).

---

## §4 — Experiment Ladder

Core is **small by design**. Every pre-registered strategy + hyperparameters + expected band is frozen in `precommit/phase{N}.json` BEFORE the phase runs.

### Phase 0 — Power / Feasibility / Plant-test (MANDATORY GATE)
- Recompute MDE per-strategy via Romano-Wolf simultaneous framework at TE anchors {4%,8%,12%}; **[R1]** report both Holm(N=2) (conservative) and effective-N-from-correlation.
- **PIT plant-tests:** (a) inject a known-future price signal → backtester must raise `PITViolation`; (b) **[R1] inject a known CPI revision → backtester must use the pre-revision (ALFRED first-vintage) value** (guards the §6.2 fix).
- Survivorship gap check (≥1% CAGR membership-gated vs ungated); cost sanity.
- **[R1] sector-label assertion:** no ticker entering any sector bucket has a missing/`'nan'` sector (guards the §6.1 fix). Per-stock GICS conditioning is descriptive-only; the confirmatory tilt uses SPDR proxies.
- Macro coverage report (per-series start floors).
- **Tier decision:** MDE > 0.30 → headline point-estimate-only (expected branch).

### Phase 1 — Trailing/derived valuation (PRE-REGISTERED CORE member: 1)
- **Hypothesis:** EDGAR-reconstructed EBITDA yield (EBITDA/EV) earns a small value premium (Loughran-Wellman EV/EBITDA ~5.28%/yr gross, mostly small-cap-attenuated).
- **Construction (PIT-safe):** decile-rank active S&P 500 on `ebitda_yield_pit`; top-decile, equal-weight, monthly rebalance, T+1, costs on. No analyst/forward inputs.
- **Pre-registered core (1):** `evebitda_yield`.
- **[R1] Expected band anchored on the in-universe sibling**, not re-halved literature: ≈ **+0.05 to +0.35** (the directly-comparable prior `value_EY` realized +0.351 and failed; `quality_value_zsum` +0.068). Re-halving Loughran-Wellman would double-count decay.
- **Descriptive-only companions [R1]:** `earnings_yield_v2` (admitted ~95% clone of `value_EY`), reported as point estimate + CI, *not* in the gate family. Report the pairwise daily-excess-return correlation matrix of all value candidates and state the effective number of independent tests.
- **Expectation:** fails gate, like `value_EY`.

### Phase 2 — Forward/analyst signals: DESCRIPTIVE-only + optional harness launch
- **Hypothesis (descriptive):** estimate-revision momentum (So 2013 ~5.8%/yr; practitioner ~7.6%/yr decile, IC 0.23) is the strongest analyst signal but is **not PIT-backtestable free**.
- **What runs now:** (a) descriptive characterization of `yfinance .upgrades_downgrades` (2012+, survivorship-contaminated) — no gate; (b) *if opted-in* launch the single-source FMP harness (§3.4) and verify clean weekly accrual; (c) freeze the **EXPLORATORY-PERMANENT** prospective precommit (`prospective_revision_momentum.json`) — *not* "confirmatory in 2032."
- **No confirmatory claim** on any snapshot-only/forward-collected signal.

### Phase 3 — PIT composite: DESCRIPTIVE-only [R1]
- **Hypothesis:** PIT-safe trailing value + quality (reuse `quality_roe_ttm`) + EBITDA yield improves the spread.
- **Strategy (1):** `pit_value_quality_evebitda_zsum` — z-sum of (earnings yield, ROE TTM, EBITDA yield), top-decile EW.
- **[R1] DESCRIPTIVE-only**, not a gate-family member: it is the prior `quality_value_zsum` (+0.068) with an EBITDA leg ⇒ ~clone, cannot independently confirm. Point estimate + CI only.

### Phase 4 — Macro × sector: ONE pre-registered tilt (CORE member: 2) + disciplined descriptive cross
Per `evidence-macro-sector`, a naive cross is multiplicity death. The pre-registered confirmatory footprint is a **single binary tilt**:
- **Hypothesis (one signal, one binary tilt, one threshold):** "Overweight defensive sectors (XLP/XLU/XLV constituents) vs cyclicals (XLK/XLY/XLI/XLB/XLE) when the variance-risk-premium is elevated" (Bollerslev-Tauchen-Zhou; VRP aggregate OOS R²~4%).
- **Construction (PIT-safe):** VRP proxy = VIX − realized vol (both PIT, not revised, deep — **no AAII/VVIX/MOVE/CNN in the confirmatory path [R1]**); threshold = pre-committed quantile (frozen). **[R1] Buckets via SPDR proxies** (survivorship-clean), *not* per-stock `sector_as_of` (which has the `'nan'`-label + reclassification traps). Coarse defensive/cyclical buckets chosen to be invariant across the 2016/2018 GICS reclassifications.
- **Pre-registered core (1):** `vrp_defensive_tilt`. Distinct hypothesis from the value signal → legitimate 2nd family member.
- **Expected band (half-lit):** +0.05 to +0.20 (BTZ aggregate, short-horizon, halved). **Expectation: fails gate.**
- **[R1] Descriptive cross discipline:** the broader 5 macro × 11 sector × 3 threshold grid is **pre-registered in full** in the precommit JSON (no post-hoc cell selection). It is reported as the **whole-grid excess-Sharpe distribution overlaid with a random-permutation null band**; **no single cell is called a "finding" unless it survives a max-statistic (Romano-Wolf / White reality-check) permutation test across all 165 cells.** This is the honest descriptive analogue of Holm and the antidote to the Molchanov-Stangl data-snooping failure mode. The hierarchical gate (cross runs as a *gate* only if the single tilt passes) protects the confirmatory p-value; the max-statistic null protects the descriptive layer.

### Phase 5 — Source-period-bias + robustness battery (MANDATORY for ANY positive point estimate)
Per `feedback_source_period_bias`, treat any positive point estimate as **guilty until** it passes. **[R1] Cross-sectional analogues** (the v1.0 sleeve-style checks were ill-posed for decile baskets):
1. **Placebo = characteristic-shuffle / random-decile null.** Permute the ranking characteristic across the cross-section (preserving rebalance calendar + turnover), rebuild the excess-Sharpe null; the real signal must beat the shuffled null. **Plus** a long-leg-mean-equalization placebo (demean the long-basket's excess return to the universe mean) — the direct computable analogue of the sleeve-mean shift.
2. **Scaling = decile-breakpoint monotonicity sweep {5,10,20,30%}** (committed). A real signal decays monotonically as the basket widens toward the universe; an artifact is flat/non-monotone. *(For `vrp_defensive_tilt` the sleeve-style defensive-overweight-magnitude sweep DOES apply and is used there.)*
3. **Sub-period robustness** — 3 sub-bootstraps (≈2007-2013 / 2013-2018 / 2018-2024, adjusted for the ~2007 macro floor) + GFC/COVID exclusion. Sign-stable or it's regime-luck.
- **Cost sensitivity:** re-run at commission anchors {0.5, 2.0, 5.0, 10.0} bps.
- **Kill:** any positive Phase-1/4 result failing *any* leg is **retracted** (kept in the family denominator for accounting, per stock-selection convention).

### Phase 6 — (Optional) Multi-signal ML — EXPLORATORY-ONLY
- Reuse `MLRanker`; add PIT-safe macro/fear features + (if accrued) forward-collected estimate features. Walk-forward refit, date-contiguous validation (R6 rule), feature-importance stability ≥50% across adjacent folds.
- **[R1] `.upgrades_downgrades` as a feature re-imports survivorship leak** (missing-rating ⇒ delisting): either **exclude it**, or restrict the ML window to ≥2013, treat missingness as **coverage-invalid ⇒ exclude the name from that fold** (R6 policy; never impute), and add a missingness-vs-delisting correlation diagnostic. Point estimates only; ML "destroyed value" in etf and spanned 0 in stock-selection.

### Pre-registered core (frozen): **2 strategies** — `evebitda_yield`, `vrp_defensive_tilt`.
Reported via Romano-Wolf simultaneous CI (headline) + Holm(N=2) (cross-check). Expectation: **0/2**. All other strategies are descriptive companions.

---

## §5 — Statistical Plan

### Gate (inherited, LOCKED from stock-selection)
PASSES iff `excess_sharpe_point > 0.20` **AND** simultaneous CI / Holm-adjusted significance holds **AND** 90% block-bootstrap `ci_lower > 0`. Estimand = **Sharpe of the daily excess series `Sharpe(strat − SPY)`** (information-ratio-like). **Critical (vti-as-challenger v1 bug):** `block_bootstrap_test` returns Sharpe-of-excess but `excess_sharpe_ci.point_estimate` returns Sharpe-diff — keep the gate consistently on Sharpe-of-excess. Benchmark = SPY (single, no shopping).

### Family + multiplicity **[R1]**
- **Pre-registered family = {`evebitda_yield`, `vrp_defensive_tilt`} = N=2** (distinct hypotheses: value vs macro). **Romano-Wolf `simultaneous_sharpe_diff_ci` is the HEADLINE multiplicity control** (it uses the empirical correlation, correct for a possibly-correlated family); Holm(N=2) is the conservative cross-check. *(v1.0 had this backwards — Holm-on-p-values assumes independence the value cluster violated.)*
- All other strategies (`earnings_yield_v2`, `pit_value_quality_evebitda_zsum`, the macro×sector cross, ML) are **DESCRIPTIVE** — point estimate + CI, no gate, no place in the family denominator.
- **Retracted-but-counted:** a core strategy that fails Phase-5 robustness stays in the N=2 denominator (artifact decomposition ≠ removal).
- **Macro×sector multiplicity** is contained structurally: 1 tilt in the family; the 165-cell cross is descriptive with a max-statistic null (§4 Phase 4).

### Block bootstrap (inherited, LOCKED)
Stationary Politis-Romano, mean block 22 days, 10,000 reps, seed 42. 90% CI for the gate; report two-sided p and one-sided `p_up`.

### Power (Phase 0, pre-registered)
Recompute MDE at TE {4%,8%,12%} × `target_sharpe_diffs` {0.05…0.50}; **[R1]** per-strategy under Romano-Wolf simultaneous framework (not one Holm scalar). Pre-commit: **MDE > 0.30 → headline point-estimate-only**. Document expected MDE ≈ +0.45 to +0.55 → expectation 0 passes. Forward-collection note: the prospective revision test's decision-power date is computed from accrued-snapshot length to reach MDE < +0.50 — but is **EXPLORATORY-PERMANENT** regardless (survivorship ceiling, §3.4).

---

## §6 — PIT & Survivorship Safeguards (incl. 3 mandatory repo fixes)

### Mandatory repo fixes (verified against the codebase by reviewers — do BEFORE any results)
1. **[R1] `sector_as_of` returns the literal string `'nan'`** (not `None`) for the 146 blank/delisted membership rows (`str(row.iloc[0]['gics_sector'])`). Because the unlabeled set is structurally correlated with delisting, any per-stock sector bucketing assigns delisted names to a phantom `'nan'` sector ⇒ **survivorship-correlated label corruption inside a strategy**. **Fix:** patch `universe.sector_as_of` to return `None` on NaN; normalize `{None,'nan','',NaN}` → MISSING everywhere; route MISSING through an explicit exclude/backfill policy; Phase-0 assertion + unit test. Prefer **SPDR proxies** for the confirmatory tilt so the gap is moot.
2. **[R1] CPI/INDPRO leak via `fredapi.get_series`** (returns latest-revised). The `revision_risk` lag-table tag is **inert** (`PITFeatureSeries` reads only `lag_days`). **Fix:** real ALFRED path in `macro_sector.py` — build `PITFeatureSeries` from `get_series_all_releases`, key `release_date` to the actual `realtime_start`, and at decision date select the latest observation with `realtime_start < decision_date`. Do **not** rely on `get_series_first_release` alone (the PIT-correct value can be a later vintage whose `realtime_start < d`). Prefer CPIAUCNS (NSA, not annually back-revised) where SA isn't required. Label INDPRO honestly as an industrial-production proxy, not "ISM PMI." Phase-0 CPI-revision plant-test guards it.
3. **[R1] Credit-spread fetcher is repo-broken.** `etf/macro/fetchers.py::fetch_credit_spread` hard-codes `BAMLH0A0HYM2`, which FRED now truncates to a rolling 3-yr window (ICE license, ~2026) → returns only 2023+ and silently NaN-pads pre-2023. **Fix:** swap to **`BAA10Y`** (Moody's Baa−10Y, public-domain, daily 1986+).

### Analyst/estimate-specific safeguards (leakiest family in the repo)
4. **No restated estimates — snapshot timestamp is the only valid release date.** `estimates.consensus_as_of(t,d)` returns rows with `snapshot_timestamp < d − 1 trading day` **[R1]** (mirror the price-path strict `<` + T+1 buffer; reject same-day). **Never** read FMP's current-stored value for a historical decision date, and **never** read FMP rows whose fiscal-period date precedes `snapshot_date` (FMP backfills past-FY "estimates" to realized numbers) **[R1]**.
5. **Forward P/E uses as-of consensus, not today's:** `forward_pe(t,d) = price_pit(t,d) / consensus_eps_as_of(t,d)`, both strictly pre-`d`. The harness makes this structurally enforced by keying on snapshot timestamp.
6. **`.upgrades_downgrades` is EXPLORATORY-ONLY** — revisable rolling store (issue #1880), starts ~2011-12, **current-listed only ⇒ survivorship-contaminated**. Never a gate input; as an ML feature, see §4 Phase 6 (exclude or coverage-invalid⇒exclude, never impute).
7. **Survivorship in analyst data is structural and incurable free [R1]** — all free analyst sources serve current-listed names only; the forward-collection panel drops any name that delists during collection. The analyst/forward layer therefore **cannot** satisfy the survivorship-free bar for a gate, now or in 2032. Only WRDS/Benzinga retain delisted-name estimate history.
8. **No reconstructed/single-compilation sentiment in the confirmatory path [R1]:** AAII xls (single current compilation), CNN F&G (methodology drift), third-party VVIX/MOVE backfills are descriptive/feature-only. Config assertion: the confirmatory macro feature set ⊆ {market-derived, not-revised} series. The `vrp_defensive_tilt` uses only VIX − realized-vol.
9. **GICS label anachronism [R1]:** 2024-snapshot labels are forward-looking for pre-2016 (Real-Estate spinout) / pre-2018 (Comm-Services). The confirmatory tilt uses coarse defensive/cyclical buckets invariant across those breaks (or SPDR proxies whose own historical constituents handle it); residual mislabel documented in Phase-5 sub-period robustness.
10. **Standard engine guards carry over:** T+1 strict `<`, `validate_fundamentals_pit` (filed < decision), `apply_delisting_returns`, `_filter_spurious_prices`, `first_test_start_min` floor.

---

## §7 — Kill Criteria & Deliverables

### Kill criteria
- **Phase 0:** MDE > 0.30 (expected) → no gate claims for headline signals; proceed descriptive + optional harness. Infra/PIT plant-test failure (price OR CPI-revision) → fix before any results. Sector-`'nan'` assertion failure → fix `sector_as_of` first.
- **Per-strategy:** fails the gate (expected 0/2). A positive point estimate failing *any* Phase-5 leg is **retracted** (kept in the N=2 denominator).
- **Macro×sector:** `vrp_defensive_tilt` fails (expected) → entire cross is descriptive-with-max-statistic-null; no hierarchical gate descendants.
- **Forward-collection:** harness fails clean weekly accrual → fix; the prospective (exploratory-permanent) test simply waits.
- **Whole-workflow stop:** after Phases 0–5 + ≥3 adversarial review rounds, with the optional harness running (if opted-in) and the prospective precommit frozen. The free-data ceiling is then reached.

### Deliverables
- `research/log.md` (read at every session start); `research/final-report.md` (Romano-Wolf simultaneous + Holm(N=2) table, per-phase results, paper-vs-net delta, the explicit data-feasibility verdict, harness launch record).
- `precommit/phase{0,1,4}.json` + `precommit/prospective_revision_momentum.json` (frozen now, exploratory-permanent).
- `data/forward_snapshots/` (if opted-in) — accruing dated parquet + sha256 manifest.
- `holdout_tracking.md` — prospective tracking, decision-power date pre-committed (mirrors international-etf / macro-exploratory / cagr-max).
- **Methodology contributions to fold back:** `ev_ebitda_pit` PIT construction, the **ALFRED real-time-vintage fetcher**, the **`BAA10Y` credit-spread fix**, the **`sector_as_of` NaN fix**, `estimates.consensus_as_of` snapshot-PIT contract, the **max-statistic descriptive-cross discipline**, cross-sectional Phase-5 placebo analogues.

---

## §8 — Open Decisions for the User (recommended defaults bolded)

1. **Free-only vs paid analyst data.** Stay free (forward-collect-only; analyst/forward signals never confirmatory) **vs** acquire **WRDS I/B/E/S** (if affiliated → true 2005+ PIT confirmatory test, the single highest-value change) / Benzinga (2011-12+, paid) / Sharadar (universe quality, no forward estimates). **Default: stay free.** A confirmatory analyst test on free data is impossible; only WRDS makes it honest.
2. **Pure-historical vs also forward-collection.** Run only the PIT-safe historical work (Phases 1/4 core + descriptive) **vs** also stand up the weekly FMP harness. **Default: do the historical core regardless; stand up the harness only if (1)=free AND you commit to a ~6-yr horizon** — and even then it is exploratory-permanent (survivorship ceiling). If you won't commit 6 years, **defer the harness** [R1].
3. **Macro×sector aggressiveness.** Single VRP/defensive tilt (N=1 in family) **vs** larger theory set **vs** broad cross (rejected — Holm death). **Default: N=1 + hierarchical + max-statistic-null descriptive cross.** Evidence is decisive it's a data-mining trap; youBet failed it twice.
4. **Delisted-tail GICS handling.** EDGAR-SIC backfill (61 CIK rows; 85 unrecoverable) **vs** SPDR proxies for all category conditioning. **Default: SPDR proxies; drop `sic_backfill.py` and its Phase-0 gate check from v1** [R1] — per-stock GICS on the delisted tail is irrecoverably incomplete free and the SPDR route sidesteps the survivorship gap.
5. **Revision-signal prospective eval date.** **Default: 2032-06-01** (~6yr weekly snapshots), recomputed in Phase 0 — but labeled **exploratory-permanent**, not confirmatory [R1]. Frozen in `precommit/prospective_revision_momentum.json` now.
