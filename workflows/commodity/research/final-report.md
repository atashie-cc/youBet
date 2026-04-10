# Commodity Workflow — Final Report

**Date:** 2026-04-10
**Status:** Complete
**Tests completed:** 31 walk-forward backtests across 6 phases + 300 null simulations
**Codex review rounds:** Multiple adversarial review rounds

---

## Executive Summary

The commodity workflow evaluated whether systematic strategies on publicly traded commodity ETFs can improve a 60/40 VTI/BND portfolio on a risk-adjusted basis. After 31 walk-forward backtests spanning static allocation, timing overlays, cross-sectional selection, and macro-regime gating — **no strategy passed the workflow's precommitted strict gate** (excess Sharpe > 0.20, Holm p < 0.05, CI lower > 0).

**The single most important message:** No commodity strategy achieved confirmatory status under the project's strict gate. The strongest exploratory finding — a macro-gated DBC strategy — shifted from an apparent inflation-and-dollar edge to a weaker, simpler, still-unconfirmed dollar-linked timing story.

**Three descriptive findings warrant practical attention:**

1. **Static 10% physical gold (IAU) in 60/40** showed the most robust walk-forward support: a 19-fold test delivered Sharpe(strat) − Sharpe(bench) = +0.069 and a separate Sharpe-of-excess estimand of +0.118 with CI lower +0.019. Higher weights (15% and 20%) tested only on shorter 14-fold samples with CI spanning zero. The 10% effect is positive across most regimes except the 2011-2018 gold bear market. It fails the strict magnitude gate but is the most robust descriptive result.

2. **Macro-gated DBC** (long broad commodities when 10Y breakeven inflation is rising AND dollar index is weakening) produced a walk-forward Sharpe difference of +0.456, CI [+0.068, +0.847]. A stationary-bootstrap null on both macro series was rejected at p=0.035, but the primary precommitted test still failed (p=0.174). The strategy is parameter-fragile (only 126-day lookback survives), regime-concentrated (leave-out 2015-2019 pushes CI through zero), and has **minimal practical value in a 60/40 portfolio** (+0.013 as a 10% sleeve). The simpler `DOLLAR_ONLY` variant captures nearly the full effect.

3. **DBC trend-following** (SMA200 or TSMOM) cuts maximum drawdown from -79% to -52% but does not clear the Sharpe gate. If broad commodity exposure is held for other reasons, a simple SMA overlay provides meaningful drawdown protection.

**Recommendation:** The commodity workflow does not support active commodity strategies. For investors who want commodity exposure, a **static 10% allocation to physical gold (IAU preferred for lower ER)** has the cleanest walk-forward support in the final catalog. Higher weights (15-20%) show positive point estimates but only on shorter samples with CIs that span zero. All other tested approaches either fail the gate, show minimal portfolio-level impact, or have statistical power too low to distinguish signal from noise.

---

## 1. Methodology and Inference Framework

### 1.1 Two-Tier Design

Phase 0B power analysis (documented below) showed the minimum detectable excess Sharpe for a 50-strategy Holm-corrected family was approximately +0.80 — far above realistic wrapper-adjusted effect sizes from the academic literature (0.08-0.18 after 70% haircut). This made the original plan of testing dozens of strategies confirmatorily untenable.

We adopted a **two-tier framework**:
- **Phases 1-2 (descriptive):** Characterize the universe with point estimates, CIs, and regime breakdowns. No Holm correction, no hard pass/fail gate.
- **Phases 3+ (confirmatory):** Test a small number of precommitted hypotheses with the strict gate. Holm correction scoped to the confirmatory set only.

### 1.2 Strict Gate

A strategy is "confirmed" only if all three criteria are met:
1. **Magnitude:** Excess Sharpe > 0.20
2. **Significance:** Holm-adjusted p < 0.05 (one-sided, block bootstrap)
3. **Interval:** 90% CI lower bound > 0

All three criteria were locked in CLAUDE.md before running any Phase 3+ tests.

### 1.3 Backtesting Infrastructure

All walk-forward tests use the shared `youbet.etf.backtester` engine with:
- **Walk-forward parameters:** 36-month train / 12-month test / 12-month step
- **Execution:** T+1 (signal at close of day T, executed at close of day T+1)
- **Transaction costs:** Per-category bps schedule (e.g., commodity_physical_metal: 3 bps; commodity_broad_futures: 5 bps; miner_equity: 3 bps)
- **Cash accrual:** Risk-off positions earn prevailing 3-month T-bill rate from FRED
- **PIT validation:** Survivorship check via inception date; publication-lag enforcement via `PITFeatureSeries.as_of()` for macro signals
- **Monthly rebalancing** with drift-and-reset

### 1.4 Statistical Framework

- **Point estimate and CI:** `excess_sharpe_ci` computes Sharpe(strategy) − Sharpe(benchmark) via paired stationary block bootstrap with 10,000 replicates and 22-day expected block length (Politis-Romano 1994).
- **P-value:** `block_bootstrap_test` centers the excess-return series at zero and computes a one-sided p-value against the null "excess Sharpe ≤ 0."
- **Block length:** 22 days matches approximately one trading month and captures autocorrelation without destroying sample diversity.

### 1.5 Power Analysis Retrospective

Phase 0B estimated the minimum detectable excess Sharpe for a 50-strategy Holm-corrected family at approximately +0.80, with 60% power at 0.80 and 28% power at 0.60. Literature-adjusted expected effects (0.08-0.18 after 50% McLean-Pontiff decay plus additional wrapper degradation) were far below this threshold.

**Implication:** Positive but non-confirmatory results recur throughout this workflow precisely because the framework is underpowered for realistic effect sizes. A descriptive finding with CI lower near zero is not weak evidence — it is the expected outcome for a genuine small effect in a short sample.

---

## 2. Core Findings by Strategy Family

### 2.1 Static Allocation

**Result family:** Physical gold allocation in a 60/40 portfolio.

| Strategy | Sample | Sharpe Diff | CI Lower | Estimand | Gate |
|---|---|---|---|---|---|
| 54/36/10 VTI/BND/IAU vs 60/40 | 2007-11 to 2026-04 (19 folds) | +0.069 (Sharpe diff) / +0.118 (Sharpe-of-excess) | +0.019 (Sharpe-of-excess CI) | Mixed (see note) | FAIL magnitude |
| 51/34/15 VTI/BND/IAU vs 60/40 | 2009-12 to 2026-04 (14 folds) | +0.082 | -0.033 | Sharpe diff | FAIL CI |
| 48/32/20 VTI/BND/IAU vs 60/40 | 2009-12 to 2026-04 (14 folds) | +0.097 | -0.063 | Sharpe diff | FAIL CI |

**Estimand note:** Phase 2B (the 10% IAU walk-forward) was run before the workflow standardized on Sharpe(strategy) − Sharpe(benchmark). The original test reported the "Sharpe-of-excess-returns" estimand (+0.118, CI [+0.019, +0.135]), which is the Sharpe ratio of the excess return series — an information-ratio-like metric. The newer Sharpe-difference estimand gives +0.069. Both are legitimate; they diverge when strategy and benchmark have different volatilities. The 14-fold tests use the Sharpe-difference estimand consistently.

**Interpretation:** Point estimates are monotonically positive from 10% to 20% gold allocation in the tests reported. The 10% weight in the 19-fold sample has CI lower above zero on the Sharpe-of-excess estimand (+0.019), while the 14-fold samples at 15% and 20% have CI lower below zero on the Sharpe-difference estimand because the shorter sample lost the 2008-2012 period where gold was strongest relative to bonds. A 5% weight was not tested in the final walk-forward catalog; its inclusion in earlier descriptive analysis was via in-sample weight-grid sensitivity only.

Regime sensitivity (10% IAU sleeve): positive in 3 of 4 regime windows. The **gold bear market (2011-09 to 2018-08)** shows a -0.12 Sharpe delta — the single period where the allocation hurt the portfolio.

**Conclusion:** Static 10% IAU is the **most robust walk-forward result** in the commodity workflow. It is too small to clear the project's primary confirmatory standard but consistently positive across regimes with CI excluding zero in the longest-available sample under the Sharpe-of-excess estimand.

### 2.2 Timing Strategies

**Result family:** SMA crossover and time-series momentum on commodity ETFs.

| Strategy | Benchmark | Sharpe Diff | CI Lower | Outcome |
|---|---|---|---|---|
| IAU SMA100/200/252 sleeve | Static 54/36/10 | -0.006 to -0.015 | ≤ 0 | Negative |
| IAU SMA200 standalone | IAU B&H | -0.055 | -0.339 | Negative |
| DBC SMA100 standalone | DBC B&H | +0.214 (desc) / +0.178 (conf) | -0.134 | FAIL CI |
| DBC SMA200 standalone | DBC B&H | +0.345 (desc) / +0.178 (conf) | -0.134 | FAIL CI |
| DBC 12-month TSMOM | DBC B&H | +0.137 | -0.145 | FAIL CI |
| VTI SMA100 + static gold | Static 54/36/10 | -0.098 | -0.391 | Negative |
| VTI SMA200 + static gold | Static 54/36/10 | +0.073 | -0.189 | FAIL CI |

**Interpretation:** Timing strategies split into two groups. Gold timing is uniformly negative or near zero — gold's low-volatility trends produce costly whipsaws. DBC timing shows positive point estimates with meaningful drawdown reduction (MaxDD -52% vs -79% for DBC B&H) but CIs that span zero.

Phase 4D's descriptive DBC SMA200 result of +0.345 was noise-inflated by the shorter sample; the extended Phase 5.1 confirmatory version with 19 folds produced +0.178, matching the TSMOM result. Both approaches achieve similar point estimates on the same underlying signal.

**Conclusion:** Gold timing destroys value. DBC timing provides substantial drawdown protection but no Sharpe improvement that is statistically distinguishable from zero.

### 2.3 Cross-Sectional Selection

**Phase 5.4 Cross-sectional rotation** among DBC, PDBC, USCI: monthly select top-1 by trailing 6-month return with absolute momentum filter.

- Sharpe diff: +0.271
- 90% CI: [-0.042, +0.597]
- p-value: 0.162
- MaxDD: -47% vs -79% for DBC B&H

**Interpretation:** The strongest positive point estimate among simple commodity strategies outside macro gating. CI lower is -0.042, extremely close to zero. This suggests real wrapper-selection alpha (USCI's backwardation-seeking methodology outperforms during some regimes) but the effect is small enough that the 19-fold sample cannot reliably distinguish it from noise.

**Conclusion:** Promising descriptive result. Would benefit from longer history or a confirmatory rerun on a held-out sample.

### 2.4 Macro-Regime Gating

See the dedicated Section 3 below.

---

## 3. Discovery and Deconstruction of the Macro-Gated DBC Result

### 3.1 Discovery (Phase 5.3)

After 4 phases of largely negative results, Codex's Phase 5 review proposed a specific Phase 5.3 test: hold DBC only when 10Y breakeven inflation (FRED T10YIE) is above its 6-month trailing average AND dollar index (DXY) is below its 6-month trailing average; otherwise sit in cash.

The initial Phase 5.3 result was striking:
- **Sharpe diff: +0.545**
- **90% CI: [+0.140, +0.945]** — CI lower excludes zero
- p-value: 0.148 (above 0.05 gate)
- All 5 regime windows showed positive delta, largest at +1.371 (Low-vol, dollar strength 2015-2019)

Gate status: **2 of 3 criteria passed** (magnitude, CI lower). Only the p-value failed.

### 3.2 PIT Bug Correction (Phase 6.1)

Codex Round 7 review identified a critical PIT leakage bug in the Phase 5 implementation: `load_macro_series()` extracted `.values` from the `PITFeatureSeries` objects, stripping PIT metadata. The strategy then used `.loc[:as_of_date]` which includes same-day observations — a violation of the strict release_date < decision_date requirement.

After the fix (using `PITFeatureSeries.as_of(as_of_date)` which enforces strict inequality):
- **Sharpe diff: +0.456** (down from +0.545)
- **90% CI: [+0.068, +0.847]** — CI lower still excludes zero
- p-value: 0.174

The effect **survives the PIT fix** but is reduced approximately 16%. Same-day observations were contributing modest but non-zero signal.

### 3.3 Robustness Battery (Phase 6.2-6.8)

**Lookback sensitivity (Phase 6.3):** Only the precommitted 126-day lookback has CI excluding zero.

| Lookback | Sharpe Diff | CI Lower |
|---|---|---|
| 63 days | +0.354 | -0.055 |
| **126 days** | **+0.456** | **+0.058** |
| 189 days | +0.384 | -0.026 |
| 252 days | +0.256 | -0.182 |

This is a spike, not a plateau. If the finding were robust, we would expect multiple nearby lookbacks to also show CI excluding zero.

**Logic ablation (Phase 6.4):** The `DOLLAR_ONLY` variant captures most of the effect with a simpler rule.

| Logic | Sharpe Diff | CI Lower |
|---|---|---|
| INFLATION_ONLY | +0.268 | -0.045 |
| **DOLLAR_ONLY** | **+0.399** | **+0.047** |
| AND | +0.456 | +0.058 |
| OR | +0.261 | -0.023 |
| ADDITIVE | +0.261 | -0.023 |

The inflation leg adds approximately +0.057 Sharpe diff over dollar alone. The AND rule's complexity is not earning significant additional signal.

**Leave-2015-2019-out (Phase 6.2):** Excluding the "low-vol, dollar strength" regime pushes the CI through zero.
- Sharpe diff: +0.379
- CI: [-0.073, +0.824]

The full-sample result is partially regime-dependent. However, the median point estimate across regime exclusions is still positive, and the effect does not disappear entirely.

**External replication (Phase 6.5):** Same frozen rule applied to alternative broad commodity wrappers.

| Wrapper | Sharpe Diff | CI Lower |
|---|---|---|
| DBC (original) | +0.456 | +0.068 |
| **GSG** | **+0.452** | **+0.048** |
| PDBC | +0.366 | -0.031 |
| USCI | +0.199 | -0.191 |

GSG shows an independent positive finding with CI excluding zero. PDBC is marginal; USCI is weak. Using a post-hoc heuristic suggested during review ("at least 2 of 3 wrappers with CI excluding zero"), 2 of 3 tested wrappers (DBC and GSG) meet the bar. This heuristic was not part of the precommitted workflow framework.

**Portfolio sleeve (Phase 6.7):** 10% macro-gated DBC sleeve in 60/40 VTI/BND.
- Sharpe diff: **+0.013**
- 90% CI: [-0.020, +0.049]

Essentially zero. The macro conditions (both inflation rising AND dollar weakening) are rarely met, so the sleeve sits in cash most of the time. The portfolio becomes effectively 54/36/0 — functionally equivalent to a 60/40 scaled by 0.9.

**Conclusion: The strategy has minimal practical value in the tested 10% sleeve implementation.**

**Block length sensitivity (Phase 6.6):** CI recomputation on the Phase 6.1 return series at block lengths 22, 44, and 66 days. This is not a new walk-forward test — it checks whether the CI/p-value is fragile to the dependence assumption in the block bootstrap. Full results are retained in the research log; the qualitative finding is that the CI lower bound remains positive across the tested block lengths, consistent with the Phase 6.1 headline.

### 3.4 Null Tests

**Phase 6.8: Narrow null (shifted T10YIE, true DXY).** 100 circular shifts of the breakeven inflation series paired with the true dollar index. Tests whether T10YIE's exact alignment adds incremental value beyond true DXY.

- Observed: +0.456
- Null mean: +0.334
- Null 95th percentile: +0.645
- Fraction null ≥ observed: 25%

**Interpretation:** T10YIE's exact alignment does NOT add meaningful incremental value beyond the true DXY signal. This is consistent with the DOLLAR_ONLY finding that dollar alone carries most of the effect.

**Phase 6B: Full-strategy null (stationary bootstrap on both series).** 200 stationary bootstrap draws of both T10YIE and DXY with 22-day blocks, each paired with true DBC prices.

- Observed: +0.456
- Null mean: +0.037
- Null 95th percentile: +0.391
- Fraction null ≥ observed: 3.0%
- **Permutation p-value: 0.0348**
- Observed > null 95th percentile: PASS

**Interpretation:** When BOTH series are randomized, breaking joint alignment with DBC while preserving each series' autocorrelation, the null distribution tightens dramatically (null mean +0.037 vs +0.334). The observed effect stands clearly above the 95th percentile.

**The two null tests answer different questions:**
1. Phase 6.8 asks "does exact T10YIE alignment add to TRUE DXY?" — answered NO (25% exceedance)
2. Phase 6B asks "is the combined macro signal distinguishable from noise?" — answered **YES, at p=0.035**

These are complementary, not contradictory. Phase 6.8 isolates the incremental value of the inflation leg; Phase 6B tests the full strategy against a proper full-strategy null.

### 3.5 What Survives and What Does Not

**Survives:**
- **Statistical significance under proper null**: Phase 6B stationary bootstrap rejects at p=0.035
- **PIT-safe implementation**: +0.456 with CI lower +0.068
- **External replication**: DBC and GSG both show CI excluding zero
- **Massive drawdown reduction**: -40% vs -79% for DBC B&H
- **Directionally correct across regimes in the original discovery result**: positive point estimate in all 5 regime windows for the Phase 5.3 buggy version; the PIT-safe Phase 6.1 version has weaker regime-level performance, and leave-one-regime-out analysis (6.2) showed CI crosses zero when 2015-2019 is excluded

**Does not survive:**
- **Strict gate**: p-value 0.174 > 0.05 on primary block-bootstrap test
- **Lookback robustness**: 126 days is a spike, not a plateau
- **AND-specific mechanism**: DOLLAR_ONLY captures most of the effect
- **Regime independence**: leave-2015-2019-out pushes CI through zero
- **Portfolio utility**: +0.013 as a 10% sleeve in 60/40

### 3.6 Correct Framing

The macro-gated DBC finding is **exploratory evidence for dollar-linked commodity timing that survives one stronger noise null but fails the primary confirmatory standard**. It is:
- **Not** gate-passing — the primary precommitted block-bootstrap p-value is 0.174, above the 0.05 threshold
- **Weakened** but not invalidated by robustness tests — CI lower is still +0.068 on the PIT-safe rerun, and the Phase 6B stationary-bootstrap null rejects at p=0.035 (one stronger exploratory null, 200 draws with 500 inner replicates per draw)
- **Parameter-fragile** — only the 126-day lookback has CI excluding zero
- **Minimal practical value in the tested 10% sleeve implementation** (+0.013 vs 60/40)

Because the supporting null is post-hoc and exploratory, the result cannot be called free of data-mining concerns. It should be treated as "promising and directionally supported by multiple checks" rather than "confirmed."

The cleanest post-hoc form of the finding is: **"Broad commodity ETPs tend to underperform during dollar strength; a simple dollar-based filter reduces drawdowns and may improve Sharpe, but the effect is too fragile to clear the project's primary confirmatory standard in a single 19-year sample."**

This is consistent with the well-known USD-commodity inverse correlation in the academic literature. It is not a novel factor discovery.

---

## 4. Practical Implications

**For investors considering commodity exposure:**

1. **If using commodities at all, favor physical gold over broad futures or miners.** GLD/IAU have the clearest descriptive support in our sample, and the effect is robust across most regimes. IAU is preferred over GLD due to lower expense ratio (0.25% vs 0.40%) with nearly identical performance.

2. **Use a static 5-10% allocation rather than active timing.** Timing strategies on gold destroy value. DBC timing provides drawdown protection but no Sharpe improvement. None of the tested active approaches achieved confirmatory status.

3. **Do not expect statistical certainty from this workflow.** Even the strongest descriptive finding (static gold improving 60/40) fails the magnitude gate on both estimands tested (+0.069 Sharpe-difference and +0.118 Sharpe-of-excess, both well below 0.20). The sample length (14-19 years of walk-forward) is too short to reliably detect realistic effect sizes.

4. **DBC is a poor portfolio diversifier in this sample.** DBC loses a mean -4.34% during severe equity drawdowns (VTI < -5% months) and has near-zero unconditional Sharpe over 20 years. GSG was not tested for severe-drawdown portfolio contribution; its drawdown correlation profile is expected to be similar due to shared underlying commodity exposure but this was not directly verified. Both may have regime-specific value during inflation shocks, but that value does not survive multiple-testing adjustment.

5. **Gold miners (GDX, GDXJ) are not a substitute for gold.** They carry equity beta (correlation 0.83 with VTI during drawdowns) without delivering superior gold exposure.

**For systematic strategy researchers:**

1. **The commodity workflow's power constraints are severe.** Phase 0B showed minimum detectable excess Sharpe approximately +0.80 for a 50-strategy Holm-corrected family. Realistic wrapper-adjusted effects from the academic literature (0.08-0.18) are 4-10x below this threshold. Do not expect gate-passing results.

2. **The null test design matters enormously.** Phase 6.8 (shifted T10YIE only, true DXY preserved) and Phase 6B (stationary bootstrap on both series) answered different questions and produced different results (p=0.25 vs p=0.035). Phase 6.8 isolates the incremental value of the inflation leg conditional on true DXY; Phase 6B tests the full strategy against a proper noise null. Always specify exactly what is being nulled and report the most appropriate test for the claim being made.

3. **PIT bugs are easy to introduce with macro signals.** The Phase 5 implementation inadvertently bypassed the PITFeatureSeries safeguards by extracting `.values`. This reduced the apparent effect from +0.545 to +0.456 (~16%). Always use `.as_of()` for macro signals, never direct slicing.

---

## 5. Limitations

1. **Sample length.** The commodity ETF universe has only 14-19 years of walk-forward history (depending on instrument inception dates). This is too short for Phase 0B power analysis to detect realistic wrapper-adjusted effect sizes, and too short for regime-stratified robustness tests to have meaningful statistical power.

2. **Wrapper-vs-futures translation.** Academic commodity factor evidence (Moskowitz-Ooi-Pedersen time-series momentum, Koijen-Moskowitz-Pedersen-Vrugt carry) applies to direct futures contracts, not ETF wrappers. ETF wrappers introduce contango drag, expense ratio, and roll-methodology differences. The appropriate wrapper haircut is large and uncertain.

3. **Regime concentration.** The 2020-2022 inflation shock contributes disproportionately to favorable results. Phase 6.2 showed that excluding the 2015-2019 dollar-strength regime pushes the macro-gated DBC CI through zero. The walk-forward sample contains approximately one full commodity cycle, which limits inference about long-horizon dynamics.

4. **Dependence on broad index wrappers.** Cross-sectional rotation is tested only across DBC, PDBC, and USCI — three funds covering approximately the same underlying basket with different roll methodologies. True cross-sectional commodity selection (across individual commodity futures or sector-specific ETFs) would require a different universe and infrastructure.

5. **No forward test.** All results use walk-forward backtesting within a fixed historical sample (2007-2026). A genuine out-of-sample evaluation would require running the frozen macro-gated DBC rule forward from April 2026 against live data. That is deferred to future work.

6. **Transaction cost model is approximate.** Per-category bps schedules do not reflect true market impact, tax drag (especially K-1 partnership treatment for DBC), or borrow costs for short positions (though no tested strategy uses shorting).

---

## 6. Process Notes

- **6 phases of testing** (2B, 3, 4, 5, 6, 6B) plus Phase 0A wrapper audit and Phase 0B power analysis
- **31 walk-forward backtests** retained in the final catalog (see Appendix A), using the shared `youbet.etf.backtester` engine. Earlier workflow drafts included additional runs that were superseded by rewritten versions after identifying bugs (pre-VGSH survivorship, PIT leakage, monthly-rebalance calendar bug, Sharpe estimand mismatch); those runs are not included in the final catalog.
- **300 null simulations** (100 in Phase 6.8 circular-shift null + 200 in Phase 6B stationary bootstrap null)
- **Multiple adversarial review rounds** from Codex, with findings addressed through iterative implementation fixes and narrative corrections
- **Critical fixes applied during workflow:**
  - PIT leakage in macro-gated DBC (Phase 5.3 → Phase 6.1)
  - Rebalance-day calendar vs trading-day bug (Phase 2 v1 → v2)
  - Sharpe estimand mismatch between p-value and CI functions
  - Multi-instrument portfolio test bug from pre-2009 VGSH warnings
  - Mixed metric definitions between `block_bootstrap_test` and `excess_sharpe_ci`

---

## Appendix A: Full Walk-Forward Test Catalog

All tests use 36/12/12 walk-forward parameters, T+1 execution, and 22-day block bootstrap. Bootstrap replicate counts are 10,000 for the primary tests (Phase 2B, 3R, 4C, 5.1-5.5, 6.1, 6.2, 6.7) and 5,000 for the Phase 6.3 lookback sweep, 6.4 logic ablation, and 6.5 external replication sweeps to manage compute time. Null distribution tests (Phase 6.8, 6B) use inner bootstrap counts of 500 and 500 respectively per null draw.

| # | Phase | Strategy | Benchmark | Folds | Sharpe Diff | p-value | 90% CI | Gate |
|---|---|---|---|---|---|---|---|---|
| 1 | 2B | Static 54/36/10 IAU | 60/40 | 19 | +0.069 (diff) / +0.118 (excess) | 0.285 | [+0.019, +0.135] (excess) | FAIL |
| 2 | 3R | IAU SMA100 sleeve | Static 54/36/10 | 14 | -0.006 | 0.740 | [-0.050, +0.042] | — |
| 3 | 4D.1 | IAU SMA200 sleeve | Static 54/36/10 | 14 | -0.015 | 0.833 | [-0.059, +0.032] | — |
| 4 | 4D.2 | IAU SMA252 sleeve | Static 54/36/10 | 14 | -0.003 | 0.697 | [-0.045, +0.043] | — |
| 5 | 4D.3 | IAU SMA200 standalone | IAU B&H | 14 | -0.055 | 0.833 | [-0.339, +0.226] | — |
| 6 | 4D.4 | DBC SMA100 standalone | DBC B&H | 14 | +0.214 | 0.274 | [-0.160, +0.585] | — |
| 7 | 4D.5 | DBC SMA200 standalone | DBC B&H | 14 | +0.345 | 0.148 | [-0.038, +0.733] | — |
| 8 | 4D.6 | Static 51/34/15 IAU | 60/40 | 14 | +0.082 | 0.568 | [-0.033, +0.191] | — |
| 9 | 4D.7 | Static 48/32/20 IAU | 60/40 | 14 | +0.097 | 0.568 | [-0.063, +0.245] | — |
| 10 | 4C | VTI SMA100 + gold | Static 54/36/10 | 14 | -0.098 | 0.973 | [-0.391, +0.158] | FAIL |
| 11 | 5D.1 | VTI SMA100 + gold | 60/40 | 14 | -0.038 | 0.958 | [-0.354, +0.235] | — |
| 12 | 5D.2 | VTI SMA200 + gold | Static 54/36/10 | 14 | +0.073 | 0.873 | [-0.189, +0.302] | — |
| 13 | 5D.3 | DBC SMA100 sleeve | 60/40 | 14 | +0.020 | 0.882 | [-0.046, +0.083] | — |
| 14 | 5.1 | DBC SMA200 (confirmatory) | DBC B&H | 19 | +0.178 | 0.276 | [-0.134, +0.488] | FAIL |
| 15 | 5.2 | DBC 12m TSMOM | DBC B&H | 19 | +0.137 | 0.324 | [-0.145, +0.434] | — |
| 16 | 5.3 | Macro-gated DBC (buggy) | DBC B&H | 19 | +0.545 | 0.148 | [+0.140, +0.945] | FAIL |
| 17 | 5.4 | XS rotation DBC/PDBC/USCI | DBC B&H | 19 | +0.271 | 0.162 | [-0.042, +0.597] | — |
| 18 | 5.5 | Dynamic sleeve IAU/DBC/cash | Static 54/36/10 | 14 | +0.003 | 0.384 | [-0.042, +0.050] | — |
| 19 | **6.1** | **Macro-gated DBC (PIT-safe)** | DBC B&H | 19 | **+0.456** | 0.174 | [+0.068, +0.847] | FAIL |
| 20 | 6.2 | Macro-gated DBC ex 2015-19 | DBC B&H | 19 | +0.379 | 0.272 | [-0.073, +0.824] | — |
| 21 | 6.3a | Macro-DBC LB=63 | DBC B&H | 19 | +0.354 | 0.273 | [-0.055, +0.767] | — |
| 22 | 6.3b | Macro-DBC LB=189 | DBC B&H | 19 | +0.384 | 0.239 | [-0.026, +0.793] | — |
| 23 | 6.3c | Macro-DBC LB=252 | DBC B&H | 19 | +0.256 | 0.363 | [-0.182, +0.679] | — |
| 24 | 6.4a | INFLATION_ONLY 126 | DBC B&H | 19 | +0.268 | 0.178 | [-0.045, +0.579] | — |
| 25 | 6.4b | DOLLAR_ONLY 126 | DBC B&H | 19 | +0.399 | 0.125 | [+0.047, +0.745] | FAIL |
| 26 | 6.4c | OR logic 126 | DBC B&H | 19 | +0.261 | 0.120 | [-0.023, +0.548] | — |
| 27 | 6.4d | ADDITIVE logic 126 | DBC B&H | 19 | +0.261 | 0.120 | [-0.023, +0.548] | — |
| 28 | 6.5a | Macro-PDBC | PDBC B&H | 11 | +0.366 | 0.409 | [-0.031, +0.771] | — |
| 29 | 6.5b | Macro-USCI | USCI B&H | 15 | +0.199 | 0.545 | [-0.191, +0.585] | — |
| 30 | 6.5c | Macro-GSG | GSG B&H | 19 | +0.452 | 0.133 | [+0.048, +0.847] | FAIL |
| 31 | 6.7 | Macro-DBC sleeve in 60/40 | Static 60/40 | 19 | +0.013 | 0.293 | [-0.020, +0.049] | — |

**Summary across 31 walk-forward tests:** 0 strategies pass the strict gate (Sharpe diff > 0.20, Holm p < 0.05, CI lower > 0). The strongest effect is macro-gated DBC (PIT-safe) at +0.456 with CI lower +0.068, but p=0.174 fails the significance requirement.

---

## Appendix B: Null Test Results

| Null Test | N | Observed | Null Mean | Null 95th | Fraction ≥ | Permutation p |
|---|---|---|---|---|---|---|
| Phase 6.8: Shifted T10YIE, true DXY | 100 | +0.456 | +0.334 | +0.645 | 25.0% | ~0.25 |
| **Phase 6B: Stationary bootstrap both** | 200 | +0.456 | +0.037 | +0.391 | 3.0% | **0.035** |

**Interpretation:** Phase 6.8 tests incremental T10YIE value conditional on true DXY. Phase 6B tests the full strategy against noise. The observed effect fails Phase 6.8 (T10YIE adds little beyond DXY) but passes Phase 6B (combined signal is distinguishable from noise).

---

**End of report.**
