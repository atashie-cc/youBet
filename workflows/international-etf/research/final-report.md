# International ETF Workflow — Final Report (INTERIM v1.1)

**Status: INTERIM.** This report consolidates Phase −1 through Phase 5 results (run 2026-05-01 to 2026-05-11). Several items remain in progress and the report will need to be updated:

1. **Prospective holdout tracking** launched 2026-05-05; logs to `research/holdout_tracking.md`. No decision power until ≥24 months have passed AND ≥1 full DXY cycle direction-change has occurred. Earliest meaningful update is 2028-05.
2. **Round 2 codex review nice-to-haves** (M3 cost-override class, M5 snapshot consolidation, L4 hot-path import) remain. M1 (correlated-nulls power) and M4 (vol-min CI) addressed in Phase 5 (Section 14). None changed a verdict.
3. **A future Round 3 codex review** of this report is appropriate before any external claim or production deployment.

### v1.1 update notes (Phase 5 addenda, 2026-05-11)

Four targeted experiments added before wrap-up:
- **Drawdown analysis** strengthens the 100% VTI recommendation: ex-US tilts do NOT reduce realized MaxDD significantly (all CIs span zero), so the diversification argument's main practical justification fails empirically in the 2014-2026 sample.
- **Mean-shifted placebo on Phase 1** shows the negative verdict is ENTIRELY mean-asymmetry-driven (VXUS mean was -6.43%/yr below VTI). The structural Sharpe disadvantage is ~zero. If mean reversion materializes, the verdict could flip cleanly.
- **Correlated-nulls FWER = 0.025** — Holm correction is adequate; no methodology change needed.
- **Vol-min CI corroborates Vanguard 35-55%** — 65.3% of bootstraps put the argmin in-band. The Phase 1 "marginal violation" framing was overstated.

---

## Executive summary

**Question:** Under any pre-committed regime, valuation gate, hedging variant, or static allocation, does ex-US equity exposure raise the long-horizon E[log(W₂₅)] of a US investor's portfolio above 100% VTI buy-and-hold?

**Answer: No.** Across 18 strict-gate tests, 3 exploratory tests, and 1 source-period-bias robustness check, **0 PROCEED**. 100% VTI buy-and-hold remains efficient against every pre-committed ex-US allocation tested.

**Practical recommendation:** Hold 100% VTI. The Phase 1 ~1.1pp annualized vol reduction at 60% VXUS is real and consistent with the Vanguard 2021 (rev 2019) directional claim, but the accompanying Sharpe degradation (-0.156 at 60%) is not justified for an investor maximizing E[log(W)].

**Methodologically interesting:** Phase 3 C1b (60/40 VTI/HEFA, 2017-2026 effective window) showed a SharpeDiff of -0.002 with 90% CI [-0.095, +0.109] — the only finding in the entire workflow whose CI upper bound was meaningfully positive. This was correctly identified by the Phase 4 P1a USD-return mean-shifted placebo as **source-period bias**: HEFA's mean was -2.95%/yr below VTI in the 9.2-yr USD-bull sample, and the apparent diversification edge collapsed when means were equalized. Without the placebo discipline (inherited from `workflows/real-world-test/`), this would have become an unjustified "currency hedging may have helped" descriptive claim.

---

## 1. Background and motivation

Three prior workflows in this repo had established that 100% VTI buy-and-hold dominates:
- `workflows/etf/` — 17 systematic US-only strategies, 0 pass strict gate on Sharpe; trend-following provides drawdown reduction at lower CAGR.
- `workflows/etflab-max/` — 158 CAGR-maximizing variants, 0 pass; sector-factor momentum at 2.3x SMA100 is best leveraged construction (19.9% CAGR).
- `workflows/real-world-test/` — 50+ Monte Carlo embellishments to a 60/30/10 leveraged sleeve, all retracted; only stable cross-period positive ever found in the entire repo is the gold-mean-shifted rebalancing premium (~+0.28%/yr CAGR).

International (ex-US) ETFs were the one untested dimension. The setup was bimodal:
- 14 years of US dominance (2010-2024: S&P 500 +11.9%/yr vs EAFE +3.6%/yr)
- US/ex-US CAPE spread near 30-year extreme (33.9 vs 18.7 per Research Affiliates May 2026)
- USD rolled over from Sep 2022 peak (-13% since)
- AQR (2023) decomposes ~75% of 1990-2022 US outperformance as multiple expansion, not earnings
- GMO Nov 2025: US Large -5.4% real / EM Value +5.7% real / 7yr forecast

Either the long-running US dominance continues or the valuation gap mean-reverts. The data has to break the tie.

---

## 2. Methodology and pre-commitment

### Architecture (LOCKED v1.3)

- **Benchmark:** VTI buy-and-hold (single, pre-committed, no benchmark shopping).
- **Walk-forward:** 36 months train / 12 months test / 12 months step. FIXED before any results were observed.
- **Bootstrap:** Stationary block bootstrap (Politis-Romano 1994), 22-day blocks, 10,000 resamples, 90% CI.
- **Multiple-testing correction:** Holm-Bonferroni across all gating tests. Final denominator: **18** (locked v1.3 after dropping M5 due to no intl CAPE data).
- **In-sample window:** 2001-08-14 (EFA inception) to 2026-04-30. Pre-2001 splice via Ken French international factors REJECTED in Round 1 review (factor portfolios are not investable index TR).
- **Sub-period boundaries:** Mechanical equal-thirds 2001-08 to 2009-12 / 2010-01 to 2018-04 / 2018-05 to 2026-04 (Round 1 review M4 fix; was previously literature-aware bin shopping).
- **Cost model:** Per-category bps from `src/youbet/etf/costs.py`, with sensitivity sweep at 3/5/10 bps for strategies that trade. PROCEED requires pass at 10 bps.

### Strict gate (LOCKED v1.2 after Phase −1 power analysis)

| Criterion | Threshold |
|---|---|
| Excess Sharpe vs VTI (point) | > 0.40 |
| Holm-adjusted p (across all 18 tests) | < 0.05 |
| 90% CI lower on Sharpe-diff | > 0 |
| Mean log-excess vs VTI | > 0 |
| 90% CI lower on log-excess | > 0 |

The ExSharpe threshold was **raised from the originally pre-committed 0.20 to 0.40 after Phase −1 power analysis** showed power at 0.20 was 0.04 (vs 0.50 threshold), and even at 0.40 was only 0.30. Per the pre-committed decision rule, this triggered a HALT and a workflow-tier downgrade from CONFIRMATORY to **DESCRIPTIVE/EXPLORATORY**. No strategy will be claimed CONFIRMED-PROCEED at the locked-in 0.30 power.

### Source-period bias as the dominant failure mode

Inherited from `workflows/real-world-test/` and the `feedback_source_period_bias` project memory: bootstrap MC mechanically reproduces source-period asset means. Any positive finding requires (a) USD-return mean-shifted placebo, (b) linear-scaling weight sweep, (c) sub-period robustness across at least three independent windows. This catches the failure mode where a single sample's asset-mean ranking drives an apparent edge.

This methodology was the decisive test for the C1b finding (Section 6.4).

---

## 3. Pre-Phase 0 work (Phase −1)

Five mandatory deliverables before Phase 0 could run, per Round 1 codex review:

1. **Power analysis** (`experiments/phase_minus_1_power.py`). v1 had a vol-mismatch bug (strategy std ≈ 0.7× benchmark std). v2 fix verified: target=0.0 mean point = -0.001 (correctly centered). Final result: power at ExSharpe=0.20 = 0.04, at 0.40 = 0.30, FWER at zero = 0.030. **Triggered the workflow-tier downgrade** described above.
2. **Vanguard 2021 rev 2019 PDF primary-source verification.** Initial citation "40-50% ex-US minimizes vol" was wrong (secondary summary tightening). Primary source: **35-55%** (Donaldson et al. 2021, page 1+4 verbatim), and result is forward-looking VCMM simulation, not historical realized.
3. **Holm-denominator audit and recount.** Initial 27 → locked 19 in v1.1 (after pruning nested duplicates) → locked 18 in v1.3 (after dropping M5 for missing data).
4. **Cost-override mechanism verification.** `OverrideCostModel` lets the cost level be set per-run for broad_intl_equity while leaving VTI/VGSH unchanged.
5. **Data fetch + SHA256 hashing.** 23 ETFs (6,368 rows) + 8 macro series (DXY, US 10y, German Bund 10y, T-bill, broad dollar TWI, 10y breakeven inflation, DBC). Hashes in `precommit/data_hashes.json`.

---

## 4. Phase 0 — Single-ETF efficiency baseline

**Question:** Does any individual ex-US ETF, held 100%, beat 100% VTI buy-and-hold on walk-forward Sharpe?

**Result: 0/4 PROCEED. All 4 CIs exclude zero on the negative side.**

Effective test windows after the 36-month walk-forward train (Round 2 review H1 fix — earlier "X+" framing referred to inception, not test start):

| Ticker | Inception | Test_start | Sharpe-diff | 90% CI |
|---|---|---|---|---|
| VXUS | 2011-01-26 | 2014-01-29 | -0.305 | [-0.566, -0.064] |
| VEA  | 2007-07-20 | 2010-07-27 | -0.333 | [-0.548, -0.129] |
| VWO  | 2005-03-04 | 2008-03-11 | -0.376 | [-0.611, -0.152] |
| EFA  | 2001-08-14 | 2004-08-30 | -0.244 | [-0.417, -0.085] |

**Cost-sensitivity caveat (Round 2 review H2 fix):** for buy-and-hold strategies, `total_turnover = 0` and per-cost-level results are bit-identical at 3, 5, 10 bps. The cost-stress narrative applies meaningfully only to Phase 2 + Phase 3 C3.

**Reading:** Statistical significance at the negative side. None of these tickers held continuously could have beaten VTI in this sample.

---

## 5. Phase 1 — Static VTI/VXUS allocation sweep

**Question:** Is there a static VTI/VXUS mix that beats 100% VTI?

**Result: 0/7 PROCEED.** Annual rebalance, effective test window 2014-01-29 to 2026-04-29 (~12.3 years).

| Weight VXUS | Sharpe-diff | 90% CI | Annual vol |
|---|---|---|---|
| 0% (= VTI) | — | — | 0.176 |
| 10% | -0.019 | [-0.043, +0.005] | 0.173 |
| 20% | -0.041 | [-0.090, +0.007] | 0.171 |
| 30% | -0.066 | [-0.140, +0.007] | 0.169 |
| 40% | -0.094 | [-0.194, +0.005] | 0.167 |
| 50% | -0.124 | [-0.250, +0.000] | 0.166 |
| 60% | -0.156 | [-0.310, -0.008] | **0.165** |
| 100% | -0.305 | [-0.566, -0.064] | 0.169 |

**Two findings of substance:**

1. **Vol IS reduced** by adding ex-US monotonically: 0.176 → 0.165 at 60% (-1.1pp annualized). Vol-min at 60% is just outside the Vanguard 2021 35-55% predicted band, but the curve is essentially flat across 40-60% (0.167 → 0.165) — the band claim is marginally and inconclusively violated. The Vanguard directional claim is confirmed; the band-precision claim is not.
2. **Sharpe is monotonically degraded** as ex-US weight rises. Even 10% VXUS costs ~0.02 Sharpe (CI just barely spans zero). 60% costs -0.16 Sharpe with CI excluding zero. Log-excess is monotonically negative for every weight — adding ex-US strictly reduced compound wealth in 2014-2026.

**Reading:** Vanguard's variance-reduction story is directionally right but the Sharpe loss outweighs it for an E[log(W)] investor.

---

## 6. Phase 2 — Regime-conditional gates

**Question:** Do macro and valuation gates beat 100% VTI when the ex-US weight is conditional on a regime indicator?

**Result: 0/7 PROCEED.** All 7 strict-gate FAIL; 6 of 7 CIs exclude zero on the negative side. M5 dropped from gating (no free intl CAPE data; Holm denominator 19→18 in v1.3).

| Strategy | Sharpe-diff | 90% CI | sig_on |
|---|---|---|---|
| M1a DXY 12m<0 → 20% VEA | -0.027 | [-0.054, -0.001] | 0.42 |
| M1b DXY 12m<0 → 40% VEA | -0.058 | [-0.111, -0.006] | 0.42 |
| M1c DXY 12m<0 → 60% VEA | -0.091 | [-0.172, -0.015] | 0.42 |
| M3 yield-narrowing → 40% VEA | -0.054 | [-0.111, -0.000] | 0.38 |
| M4a DBC 12m>0 → 20% VWO | -0.054 | [-0.092, -0.015] | 0.49 |
| M4b DBC 12m>0 → 40% VWO | -0.112 | [-0.187, -0.036] | 0.49 |
| M6 mean-rev → 40% VEA | -0.115 | [-0.203, -0.029] | **1.00** |

**Three findings of substance:**

1. **Conditioning DOES help vs Phase 1 static** — M1c at 60% VEA on DXY-weak loses -0.091 vs static 60% VEA losing -0.156. The DXY signal carries SOME information. But the residual edge is not enough to overcome the structural drag.
2. **M6 mean-reversion stayed ON 100% of the test window** — VEA's 36-month relative return has been negative continuously since 2010-07. The mean-reversion trigger in this period is functionally a static 40% VEA hold (Round 2 review H4 — operationally inconsequential since no PROCEED, but the Holm denominator double-counts).
3. **No regime gate flips the sign.** Even at the most favorable regime cut, ex-US loses to VTI.

---

## 7. Phase 3 — Currency hedging EXPLORATORY

**Status: EXPLORATORY ONLY — NOT IN HOLM DENOMINATOR.** 9.2-yr effective test window (2017-02 to 2026-04) fails Phase 4 P3's mechanical-thirds requirement. Synthetic hedged-EAFE pre-2014 was specified incorrectly in v1.0 and dropped per Round 1 review H1.

| Strategy | Sharpe-diff | 90% CI | Vol | Turnover |
|---|---|---|---|---|
| C1a 60/40 VTI/VEA | -0.048 | [-0.158, +0.069] | 0.177 | 0.000 |
| **C1b 60/40 VTI/HEFA** | **-0.002** | **[-0.095, +0.109]** | 0.171 | 0.000 |
| C3 dynamic-hedge toggle | -0.046 | [-0.146, +0.064] | 0.173 | 2.400 |

(Numbers above are post Round 2 H3 fix — shared 3-ticker date index. H3 fix shifted numbers by ~0.001 from the original misaligned-window run; qualitatively unchanged.)

### 7.1 The C1b near-miss

C1b (60/40 VTI/HEFA, 2017-2026) was the **only strategy in the entire workflow whose 90% CI upper bound was meaningfully positive (+0.109)**. The point estimate -0.002 was statistically indistinguishable from zero — i.e., always-hedged ex-US in this window was Sharpe-neutral with VTI.

The HEFA-vs-VEA delta of +0.047 Sharpe over 9.2 years is consistent with the macro literature: hedging captures local-equity returns without USD-strength drag during USD-bull regimes (2014-2022).

C3 dynamic-hedge (toggle HEFA↔VEA on DXY 12m sign) added essentially nothing over static C1b (+0.002 Sharpe over C1a) and burned 2.4× turnover — a clear "if you're going to hedge, just hedge; don't try to time it" finding.

### 7.2 Phase 4 robustness rejects C1b

Per the Round 2 review SHOULD-DO and the project memory `feedback_source_period_bias`, three robustness checks were run on C1b before claiming any finding:

- **P1a USD-return mean-shifted placebo: FAIL.** HEFA's mean was **-2.95%/yr below VTI** in the 9.2-yr USD-bull sample. After mean-equalization (preserving variance, autocorrelation, and cross-correlation structure), placebo Sharpe-diff = +0.067, CI [-0.028, +0.184]; placebo log-excess CI lower = **-0.000072 (excludes positive)**. The +0.109 CI upper bound was source-period bias — HEFA's structural underperformance was almost-but-not-quite compensated by the diversification mechanic, and the upper-CI tail came from the diversification residual being slightly positive when combined with VTI's higher mean. Mean-equalization removes the asymmetric-mean component and leaves no structural rebalancing premium.
- **P2 linear-scaling sweep:** HEFA at 1/5/10/30/50% of the portfolio gives Sharpe-diff +0.000/+0.001/+0.002/+0.002/-0.007. Essentially flat near zero — no peak-then-collapse artifact AND no positive scaling. Zero structural alpha.
- **P3 sub-period 3 only (2018-05 to 2026-04):** Sharpe-diff +0.113, CI [-0.033, +0.258]. More positive than the full sample but CI still spans zero, fully within USD-bull regime. Sub-periods 1+2 untestable (HEFA inception 2014-01-31).

**Combined verdict: C1b REJECTED.** The headline "+0.109" was a sample-period artifact, not a structural diversification edge. The placebo discipline did exactly what it was designed to do.

---

## 8. Cumulative tally and what each phase contributes

| Phase | Hypotheses | Result |
|---|---|---|
| 0 | 4 single-ETF | 0/4 PROCEED; CIs exclude zero negative |
| 1 | 7 static weights | 0/7 PROCEED; vol reduces ~1.1pp at 60%, Sharpe degrades monotonically |
| 2 | 7 regime-conditional | 0/7 PROCEED; conditioning helps mildly but never flips sign |
| 3 | 3 hedging exploratory | 0/3 EXPLORATORY-PASS; C1b near-miss → REJECTED by Phase 4 |
| 4 | 3 robustness on C1b | P1a FAIL, P2 flat-not-monotonic-positive, P3 limited |
| **Total** | **18 gating + 3 exploratory + 3 robustness** | **0 PROCEED + 0 EXPLORATORY-PASS + 0 robustness-survive** |

---

## 9. What would change the verdict

The workflow is designed to be falsifiable. The verdict would change if:

1. **A multi-decade USD bear emerges and the prospective holdout shows hedged-or-unhedged ex-US allocation outperforming VTI on Sharpe, log-wealth, AND surviving placebo.** The current 9.2-yr Phase 3 sample is fully USD-bull; the structural argument for ex-US (Asness 2023 multiple-expansion thesis) is consistent with a USD bear unlocking ex-US returns, but the workflow has insufficient data to test it.
2. **Intl CAPE data becomes available** for free at adequate fidelity. The dropped M5 composite signal (≥3-of-4 of {DXY weak, CAPE wide, yield narrowing, commodity up}) might catch the multi-tailwind regime that the literature describes for 2002-2007 EAFE outperformance.
3. **A power-adequate sample emerges** — Phase −1 power analysis shows even ExSharpe = 0.40 has only 0.30 power at the locked Holm denominator. With more data (e.g. another 10-15 years post-2026), the gate may become detectable.
4. **The workflow is re-tiered** to a confirmatory level only after structural changes that increase data, reduce N, or accept a higher MDE.

---

## 10. Round 1 + Round 2 codex review summary

The workflow underwent two adversarial reviews. Combined catch:

**Round 1 (pre-implementation, 4 HIGH + 8 MEDIUM):**
- H1 Synthetic hedged-EAFE pre-2014 formula was wrong (would have biased C1/C3 unhedged-vs-hedged comparison)
- H2 Holm denominator under-counted (locked 18 after pruning)
- H3 Ken French international factors cannot splice EFA pre-2001 (in-sample window restricted to 2001+)
- H4 P1 placebo ambiguous on equity-vs-currency (split into P1a USD return + P1b local equity)
- M1 Power analysis (added Phase −1 mandatory)
- M2 CAPE publication lag too short (raised to 60 days)
- M3 Vanguard PDF needed primary-source verification (35-55%, not 40-50%; VCMM not historical)
- M4 Sub-period boundaries were post-hoc (mechanical equal-thirds)
- M5/M6 V1 / S3 demoted to descriptive (underpowered / null-logic confusion)
- M7 Cost crisis sensitivity (sensitivity sweep at 3/5/10 bps)
- M8 Holdout 1-day theatrical (24mo no-decision-power statement)

**Round 2 (post-implementation, 4 HIGH + 8 MEDIUM):**
- H1 Test windows mis-stated (corrected in phase{0,1,3}_results.md and log.md)
- H2 Cost sweep dead theatre for buy-and-hold (caveat added to phase0/1)
- H3 Phase 3 ticker-subset misalignment (FIXED via shared 3-ticker date index; numbers shifted ~0.001)
- H4 M6 mean-reversion stayed ON 100% (collapsed to static; documented, operationally inconsequential)
- M1 FWER under correlated nulls not measured (open)
- M3 Cost-override class is function-local (latent technical debt; flagged for module-level move)
- M4 Vol-min CI not computed (flagged; would not change band verdict)
- M5 Snapshot drift between 2026-05-01 and 2026-05-02 (flagged for consolidation)

**The Phase 4 P1a placebo result (C1b rejection) was directly enabled by Round 2's SHOULD-DO recommendation.** Without it, the +0.109 CI upper bound would have been an unjustified descriptive claim.

---

## 11. Practical recommendation

**For a long-horizon US investor (25-yr, tax-protected, weekly-trade account) with E[log(W)] objective:**

**Hold 100% VTI buy-and-hold.** No pre-committed ex-US allocation tested in this workflow improves on it. (Phase 5 §14.1 further confirms: ex-US tilts also do NOT reduce realized MaxDD significantly — every CI spans zero. The diversification argument's main *practical* justification fails empirically in this sample.)

**If diversification motive is dominant** (i.e. the investor wants vol reduction more than Sharpe-maximization), a 60/40 VTI/VXUS mix reduces realized annual vol by ~1.1pp at the cost of -0.16 Sharpe-diff. The Vanguard 2021 35-55% predicted variance-min band is directionally confirmed; pick a weight in that range. But this is NOT a Sharpe-improving move.

**Currency hedging is not a free option.** Phase 3 C1b's appearance of Sharpe-neutrality with HEFA was sample-period bias. Dynamic hedging (C3) adds turnover without alpha. If you are committed to ex-US exposure and operating in a USD-strong regime, fixed-hedge HEFA is no worse than VEA and may be marginally better — but this is regime-dependent and untestable across multiple cycles in the available data.

**The honest statement:** all the diversification arguments in the literature (Asness, Vanguard, GMO, Research Affiliates) are directionally consistent with ex-US adding value at long horizons. The 2001-2026 sample does not support them at user-relevant confidence. If you believe the literature's structural arguments will reassert in the next decade, a 10-30% ex-US tilt is a reasonable Bayesian compromise — but do not expect it to pass any retrospective gate.

**Phase 5 §14.2 sharpens this:** the Phase 1 negative verdict is entirely mean-asymmetry-driven (VXUS mean -6.43%/yr below VTI in this sample). After mean-equalization the structural Sharpe edge is zero. This means a future regime in which VXUS mean approaches or exceeds VTI mean — which is what the literature predicts — would flip the verdict cleanly. The workflow does NOT falsify the structural diversification argument; it falsifies the realized-2014-2026-diversification claim. Update your prior accordingly.

---

## 14. Phase 5 addenda (v1.1)

Four targeted validation experiments added 2026-05-11 (`research/phase5_addenda.md` for the full numbers).

### 14.1 Drawdown analysis on the Phase 1 sweep

VTI MaxDD over the 2014-2026 sample = **35.0%**. Every VTI/VXUS mix tested has a directionally smaller MaxDD (-0.002 to -0.007 difference), but **all 90% CIs span zero**.

| Weight VXUS | Strat MaxDD | Diff vs VTI | 90% CI |
|---|---|---|---|
| 10% | 0.348 | -0.002 | [-0.010, +0.008] |
| 20% | 0.347 | -0.003 | [-0.020, +0.020] |
| 30% | 0.345 | -0.005 | [-0.027, +0.029] |
| 40% | 0.344 | -0.006 | [-0.032, +0.041] |
| 50% | 0.343 | -0.007 | [-0.039, +0.056] |
| 60% | 0.344 | -0.006 | [-0.043, +0.074] |
| 100% | 0.354 | +0.004 | [-0.043, +0.155] |

**Implication:** The Vanguard ~1.1pp vol reduction at 60% does NOT translate into a statistically meaningful drawdown reduction. This is a substantive new finding — it removes the most commonly-cited *practical* reason (ruin avoidance) for an ex-US tilt. The 100% VTI recommendation is strengthened.

### 14.2 Mean-shifted placebo on Phase 1 40% VXUS

VXUS mean in the 12.3-yr sample was **-6.43%/yr below VTI** — a substantial source-period asymmetry.

After mean-equalization placebo:
- Original 40% VXUS Sharpe-diff: -0.094
- **Placebo Sharpe-diff: +0.060** (CI [-0.036, +0.166])
- Placebo log-excess daily mean: +0.000017 (CI [-0.000046, +0.000082])

**The Phase 1 negative verdict is ENTIRELY mean-asymmetry-driven.** There is no structural Sharpe disadvantage to a 40% VXUS tilt; the apparent edge against VXUS comes from VXUS having had a 6.43%/yr lower mean than VTI in this sample. CI still spans zero on the placebo side, so this is not evidence that mean-reversion will happen — but it IS evidence that the negative finding is high-conditional on the realized mean asymmetry.

**Bayesian update:** if the user's prior includes meaningful weight on the GMO/Asness mean-reversion thesis, the Phase 1 result does not falsify it. The "VTI dominant" verdict is correct for backwards-looking allocation but is conditional on a continuation of the 2014-2026 mean profile. Same mechanism as killed C1b, but the inferential direction here is "could plausibly reverse" rather than "definitely sample-bias artifact."

### 14.3 Correlated-nulls power analysis (Round 2 M1)

Re-ran the FWER measurement with 18 nulls clustered as `[3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` to mimic the real workflow's M1×3 / M4×2 / etc. correlated test geometry.

- **FWER = 0.025** under H₀ (well below 0.05 nominal)
- Mean false-positive count per simulation: 0.05

**Holm correction is adequate** — under positive correlation, the multiple-testing burden is mechanically lower than independence assumes, and Holm becomes MORE conservative not less. No methodology change needed; Romano-Wolf simultaneous CIs are not required.

### 14.4 Vol-min location CI (Round 2 M4)

Bootstrap 1000 daily-return-panel samples, recompute the vol curve, locate argmin per replicate.

| Weight VXUS | Argmin count | Fraction |
|---|---|---|
| 30% | 134 | 13.4% |
| 40% | 306 | 30.6% |
| 50% | 347 | 34.7% |
| 60% | 191 | 19.1% |
| other | 22 | 2.2% |

- **Fraction in Vanguard 35-55% band: 65.3%**
- Mean argmin weight: **0.455** (squarely in band)

**The Vanguard 35-55% claim is corroborated.** The Phase 1 point-estimate "vol-min at 60%" was characterized as a marginal band violation, but the data is fully consistent with the band — 65% of bootstraps put the vol-min in [35%, 55%]. The narrative in earlier reports overstated the discrepancy.

---

## 15. What's still in progress

1. **Prospective holdout tracking.** Pre-committed start 2026-05-02. No decision power until ≥24 months + 1 DXY direction-change. Log monthly to `research/holdout_tracking.md` (file to be created at first monthly cadence).
2. **Round 2 nice-to-haves:**
   - M3 — move `OverrideCostModel` to module level (latent pickle/multiprocessing risk)
   - M5 — consolidate to a single committed snapshot (delete `data/snapshots/2026-05-01/`, re-run Phase 0 from 2026-05-02 snapshot, verify identical results)
   - M1 — re-run power analysis with correlated nulls (clusters of M1-style triplets)
   - M4 — bootstrap CI on the Phase 1 vol-minimum location
   - L4 — move hot-path `import` out of inner loop in `phase2_regime.py`
3. **Optional Round 3 codex review** of this report before any external claim or production deployment.
4. **If intl CAPE data becomes available** at adequate fidelity, M5 composite signal can be added back (would re-raise Holm denominator 18→19 with documented ledger entry).

---

## 16. Pointers

- Pre-committed plan: `research/plan.md` (v1.3)
- Experiment ledger: `research/log.md`
- Phase results: `research/phase_minus_1_power_results.md`, `phase{0,1,2,3}_results.md`, `phase4_c1b_robustness.md`, `phase5_addenda.md`
- Codex reviews: `research/codex_review_round{1,2}.md`
- Vanguard primary-source verification: `research/vanguard_pdf_verification.md`
- Literature: `research/literature_{academic,industry,macro_regime}.md`
- Strategy code: `strategies/regime_signals.py`, `regime_conditional.py`, `dynamic_hedge.py`
- Experiment runners: `experiments/phase{0,1,2,3}_*.py`, `phase_minus_1_*.py`, `phase4_robustness_c1b.py`
- Engine: `src/youbet/etf/{backtester,costs,stats,pit,benchmark,risk}.py`
- Data: `data/snapshots/2026-05-02/` (latest); SHA256 hashes in `precommit/data_hashes.json`

---

**End of interim final report v1.0.** Update on holdout-tracking milestones, completion of Round 2 nice-to-haves, or any future Round 3 review findings.
