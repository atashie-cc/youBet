# International ETF Workflow — Pre-Committed Plan

**Date pre-committed:** 2026-05-01 (v1.0); 2026-05-01 (v1.1 after Round 1 codex review — 4 HIGH + 8 MEDIUM fixes applied)
**Author:** claude (with arik approval)
**Status:** Pre-commit. NO returns data has been examined for any candidate strategy. NO regression has been run. NO weight has been tuned. This document is the locked spec; deviations require a dated, signed entry in `research/log.md` BEFORE re-running.

**Commercial-interest disclosure (per L1):** AQR (Asness 2011/2023) and GMO (Grantham forecasts) cited in the literature reviews have direct commercial exposure to ex-US allocation. Their valuation-mean-reversion case is directionally consistent with internal incentive but not invalidated by it. Vanguard (cited for variance-min claim) similarly runs international funds. Bogle/nisiprius (US-only camp) have no equivalent commercial stake. Treat the asymmetry as informative, not disqualifying.

---

## 0. Question

Under any pre-committed regime, valuation gate, hedging variant, or static allocation, does ex-US equity exposure raise the long-horizon **E[log(W₂₅)]** of a US investor's portfolio above 100% VTI buy-and-hold?

The null is: **VTI buy-and-hold is efficient on Sharpe, log-wealth, and worst-case rolling 10y CAGR.** This null has survived 17+ US-only strategies (etf/), 158 CAGR variants (etflab-max/), and 50+ Monte Carlo embellishments (real-world-test/).

The prior is informed:
- US has dominated 2010-2024 (S&P 500 +11.9% vs EAFE +3.6% annualized).
- US/ex-US CAPE spread is near 30-yr extreme (33.9 vs 18.7 per Research Affiliates May 2026).
- DXY has rolled over from Sep 2022 peak (-13% since).
- AQR (2023) decomposes ~75% of 1990-2022 US outperformance as multiple expansion, not earnings.
- Vanguard 2021 (rev. 2019) VCMM forward simulation: 35-55% ex-US minimizes variance for non-euro-area domiciles (verified against primary PDF — see `vanguard_pdf_verification.md`).
- GMO Nov 2025: US Large -5.4% real / EM Value +5.7% real / 7yr forecast.

The literature is therefore **bimodal**: 14 years of US dominance vs. a near-extreme valuation gap and a rolling-over USD. The data has to break the tie.

---

## 1. Authoritative Gate (LOCKED v1.2 after Phase -1 power analysis)

**PHASE -1 POWER ANALYSIS RESULT (2026-05-01):**
- Power at ExSharpe=0.20: **0.04** (decisively below 0.50 threshold)
- Power at ExSharpe=0.40: **0.30** (also below 0.50)
- FWER at ExSharpe=0.0: 0.030 (well-controlled, < 0.05 nominal)
- Per pre-committed decision rule: **HALT** triggered. Gate raised from 0.20 to 0.40 (fallback); workflow tier downgraded from CONFIRMATORY to **DESCRIPTIVE/EXPLORATORY**.

| Criterion | Threshold (LOCKED v1.2) |
|---|---|
| Excess Sharpe vs VTI buy-and-hold (point) | > 0.40 |
| Holm-adjusted p-value across ALL tested strategies × variants | < 0.05 |
| 90% stationary-block-bootstrap CI on excess Sharpe — lower bound | > 0 |
| Mean log-excess vs VTI (point) | > 0 |
| 90% bootstrap CI on log-excess — lower bound | > 0 |

**Workflow tier: DESCRIPTIVE/EXPLORATORY.** Even at the raised gate (0.40), power is only 0.30 — meaning a true ExSharpe = 0.40 effect would be missed 70% of the time. Therefore:

1. **No strategy will be claimed CONFIRMED-PROCEED.** Any PROCEED label is conditional and exploratory.
2. **All results reported as point estimates + CIs.** The CI is the primary inference vehicle, not the gate.
3. **Phase 4 robustness (P1-P5) and prospective holdout (≥24 months) become the binding cross-validation**, not the strict gate.
4. **The honest answer space is now bimodal**: (a) most likely outcome — every ex-US strategy has CI overlapping zero, consistent with VTI-efficiency at the available sample's resolution; (b) rare outcome — a strategy with implausibly large effect (0.5+) clears the gate AND survives Phase 4.

This mirrors `cagr-max` workflow (0/65 PASS, structurally unpowered per E0) and is a HONEST outcome of pre-committed methodology, not a failure.

**Multiplicity scope:** the Holm correction is applied across all strategies, all parameter variants, and all phases. With ~30+ tests planned, raw p<0.05 buys nothing. This is identical to etf/ and commodity/ workflow practice.

---

## 2. Pre-Committed Hypothesis Catalog

Synthesized from the three literature files. Duplicates merged. Each hypothesis has: ID, claim, signal definition, threshold, expected sign, mechanism, source. The "Phase" column maps each test to the phase that owns it.

**In-sample window (LOCKED v1.1):** 2001-08-14 (EFA inception) to 2026-04-30. Pre-2001 splice using Ken French factor portfolios is REJECTED (H3 fix — Ken French exposes long-short factor portfolios, not investable index TR; pre-1990 dev-ex-US factor data unreliable for splicing). Phase 4 sub-period robustness MAY use Ken French Mkt-RF + RF as approximate pre-2001 supplement, explicitly labeled "approximate, factor-portfolio-derived, not investable."

### A. Static-allocation hypotheses (Phase 1)

| ID | Claim | Test | Source | Gating? | Phase |
|---|---|---|---|---|---|
| **S1** | A US-investor static VTI/VXUS mix has its REALIZED 10yr-rolling vol minimum within the **35-55%** ex-US range Vanguard 2021 (rev. 2019) predicted via VCMM forward simulation (Donaldson et al. 2021, page 1+4 verbatim — see `vanguard_pdf_verification.md`). NOTE: Vanguard's claim is forward-looking VCMM, not historical realized; historical confirmation in their Figure A-1 appendix only. | Sweep VTI/VXUS at 0/10/20/30/40/50/60% ex-US. Rolling 120m REALIZED vol of monthly returns 2001-08 through 2026-04. Test: vol-minimum location ∈ [35%, 55%] ex-US. | Vanguard 2021 (rev. 2019), Donaldson et al. | YES — vol-minimum location | 1 |
| **S2** | Over rolling 10yr+ horizons, the 60/40 US/ex-US worst-case dominates the worst-case of 100% US AND 100% ex-US. Effect grows monotonically with horizon. | Min-of-rolling at horizons {3, 5, 10}yr for {VTI, ex-US, 60/40, 80/20}. (Dropped 1y, 15y, 20y — sample too short for 15-20y rolling.) | Asness 2011 | YES — min-diff CI | 1 |
| **S3** | (DESCRIPTIVE, NOT GATING — M6 fix) Paired rolling 10y CAGR difference VTI vs 60/40 VTI/VXUS. Reported with CI. Failing to reject zero ≠ evidence FOR null; reported as "consistent with null" only when power ≥ 80% to detect ExSharpe ≥ 0.20 (per Phase -1 power analysis). | Paired-bootstrap. | Bogle/nisiprius/Larimore | NO — descriptive | 1 |
| **S4** | At 10y horizons, US/ex-US realized correlation is materially below short-horizon correlation. | Rolling 1m, 12m, 60m, 120m correlation 2001-08 through 2026-04 (and 1990-2026 supplement using Ken French dev-ex-US Mkt-RF as APPROXIMATE). | Asness 2011 | NO — descriptive | 1 |

### B. Valuation-conditional hypotheses (Phase 2)

| ID | Claim | Test | Source | Gating? | Phase |
|---|---|---|---|---|---|
| **V1** | (INFORMATIVE, NOT GATING — M5 fix; horizon overlap kills statistical power) When (US CAPE − ex-US CAPE) is in top quintile, ex-US outperforms US over the next 3y. (Was 10y; reduced to 3y for ~8 effective independent windows in 2001-2026.) Walk-forward gate strategy: increase ex-US weight to 20% when CAPE spread in top quintile of pre-rebalance trailing distribution. | Walk-forward CAPE-quintile gate; bootstrap CI on excess-Sharpe vs VTI. Block-bootstrap with 24-month blocks (not 22-day) to handle horizon overlap. | Asness 2023, Research Affiliates | NO — informative | 2 |
| **V2** | (READ-ONLY DECOMPOSITION, NOT IN HOLM DENOMINATOR) Sorting countries by trailing CAPE and overweighting the cheapest tercile produces excess-Sharpe vs cap-weighted ACWI ex-US. | MSCI country indices 2001-2026, CAPE-tilted vs cap-weighted. | Arnott/Research Affiliates | NO — read-only | 2 |
| **V3** | (READ-ONLY DECOMPOSITION, NOT IN HOLM DENOMINATOR) The 2001-2024 US-vs-International outperformance is ~75% multiple expansion. | Decompose total return = dividend + earnings growth + multiple change. | AQR 2023 | NO — read-only | 2 |

### C. Macro-regime hypotheses (Phase 2)

| ID | Claim | Test | Source | Gating? | Phase |
|---|---|---|---|---|---|
| **M1** | When DXY trailing 12m return < 0, ex-US outperforms US over the next 12m. | Walk-forward gate: hold {weight} ex-US (VEA) when DXY 12m < 0, else 0% ex-US (VTI). Test {weight ∈ 20%, 40%, 60%}. | RBC, Crescat, multiple | YES (×3) | 2 |
| ~~M2~~ | ~~Stronger version of M1: DXY 12m < -5% → VEA outperforms~~ | **DROPPED v1.1 (H2 fix):** nested duplicate of M1; threshold post-literature-tuning would be data-snooping. | — | — | — |
| **M3** | When (US 10y − German Bund 10y) spread is narrowing (12m change < 0), ex-US > VTI over next 6-12m. | Walk-forward gate on rate-diff change, sign(Δ) only (not threshold). Test 40% ex-US allocation. | MacroMicro / JPM | YES (×1) | 2 |
| **M4** | When DBC 12m return > 0 (sign only), VWO > VTI over next 12m. | Walk-forward gate on commodity uptrend; sign threshold (not +10%, to avoid post-literature tuning). Test {20% VWO, 40% VWO}. | 2003-07 BRICs era | YES (×2) | 2 |
| ~~M5~~ | ~~Composite signal: ≥3 of 4 of {DXY 12m<0, CAPE spread > median, US-Bund narrowing, DBC 12m>0}~~ | **DROPPED v1.3 (2026-05-02):** intl CAPE data unavailable from free sources at adequate fidelity (Siblis Research not bulk-downloadable; MSCI / Barclays paywalled). Composite without CAPE is only 3-of-3 which is over-strict. Drop entirely; revisit only if intl CAPE feed acquired. | — | — | — |
| **M6** | After ex-US 36m relative return < 0 (not −30%), ex-US 5y forward return > US (mean-reversion). | Walk-forward gate: increase ex-US weight from 0% to 40% when 36m rel-return is negative (sign only). | GMO, AQR Value Everywhere | YES (×1) | 2 |

### D. Currency-hedging hypotheses (Phase 3) — short-sample exploratory

**Phase 3 sample window restricted to 2014-01-31 onwards (HEFA inception).** v1.1 fix per H1: synthetic hedged-EAFE pre-2014 reconstruction was specified incorrectly (local-ccy + DXY ignores rate-differential carry, which has been +200-500 bps for USD investors hedging EUR/JPY 2022-2024). Building it correctly requires GDP-weighted foreign short rates from FRED with monthly recompute; out-of-scope for v1.1. Phase 3 results are **EXPLORATORY** (12yr sample fails Phase 4 P3's three-window requirement) and CAN NOT pass strict gate as currently sampled. They report directional evidence only.

| ID | Claim | Test | Source | Gating? | Phase |
|---|---|---|---|---|---|
| **C1** | Over 2014-2026, hedged ex-US (HEFA) has Sharpe within ±0.10 of unhedged (VEA) at 40% ex-US allocation, with materially lower 1yr σ. | Rolling Sharpe at 1y, 3y, 5y horizons. Real HEFA returns only — no synthetic pre-2014. | Campbell-Serfaty-Viceira 2010, Vanguard LaBarge 2014 | NO — exploratory | 3 |
| ~~C2~~ | ~~Per-currency selective hedging~~ | **DROPPED v1.1:** requires GDP-weighted per-currency hedge construction; building correctly is a workflow of its own. Re-add only if Phase 3 C1/C3 produce ambiguous results worth the engineering. | — | — | — |
| **C3** | (EXPLORATORY) Hedged beats unhedged when DXY 12m > 0 (USD-strengthening), unhedged beats when DXY 12m < 0. (Sign only, not ±5% threshold.) | Walk-forward toggle: HEFA when DXY 12m>0, VEA when DXY 12m<0. Single weight 40%. | Castro/Hamill/Harvey 2024, Morningstar | NO — exploratory | 3 |

### E. Falsification / placebo hypotheses (Phase 4 — MANDATORY)

| ID | Claim | Test | Source | Phase |
|---|---|---|---|---|
| **P1a** (primary) | **Mean-shifted USD-return placebo:** shift ex-US monthly USD returns to match VTI's mean (preserving correlation/vol structure). If benefit survives, it's structural rebalancing premium. If it disappears, source-period bias. | Re-run all PROCEEDing strategies from Phases 1-2 with ex-US USD returns mean-shifted to VTI's mean. | real-world-test workflow; gold-mean-shifted precedent | 4 |
| **P1b** (secondary, hedging only) | **Mean-shifted local-equity-component placebo:** decompose ex-US USD return into local-equity + FX components; mean-shift only the local-equity component. Required for any C1/C3 (Phase 3) claim — both P1a AND P1b must pass log-excess > 0 with CI lower > 0. | Re-run hedging variants with local-equity-component mean-shifted. | H4 fix from Round 1 review | 4 |
| **P2** | **Linear-scaling sweep:** if positive log-excess scales monotonically with ex-US weight 1%/5%/10%/30%/50%, finding is structural. If it peaks then collapses, it's an artifact. | Sweep {1%, 5%, 10%, 30%, 50%} ex-US for any PROCEEDing static or regime strategy. | real-world-test workflow | 4 |
| **P3** | **Mechanical equal-thirds sub-bootstrap (M4 fix):** sample window 2001-08 to 2026-04 (~24.7yr) is split into three equal-length sub-periods: 2001-08 to 2009-12, 2010-01 to 2018-04, 2018-05 to 2026-04. Sign-flip in any period = retract. (Pre-1990 and 1990-2001 supplements via Ken French dev-ex-US Mkt-RF+RF are APPROXIMATE-only, not gating.) | Re-run on each non-overlapping window with same gate criteria. | real-world-test workflow + M4 mechanical-split fix | 4 |
| **P4** | **Randomized regime placebo:** if a regime gate "works," shuffle the regime indicator timestamps and re-test 1000 times. Real signal: shuffled distribution does NOT cover real result at α=0.05. | Time-shuffle all macro signals. | macro-exploratory workflow | 4 |
| **P5** | **Codex adversarial review:** before claiming any PROCEED, submit code + spec + result to Agent(general-purpose) with adversarial-review prompt. Mandatory before any "final" claim per project memory. | Standard practice. | feedback_codex_review | 4 |

---

## 3. Phased Experiment Schedule

### Phase −1 — Pre-Phase 0 prep (NEW v1.1, MANDATORY before Phase 0)

Per Round 1 review M1 + reviewer's recommended Phase -1 work. NONE of these are gated tests — they prepare the test environment.

1. **Power analysis (`experiments/phase_minus_1_power.py`)** — simulate two correlated return series (corr 0.7 to match Asness 2011 short-horizon) with known excess-Sharpe ∈ {0.10, 0.20, 0.30, 0.40, 0.50}. Run engine walk-forward 36/12/12 + stationary block bootstrap 22d + Holm-correct across N (calibrated against the §6 denominator). Compute power at each effect size. **Decision rule:** if power at ExSharpe = 0.20 < 0.50, raise gate threshold to ExSharpe > 0.40 OR cut N. Lock the chosen threshold in `config.yaml` and document in `research/log.md` BEFORE Phase 0.
2. **Vanguard 2019 PDF primary-source verification (M3 fix)** — fetch PDF, extract exact sentence describing the variance-min claim, paste into `literature_industry.md` with page number. If it diverges from "40-50% ex-US," update S1 specification.
3. **Holm-denominator audit and recount (H2 fix)** — list every config-tuple-level test (weight × signal-threshold × phase). Get a hard number, lock it in `config.yaml` `bootstrap.holm_denominator`.
4. **Cost-model crisis sensitivity (M7 fix)** — `experiments/phase_minus_1_cost_sensitivity.py` showing ex-US ETF spread regimes (Mar 2020, Aug 2015, Dec 2018). Pre-commit a regime-aware multiplier OR commit to running all PROCEED candidates at cost ∈ {3, 5, 10} bps and requiring pass at 10 bps.
5. **Pre-fetch all data, hash, freeze (real-world-test discipline)** — fetch every ticker and macro series; SHA256 each parquet; write `precommit/data_hashes.json`. Re-fetch and re-hash before Phase 4.

If any Phase -1 step changes plan parameters, restart the lit→plan→codex-review cycle. Round 2 review goes faster.

### Phase 0 — Single-ETF efficiency baseline

**Question:** Does any individual ex-US ETF, held 100%, beat VTI on walk-forward Sharpe and log-wealth? If no, regime-conditioning has a steep hill.

**Strategies tested (4):** 100% VXUS, 100% VEA, 100% VWO, 100% EFA. Each uses real ETF returns from inception forward; PIT survivorship guard handles differing windows.

**Window:** 2001-08-14 (EFA inception) to 2026-04-30. Walk-forward 36/12/12. Pre-2001 splice REJECTED v1.1 per H3 (Ken French exposes factor portfolios, not investable index TR).

**Pass criterion:** strict gate. Almost certainly all FAIL given prior literature; this is a negative-result baseline.

**Expected output:** `experiments/phase0_efficiency.py`, `research/phase0_results.md`.

### Phase 1 — Static allocations

**Question:** Is there a static VTI/VXUS mix that beats 100% VTI on the strict gate?

**Strategies tested (8 weights, 1 rebalance regime):** weights ∈ {0%, 10%, 20%, 30%, 40%, 50%, 60%, 100%} VXUS in equity. **Rebalance: annual ONLY** (H2 fix; the original "±7pp drift OR annual" doubled the test count). Drift-band variant moved to Phase 4 sensitivity, not gating.

**Effective gating tests:** 7 (excluding 0% which equals VTI benchmark; 0% is reported as a sanity check, not a separate test).

**Pre-committed prior:** S1 says variance-min is at the Vanguard-published range (verified in Phase -1); S3 is descriptive only.

**Output:** `experiments/phase1_static.py`, `research/phase1_results.md`.

### Phase 2 — Regime-conditional gates

**Question:** Do macro and valuation gates beat 100% VTI when ex-US weight is conditional?

**Strategies tested (8 gating, after H2 prune):**
- M1 × {20%, 40%, 60%}: DXY 12m<0 → ex-US weight, else VTI (3 tests)
- M3: US-Bund yield-diff sign-of-change → 40% ex-US (1 test)
- M4 × {20% VWO, 40% VWO}: DBC 12m>0 → ex-US-EM weight (2 tests)
- M5: composite ≥3-of-4 → 50/50 VEA+VWO blend (1 test)
- M6: ex-US 36m rel-return<0 → 40% ex-US (1 test)
- V1 (3y forward, single weight 20%) — **NOT GATING** per M5 fix; reported but not in Holm denominator
- V2, V3 (read-only decomposition, not allocation) — NOT GATING

**Dropped v1.1:** M2 (nested DXY duplicate), V1×40% (single-weight only), original ±10% / -30% / +10% / -5% thresholds tightened to sign-of-change to avoid post-literature-tuning.

**Pre-committed prior:** macro-exploratory workflow E4 paper-validated mechanism, but investable path constrained. Expect most M-gates to fail strict gate after Holm correction.

**Output:** `experiments/phase2_regime.py`, `research/phase2_results.md`.

### Phase 3 — Currency hedging

**Question:** Does hedging or per-currency hedging materially shift the verdict?

**Strategies tested (2, 2014-2026 window only, EXPLORATORY):**
- C1: HEFA vs VEA at 40% ex-US allocation
- C3: dynamic hedge — HEFA when DXY 12m>0, VEA when DXY 12m<0, single weight 40%

**Phase 3 results CANNOT pass strict gate** because 2014-2026 (12yr) fails Phase 4 P3's three-window requirement. Phase 3 is reported as directional evidence only and is NOT in the Holm denominator. C2 (per-currency) DROPPED v1.1 per H1 fix (proper construction is its own workflow).

**Pre-committed prior:** unhedged 30yr beats hedged in current data (Morningstar $24k vs $22k); hedge regime conditioning may flip it but 12yr is too short to commit to.

**Output:** `experiments/phase3_hedging.py`, `research/phase3_results.md`.

### Phase 4 — Robustness (MANDATORY before any final claim)

For every strategy that passed strict gate in Phases 0-2 (Phase 3 exploratory only):
1. **P1a mean-shifted USD-return placebo** — re-run with ex-US USD returns mean-shifted to VTI mean. Pass: log-excess ≥ 0 with CI lower ≥ 0. (For Phase 3 hedging variants, P1b also required: local-equity-component mean-shifted; both must pass.)
2. **P2 linear-scaling sweep** — 1/5/10/30/50% sweep. Pass: monotonic in weight.
3. **P3 mechanical equal-thirds sub-bootstrap** — 2001-08 to 2009-12, 2010-01 to 2018-04, 2018-05 to 2026-04. Pass: positive sign in ALL three sub-periods. (Optional pre-2001 supplement using Ken French dev-ex-US Mkt-RF+RF as APPROXIMATE; informative not binding.)
4. **P4 regime-shuffle** — 1000-shuffle null distribution. Pass: real result outside 95% of shuffled.
5. **P5 Codex adversarial review** — Agent(general-purpose) with adversarial prompt. Pass: no HIGH-severity findings unaddressed.

**Output:** `experiments/phase4_robustness.py`, `research/phase4_results.md`. Final report only after all 5 sub-tests pass per claim.

---

## 4. Implementation Notes

### Reusable from existing engine
- `src/youbet/etf/backtester.py` — walk-forward, T+1, PIT, costs.
- `src/youbet/etf/stats.py` — stationary block bootstrap (22d), Holm correction, excess Sharpe CIs.
- `src/youbet/etf/transforms.py` — stateful normalizer with drift monitoring.
- `src/youbet/etf/costs.py` — `broad_intl_equity` already at 3 bps one-way; HEFA/HEDJ expense ratios encoded in universe CSV.
- `src/youbet/etf/macro/fetchers.py` — yield curve, credit spread, CAPE (US), VIX, PMI.
- `src/youbet/commodity/macro/fetchers.py` — DXY (move/re-register into etf/macro).

### New code required
- `src/youbet/etf/macro/intl_fetchers.py` (new):
  - `fetch_german_bund_10y()` from FRED `IRLTLT01DEM156N` (monthly, 30-day publication lag)
  - `fetch_intl_cape()` from Siblis Research (or fall back to MSCI EAFE Shiller-equivalent)
  - `fetch_dbc_returns()` from yfinance (commodity index)
  - `fetch_dxy()` (relocate / re-register from commodity macro)
- `workflows/international-etf/strategies/`:
  - `static_allocation.py` — fixed-weight VTI/VXUS with rebalance bands
  - `regime_conditional.py` — generic gate strategy with pluggable signal
  - `hedged_overlay.py` — VEA→HEFA toggle on DXY signal
- Pre-1995 splicing utility: `src/youbet/etf/data.py::splice_with_index(etf_series, msci_index_series)` to extend EFA/EEM history using Ken French international factors (already in `src/youbet/factor/data.py`).

### Data fetches required (not yet executed)
- VTI, VXUS, VEA, VWO, EFA, EEM, IEFA, IEMG, SCHF, IXUS, HEFA, HEDJ, DBEF (yfinance)
- VGSH, BIL, SGOV (yfinance)
- DXY (`DX-Y.NYB`), DBC, GSCI proxy (yfinance)
- DTWEXBGS, DGS10, IRLTLT01DEM156N, TB3MS (FRED)
- US CAPE (Shiller online); ex-US CAPE (Siblis or splice via P/E + smoothing)
- Ken French international factors (already cached for factor-timing workflow; reuse)

---

## 5. Pre-Committed Holdout

**Holdout-start date:** 2026-05-02. Any strategy that passes Phase 4 enters a prospective tracking window with monthly results logged to `research/holdout_tracking.md`. No re-fitting after this date.

**Holdout has NO statistical decision power until ≥ 24 months have passed AND ≥ 1 full DXY cycle direction-change has occurred** (M8 fix). The frozen tracker is for accountability, not validation. The binding cross-validation is Phase 4 P3 (mechanical equal-thirds sub-bootstrap on 2001-2026 in-sample data). This mirrors macro-exploratory and cagr-max frozen-tracker discipline but is honest about the power constraint.

---

## 6. Multiplicity-Correction Tally (LOCKED v1.1 after H2 prune)

| Phase | Test | Count |
|---|---|---|
| 0 | VXUS, VEA, VWO, EFA single-ETF efficiency | 4 |
| 1 | Static weight sweep at {10, 20, 30, 40, 50, 60, 100}% ex-US, annual rebalance | 7 |
| 2 | M1×3 (DXY<0 at 20/40/60%) + M3 (yield-diff narrowing at 40%) + M4×2 (DBC>0 at 20/40% VWO) + M6 (ex-US 36m<0 at 40%) | 7 |
| 3 | C1, C3 — EXPLORATORY ONLY, NOT IN HOLM DENOMINATOR (12yr sample fails Phase 4 P3) | 0 |
| **Total Holm denominator (LOCKED v1.3, M5 dropped)** | | **18** |

**Descriptive-only / read-only (not in denominator):** S3, S4, V1, V2, V3.

Pre-Phase-0 power analysis (Phase −1) MAY raise this denominator if it shows additional implicit tests. Lock the final number in `config.yaml` `bootstrap.holm_denominator` BEFORE Phase 0.

Holm-adjusted threshold (Bonferroni-equivalent worst case): raw p must be < 0.05 / 19 ≈ 0.0026 to pass at the most-conservative rank. Phase −1 power analysis must demonstrate that at the locked gate ExSharpe threshold, the realistic effect-size MDE is detectable with power ≥ 0.50; if not, raise the gate threshold from 0.20 to 0.40 BEFORE Phase 0.

Any "borderline" raw p must be reported as INCONCLUSIVE.

---

## 7. Failure Mode Watch (from prior workflows)

| Mode | Signal | Mitigation |
|---|---|---|
| Source-period bias | Bootstrap MC reproduces source-period asset means | P1 mandatory |
| Regime gate "works" only in-sample | Lookahead in regime construction | P4 + Codex review |
| Sign-flip in sub-period | 2010-2026 dominates result | P3 mandatory |
| Multiplicity inflation | Many M-gates × signal variants | Single Holm across all 27 tests |
| Carry-flattering | One asset's source-period mean drives result | Linear-scaling sweep P2 + sub-period P3 |
| Composite gate over-fitting | M5 has 4 signals × thresholds | Pre-commit thresholds; no post-hoc tuning |
| Hedged-vs-unhedged short sample | HEFA inception 2014 | Use synthetic hedged from MSCI EAFE local-ccy + DXY pre-2014 |

---

## 8. Decision Tree After Phase 4

- **All strategies fail strict gate AND P1-P5:** workflow concludes "VTI buy-and-hold remains efficient; international diversification benefit not observable in 1985-2026 returns at user-relevant horizons. Recommend 100% VTI."
- **One strategy passes ALL gates:** workflow recommends that strategy, registers prospective holdout starting 2026-05-02. Confidence is conditional on holdout outcome.
- **Multiple strategies pass:** report Pareto front; user chooses based on log-wealth vs simplicity vs taxable-account considerations.
- **Conflicting results (some periods, some hedging variants):** workflow concludes "international diversification benefit is regime-dependent in ways that cannot be pre-committed reliably; treat as exploratory. Recommend small (5-10%) static ex-US tilt as a hedge against US-permanent-dominance fragility, sized like the placebo-confirmed gold sleeve from real-world-test."

---

## 9. Locked Inputs Checklist (LOCKED v1.1)

- [x] Benchmark: VTI buy-and-hold
- [x] Walk-forward: 36/12/12
- [x] In-sample window: **2001-08-14 to 2026-04-30** (no Ken French splice for gating tests; pre-2001 only as approximate Phase 4 supplement)
- [x] Bootstrap: stationary Politis-Romano, 22-day blocks, 10,000 resamples, 90% CI
- [x] Multiplicity: Holm across **19** Phase-0-to-2 gating tests (Phase 3 exploratory not in denominator)
- [x] Gate ExSharpe threshold: 0.20 — **subject to Phase −1 power-analysis revision; if MDE > 0.20, raise to 0.40 BEFORE Phase 0**
- [x] Cost schedule: per `src/youbet/etf/costs.py` (broad_intl_equity 3 bps one-way + ETF expense ratio drag); **regime-aware multiplier or {3, 5, 10} bps sensitivity sweep TBD in Phase −1 cost analysis**
- [x] Cash rate: FRED TB3MS (3-month T-bill)
- [x] Total return only (adjusted close)
- [x] Survivorship guard: inception date enforcement via PIT
- [x] T+1 execution
- [x] Rebalance frequency: monthly signal cadence; **annual rebalance** (drift-band variant moved to Phase 4 sensitivity, not gating)
- [x] Phase ordering: −1 → 0 → 1 → 2 → 3 → 4 (no skipping; Phase −1 mandatory before Phase 0)
- [x] Holdout date: 2026-05-02 (no decision power until ≥ 24 months + 1 DXY cycle change)
- [x] Sub-period boundaries (mechanical equal-thirds): 2001-08 to 2009-12, 2010-01 to 2018-04, 2018-05 to 2026-04
- [x] CAPE publication lag: US 30 days, intl **60 days** (M2 fix)

Drift on any locked input requires a dated entry in `research/log.md` AND re-running ALL Phases 0-2 (and Phase 3 as exploratory) under the new lock.
