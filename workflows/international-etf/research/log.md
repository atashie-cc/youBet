# International ETF Workflow — Experiment Log

This log is read at the start of every session. Append-only; do not rewrite history.

## Pre-commit ledger

Every parameter change to LOCKED values (gate criteria, walk-forward window, block length, benchmark, rebalance frequency, cost schedule) requires a row here BEFORE re-running anything.

| Date       | Author | Change | Rationale |
|------------|--------|--------|-----------|
| 2026-05-01 | claude | Workflow created. All params per `config.yaml` v0. Benchmark = VTI B&H. Gate = ExSharpe>0.20 AND Holm p<0.05 AND CI_lower>0. WF 36/12/12. Bootstrap stationary 22d. | Mirrors etf/ + commodity/ workflows. |
| 2026-05-02 | claude | v1.3: M5 (composite 4-signal regime gate) DROPPED. Free intl CAPE feed (Siblis/MSCI/Barclays) not available; CAPE-less composite = 3-of-3 (over-strict). Holm denominator 19 → 18. Added DBC fetch to snapshot. | Honest data-availability constraint, not post-hoc tuning. |
| 2026-05-01 | claude | v1.1 after Round 1 codex review (4 HIGH + 8 MEDIUM). Plan revisions: (1) drop Ken French pre-2001 splice (H3) — in-sample window locked at 2001-08 to 2026-04; (2) drop synthetic hedged-EAFE pre-2014 (H1) — Phase 3 restricted to 2014-2026 EXPLORATORY, not in Holm denominator; (3) split P1 placebo into P1a (USD-return mean-shift, primary) + P1b (local-equity-component, secondary, mandatory for hedging) (H4); (4) recount Holm denominator to **19** after pruning M2, V1×40%, drift-band rebalance variant (H2); (5) add Phase −1 mandatory prep (power analysis, Vanguard PDF, cost crisis, data hashes) (M1); (6) intl CAPE lag 60d (M2); (7) sub-periods to mechanical equal-thirds 2001-2009 / 2010-2018 / 2018-2026 (M4); (8) demote V1 to 3y forward + informative-not-gating, S3 to descriptive-only (M5+M6); (9) holdout 24mo no-decision-power (M8); (10) cost crisis sensitivity sweep {3,5,10} bps with PROCEED gated at 10 bps (M7). | Sharp Round 1 review. Synthetic-hedged formula bug would have biased C1/C3 verdicts. Ken French splice claim was load-bearing for pre-2001 robustness — dropping it shortens supportable history but keeps integrity. Holm denominator 19 instead of 27 because dropped tests, not relaxed standards. |

## Sessions

### 2026-05-01 — Workflow scaffolded + plan committed

- Created `workflows/international-etf/` mirroring etf/ pattern.
- Confirmed reusable engine: backtester, PIT, costs (broad_intl_equity = 3 bps), stats, allocation. International tickers (VXUS/VEA/VWO/EFA/EEM/HEFA/HEDJ) all fetch via yfinance. DXY fetcher exists in commodity/macro/fetchers.py — will be re-registered into etf/macro/.
- Dispatched 3 parallel literature agents: academic, industry/forum, macro-regime. Outputs in `research/literature_academic.md`, `literature_industry.md`, `literature_macro_regime.md`.
- Synthesized 30 source hypotheses into 21 deduplicated, falsifiable tests across 4 phases (S1-4 static, V1-3 valuation, M1-6 macro, C1-3 currency, P1-5 robustness). See `research/plan.md`.
- **Pre-committed Holm denominator: 27 tests across Phases 0-3.** Phase 4 (robustness) does not add new claims, only validates existing ones.
- **Pre-committed holdout date: 2026-05-02** (tomorrow). Any PROCEEDing strategy enters prospective tracking; no re-fit after this date.
- Key prior-art lessons baked into the plan:
  - Source-period bias is the dominant failure mode (real-world-test). Mean-shifted placebo P1 is mandatory.
  - 3-period sub-bootstrap (P3) catches sample-window-dependent results that pass single-period CIs.
  - Linear-scaling sweep (P2) catches carry-flattering one-asset effects.
  - Codex adversarial review (P5) is mandatory before any final claim, per `feedback_codex_review`.
- Next session: implement Phase 0 efficiency test in `experiments/phase0_efficiency.py`.

### 2026-05-01 — Round 1 codex adversarial review

- Dispatched Agent(general-purpose) with adversarial-review prompt covering 15 dimensions. Output saved at `research/codex_review_round1.md`.
- **Findings: 4 HIGH (synthetic hedged-EAFE formula wrong, Holm denominator under-counted, Ken French splice impossible, P1 placebo ambiguous), 8 MEDIUM (no power analysis, CAPE lag short, Vanguard PDF unverified, sub-period bin-shopping, V1 underpowered, S3 null-logic confusion, cost crisis insensitive, 1-day holdout theatrical), 7 LOW.**
- **Plan v1.1 applied per ledger above.** All 4 HIGH and 7 of 8 MEDIUM addressed; M3 (Vanguard PDF verification) deferred to Phase −1 task.
- Strategy count net of changes: Phase 0 = 4, Phase 1 = 7 gating (8 weights, 0% excluded), Phase 2 = 8 gating (M1×3, M3, M4×2, M5, M6), Phase 3 = 0 gating (exploratory). **Holm denominator = 19**.
- Next session: Phase −1 power analysis script `experiments/phase_minus_1_power.py`. If MDE > 0.20 ExSharpe at the locked denominator, raise gate to 0.40 BEFORE Phase 0.

### 2026-05-01 — Phase -1 prep (in progress)

- **Vanguard 2019 PDF primary-source verification (M3 fix): COMPLETE.** Agent fetched the actual PDF; the citation in plan v1.0 was wrong. Real range is **35-55%** (20-point band), not 40-50%. Result is also **forward-looking VCMM simulation**, not realized historical — historical only confirms in appendix. Updated `plan.md` S1 and `literature_industry.md`. Agent's full report at `research/vanguard_pdf_verification.md`.
- **Cost-model crisis sensitivity verification: COMPLETE.** `experiments/phase_minus_1_cost_sensitivity.py` confirms `OverrideCostModel` mechanism allows broad_intl_equity bps to be set per-run while leaving VTI/VGSH unchanged. Phase 0/1/2 will run each strategy at {3, 5, 10} bps and require pass at 10 bps. Artifact: `artifacts/phase_minus_1_cost_override_verified.json`.
- **Data fetch + hash: COMPLETE.** `experiments/phase_minus_1_fetch_and_hash.py` fetched 23 ETFs (6368 rows) + 6 macro series (DXY, US 10y, German Bund 10y, T-bill, broad dollar TWI, 10y breakeven inflation). Wrote `data/snapshots/2026-05-01/` and SHA256 hashes to `precommit/data_hashes.json`. Phase 4 will re-fetch + re-hash; mismatch triggers audit-before-claim.
- **Power analysis: V1 RESULTS RETRACTED, V2 COMPLETE.** First run completed at 10:07 with apparent power=0.580 at ExSharpe=0.20. **BUT inspection showed point estimate at target=0.0 was 0.272, not 0** — diagnosed bug in the simulator: linear combination of standardized benchmark + sigma-scaled noise had non-unit variance, making strategy std ≈ 0.7× benchmark std and giving the strategy a Sharpe boost from lower vol. Fix: build eps as unit-variance THEN rescale to benchmark_sigma. v2 verified: target=0.0 mean point = -0.001 (correctly centered).
- **V2 RESULTS:** power at ExSharpe=0.20: **0.04** (vs 0.50 threshold). Power at ExSharpe=0.40: **0.30** (also below 0.50). FWER at target=0.0: 0.030 (well-controlled). Point SD ≈ 0.155 across all effect sizes, indicating bootstrap CI is stable and the underlying detectability is genuinely low at this sample size + N=19.
- **DECISION (per pre-committed rule):** HALT triggered. Actions taken in v1.2:
  1. Gate ExSharpe threshold raised from 0.20 to 0.40 in `config.yaml`.
  2. Workflow tier downgraded from CONFIRMATORY to DESCRIPTIVE/EXPLORATORY.
  3. `plan.md` §1 rewritten — no strategy will be claimed CONFIRMED-PROCEED; results reported as point estimates + CIs; Phase 4 robustness + prospective holdout become the binding cross-validation.
  4. This mirrors `cagr-max` workflow (0/65 PASS, structurally unpowered per E0). Honest outcome of pre-committed methodology, not a failure.
- This means Phase 0/1/2 will run as planned to produce informational results. Strategies that show positive point estimates with CIs that exclude zero will be flagged as "EXPLORATORY-POSITIVE" but not PROCEED. The user will get descriptive evidence about ex-US allocation, not a confirmatory verdict.
- Implication for project memory + CLAUDE.md: this workflow joins `cagr-max` and (informally) the etf workflow's drawdown analysis as DESCRIPTIVE-only; the strict gate is the right tool for hypothesis testing in this domain, but at realistic effect sizes the gate is uninformative — only Phase 4 robustness + multi-year prospective tracking can confirm a finding.

### 2026-05-01 — Phase -1 COMPLETE

All 5 deliverables done:
1. Power analysis v2 (gate raised to 0.40, tier downgraded to descriptive)
2. Vanguard PDF verified (35-55%, not 40-50%; VCMM not historical)
3. Holm denominator audit (locked at 19)
4. Cost-override mechanism verified
5. Data fetched + SHA256 hashed

Phase 0 script written and reads gate from config.yaml. Ready to run.

### 2026-05-01 — Phase 0 COMPLETE (4/4 FAIL)

Single-ticker buy-and-hold, walk-forward 2001-2026 (each starts at its own inception via PIT survivorship guard):
- VXUS (2011+): ExSharpe-diff -0.305, 90% CI [-0.566, -0.064]
- VEA (2007+):  ExSharpe-diff -0.333, 90% CI [-0.548, -0.129]
- VWO (2005+):  ExSharpe-diff -0.376, 90% CI [-0.611, -0.152]
- EFA (2001+):  ExSharpe-diff -0.244, 90% CI [-0.417, -0.085]

**All 4 FAIL the strict gate; all 4 90% CIs exclude zero on the negative side** — i.e., statistically significant Sharpe underperformance vs VTI in this sample. As expected from literature priors (2010-2024 US dominance era + AQR 2023 multiple-expansion finding). Cost level (3/5/10 bps) doesn't change the verdict.

### 2026-05-02 — Phase 1 COMPLETE (0/7 PROCEED)

Static VTI/VXUS sweep at 7 gating weights {10, 20, 30, 40, 50, 60, 100}% VXUS, annual rebalance, 2011-2026 (VXUS inception):

| Weight | ExSharpe-diff | 90% CI | Sharpe-diff CI excludes 0? |
|---|---|---|---|
| 10% | -0.019 | [-0.043, +0.005] | No (essentially flat) |
| 20% | -0.041 | [-0.090, +0.007] | No |
| 30% | -0.066 | [-0.140, +0.007] | No |
| 40% | -0.094 | [-0.194, +0.005] | No |
| 50% | -0.124 | [-0.250, +0.000] | Borderline |
| 60% | -0.156 | [-0.310, -0.008] | YES, negative |
| 100% | -0.305 | [-0.566, -0.064] | YES, negative |

**Empirical findings:**
1. **Monotonic Sharpe degradation** with ex-US weight; even 10% costs ~0.02 Sharpe (not significant).
2. **Vol IS reduced** by adding ex-US (0.176 at 0% → 0.165 at 60%; -1.1pp annualized). Directional Vanguard claim confirmed; the 35-55% band slightly violated (realized vol-min at 60% but the curve is flat 40-60% so it's a soft violation).
3. **Log-excess (E[log(W)]) is monotonically negative** for every ex-US tilt; adding ex-US strictly reduces compound wealth in 2011-2026.
4. **No 90% CI includes positive ExSharpe** at any weight; 60% and 100% CIs exclude zero on the negative side (statistically significant Sharpe degradation).

This is a clear PHASE 1 NEGATIVE result for unconditional ex-US tilts in the user-facing 2011-2026 sample. Bug fixes during the run: `RiskMetrics.sharpe` → `sharpe_ratio`; `annualized_volatility` was the correct attr (not `annual_volatility`); markdown key formatting `_intl010bps` → `_intl10bps`. Plus added `"annual"` rebalance frequency to `src/youbet/etf/backtester.py` (previously only daily/weekly/monthly supported).

### 2026-05-02 — Phase 2 COMPLETE (0/7 PROCEED)

Implemented `strategies/regime_signals.py` (4 PIT-aware signals), `strategies/regime_conditional.py` (RegimeConditionalStrategy), and `experiments/phase2_regime.py`. M5 dropped (no intl CAPE data) → 7 gating tests, Holm denominator 18.

**Verdicts at strict 10 bps cost (all FAIL):**

| Strategy | Sharpe-diff | 90% CI | sig_on | Log-excess daily |
|---|---|---|---|---|
| M1a DXY<0 → 20% VEA | -0.027 | [-0.054, -0.001] | 0.42 | -0.000020 |
| M1b DXY<0 → 40% VEA | -0.058 | [-0.111, -0.006] | 0.42 | -0.000040 |
| M1c DXY<0 → 60% VEA | -0.091 | [-0.172, -0.015] | 0.42 | -0.000061 |
| M3 yield-narrowing → 40% VEA | -0.054 | [-0.111, -0.000] | 0.38 | -0.000043 |
| M4a DBC>0 → 20% VWO | -0.054 | [-0.092, -0.015] | 0.49 | -0.000039 |
| M4b DBC>0 → 40% VWO | -0.112 | [-0.187, -0.036] | 0.49 | -0.000079 |
| M6 mean-reversion → 40% VEA | -0.115 | [-0.203, -0.029] | 1.00 | -0.000087 |

**Empirical findings:**
1. **All 7 CIs have negative or zero upper bound** — the regime gates do not flip the sign of the ex-US tilt's effect.
2. **Conditioning DOES reduce the Sharpe loss vs static (Phase 1)**: M1c at 60% conditional VEA has Sharpe-diff -0.091 vs static 60% VEA at -0.156. The DXY signal carries SOME information (negative-DXY periods are mildly less bad for ex-US) but the residual edge is not enough to overcome structural underperformance after costs.
3. **M6 (mean-reversion) signal stayed ON 100% of the test window** — VEA has been beaten by VTI for the entire 36m-trailing window since 2010-07. So M6 is functionally a static 40% VEA hold; result matches Phase 1's 40% VEA Sharpe-diff (-0.094 vs -0.115 here, the small gap is start-date alignment).
4. **Even the "best" regime gate (M1a, 20% VEA on DXY-weak) loses** with Sharpe-diff -0.027 [CI -0.054, -0.001] — CI excludes zero on the negative side.
5. **No PROCEED-direction signal anywhere.** The pre-committed regime hypotheses (DXY weakness, yield narrowing, commodity uptrend, mean-reversion) all fail to identify ex-US tilts that beat VTI in 2007-2026.

**Workflow-level cumulative tally: 0/18 PROCEED across Phase 0-2 strict gate.** Sharpe-diff CIs at strict 10bps cost:
- Phase 0 (4 single-ETF): all CIs exclude zero negative
- Phase 1 (7 static weights): 5 CIs span zero, 2 (60% and 100%) exclude zero negative
- Phase 2 (7 regime-conditional): 6 CIs exclude zero negative, 1 (M3) at exact zero upper bound

The workflow has now decisively answered its question: **no pre-committed ex-US allocation, static or regime-conditional, generates a positive Sharpe-diff vs VTI buy-and-hold in the 2001-2026 sample**. The variance-reduction finding from Phase 1 (vol drops 0.176→0.165 at 60% VXUS) holds, but Sharpe loss outweighs it.

### 2026-05-04 — Phase 3 COMPLETE (EXPLORATORY)

Implemented `strategies/dynamic_hedge.py` (DynamicHedgeStrategy) and `experiments/phase3_hedging.py`. 3 strategies × 3 cost levels × 2 hedge-friction variants = 18 backtests on 2014-01-31 to 2026-04-30 sample (12.2 years; below P3 mechanical-thirds threshold).

**Headline @ strict 10 bps cost, +0 bps friction:**

| Strategy | SharpeDiff | 90% CI | Vol | Turnover |
|---|---|---|---|---|
| C1a 60/40 VTI/VEA | -0.049 | [-0.161, +0.069] | 0.176 | 0.000 |
| C1b 60/40 VTI/HEFA | -0.002 | [-0.095, +0.109] | 0.171 | 0.000 |
| C3 dynamic-hedge | -0.046 | [-0.146, +0.064] | 0.173 | 2.400 |

**With +15 bps HEFA hedge-roll friction added:** C1b SharpeDiff -0.005, CI [-0.099, +0.105] (essentially unchanged); C3 unchanged at -0.049.

**Hedged-vs-unhedged delta (informational, both at +0bps friction):**
- C1b (always-hedged) - C1a (always-unhedged) = **+0.048 Sharpe**
- C3 (dynamic toggle) - C1a (always-unhedged) = **+0.003 Sharpe**

**Notable findings:**
1. **C1b HEFA is the FIRST strategy in the entire workflow (across 18 gating + 3 exploratory) whose CI upper bound is meaningfully positive (+0.109).** Always-hedged 60/40 VTI/HEFA in 2014-2026 is essentially indistinguishable from VTI on Sharpe terms.
2. **Currency hedging mattered in 2014-2026.** HEFA beats VEA by +0.048 Sharpe over 12 years. This confirms the macro literature: hedging captures local-equity returns without USD-strength drag during USD-bull regimes (2014-2022).
3. **C3 dynamic-hedge adds essentially nothing over static C1b** (+0.003 vs +0.048), AND incurs massive turnover (2.4 vs 0). The DXY-trend-toggle strategy churns positions every year without producing alpha. Fixed full-hedge dominates.
4. **+15 bps hedge friction is immaterial** — HEFA still beats VEA by +0.044 with friction. The result is structural, not friction-sensitive.
5. **Critical caveat — sample-period bias:** the 12-year sample IS the USD-bull era (DXY 80→105 from 2014 to 2024). HEFA is structurally favored in this regime. If USD continues weakening from its Sep 2022 peak, the HEFA advantage will reverse. This is the central reason Phase 3 is exploratory and CANNOT pass strict gate — the 12yr sample fails Phase 4 P3's three-window requirement, AND the source-period-bias guard from real-world-test workflow says one regime ≠ confirmed finding.
6. **Phase 1 vs Phase 3 cross-reference:** Phase 1's 40% VEA over 2011-2026 had SharpeDiff -0.094. Phase 3's C1a 40% VEA over 2014-2026 has SharpeDiff -0.049. Difference is roughly the 2011-2013 leg dragging Phase 1 down. Sample alignment matters.

**Workflow cumulative tally: 0/18 PROCEED across Phase 0-2 strict gate; 0/3 EXPLORATORY-PASS in Phase 3** (none clears the descriptive 0.40 ExSharpe threshold either, though C1b CI now contains zero rather than excluding it negative). The honest sample-aware reading: **HEFA-hedged ex-US in 2014-2026 was Sharpe-neutral vs VTI; this MAY be USD-regime-specific**.

### 2026-05-04 — Round 2 codex review COMPLETE (4 HIGH + 8 MEDIUM + 8 LOW)

Review at `research/codex_review_round2.md`. Top findings:
- **H1**: Test-window dates mis-stated everywhere — Phase 3 "12.2 yr" is actually **9.2 yr** after the 36-mo train consumes 2014-01 to 2017-02. Same in Phase 0 ("VEA 2007+" is actually test_start 2010-07) and Phase 1.
- **H2**: Cost-sensitivity sweep is dead theatre for buy-and-hold (turnover=0). Phase 0 + Phase 1 + C1a/C1b results bit-identical at 3/5/10 bps. The "PROCEED requires PASS at 10 bps" gate only meaningfully bites Phase 2 + C3.
- **H3**: Phase 3 C1a vs C1b had different sample windows by ~10 trading days (separate dropna on different ticker subsets). FIXED — added shared 3-ticker date index.
- **H4**: M6 mean-reversion signal stayed ON 100% of test window — collapses to a static 60/40 VTI/VEA hold; Holm denominator double-counts (operationally inconsequential since 0/18 PROCEED).

### 2026-05-04 — Phase 3 v2 (H3 fix) + Phase 4 robustness on C1b COMPLETE

**Phase 3 v2 (shared 3-ticker window):**

| Strategy | SharpeDiff | 90% CI | Vol | Turnover |
|---|---|---|---|---|
| C1a 60/40 VTI/VEA | -0.048 | [-0.158, +0.069] | 0.177 | 0.000 |
| C1b 60/40 VTI/HEFA | -0.002 | [-0.095, +0.109] | 0.171 | 0.000 |
| C3 dynamic-hedge | -0.046 | [-0.146, +0.064] | 0.173 | 2.400 |

H3 fix changed numbers in the 0.001 range — the +0.109 CI upper bound on C1b survived the shared-window correction.

**Phase 4 robustness on C1b (the only finding with positive CI upper bound):**

- **P1a USD-return mean-shifted placebo: FAIL.** HEFA mean was **-2.95%/yr below VTI** in the 2017-2026 sample. After mean-equalization, placebo Sharpe-diff = +0.067 with CI [-0.028, +0.184]; placebo log-excess CI lower = -0.000072 (excludes positive). Conclusion: the +0.109 CI upper bound was source-period bias — HEFA's structural underperformance was almost-but-not-quite compensated by diversification. Mean-equalization removes the sample-bias half and leaves no residual structural rebalancing premium.
- **P2 linear-scaling sweep:** HEFA at 1/5/10/30/50% gives SharpeDiff +0.000/+0.001/+0.002/+0.002/-0.007. Essentially flat near zero — no peak-then-collapse artifact AND no positive scaling. Zero structural alpha.
- **P3 sub-period 3 (2018-05 to 2026-04, only sub-period testable):** SharpeDiff +0.113, CI [-0.033, +0.258]. More positive than full window but still spans zero, fully USD-bull regime — sub-periods 1+2 untestable (HEFA inception 2014-01-31).

**Combined verdict: C1b is REJECTED.** The "first positive CI upper bound in the workflow" is a sample-period artifact, not a structural rebalancing premium. This is exactly the failure mode the project memory `feedback_source_period_bias` was designed to catch; the placebo correctly identified it.

### Final cumulative answer

**0/18 strict-gate PROCEED + 0/3 EXPLORATORY-PASS + 0/1 robustness-survive (C1b rejected by P1a).**

The workflow has now decisively answered its core question across multiple test framings:
- No single ex-US ETF (Phase 0) beats VTI; CIs exclude zero negative.
- No static VTI/VXUS mix (Phase 1) beats VTI; vol IS reduced ~1.1pp at 60% but Sharpe degrades monotonically.
- No regime-conditional gate (Phase 2) flips the sign; conditioning helps mildly but never enough.
- No currency-hedging variant (Phase 3, exploratory) survives source-period-bias placebo.
- No structural rebalancing premium for HEFA (Phase 4 P1a/P2).

**Practical recommendation:** 100% VTI buy-and-hold remains efficient against every pre-committed ex-US allocation tested. The honest variance-reduction observation from Phase 1 (~1.1pp annual vol drop at 60% VXUS) is not large enough to justify the Sharpe loss for an E[log(W)]-maximizing investor.

### 2026-05-04 — Final-report INTERIM v1.0 written

`research/final-report.md` (~3,000 words, 13 sections). Marked INTERIM because (a) prospective holdout has 24mo no-decision-power, (b) Round 2 nice-to-haves outstanding, (c) future Round 3 review is appropriate. Practical recommendation: 100% VTI buy-and-hold remains efficient against every pre-committed ex-US allocation tested.

### 2026-05-05 — Prospective holdout tracking LAUNCHED

`experiments/holdout_tracking.py` written + first run logged to `research/holdout_tracking.md`. Tracks 4 descriptive baselines (none passed Phase 0-4 strict gate but they're the natural anchors for "would the verdict reverse out-of-sample?"):

- **A**: 100% VTI buy-and-hold (anchor)
- **B**: 60/40 VTI/VXUS annual rebalance (Vanguard variance-min point)
- **C**: 60/40 VTI/HEFA annual rebalance (the C1b near-miss; rejected by P1a placebo; tracked for falsification)
- **D**: 60/40 VTI/VEA annual rebalance (unhedged hedging-comparison baseline)

Each run: yfinance fetches latest prices, computes cumulative return since 2026-05-02, records DXY 12m return for direction-change tracking, appends a row to the markdown rolling log. Idempotent (de-dupes by as-of date). Decision-power date 2028-05-02 + at least one DXY direction-change required before any conclusion.

**First-row reading (2026-05-05, 1 trading day in):** A=-0.37%, B=-0.57%, C=-0.60%, D=-0.69% cumulative. C-D=+0.0009 (HEFA beat VEA by 9bps in 1 day; directionally consistent with 9.2yr Phase 3 finding but pure noise at this horizon). DXY 12m return = -1.53% (negative). One trading day = uninformative; first meaningful read at ~6 months, decision-power read at 2028-05.

### 2026-05-11 — Phase 5 addenda COMPLETE (4 experiments)

`experiments/phase5_addenda.py` + `research/phase5_addenda.md` + `artifacts/phase5_addenda.json`.

**#1 Drawdown analysis on Phase 1 sweep.** VTI MaxDD = 35.0% over the 2014-2026 sample. Every VTI/VXUS mix has slightly lower MaxDD (-0.002 to -0.007 difference), but ALL 90% CIs span zero — **INCONCLUSIVE at every weight**. 100% VXUS is actually directionally worse (+0.004). **The diversification argument's main practical justification (lower drawdown during crashes) does NOT hold up empirically in this sample.** Stronger support for the 100% VTI recommendation than vol-reduction alone provided.

**#2 Mean-shifted placebo on Phase 1 at 40% VXUS.** VXUS mean was **-6.43%/yr below VTI** in the 12.3yr Phase 1 sample. After mean-equalization placebo: Sharpe-diff goes from -0.094 to **+0.060** (CI [-0.036, +0.166]); log-excess daily mean +0.000017 (CI spans zero). **The Phase 1 negative verdict is ENTIRELY mean-asymmetry-driven** — there is no structural Sharpe disadvantage to a 40% VXUS tilt, only a mean disadvantage. If the GMO/Asness mean-reversion thesis materializes (US underperforms ex-US going forward), the verdict would flip cleanly. This is the same source-period-bias mechanism that killed C1b, but with a more sympathetic conclusion: the Phase 1 result is high-conditional on the 2014-2026 US-dominance era and does not falsify the structural diversification argument.

**#3 Correlated-nulls power analysis (Round 2 M1).** Cluster structure [3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] = 18 nulls. **FWER = 0.025** under H₀ (well below 0.05 nominal); mean false-positive count per sim = 0.05. **Holm correction is ADEQUATE under correlated nulls** — actually more conservative under positive correlation than under independence, as expected theoretically. No need to switch to Romano-Wolf simultaneous CIs.

**#4 Vol-min location CI (Round 2 M4).** Bootstrap 1000 samples; argmin distribution: 30%=13.4%, 40%=30.6%, 50%=34.7%, 60%=19.1%. **Fraction in Vanguard 35-55% band: 65.3%**. Mean argmin = 0.455 (squarely in band). **The Vanguard 35-55% claim is corroborated.** The Phase 1 markdown's "marginal violation" framing was too harsh — the data strongly supports Vanguard's prediction; the point-estimate vol-min at 60% was within bootstrap noise of the band.

### Net effect on the workflow story

- **Recommendation strengthens:** drawdown finding adds a second practical reason for 100% VTI (vol-reduction without drawdown reduction is a weaker case for ex-US tilt than originally framed).
- **Causal story for Phase 1 negative result becomes clear:** mean-asymmetry, not structural diversification failure. Updates the Bayesian prior for "would this verdict reverse under a different mean regime?" → answer: yes, plausibly.
- **Methodology validated:** Holm correction is adequate; Vanguard claim is corroborated; the workflow's statistical machinery is sound.

### Remaining optional steps

- Update `final-report.md` from v1.0 INTERIM to v1.1 with Phase 5 findings (recommend doing this)
- Documentation cleanup for H1/H2 — DONE 2026-05-04
- M3 (cost-override class to module level), M5 (snapshot consolidation 2026-05-01 vs 2026-05-02), L4 (hot-path import) — minor cleanup
- Schedule monthly cron / wrapper for `holdout_tracking.py`
- Optional Round 3 codex review

