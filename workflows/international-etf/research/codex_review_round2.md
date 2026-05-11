# Codex Adversarial Review — Round 2 (post-implementation)

**Date:** 2026-05-04
**Reviewer:** Agent(general-purpose), adversarial-review prompt
**Subject:** `workflows/international-etf/` — code, artifacts, results.md across Phase −1 / 0 / 1 / 2 / 3
**Status of code:** all 4 phases run. Phase 4 (P3 sub-bootstrap, P1 placebo) NOT yet run.

---

## TL;DR

1. **HIGH — Test windows are mis-stated everywhere.** `phase3_results.md`, `log.md`, and the "C1b HEFA 12yr" framing all claim 2014-01-31 to 2026-04-30 (12.2 yr). The actual sample is **2017-02-15 to 2026-04-29 = 9.2 yr** (HEFA inception + 36-mo train + first fold). Same class of bug in Phase 0 (VEA "2007+" actually 2010-07-27) and Phase 1 ("2011+" actually 2014-01-29). The +0.109 CI upper bound finding is on a 9-year subset of the USD-bull era, not 12 years.
2. **HIGH — Cost sensitivity sweep is dead theatre for buy-and-hold.** Phase 0 and Phase 1 results are bit-identical at 3 / 5 / 10 bps because `BuyAndHold` has zero rebalance turnover — `total_cost_drag = 0.0` everywhere. The "PROCEED requires PASS at 10 bps" gate (`config.yaml:110`) only ever bites Phase 2 and C3. The published cost-stress narrative is misleading.
3. **HIGH — Phase 3 C1a vs C1b have different sample windows by ~10 trading days** (C1a starts 2017-02-01, C1b starts 2017-02-15; see `phase3_results.json` lines 33 vs 120). The +0.048 "hedging delta" is contaminated by sample misalignment, not a clean apples-to-apples comparison.
4. **MEDIUM — Power analysis FWER table at ExSh = 0.20 reports 0.070, but plan claims FWER controlled.** `phase_minus_1_power_results.md:15` shows null-pass rate climbs to 0.07 at target=0.20 and 0.32 at target=0.40 — i.e. with one true effect in the family the chance of ANY of the 18 nulls also passing rises sharply. The headline "FWER 0.030" is only true at exact-null target=0.0; the realistic 18-correlated-nulls scenario was never measured.
5. **MEDIUM — M5 was DROPPED but Holm denominator is still 18, not 11 effective tests.** With 0/18 PROCEED across the workflow there's no harm done, but the rationale "M5 dropped, denominator 18" doesn't match the actual lock chain (was 19 under v1.1; now 18; effective tested-and-counted = 4+7+7 = 18 ✓). Phase 0/1/2 each ran their own per-phase Holm separately rather than one workflow-wide Holm of 18; this is more conservative than the plan, not a defect.

---

## HIGH severity findings

### H1. Test-window dates mis-stated across results.md and log.md

- **What.** `phase3_results.md:5` declares "Sample: 2014-01-31 to 2026-04-30 (12.2 years)". `log.md:130` also claims 12.2 yr. **The actual sample in `phase3_results.json` is 2017-02-15 to 2026-04-29, n_test_days=2311, ≈9.2 years** (lines 120-122, 207-209, 381-383). The walk-forward 36-month TRAIN window consumes Jan 2014 – Jan 2017 before any test starts, so the first test fold begins 2017-02. C1a VEA gets 2017-02-01, C1b HEFA gets 2017-02-15 (different inception offsets in the train window).
- **Same bug in Phase 0.** `log.md:69-72`: "VXUS (2011+)", "VEA (2007+)", "VWO (2005+)", "EFA (2001+)". Actual `phase0_results.json` test_starts: VXUS=2014-01-29, VEA=2010-07-27, VWO=2008-03-11, EFA=2004-08-30. The 2.5–3yr train pre-roll is omitted from every "X (year+)" caption.
- **Same in Phase 1.** `log.md:78`: "Static VTI/VXUS sweep ... 2011-2026 (VXUS inception)". Actual Phase 1 test_start = 2014-01-29 (need to confirm in artifacts). 12-year sample, not 15.
- **Why this matters for the C1b finding.** A 9-year sample over 2017-2026 is even MORE concentrated in the USD-bull era than 12-year would have been. The +0.109 CI upper bound is conditional on a window that excludes the early-2014 EM-strength leg AND the 2017 USD reversal. Reframe Phase 3 honestly as 9.2 years.
- **Fix.** (a) Globally search-replace "12.2 yr" → "9.2 yr" and "(2014+)" → "(2017+)" across `phase3_results.md`, `log.md`, and the in-script print statements (`phase3_hedging.py:165, 270`). (b) Add a "test_window vs raw_window" disambiguation row in each `phase_X_results.md`. (c) Update `final-report.md` if drafted.
- **Cite.** `experiments/phase3_hedging.py:68-69, 99-100, 270`; `research/phase3_results.md:5`; `research/log.md:130, 152`; artifact `artifacts/phase3_results.json:33,62,120,207`.

### H2. Cost sensitivity sweep is dead theatre for B&H

- **What.** `phase0_results.json` has the same `sharpe_diff_point = -0.30455...` across 3 / 5 / 10 bps for VXUS (lines 264, 290, 316), and same for VEA / VWO / EFA / Phase-1 weights / Phase-3 C1a/C1b. `total_turnover = 0.0` and `total_cost_drag = 0.0` for every BuyAndHold run. The reason is structural: `Backtester._run_fold` only calls `cost_model.rebalance_cost` when `len(current_weights) > 0` (line 310), and the FIRST rebalance per fold has no `current_weights`, so the initial purchase is free. Subsequent rebalances within the same B&H allocation produce zero turnover. Cost level is never exercised.
- **Why this matters.** The Phase −1 cost-crisis-sensitivity ledger (`log.md:43`) and `config.yaml:110` (`pass_at_strict_cost_bps: 10`) imply the gate has a meaningful cost-stress component. For Phase 0 and Phase 1 (and C1a/C1b in Phase 3), it does not. Only Phase 2 (turnover 5.2-15.6) and C3 (turnover 2.4) are actually cost-tested. The narrative "all four FAIL at 10 bps cost level" is true but vacuous — they would fail at 100 bps, since cost is not in the path.
- **Fix.** (a) Document this honestly in `phase0_results.md` and `phase1_results.md`: "B&H total_turnover = 0; cost level immaterial." (b) For a meaningful cost stress on B&H, would need to model expense-ratio + holding cost. (c) For Phase 4 robustness, restrict the {3,5,10} bps gate to strategies with non-zero turnover. The current cost-sweep adds zero information for 11 of 18 gating tests.
- **Cite.** `phase0_results.json:259, 285, 311` (turnover=0, total_cost_drag=0 across cost levels); `Backtester._run_fold` at `src/youbet/etf/backtester.py:308-319`.

### H3. C1a vs C1b sample-window misalignment biases the +0.048 hedging delta

- **What.** Phase 3 reports `+0.048` SharpeDiff between C1b (always-HEFA) and C1a (always-VEA), and the headline claim is "currency hedging mattered in 2014-2026". `phase3_hedging.py:98-100` slices `prices[tickers].dropna(how="any")` SEPARATELY for each strategy: C1a tickers are `[VTI, VEA]`, C1b is `[VTI, HEFA]`, C3 is `[VTI, HEFA, VEA]`. VEA's first valid date is earlier than HEFA's, so the dropna filter produces:
  - C1a (VTI+VEA): test_start 2017-02-01, 2321 days
  - C1b (VTI+HEFA): test_start 2017-02-15, 2311 days
  - C3 (VTI+HEFA+VEA): test_start 2017-02-15, 2311 days
- The 10-day misalignment is small but the C1a-vs-C1b comparison is NOT apples-to-apples. C1b's VTI baseline Sharpe is computed over a strictly later window than C1a's. The +0.048 is partially "VEA missed the 2017-02-01 to 2017-02-15 leg of VTI".
- **Fix.** In `phase3_hedging.py:84-103` compute a SHARED date index across all 3 strategies (`prices[["VTI","VEA","HEFA"]].dropna(how="any")`) and pass identical sub-prices to each backtest. Re-run; expect the +0.048 to shift by 1-3 bps Sharpe.
- **Cite.** `experiments/phase3_hedging.py:98-100`; `artifacts/phase3_results.json:33, 120`.

### H4. M6 (mean-reversion) signal hard-codes 100% on with no diagnostic

- **What.** `phase2_results.json` shows M6_rev_VEA40 has `signal_on_fraction = 1.0` (always on). The strategy reduces to a pure 60/40 VTI/VEA static hold. **This is correct**: `ex_us_36m_negative_signal` requires VEA 36m return < VTI 36m return, which has held continuously since ~2010-07. The result matches Phase 1's 40% VEA = -0.094 (close to Phase 2's M6 -0.115; gap is start-date alignment per the log). **This is not a bug, but it IS an unacknowledged duplication: M6 is mechanically identical to a Phase 1 static test.** Holm denominator double-counts.
- **Fix.** Either (a) drop M6 from the Holm denominator (correctly noting 0/17, not 0/18) or (b) document in `phase2_results.md` that "M6 collapsed to Phase 1's 40% VEA static; Holm-conservative recount available on request". Operationally inconsequential since 0/18 PROCEED, but methodologically loose.
- **Cite.** `phase2_results.json:813-814`; `regime_signals.py:112-126`; `log.md:117-118`.

---

## MEDIUM severity findings

### M1. Power analysis FWER under correlated nulls is not measured

- `phase_minus_1_power_results.md:13-19` reports FWER at target=0.0 = 0.030 (good), but at target=0.20 = 0.07 and at target=0.40 = 0.32. The latter two are NOT FWER (which is defined under the global null) — they're "any-pass rates" when there IS a true effect, and they reflect the presence of the target candidate passing PLUS some null contamination. The simulator's 18 nulls are independent draws (`run_one_simulation:289-297`); in the real workflow, M1a/M1b/M1c (DXY at 3 weights) would be ~99% correlated with each other and the Holm correction is inadequate for that geometry. The "FWER controlled" claim is technically right at H_0 = 0 but is not stress-tested in the realistic case.
- **Fix.** Add a second power-analysis run where the 18 nulls are clustered into 6 correlated triplets (mimicking M1 weight-sweep variants). Compare FWER. If FWER blows out > 0.10, switch to Romano-Wolf simultaneous CIs (already in `src/youbet/etf/stats.py`). For descriptive-tier workflow this is documentation, not a verdict change.
- **Cite.** `experiments/phase_minus_1_power.py:280-297`; `phase_minus_1_power_results.md:13-19`.

### M2. Phase 1 vs Phase 3 sample-window difference attributed to "mostly start date"

- `log.md:152` says "Phase 1's 40% VEA over 2011-2026 had SharpeDiff -0.094. Phase 3's C1a 40% VEA over 2014-2026 has SharpeDiff -0.049. Difference is roughly the 2011-2013 leg dragging Phase 1 down." But Phase 1 actually used VXUS, not VEA. **This is a different ticker on a different window** — not directly comparable. VXUS holds ~30% EM (VEA 0%); VXUS includes Canada (VEA does too); but the post-2014 EM drag is a known feature of VXUS-vs-VEA. The narrative needs restating: Phase 1 = VTI/**VXUS** 2014-2026; Phase 3 C1a = VTI/**VEA** 2017-2026.
- **Fix.** Run a Phase 4 P3 calibration: VTI/VEA 40% over the SAME 2017-2026 window as C1a, vs 40% VXUS over the SAME 2017-2026 window. Should be ≤5 bp Sharpe apart.
- **Cite.** `phase1_static.py:99-104`; `phase3_hedging.py:138-139`; `log.md:151-152`.

### M3. Cost-override is not documented as a function-local subclass

- `phase0_efficiency.py:119-130` defines `class OverrideCostModel(CostModel)` INSIDE `build_cost_model()`. The closure captures `intl_bps_one_way`. This works because the backtester only ever holds the instance and never pickles or deepcopies it. **But:** if a future user adds parallel execution via `multiprocessing` or `joblib`, the local-class instance cannot be pickled and will fail. This is latent technical debt, not a current bug. Move the subclass to module level with a constructor parameter.
- **Cite.** `phase0_efficiency.py:122-128`; `phase_minus_1_cost_sensitivity.py:41-47`.

### M4. Vanguard 35-55% band claim test on 8 weights is point-estimate-only

- Phase 1 found vol-min at 60% VXUS (`phase1_results.md:11-12`), characterized as "soft violation" of the 35-55% Vanguard band. The vol-curve table (40%=0.167, 50%=0.166, 60%=0.165) shows differences in the third decimal. **No CI on the vol-min LOCATION is computed.** A bootstrap on the vol curve would almost certainly show 40 / 50 / 60% indistinguishable. The "marginal violation" framing is at single-grid-point precision, no inference.
- **Fix.** Bootstrap the vol curve: for each B sample, locate argmin among {40, 50, 60}%; report fraction-in-band. If fraction-in-35-55% > 50%, the Vanguard claim is corroborated. Cheap to compute.
- **Cite.** `phase1_static.py:136-159`; `phase1_results.md:11-22`.

### M5. Snapshot path drift between Phase 0 (2026-05-01 snap) and later phases (2026-05-02 snap)

- Two snapshots exist: `data/snapshots/2026-05-01/` and `data/snapshots/2026-05-02/`. Hashes differ (`d91d4e2c...` vs `c7026ef6...`). `precommit/data_hashes.json` records 2026-05-02 only. `_latest_snapshot_dir()` (regime_signals.py:34-39) returns "latest by reverse-sorted name", so all signal computation reads 2026-05-02. But Phase 0 was run on 2026-05-01 (per `phase0_results.json:2`), almost certainly against the 2026-05-01 snapshot before it was regenerated for DBC. Phase 1, 2, 3 used 2026-05-02. **Strictly, Phase 0's results are not reproducible from the committed hash.** Either delete 2026-05-01 snapshot and re-run Phase 0, or commit two hash files.
- **Fix.** Delete `data/snapshots/2026-05-01/`, re-run Phase 0 from the 2026-05-02 snapshot, verify identical results to current `phase0_results.json`. Recompute / lock single hash.
- **Cite.** `data/snapshots/` directory; `regime_signals.py:34-39`; `precommit/data_hashes.json:2-3`.

### M6. Phase 4 P3 mechanical-thirds 2001-08 to 2009-12 will crash for VXUS / HEFA

- `config.yaml:50-59` locks sub-period boundaries 2001-08 / 2010-01 / 2018-05. **VXUS inception 2011-01** — sub-period 1 has no VXUS data. **VEA inception 2007-07** — sub-period 1 has only ~2 years (after 36mo train: nothing). **HEFA inception 2014-01** — sub-periods 1 AND 2 have no HEFA data.
- This is BY DESIGN per plan v1.1 (in-sample window 2001-08 to 2026-04, with EFA being the only ticker that spans). But Phase 4 is supposed to re-test PROCEEDing strategies on each sub-period. There are no PROCEEDing strategies, so nothing crashes today. **But** if the user runs Phase 4 P3 on C1b (HEFA) anyway "to be thorough", it will crash or silently fall back to VTI-only weights via the `validate_universe_as_of` survivorship guard. Not currently exercised, but the design needs a documented rule: "P3 sub-periods skip strategies whose tickers have no inception data; report N/A, not a sign-flip."
- **Fix.** Add to `phase4_robustness.py` (when written) an explicit check: for each (strategy, sub-period) pair, if any ticker's inception > sub-period end, skip and emit "N/A — no sample". Document in plan.md §3 Phase 4. Note this means C1b HEFA can ONLY be tested on sub-period 3 (2018-05 to 2026-04) — which is 8 years of the same USD-bull era, providing zero independent sub-period evidence.
- **Cite.** `config.yaml:50-59`; `data/reference/international_universe.csv:18` (HEFA inception 2014-01-31); plan.md §3 Phase 4.

### M7. "log_excess CI lower" sometimes positive in Phase 1 yet headline says all negative

- `phase1_results.md:30-31` table shows weights 10/20/30/40% have log-excess CI lower in [-0.000004, -0.000020] — close to zero, all negative but borderline. The narrative "log-excess monotonically negative for every ex-US tilt" is technically correct on point estimates, but the 10% weight CI [-0.000036, -0.000004] is essentially "not distinguishable from zero log-wealth loss". Could be reframed as "below 30% VXUS the log-excess loss is undetectable; above 30% it's significant".
- **Fix.** Soften narrative in `phase1_results.md:39` and `log.md:91`. This is interpretive, not a bug.
- **Cite.** `phase1_results.md:28-34`.

### M8. Bund publication-lag uses `pd.Timedelta(days=30)` — calendar days, not business days

- `regime_signals.py:62`: `series.shift(freq=pd.Timedelta(days=lag_days))`. A monthly Bund release timestamped 2020-01-31 becomes timestamped 2020-03-01 (30 calendar days). For consumption via `signal_at_date(signal, t).iloc[-1]` where signal index is later resampled / aligned to business days, this is functionally correct (release date is end-of-Jan + 30 days, real-world OECD release is typically mid-Feb to early-Mar). 30 calendar days ≈ 22 business days; if real publication is faster (~20 calendar days for monthly OECD), the gate is conservative. No bug, but worth a single-line comment.
- **Cite.** `regime_signals.py:56-62`; OECD IRLTLT01 release calendar.

---

## LOW severity / nice-to-have

- **L1.** `phase3_results.md` Markdown body has an unrendered en-dash in the title line ("# Phase 3 — Currency Hedging Exploratory") that the writer turned into an HTML "■" in the file (line 1 reads "Phase 3 �"). Encoding issue from the Python script; harmless.
- **L2.** `phase_minus_1_power.py:80-82` uses `bootstrap_n=600` for the power sims and `n_bootstrap=10_000` for the real backtests. The factor-of-16 mismatch is acknowledged in the docstring but worth adding to the published power table.
- **L3.** `Backtester._get_rebalance_dates` for "annual" frequency uses `to_period("Y").first()` (line 211-213). With a 12-month test window starting in February (e.g. fold starts 2014-01-29), the FIRST trading day of 2014 is OUTSIDE the test window, so no annual rebalance fires for that fold's first calendar year — strategy holds initial weights for ~11 months unrebalanced. Acceptable for B&H but a quirk worth documenting in `backtester.py:202-214`.
- **L4.** `phase2_regime.py:137-138` has an `import` inside the `for t in test_dates` loop: `from strategies.regime_signals import signal_at_date`. Hot-path import in a tight loop over 4000+ days × 7 strategies × 3 cost levels. Move to module-level. Negligible perf cost given that signal_at_date is microseconds, but bad form.
- **L5.** `regime_signals.py:34-39`'s `_latest_snapshot_dir` sorts directory names lexicographically, which for ISO-format dates is correct, but a future "2026-05-02-rerun" suffix would silently shadow "2026-05-02". Not a current issue.
- **L6.** Phase 0 narrative says VWO test_start 2008-03 implies "2005+ inception". The 36-month train means the EFFECTIVE evaluation period starts 2008-03 — which is right at the GFC peak. VWO's first test year is its worst; this likely contributes to its -0.376 (worst of 4) point estimate. Consider noting "GFC-anchored start" as a sample-period caveat.
- **L7.** `config.yaml:110` `pass_at_strict_cost_bps: 10` is meaningful for Phase 2 + C3 only. State this in the gate doc.
- **L8.** `dxy_12m_positive_signal` (regime_signals.py:74-84) and `dxy_12m_negative_signal` (lines 65-71) read DXY twice and compute `pct_change(252)` twice. Trivial, but cleaner to compute once and derive both.

---

## What the implementation got RIGHT

1. **Phase −1 power analysis was actually run, the v1 vol-mismatch bug was caught, v2 fix was correct** (`simulate_strategy_returns:120-134` builds eps as unit-variance then rescales — variance of `eps_unit` is `corr^2 + (1-corr^2) = 1` ✓). Even better: the user honored the pre-committed HALT decision rule and downgraded the workflow to descriptive-only when v2 power came in below threshold. Discipline.
2. **PIT publication-lag direction is correct.** `_apply_publication_lag` shifts forward, `signal_at_date` strict-`<`. No same-day leak even at 0-lag DXY.
3. **NaN-safe portfolio return logic** (`backtester.py:344-358`) treats pre-inception or missing-asset weights as cash earning T-bill, not zero. Correct handling for VEA pre-inception in Phase 3 multi-ticker subsets.
4. **Survivorship guard via `validate_universe_as_of`** is applied to BOTH strategy AND benchmark (lines 273-293), preventing differential-survivorship comparison artifacts.
5. **Holm-within-phase rather than across-phases is conservative.** Plan v1.3 declared denom 18 but Phase 0/1/2 each ran their own 4 / 7 / 7 within-phase Holm — this is more conservative (smaller correction families, but applied at the result-table level the user actually reads), not less. With 0/18 PROCEED there's no harm.
6. **All 7 Phase-2 CIs exclude zero on the negative side** at strict 10 bps is a clean, decisive negative result. Not a power-failure-of-detectability mush.
7. **Plan ledger discipline** — every parameter change has a dated row. The v1.3 entry dropping M5 (intl CAPE unavailable) recounts the denominator from 19→18 explicitly.
8. **Round 1 H1 (synthetic-hedged formula bug) was correctly addressed by REMOVING the test, not papering over it.** Phase 3 is real-HEFA-only and explicitly EXPLORATORY/NOT-IN-HOLM.
9. **C1b CI [-0.095, +0.109] is reported with the correct disclaimer** (`log.md:147-151`): "first whose CI upper bound is meaningfully positive" — not over-claimed as PROCEED.

---

## Recommended next steps before final report (priority order)

1. **MUST-FIX before any external claim: H1, H2, H3.**
   - H1: Globally correct test-window dates ("9.2 yr", "2017-2026" for Phase 3; "2014+" for Phase 1; per-ticker effective starts for Phase 0). Touches `log.md`, `phase{0,1,3}_results.md`, scripts' print headers.
   - H2: Add a one-line caveat to Phase 0 + Phase 1 results.md: "B&H total_turnover = 0; cost-level sweep is structurally a no-op for these tests."
   - H3: Re-run Phase 3 with a SHARED 3-ticker (VTI+VEA+HEFA) date index. Re-publish `phase3_results.json`, `phase3_results.md`. Expect C1b -0.002 to shift by ≤3 bp; the +0.048 hedging delta to shift by ≤5 bp. The qualitative finding will not change but the numbers will.

2. **SHOULD-DO before Phase 4 / final report.**
   - **Run Phase 4 P3 mechanical-thirds even though no strategy PROCEEDed** — specifically on C1b HEFA — but document that only sub-period 3 (2018-05 to 2026-04) is testable. This will tell the user whether the 2018-2026 sub-window alone can produce the +0.109 CI upper bound, or whether the 2017-2018 leg is doing the work.
   - **Run P1a USD-return mean-shifted placebo on C1b**: take HEFA's monthly USD return, shift to match VTI's mean, re-run C1b. If the +0.109 CI upper bound disappears or flips, the finding is source-period bias (the dominant failure mode per `feedback_source_period_bias` memory). Cheap, decisive.
   - **Run the linear-scaling sweep P2 on C1b**: 1/5/10/30/50% HEFA in VTI. If the SharpeDiff is monotonic in weight, the structural-rebalancing-premium story is supported; if it peaks then collapses, it's an artifact.
   - **Add a single Phase-4 calibration run**: 40% VEA in VTI over the SAME 2017-2026 window as C1a, to validate H4 / M2's claim that VXUS-vs-VEA on different windows confounds Phase 1 and Phase 3.

3. **NICE-TO-HAVE.**
   - M1: re-run power analysis with correlated nulls (clustered into M1-style triplets).
   - M4: bootstrap the Phase 1 vol curve to put a CI on the vol-min location.
   - M5: collapse to a single committed snapshot (delete 2026-05-01 dir, re-verify Phase 0 reproduces).

4. **Final-report state assessment.** Current state is **NOT ready for final report** until at minimum H1+H2+H3 are fixed and at minimum **P1a placebo on C1b** is run. Without P1a the +0.109 CI upper bound is unfalsifiable — the prior workflows (real-world-test, gold-mean-shifted) make clear this MUST be done before any descriptive claim of "hedging may have helped". With those four fixes (H1/H2/H3 housekeeping + P1a + P2 + Phase-4-P3-on-C1b sub-period-3-only), the workflow has a defensible final report. Without them, the headline "C1b CI upper bound +0.109" is exactly the kind of unguarded finding the project memory warns against.

5. **Workflow tier remains DESCRIPTIVE/EXPLORATORY.** Even with all fixes, the strict gate cannot pass at the locked-in 0.30 power. The honest outcome: "VTI is at minimum CI-overlap-with-zero efficient against every pre-committed ex-US allocation in 2001-2026; the strongest exception is HEFA-hedged in 2017-2026, which after sample-window correction is essentially Sharpe-neutral; this MAY be USD-regime-specific and CANNOT be confirmed without a fresh 2026+ holdout."

---

**End of Round 2 review.**
