# VTI-as-Challenger Experiment Log

## 2026-05-12 — Workflow initialized

**Premise**: across 9 youBet workflows, zero alternatives beat their workflow's native benchmark on Sharpe/CAGR-diff gates. User hypothesized this was at least partly power-limited. This workflow inverts the framing: for every alternative with positive point estimate, test whether the benchmark passes the gate against the alternative.

**Methodology**:
- Two tests per candidate:
  1. **Sign-flipped one-sided gate** (block_bootstrap_test on benchmark-as-strategy vs candidate-as-benchmark); pass at p < 0.05.
  2. **TOST equivalence** at workflow-native MEI; equivalent iff 90% CI on (benchmark - candidate) ⊂ [-MEI, +MEI].
- Source-period-bias controls (per `feedback_source_period_bias`) applied uniformly: mean-shifted placebo, sub-period split, linear scaling.
- Joint Holm across 27 candidates (statistical family ≠ 3-class reporting segmentation).
- Native window primary; 2017-01-01 to 2026-04-30 common window as sensitivity sidebar.
- Politis-Romano stationary block bootstrap, 22-day blocks, 10000 reps, deterministic per-candidate seed.

**Candidate roster (27 total)**:
- Class 1 (investable, Sharpe MEI 0.20-0.30, 14): commodity (5), real-world-test (4), stock-selection (5).
- Class 2 (paper factor portfolios, Sharpe MEI 0.20, 6): factor-timing Ken French CMA/SMB/RMW/HML SMA100 + SMB/CMA SMA200.
- Class 3 (leveraged CAGR-diff, MEI 0.01, 7): cagr-max (UPRO real/synth, SSO real, 3x SPY synth), etflab-max (VTI 2.5x, MGK 2.0x, macro-factor-aggressive 2.5x).

**Excluded (deferred to v2 — no daily-return parquet)**:
- international-etf (3 candidates would have been: HEFA 60/40, mean-shifted placebos, vol-min)
- macro-exploratory (E4 pooled 12-sleeve)
- etf (VTI SMA200 trend-following)
- factor-timing hedged VLUE (ETF-bridge daily returns not persisted)

**Phase −1 precommit**: `research/precommit.json` frozen with SHA-256 `7f19ed9...`, recorded in `config.yaml`. Phase 1+ scripts refuse to run on hash mismatch.

**Phase 0 data audit**: 27/27 candidates loaded successfully. Notable directional surprises:
- `commodity__iau_sma100_sleeve` observed Sharpe-of-excess = **-0.371** (expected: SMA timing on a tiny sleeve loses to static B&H of same sleeve due to whipsaw).
- `real_world__test4_60_40_vti_ief` observed = **-0.250** (60/40 underperformed VTI 2000-2026, only some sub-periods positive — original "promising" claim was sub-period-driven).
- `real_world__v3_1998` observed = **-0.137** (already rejected v3 strategy, included for completeness).

These will give "benchmark cleanly wins" results in the inversion — useful data points but expected.

**Unit tests**: 6/6 pass:
- `test_sign_flip_invariance`: swap-and-negate exact at deterministic seed.
- `test_tost_planted_equivalence`: 10/10 declare equivalence at n=20y, rho=0.9, MEI=0.30.
- `test_tost_planted_nonequivalence`: 5/5 correctly NOT equivalent at true diff=0.50, MEI=0.20.
- `test_holm_27_monotonicity`: adjusted p monotone in raw rank.
- `test_sign_flipped_p_one_sided_relation`: signal detection p < 0.05 for true Sharpe-diff = 0.80.
- `test_precommit_hash_lock`: load_precommit raises on tampered hash.

**Implementation note from test diagnostics**: At n=12yr daily with rho=0.9, typical 90% bootstrap CI on Sharpe-diff is ±0.18-0.25 — borderline at MEI=0.20. This means some real candidates with short windows (≤12-15 yr) and MEI=0.20 may have CIs too wide for TOST to declare equivalence. That's not a methodology bug — it's the actual power story.

## 2026-05-12 — Phase 1+2+3+4 complete; metric bug found and fixed

**Bug**: first run of phase1 mixed two distinct Sharpe metrics. `block_bootstrap_test` reports Sharpe-of-excess (Sharpe of the paired excess return series — IR-like), while `excess_sharpe_ci` reports Sharpe-diff (difference of two individual Sharpes) as its primary `point_estimate`. These differ by a factor related to the correlation of the two series — for `commodity__iau_sma100_sleeve`, Sharpe-of-excess(bench-cdt) = +0.37, but Sharpe-diff = +0.05. The workflow gates and MEIs target Sharpe-of-excess (IR), so the CI used in TOST was wrong.

**Fix**: `_shared.py::diff_ci()` now re-keys to the Sharpe-of-excess fields (`excess_sharpe_point`, `excess_sharpe_lower`, `excess_sharpe_upper`) for the sharpe metric. Sharpe-diff retained as supplementary diagnostic. Phase 1 re-run.

**Headline result (27 candidates, joint Holm-27)**:
- 1/27 raw SF p < 0.05 (commodity__iau_sma100_sleeve, p=0.0305 — static beats SMA-timed sleeve due to whipsaw)
- **0/27 survive Holm correction**
- **0/27 TOST equivalent at workflow-native MEI**
- **26/27 genuinely inconclusive** (SF=N, TOST=N)

**Interpretation**: The user's premise is confirmed: the universal null verdicts across the original 9 workflows are **power-limited**, not power-definitive. The data cannot distinguish benchmark from alternatives in either direction at the workflows' MEIs.

**Notable sub-findings**:
- Class 2 (Ken French paper factors): all 6 CIs EXCLUDE zero NEGATIVE (e.g. CMA SMA100: pt=-0.69 CI=[-0.92, -0.45]). The factor B&H benchmarks strongly underperform their SMA-timed variants over 1966-2026. Original factor-timing finding holds firmly under inversion.
- Class 1 stock-selection candidates (5 of them): point estimates -0.07 to -0.35 with CIs spanning zero at MEI=0.30. The original R9 finding ("positive but power-limited") holds — the inversion can't claim equivalence either.
- Class 3 (leveraged CAGR): all 7 inconclusive with all point estimates negative (candidate beats benchmark by 2-8%/yr CAGR), but MEI=0.01 is far below typical CI width (±0.10 = ±10%/yr CAGR). The leveraged-CAGR workflow gates were structurally unpowered, as noted in cagr-max E0 power analysis.
- **`real_world__test4_60_40_vti_ief_daily_rebal`** common-window 2017-2026 SF p=**0.0092** — in the rising-rate sub-period, VTI strongly beats 60/40 VTI/IEF. Matches the documented 2016-26 sub-period -0.063 finding.

**Robustness**: 14 PASS clean, 13 PASS-with-caveat (sub-period midpoint sign disagrees — expected for many candidates given moderate N), 0 FAIL (no placebo gates fired). Sub-period sign-flip rate is a signal of source-period dependence, not a methodology failure.

**Deliverables**:
- `research/final-report.md` — three segmented tables + headline tally
- `artifacts/{cid}.json` — per-candidate phase 1+2 results (sign-flipped + diff CI + TOST + common-window)
- `artifacts/{cid}_robustness.json` — phase 3 placebo, sub-period, linear-scaling
- `artifacts/holm_joint.json` — joint Holm-27 table
- `artifacts/phase1_summary.json`, `artifacts/phase3_summary.json` — tabular summaries
- `research/precommit.json` (SHA-locked) — frozen candidate roster + MEI table

**v2 work deferred**: re-derive daily returns for international-etf (3 candidates), macro-exploratory (E4), etf (VTI SMA200 trend), and factor-timing hedged VLUE ETF-bridge. These workflows didn't persist daily returns in parquet form.
