# Phase 5 — Post-final-report Addenda

- Run date: 2026-05-11
- Four targeted validation experiments suggested as pre-wrap-up additions.

## #1 Drawdown analysis on Phase 1 sweep

- Benchmark (100% VTI) MaxDD: **0.350**

| Weight VXUS | Strat MaxDD | Diff vs VTI | 90% CI | Verdict |
|---|---|---|---|---|
| 10% | 0.348 | -0.002 | [-0.010, +0.008] | Inconclusive |
| 20% | 0.347 | -0.003 | [-0.020, +0.020] | Inconclusive |
| 30% | 0.345 | -0.005 | [-0.027, +0.029] | Inconclusive |
| 40% | 0.344 | -0.006 | [-0.032, +0.041] | Inconclusive |
| 50% | 0.343 | -0.007 | [-0.039, +0.056] | Inconclusive |
| 60% | 0.344 | -0.006 | [-0.043, +0.074] | Inconclusive |
| 100% | 0.354 | +0.004 | [-0.043, +0.155] | Inconclusive |

## #2 Mean-shifted placebo on Phase 1 (40% VXUS)

- Delta applied: +0.000255/day (+0.0643/yr)
- Raw VXUS mean: +0.000311/day; raw VTI mean: +0.000567/day
- Original Phase 1 40% VXUS Sharpe-diff: -0.094
- **Placebo Sharpe-diff (VXUS mean-shifted to VTI's): +0.060** (CI [-0.036, +0.166])
- Placebo log-excess daily mean: +0.000017 (CI [-0.000046, +0.000082])

## #3 Correlated-nulls power analysis (Round 2 review M1)

- Cluster structure: [3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- FWER (any null pass at the 0.20-gate baseline): **0.025** (nominal Holm target: 0.05)
- Mean pass count per sim under H_0: 0.05

## #4 Phase 1 vol-min location CI (Round 2 review M4)

- Bootstrap N: 1000
- Argmin distribution (weight VXUS):
  - 0%: 1 (0.1%)
  - 10%: 2 (0.2%)
  - 20%: 19 (1.9%)
  - 30%: 134 (13.4%)
  - 40%: 306 (30.6%)
  - 50%: 347 (34.7%)
  - 60%: 191 (19.1%)
- **Fraction in Vanguard 35-55% band: 0.653**
- Mean argmin weight: 0.455
