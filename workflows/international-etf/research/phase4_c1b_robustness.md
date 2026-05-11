# Phase 4 Robustness — C1b (60/40 VTI/HEFA) only

- Run date: 2026-05-04
- Subject: the only Phase 0-3 finding whose CI upper bound is positive (+0.109 in unfixed Phase 3)
- Per Round 2 review SHOULD-DO + project memory `feedback_source_period_bias`

## P1a — USD-return mean-shifted placebo

- Delta applied: +0.000117/day (+0.0295/yr annualized)
- Raw HEFA mean: +0.000434/day; raw VTI mean: +0.000551/day
- Placebo Sharpe-diff: **+0.067** (CI [-0.028, +0.184])
- Placebo log-excess daily: +0.000004 (CI [-0.000072, +0.000080])
- **P1a verdict:** FAIL — finding was source-period bias

## P2 — Linear-scaling sweep

| Weight HEFA | SharpeDiff | 90% CI |
|---|---|---|
| 1% | +0.000 | [-0.002, +0.003] |
| 5% | +0.001 | [-0.010, +0.014] |
| 10% | +0.002 | [-0.020, +0.028] |
| 30% | +0.002 | [-0.067, +0.083] |
| 50% | -0.007 | [-0.127, +0.132] |

**P2 verdict:** PASS — monotonic in weight (structural rebalancing-premium support)

## P3 — Sub-period 3 only (2018-05 to 2026-04)

- Sub-period 3 SharpeDiff: **+0.113** (CI [-0.033, +0.258])
- **CAVEAT:** Sub-periods 1+2 not testable (HEFA inception 2014-01-31). Sub-period 3 is fully within USD-bull era; provides limited independent evidence.

## Combined verdict

- P1a: FAIL
- P2:  PASS
- P3:  positive (limited evidence)
