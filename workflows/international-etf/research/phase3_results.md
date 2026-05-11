# Phase 3 � Currency Hedging Exploratory

- Run date: 2026-05-04 (v2 after Round 2 review H3 fix: shared 3-ticker date index across all 3 strategies)
- **Status: EXPLORATORY ONLY — NOT IN HOLM DENOMINATOR**
- Raw sample window: 2014-01-31 to 2026-04-30 (12.2 years)
- **EFFECTIVE test window after 36-mo train: 2017-02-14 to 2026-04-29 (~9.2 years)** (H1 fix)
- Workflow tier: **descriptive_exploratory**
- Gate (informational only): ExSharpe > 0.4, Holm p < 0.05, CI_lower > 0
- Cost-sensitivity caveat (H2 fix): C1a and C1b are buy-and-hold with `total_turnover = 0`; cost level is structurally a no-op for them. Only C3 (turnover 2.4) is meaningfully cost-sensitive.

## Phase 4 robustness on C1b — REJECTED by P1a placebo (per Round 2 review SHOULD-DO)

The +0.109 CI upper bound on C1b prompted Round 2's recommended robustness checks. Run via `experiments/phase4_robustness_c1b.py` (results in `phase4_c1b_robustness.md`):

- **P1a USD-return mean-shifted placebo: FAIL.** HEFA's mean was -2.95%/yr below VTI in this 9.2-yr sample. After mean-equalization, placebo Sharpe-diff = +0.067 with CI [-0.028, +0.184]; placebo log-excess CI lower = -0.000072 (excludes positive). Conclusion: the +0.109 CI upper bound was source-period bias from HEFA's underperformance being almost-but-not-quite compensated by structural diversification — there is NO structural rebalancing-premium component beyond the mean-shift artifact.
- **P2 linear-scaling sweep:** sweep at HEFA in {1, 5, 10, 30, 50}% of the portfolio shows essentially flat Sharpe-diff (+0.000 to +0.002 from 1% to 30%; -0.007 at 50%). No peak-then-collapse but also no positive scaling — confirms zero structural alpha.
- **P3 sub-period 3 only:** SharpeDiff +0.113, CI [-0.033, +0.258] — more positive than full sample but still spans zero, fully within USD-bull era; sub-periods 1+2 untestable (HEFA inception 2014).

**Combined verdict: C1b is REJECTED.** The headline "+0.109 CI upper bound" is a sample-period artifact and does not reflect structural ex-US diversification value.

## Headline comparison @ strict 10 bps cost

| Strategy | HEFA friction | SharpeDiff | 90% CI | Vol | Turnover |
|---|---|---|---|---|---|
| C1a_VEA40 | +0 bps | -0.048 | [-0.158, +0.069] | 0.177 | 0.000 |
| C1b_HEFA40 | +0 bps | -0.002 | [-0.095, +0.109] | 0.171 | 0.000 |
| C3_dynhedge40 | +0 bps | -0.046 | [-0.146, +0.064] | 0.173 | 2.400 |
| C1a_VEA40 | +15 bps | -0.048 | [-0.158, +0.069] | 0.177 | 0.000 |
| C1b_HEFA40 | +15 bps | -0.005 | [-0.099, +0.105] | 0.171 | 0.000 |
| C3_dynhedge40 | +15 bps | -0.049 | [-0.149, +0.061] | 0.173 | 2.400 |

## Hedged-vs-unhedged delta (informational)

| Friction | C1b HEFA - C1a VEA | C3 dyn - C1a VEA |
|---|---|---|
| +0 bps | +0.047 | +0.002 |
| +15 bps | +0.043 | -0.000 |

See `artifacts/phase3_results.json` for full per-cost breakdown.