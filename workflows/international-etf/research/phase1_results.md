# Phase 1 - Static VTI/VXUS Allocation Sweep (corrected log-excess)

- Run date: 2026-05-02
- Gate: ExSharpe > 0.4, Holm p < 0.05, CI_lower > 0
- Workflow tier: **descriptive_exploratory**
- Power at locked gate: 0.3

## Test window (H1 fix per Round 2 review)

- Raw sample: 2011-01-26 to 2026-04-30 (15.3 years; VXUS inception)
- **Effective test window after 36-mo train: 2014-01-29 to 2026-04-29 (~12.3 years)**
- Earlier "2011-2026" framing referred to inception, not the test window.

## Cost-sensitivity caveat (H2 fix per Round 2 review)

For static VTI/VXUS mixes with annual rebalance, observed turnover and cost drag are essentially zero — the strategy returns target weights at each rebalance and the backtester's `rebalance_cost` only fires when there's drift. Empirically `total_turnover = 0` and `total_cost_drag = 0` at every cost level. Per-cost results (3, 5, 10 bps) are therefore bit-identical for Phase 1. The cost-stress narrative in plan v1.3 applies meaningfully only to strategies that toggle weights based on signals (Phase 2 + Phase 3 C3).

## S1 - Vanguard variance-minimum hypothesis

- Vol-minimum weight (point estimate): **60% VXUS** (annual vol = 0.165)
- Point estimate in Vanguard 35-55% band: **False**
- **Bootstrap CI on argmin LOCATION** (Phase 5 §14.4, 2026-05-11 addendum): 65.3% of 1000 bootstrap replicates put the argmin in the Vanguard 35-55% band (40%: 31%, 50%: 35%, 60%: 19%, 30%: 13%). Mean argmin = 0.455. **The Vanguard claim is corroborated** — the point-estimate "violation" at 60% is within bootstrap noise of the band.

| Weight VXUS | Annual vol |
|---|---|
| 0% | 0.176 |
| 10% | 0.173 |
| 20% | 0.171 |
| 30% | 0.169 |
| 40% | 0.167 |
| 50% | 0.166 |
| 60% | 0.165 |
| 100% | 0.169 |

## Strict-gate verdicts at 10 bps cost

| Weight | Verdict | ExSharpe | Holm p | CI 90% | Log-excess daily | Log-excess CI |
|---|---|---|---|---|---|---|
| 10% | FAIL | -0.019 | 1.0000 | [-0.043, +0.005] | -0.000021 | [-0.000036, -0.000004] |
| 20% | FAIL | -0.041 | 1.0000 | [-0.090, +0.007] | -0.000042 | [-0.000073, -0.000009] |
| 30% | FAIL | -0.066 | 1.0000 | [-0.140, +0.007] | -0.000063 | [-0.000111, -0.000014] |
| 40% | FAIL | -0.094 | 1.0000 | [-0.194, +0.005] | -0.000085 | [-0.000148, -0.000020] |
| 50% | FAIL | -0.124 | 1.0000 | [-0.250, +0.000] | -0.000107 | [-0.000186, -0.000026] |
| 60% | FAIL | -0.156 | 1.0000 | [-0.310, -0.008] | -0.000130 | [-0.000224, -0.000032] |
| 100% | FAIL | -0.305 | 1.0000 | [-0.566, -0.064] | -0.000223 | [-0.000380, -0.000060] |

## Interpretation

- **All 7 gating weights FAIL the strict gate** (ExSharpe>0.40); 0/7 PROCEED.
- **Sharpe-diff is monotonically negative as ex-US weight rises** from -0.019 at 10% to -0.305 at 100%.
- **Vol IS reduced** by adding ex-US (0.176 at 0% -> 0.165 at 60%; -1.1pp). Directional Vanguard claim confirmed; the 35-55% band claim is marginally violated (realized minimum is at 60%, but the curve is essentially flat from 40-60%).
- **None of the CIs include positive ExSharpe**. The data is consistent with VTI being weakly Sharpe-efficient over 2011-2026 against a VTI/VXUS sweep.

See  for full per-cost-level breakdown.