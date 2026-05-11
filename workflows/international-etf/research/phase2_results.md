# Phase 2 — Regime-Conditional Gates Results

- Run date: 2026-05-02
- Gate: ExSharpe > 0.4, Holm p < 0.05, CI_lower > 0, log-excess CI_lower > 0
- Workflow tier: **descriptive_exploratory** (power 0.3 at gate)
- Holm denominator (workflow-wide): 18
- M5 dropped (no intl CAPE data); within-Phase-2 Holm: 7 strategies per cost level

## Verdicts at strict 10 bps cost

| Strategy | Verdict | SharpeDiff | Holm p | 90% CI | sig_on frac | Log-excess daily |
|---|---|---|---|---|---|---|
| M1a_dxy_VEA20 | FAIL | -0.027 | 1.0000 | [-0.054, -0.001] | 0.42 | -0.000020 |
| M1b_dxy_VEA40 | FAIL | -0.058 | 1.0000 | [-0.111, -0.006] | 0.42 | -0.000040 |
| M1c_dxy_VEA60 | FAIL | -0.091 | 1.0000 | [-0.172, -0.015] | 0.42 | -0.000061 |
| M3_bund_VEA40 | FAIL | -0.054 | 1.0000 | [-0.111, -0.000] | 0.38 | -0.000043 |
| M4a_dbc_VWO20 | FAIL | -0.054 | 1.0000 | [-0.092, -0.015] | 0.49 | -0.000039 |
| M4b_dbc_VWO40 | FAIL | -0.112 | 1.0000 | [-0.187, -0.036] | 0.49 | -0.000079 |
| M6_rev_VEA40 | FAIL | -0.115 | 1.0000 | [-0.203, -0.029] | 1.00 | -0.000087 |

See `artifacts/phase2_results.json` for full per-cost breakdown.