# Phase 0 � Single-ETF Efficiency Test Results

- Run date: 2026-05-01
- Gate: ExSharpe > 0.4, Holm p < 0.05, CI_lower > 0.0, log_excess > 0.0
- Holm denominator (workflow-wide): 19
- Within-phase Holm: 4 candidates per cost level, applied separately

## Test windows (H1 fix per Round 2 review)

These are the EFFECTIVE test windows after the 36-month walk-forward train consumes the leading data:

| Ticker | Inception | Effective test_start | n_test_days |
|---|---|---|---|
| VXUS | 2011-01-26 | 2014-01-29 | ~3,098 |
| VEA  | 2007-07-20 | 2010-07-27 | ~3,961 |
| VWO  | 2005-03-04 | 2008-03-11 | ~4,565 |
| EFA  | 2001-08-14 | 2004-08-30 | ~5,448 |

Earlier "VXUS 2011+" / "VEA 2007+" framing in `log.md` referred to inception, not test window.

## Cost-sensitivity caveat (H2 fix per Round 2 review)

For a buy-and-hold strategy with no rebalancing turnover, the cost level is structurally a no-op: `total_turnover = 0` and `total_cost_drag = 0` at every cost level (3, 5, 10 bps). The "PROCEED requires PASS at 10 bps" framing applies meaningfully only to strategies that toggle weights (Phase 2 regime gates, Phase 3 C3 dynamic-hedge). For the Phase 0 single-ETF tests, all 3 cost-level results are bit-identical.

## Verdicts at strict 10 bps cost gate

| Ticker | Verdict | ExSharpe | Holm p | CI 90% | Log-excess CI |
|---|---|---|---|---|---|
| VXUS | FAIL | -0.305 | 1.0000 | [-0.566, -0.064] | see JSON |
| VEA | FAIL | -0.333 | 1.0000 | [-0.548, -0.129] | see JSON |
| VWO | FAIL | -0.376 | 1.0000 | [-0.611, -0.152] | see JSON |
| EFA | FAIL | -0.244 | 1.0000 | [-0.417, -0.085] | see JSON |

See `artifacts/phase0_results.json` for full per-cost-level breakdown.