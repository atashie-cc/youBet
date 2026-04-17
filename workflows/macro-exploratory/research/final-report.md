# Macro-Exploratory Workflow — Final Report

**Status**: COMPLETE (20 experiments, 5 rounds, 5 Codex adversarial reviews, 8 bugs fixed)
**Period**: 2026-04-14 to 2026-04-17
**Track**: Exploratory (no Holm across experiments)

## Executive Summary

This workflow tested 20 experiments across 5 rounds to find CAGR- and Sharpe-maximizing macro strategies. One mechanism survived all tests: **E4's pooled independent factor-vs-cash timing** — 12 sleeves of {US, Dev ex-US, Europe, Japan} x {CMA, HML, RMW} timed by SMA100, equal-weighted with annual rebalance. Paper Sharpe 1.57, CAGR 3.7%, MaxDD -5.7%.

The mechanism is empirically validated on paper:
- Random null: p=0.007 (beats 299/300 block-randomized Markov signals)
- SMA window: 6/6 windows (50-200) produce positive Sharpe-diff
- Quasi-holdout: ExS +0.845 on post-2016 data (stronger than full sample)
- Prospective holdout: committed 2026-04-16 for future Ken French data

But investable implementation is severely constrained:
- Only 1 of 12 sleeves has a validated ETF bridge (hedged VLUE, net ExS +0.453)
- CMA (most robust factor) has no ETF vehicle
- RMW doesn't transmit through QUAL (ExS +0.042)
- 8 of 12 sleeves have no viable real-world instrument
- Leveraging the paper pool (6x) reaches only 12.3% CAGR vs 21.6% for 3x VTI SMA
- Breadth-gated VTI suffers -85% MaxDD through GFC (factor signal is orthogonal to market)

**Most practical finding**: VTI SMA100 drawdown overlay (from factor-timing Phase 11, not from this workflow) — no shorting, any account, any capital level. This single finding has the highest conviction-to-complexity ratio across the entire research program.

## Experiment Roster

### Round 1 — Initial exploratory sweep (2026-04-14)

| # | Exp | Focus | Result | Key finding |
|---|-----|-------|--------|-------------|
| 1 | E10 | GEM baseline | FAIL | Pipeline smoke test; GEM doesn't beat VTI |
| 2 | E2 | Leveraged factor SMA | FAIL | Leverage destroys Sharpe on individual factors |
| 3 | **E4** | **Pooled 12-sleeve** | **PASS** | **ExS +0.716, Sharpe 1.57 (paper)** |
| 4 | E9 | Daily BSC vol-managed | FAIL | Collapsed to capped leverage on low-vol factors |
| 5 | E6b | GEM + macro circuit breaker | FAIL | Overlay is cost without protection |
| 6 | E1m | Multi-signal gate | FAIL | Signals correlated, AND gate under-leverages |

### Round 2 — Codex-proposed extensions (2026-04-16)

| # | Exp | Focus | Result | Key finding |
|---|-----|-------|--------|-------------|
| 7 | E12 | Canonical BSC UMD | FAIL (0.54) | Near-miss; BSC mechanism real but moderate |
| 8 | **E13** | **E4 quasi-holdout** | **PASS** | **ExS +0.845 post-2016; mechanism robust** |
| 9 | E14 | Investable ETF bridge | FAIL (-0.32) | Costs destroy: net Sharpe -0.316 |
| 10 | E15 | Weekly cadence | FAIL (0.54) | 19.4% collapse, above 15% threshold |

### Round 3 — OOS and UMD (2026-04-16)

| # | Exp | Focus | Result | Key finding |
|---|-----|-------|--------|-------------|
| 11 | E16a | UMD-only pool | FAIL (-0.07) | Cross-region UMD correlation 0.47 kills diversification |
| 12 | E16b | Expanded 16-sleeve | FAIL (0.45) | Adding UMD dilutes E4 by 39% |
| 13 | E17 | Unhedged factor-signal ETF | FAIL (-0.31) | Reduces DD but misses upside; concordance 52% |
| 14 | E18a | Monthly OOS | FAIL (0.55) | Transfers but 24% degradation |
| 15 | E18b | Asia-Pac absorption | FAIL (0.50) | 26.7% degradation (above 20% threshold) |

### Round 4 — CAGR maximization (2026-04-16)

| # | Exp | Focus | Result | Key finding |
|---|-----|-------|--------|-------------|
| 16 | E21 | CAGR frontier | Diagnostic | No peak through 6x; Kelly negative |
| 17 | E19 | Leveraged pool 4-6x | FAIL (gate) | Timing preserves under leverage but CAGR base too small |
| 18 | E20 | Breadth-gated VTI | FAIL (-0.10) | -85% MaxDD (GFC); factor signal bad for equity timing |

### Round 5 — Critical assessment (2026-04-16/17)

| # | Exp | Focus | Result | Key finding |
|---|-----|-------|--------|-------------|
| 19 | E23 | SMA window sweep | **ROBUST 6/6** | Not overfit; smooth plateau, shorter slightly better |
| 20 | E22 | Random signal null | **p=0.007** | Timing edge real; beats 299/300 Markov nulls |
| 21 | E24 | Signal-factor permutation | p=0.096 | Borderline; ~34% factor-specific, ~66% diversification |
| 22 | E25 | Prospective commitment | Protocol | Frozen for future Ken French data evaluation |

## Mechanism Understanding

### What E4's edge is

E4 works because it pools 12 weakly-correlated binary timing signals (SMA100 on/off for each factor-region sleeve). The pool's exceptional Sharpe (1.57) comes from:

1. **~66% diversification of switching noise** (E24 permutation: random signal-factor pairings still produce mean Sharpe-diff +0.417). Having 12 independent binary switches that are each "right more often than wrong" and nearly uncorrelated creates a portfolio with very low residual variance.

2. **~34% factor-specific alpha** (E24 permutation: correct pairing adds +0.219 on top). Each factor's SMA does capture something real about that factor's dynamics — but this contribution is borderline significant (p=0.096).

3. **Cash optionality** (factor-timing Phase 13): the mechanism works by choosing in-market vs cash, NOT by rotating between risk assets. This is the strongest finding across the entire factor-timing + macro-exploratory research program.

### What E4's edge is NOT

- **Not a construction artifact**: random Markov signals with the same autocorrelation produce mean Sharpe-diff +0.313, far below E4's +0.635 (p=0.007)
- **Not SMA100 overfit**: all windows 50-200 produce positive results with a smooth plateau
- **Not cross-regional diversification alone**: E16's UMD-only pool (same regions, one factor) fails because UMD signals are correlated across regions (0.47)
- **Not investable through current ETF vehicles**: only 1/12 sleeves has a validated bridge

## Implementation Assessment

### Available vehicles

| Factor | Best US ETF | Best Intl ETF | Bridge validated? | Issue |
|--------|-----------|--------------|-------------------|-------|
| HML (value) | VLUE (0.15%, $11B) | EFV/IVLU (0.30%, $4-27B) | US: Yes | Only validated sleeve |
| RMW (quality) | QUAL (0.15%, $47B) | IQLT (0.30%, $13B) | US: Fails (ExS +0.04) | Factor loading too weak |
| CMA (investment) | **None** | **None** | N/A | **No pure CMA ETF exists** |

### Quantified outcome scenarios

| Scenario | Sleeves | Net ExSharpe | Excess CAGR | MaxDD | P(neg 5yr) |
|----------|---------|-------------|-------------|-------|------------|
| Paper E4 (upper bound) | 12 | +1.04 | ~6% | -9% | <0.1% |
| Hedged VLUE only (validated) | 1 | **+0.45** | ~3-4% | -6% | **16%** |
| Multi-sleeve estimate | 3-4 | ~0.35-0.55 | ~2-3% | -15% | ~16% |
| CMA vehicle emerges | 8-12 | ~0.55-0.80 | ~3-5% | -12% | ~7% |

### Cost reality

Per-sleeve annual costs for hedged factor ETF timing (Phase 10 base case):
- Borrow: 0.14% (VTI short, exposure-weighted)
- Switching: 0.26% (weekly, 4.5 switches/yr x 3 bps)
- Margin drag: 0.99% (Reg T 50% excess collateral at 5%)
- **Total: ~1.40%/yr per sleeve**
- Break-even: ~4%/yr — substantial headroom for the validated VLUE sleeve

### Academic context

| Paper | Method | Sharpe | E4 comparison |
|-------|--------|--------|---------------|
| Gupta & Kelly (2019) | Factor momentum, 65 factors | 0.84 | E4 higher due to geographic diversification |
| Haddad & Kozak (2020) | Factor timing, OOS | 0.44-0.71 | E4 consistent with upper range |
| Moskowitz et al (2012) | TSMOM, 58 futures | ~1.0 | E4 uses same mechanism (trend-following) |
| Neuhierl et al (2024) | 300+ factors, 39 signals | 1.3-1.8 | E4 in range; value/profitability strongest |

No published paper tests E4's specific construction (pooled independent factor-vs-cash timing across regions). The diversification of independent timing signals is a genuine gap in the literature.

## Practical Recommendations

### For most investors: VTI SMA100 drawdown overlay

The highest-conviction, lowest-complexity finding:
- **Hold VTI** when VTI price > 100-day SMA; **hold VGSH** when below
- Check weekly (Friday close)
- No shorting, no margin, works in any account, any capital level
- Phase 11: Sharpe 0.897 vs VTI B&H 0.808, dramatically reduced MaxDD

### For sophisticated investors ($200K+ in margin account): add hedged VLUE

- Long VLUE / short beta-weighted VTI (rolling 60-day beta)
- SMA100 on the hedged spread, weekly checking
- Net cost ~1.4%/yr, net ExSharpe ~+0.45
- 16% probability of negative excess return over 5 years
- Requires: margin account, short-selling capability, active monitoring

### What NOT to do

- Don't try to implement the full 12-sleeve pool (8 sleeves have no vehicle)
- Don't use factor breadth to time VTI leverage (-85% MaxDD through GFC)
- Don't expect paper Sharpe 1.57 in practice (realistic net Sharpe: 0.4-0.8)
- Don't confuse the paper mechanism (validated) with the implementation (constrained)

## Open Questions

1. **Prospective holdout** (E25): frozen construction committed 2026-04-16. Evaluate when Ken French publishes data through 2027-02+.
2. **CMA vehicle**: if Dimensional, Avantis, or another provider launches a pure investment-factor ETF, rerun the Phase 3 bridge test immediately.
3. **Factor-specificity**: E24's p=0.096 is borderline. A larger permutation test (1000+) with derangement-only permutations would sharpen this.
4. **Data-dependent null**: E22's Markov null is return-independent. A stronger null would test against other data-dependent timing rules.

## Codex Review Summary

5 adversarial review rounds across the workflow, 8 bugs found and fixed:
1. E14 cost double-counting (exposure-weighted × exposure)
2. E14 benchmark mislabeled (unhedged B&H called "net of costs")
3. E12 RF=0 pre-1963 (undercharged leveraged months)
4. E12 variance regime break (monthly vs daily estimator)
5. E19 leveraged B&H benchmark was actually unlevered
6. E20 missing GFC (VTI cache started 2011, not 2003)
7. E20 T-bill annualization (used annual rate as daily)
8. E22/E24 Sharpe-diff discrepancy (simplified pooling inflated from +0.635 to +0.783)
