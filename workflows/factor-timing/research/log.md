# Factor Timing Research Log

## Workflow Overview
Test whether timing rules (SMA, vol-targeting) applied to Ken French factor portfolios
can improve risk-adjusted returns over static factor exposure. 62.6 years of daily data
(1963-2026), 15,770 observations, 60 walk-forward folds.

Secondary objective: measure pre-publication vs post-publication Sharpe decay per factor.

## Status: COMPLETE (Phase A + Phase B). 3 Codex review rounds, 7 bugs fixed.

---

## Codex Review (Round 1) — 3 Bugs Fixed

1. **[BLOCKER] Fold-boundary overlap** — Adjacent folds shared a boundary date, causing
   duplicate dates in overall_returns. Fixed: use exclusive upper bound (`< test_end`).
2. **[WARNING] Estimand mismatch** — p-value tested `Sharpe(strat - bench)` but CI measured
   `Sharpe(strat) - Sharpe(bench)`. Fixed: use `excess_sharpe_lower/upper` (same estimand).
3. **[WARNING] Double RF subtraction** — `compute_metrics()` subtracted 4% RF from factor
   returns that are already excess returns. Fixed: default `rf_rate=0.0`.

Additional Codex concerns (not bugs, but noted for interpretation):
- Bootstrap at resolution floor (2000 replicates → min Holm p = 0.009). True p-values
  may be smaller but are unresolvable at this resolution.
- Missing robustness checks (different SMA windows, sub-period analysis, random-timing null).

---

## Phase 0: Power Analysis — PROCEED

**Analytical MDE at 80% power: +0.46 excess Sharpe** (passes kill gate of 0.50).

- SE(Sharpe) = 1/sqrt(62.6) = 0.126
- Holm-adjusted alpha = 0.05/18 = 0.00278
- Critical z = 2.773
- Critical excess Sharpe (stat only) = 0.351
- 60 walk-forward folds (3x more than ETF workflows)

---

## Phase 1: Factor Timing — 8/18 PASS (Corrected Results)

**First strategies to pass the strict gate in the entire youBet framework
(after 206+ failures across ETF, ETF-CAGR, and Commodity workflows).**

### Gate: ExSharpe > 0.20 AND Holm p < 0.05 AND CI lower > 0

All metrics below use consistent estimand: Sharpe(strategy - benchmark).

### Passing Strategies

| Strategy | ExSharpe | Holm p | 90% CI | B&H Sharpe | Timed Sharpe |
|---|---|---|---|---|---|
| CMA_sma_100 | +0.687 | 0.009 | [+0.460, +0.932] | 0.526 | 1.188 |
| SMB_sma_100 | +0.641 | 0.009 | [+0.417, +0.872] | 0.137 | 0.915 |
| UMD_vol_target | +0.615 | 0.009 | [+0.382, +0.859] | 0.546 | 0.896 |
| RMW_sma_100 | +0.582 | 0.009 | [+0.299, +0.862] | 0.520 | 1.104 |
| HML_sma_100 | +0.535 | 0.009 | [+0.308, +0.787] | 0.380 | 1.009 |
| SMB_sma_200 | +0.505 | 0.013 | [+0.280, +0.737] | 0.137 | 0.768 |
| CMA_sma_200 | +0.497 | 0.028 | [+0.246, +0.738] | 0.526 | 1.005 |
| HML_sma_200 | +0.452 | 0.013 | [+0.228, +0.696] | 0.380 | 0.977 |

### Failing Strategies

| Strategy | ExSharpe | Holm p | Reason |
|---|---|---|---|
| RMW_sma_200 | +0.377 | 0.190 | Fails p-value (borderline) |
| UMD_sma_100 | +0.096 | 1.000 | Small effect |
| UMD_sma_200 | +0.091 | 1.000 | Small effect |
| Mkt-RF_sma_100 | +0.072 | 1.000 | Small effect |
| Mkt-RF_sma_200 | +0.068 | 1.000 | Small effect |
| HML_vol_target | -0.245 | 1.000 | NEGATIVE |
| Mkt-RF_vol_target | -0.255 | 1.000 | NEGATIVE |
| CMA_vol_target | -0.330 | 1.000 | NEGATIVE |
| RMW_vol_target | -0.544 | 1.000 | NEGATIVE |
| SMB_vol_target | -0.604 | 1.000 | NEGATIVE |

### Key Findings

1. **SMA trend filters improve Sharpe on paper factor portfolios.** 8 of 12 SMA strategies
   pass the strict gate. After Codex RF fix, all benchmark Sharpes are POSITIVE (SMB: 0.137,
   HML: 0.380, RMW: 0.520, CMA: 0.526), so this is not merely "avoiding negative-Sharpe
   factors" — it is genuine timing improvement on top of positive-return factors.

2. **Vol-targeting DESTROYS value on all factors except UMD.** 5 of 6 vol-target strategies
   produce negative excess Sharpe. Vol-targeting increases exposure during high-vol periods
   (because vol_target / realized_vol < 1 only when vol > target). For most factors, high-vol
   episodes coincide with drawdowns, amplifying losses. Only UMD benefits because momentum
   crashes are pure-vol events where reduced exposure helps.

3. **SMA100 dominates SMA200 on every factor.** Consistent with ETF workflow finding.

4. **Mkt-RF does NOT benefit from timing** (ExSharpe +0.07, p=1.0). Consistent with the ETF
   finding that VTI/market is Sharpe-efficient. Note: Mkt-RF is market excess return, not
   strictly identical to VTI (which includes RF), but the conclusion is directionally the same.

5. **CMA benefits most from SMA timing** (+0.687 excess Sharpe) despite being the factor with
   the highest post-publication decay (118% in Phase 2). This suggests the timing value comes
   from avoiding the prolonged drawdown periods, not from the factor premium itself.

### PAPER PORTFOLIO CAVEAT

These are hypothetical long-short factor portfolios from CRSP. They cannot be directly
invested in. Implementation via factor ETFs would introduce:
- Tracking error (ETFs imperfectly replicate factors)
- Expense ratios (0.15-0.20% annually)
- Shorting costs (not reflected in paper portfolios)
- Capacity constraints (small-cap legs have limited capacity)
- Rebalancing frequency mismatch (academic = annual/monthly; ETFs = quarterly)

### Open Questions (from Codex review) — RESOLVED in Phase 1B
- **Random-timing null:** RESOLVED — see Phase 1B below.
- **Sub-period robustness:** RESOLVED — see Phase 1B below.
- **Bootstrap resolution:** Outstanding — would require memory optimization or chunked bootstrap
  to run 10K+ replicates on 15K-day series.

---

## Phase 1B: Robustness Checks — Random Null + Sub-Period

### Test 1: Random-Timing Null (Exposure-Matched)

**Question:** Does random timing with the SAME average exposure as SMA produce
similar excess Sharpe? 500 random simulations per factor.

| Factor | SMA Exp | SMA ExSh | Null Mean | Null Std | Null 95th | Rank | p-value |
|---|---|---|---|---|---|---|---|
| Mkt-RF | 64% | +0.072 | -0.103 | 0.119 | +0.094 | 38/500 | 0.078 |
| SMB | 49% | +0.641 | +0.261 | 0.111 | +0.426 | 0/500 | 0.002 |
| HML | 55% | +0.535 | +0.055 | 0.099 | +0.226 | 0/500 | 0.002 |
| RMW | 61% | +0.582 | +0.078 | 0.116 | +0.286 | 0/500 | 0.002 |
| CMA | 55% | +0.687 | +0.120 | 0.103 | +0.287 | 0/500 | 0.002 |
| UMD | 65% | +0.096 | -0.123 | 0.135 | +0.115 | 29/500 | 0.060 |

**Interpretation:**
- **SMB, HML, RMW, CMA: SMA timing is NOT explained by reduced exposure alone.**
  All 4 factors show SMA excess Sharpe far above the 95th percentile of the random-timing
  null (p < 0.002). The SMA signal contains genuine timing information.
- **Critical nuance for SMB:** Random 49% exposure produces +0.261 mean excess Sharpe
  (vs SMA +0.641). So ~40% of SMB's timing effect IS from reduced exposure to a weak
  factor, but the remaining ~60% is from timing WHERE to be exposed.
- **Mkt-RF and UMD:** Not significant against random null (p=0.08 and 0.06). Consistent
  with these being the two factors where SMA timing failed the formal gate.

### Test 2: Sub-Period Robustness (Pre-1992 vs Post-1992)

| Factor | Period | Years | B&H Sh | SMA Sh | ExSharpe | Consistent? |
|---|---|---|---|---|---|---|
| Mkt-RF | pre_pub | 28.9 | +0.306 | +1.131 | +0.501 | |
| Mkt-RF | post_pub | 33.7 | +0.511 | +0.637 | -0.166 | INCONSISTENT |
| SMB | pre_pub | 28.9 | +0.258 | +1.704 | +1.264 | |
| SMB | post_pub | 33.7 | +0.072 | +0.435 | +0.285 | CONSISTENT |
| HML | pre_pub | 28.9 | +0.871 | +2.047 | +1.289 | |
| HML | post_pub | 33.7 | +0.162 | +0.513 | +0.296 | CONSISTENT |
| RMW | pre_pub | 28.9 | +0.689 | +2.195 | +1.433 | |
| RMW | post_pub | 33.7 | +0.490 | +0.768 | +0.208 | CONSISTENT |
| CMA | pre_pub | 28.9 | +0.944 | +2.008 | +1.170 | |
| CMA | post_pub | 33.7 | +0.286 | +0.735 | +0.472 | CONSISTENT |
| UMD | pre_pub | 28.9 | +1.095 | +1.833 | +0.637 | |
| UMD | post_pub | 33.7 | +0.333 | +0.359 | -0.098 | INCONSISTENT |

**Interpretation:**
- **4 of 6 factors are CONSISTENT** (same-sign excess Sharpe in both periods): SMB, HML,
  RMW, CMA. These are exactly the 4 factors where SMA100 passes the formal gate.
- **Effects are much larger pre-publication** (excess Sharpe 1.0-1.4) than post-publication
  (0.2-0.5). This is consistent with factor publication decay (Phase 2 found 77% average
  decay). The timing effect persists post-publication but is substantially weaker.
- **Mkt-RF and UMD are INCONSISTENT** — positive pre-pub, negative post-pub. This confirms
  that SMA timing on these factors is unreliable, consistent with their failure in Phase 1.
- **Post-publication effects still positive** for SMB (+0.285), HML (+0.296), RMW (+0.208),
  CMA (+0.472). These are smaller but still meaningful — and this is the investable era.

---

## Phase 2: Publication-Lag Alpha Decay Measurement

### Factor-Specific Decay Rates

| Factor | Pub Date | Pre Sharpe | Post Sharpe | Decay | Statistically Significant? |
|---|---|---|---|---|---|
| SMB | 1992-06 | +0.350 | +0.068 | 81% | INCONCLUSIVE (CI spans zero) |
| HML | 1992-06 | +0.905 | +0.200 | 78% | YES (CI excludes zero) |
| RMW | 2013-09 | +0.563 | +0.364 | 35% | INCONCLUSIVE (only 12.5yr post) |
| CMA | 2013-09 | +0.705 | -0.128 | 118% | YES (CI excludes zero) |
| UMD | 1993-03 | +1.216 | +0.341 | 72% | YES (CI excludes zero) |

### Summary
- **Average decay: 77%** (worse than McLean-Pontiff's 58%)
- **Median decay: 78%**
- **Quality (RMW) has the least decay: 35%** — confirming its reputation as most robust
- **CMA has 118% decay** (post-publication Sharpe is NEGATIVE)

### Factor-Specific Haircuts

| Factor | Multiply published Sharpe by |
|---|---|
| RMW (quality) | 0.65 |
| UMD (momentum) | 0.28 |
| HML (value) | 0.22 |
| SMB (size) | 0.19 |
| CMA (investment) | 0.00 (dead) |

---

---

## Phase 1C: SMA Deep Dive — Window Sensitivity + Regime Decomposition

### Part 1: SMA Window Sensitivity (50-250 days)

All 8 SMA windows tested on each of the 4 passing factors produce **positive excess Sharpe**.
No window produces a negative result. The effect is not specific to SMA100.

| Factor | CV (robustness) | Best Window | Worst Window | All Positive? | Verdict |
|---|---|---|---|---|---|
| SMB | 18% | 50 (+0.798) | 250 (+0.444) | YES | ROBUST |
| HML | 18% | 50 (+0.672) | 175 (+0.372) | YES | ROBUST |
| RMW | 25% | 50 (+0.723) | 250 (+0.335) | YES | MODERATE |
| CMA | 22% | 50 (+0.953) | 250 (+0.489) | YES | MODERATE |

**Key finding:** Shorter windows (50-100) consistently outperform longer windows (200-250).
This is the opposite of overfitting — SMA100 is NOT the best window, so the Phase 1
result at 100 is conservative. The monotonic decline from 50→250 suggests genuine
trend-following alpha that responds faster with shorter lookbacks.

### Part 2: Drawdown Reduction (All 6 Factors)

SMA100 provides drawdown reduction on ALL 6 factors, including Mkt-RF and UMD where
Sharpe improvement failed the gate.

| Factor | B&H MaxDD | SMA MaxDD | DD Reduction | Sharpe Gate? |
|---|---|---|---|---|
| RMW | -41.8% | -13.3% | 68% | PASS |
| CMA | -28.5% | -11.1% | 61% | PASS |
| SMB | -62.8% | -26.6% | 58% | PASS |
| HML | -56.3% | -25.4% | 55% | PASS |
| Mkt-RF | -59.2% | -36.0% | 39% | fail |
| UMD | -63.3% | -44.5% | 30% | fail |

This mirrors the ETF workflow finding exactly: trend following provides genuine
drawdown reduction even when Sharpe improvement is inconclusive. Mkt-RF drawdown
reduction (39%) is comparable to the ETF workflow's VTI result (64% with SMA200).

### Part 3: Regime Decomposition

**ALL timing alpha is bear-driven.** SMA generates positive excess returns during
bear factor regimes (trailing 6-month return < 0) and slightly negative during bull
regimes. The mechanism is clear: SMA exits the factor during prolonged drawdowns
(earning RF instead) and re-enters during recoveries.

| Factor | Bear % of Time | Bear Excess | Bull Excess | Source |
|---|---|---|---|---|
| Mkt-RF | 32% | +17.0% ann | -6.6% ann | BEAR-DRIVEN |
| SMB | 50% | +9.9% ann | -1.4% ann | BEAR-DRIVEN |
| HML | 42% | +9.8% ann | -1.2% ann | BEAR-DRIVEN |
| RMW | 35% | +7.8% ann | -0.6% ann | BEAR-DRIVEN |
| CMA | 41% | +7.1% ann | -0.6% ann | BEAR-DRIVEN |
| UMD | 29% | +12.1% ann | -3.9% ann | BEAR-DRIVEN |

The bull-regime drag is small (-0.6% to -1.4% for passing factors), meaning SMA
doesn't significantly hurt during uptrends. The asymmetry explains why it works:
the benefit of avoiding crashes far exceeds the cost of late re-entry.

---

---

## Phase 3: Factor ETF Bridge Test — REVISED: Bridge Exists for Value (Hedged)

### Stage A: Factor Loading Test — ALL 3 ETFs QUALIFY (Corrected After Codex R2)

**Original (WRONG):** Raw return correlation said no ETF qualifies (corr -0.15 to 0.30).
**Corrected:** Multi-factor regression shows all 3 ETFs have highly significant factor loadings.

| ETF | Factor | b_mkt | b_factor | t_factor | R2 | Qualified |
|---|---|---|---|---|---|---|
| VLUE | HML | 0.973 | **+0.387** | 46.8 | 0.887 | YES |
| QUAL | RMW | 0.971 | **+0.168** | 21.9 | 0.960 | YES |
| SIZE | SMB | 0.886 | **+0.203** | 17.3 | 0.847 | YES |

All t-statistics > 17. The ETFs DO capture their target factors — the low raw
correlation was entirely from market beta domination (b_mkt ~0.89-0.97).

### Stage B: Unhedged Timing — ALL NEGATIVE (Expected)

SMA on raw ETF prices doesn't work because the market beta (~0.95) overwhelms
the factor signal. SMA on VLUE is essentially SMA on the market.

| ETF | ExSharpe | DD Reduction |
|---|---|---|
| VLUE | -0.089 | 45% |
| QUAL | -0.174 | 36% |
| SIZE | -0.203 | 31% |

### Stage C: Hedged Timing — VALUE BRIDGE WORKS (Holm p = 0.045)

Market-neutralized timing (SMA on ETF - beta*VTI spread) isolates the factor timing signal.

| ETF | Factor | Paper ExSh | Hedged ExSh | Holm p | DD Reduction |
|---|---|---|---|---|---|
| VLUE | HML | +0.535 | **+0.739** | **0.045** | **78%** |
| SIZE | SMB | +0.641 | +0.224 | 1.000 | 35% |
| QUAL | RMW | +0.582 | +0.042 | 1.000 | -2% |

**Hedged VLUE (value) passes the Holm-corrected gate.** The excess Sharpe (+0.739)
exceeds the paper result (+0.535). MaxDD reduction from -38.2% to -8.4% (78%).

**Interpretation:** The value factor timing signal transmits from paper to practice
when market beta is hedged out. This requires a long VLUE / short VTI position,
which is implementable with a margin account but not with simple ETF holdings.

SIZE shows positive direction (+0.224) but fails the gate. QUAL shows no effect.

### Implication for Option 3

The multi-strategy portfolio concept is partially rehabilitated: a market-hedged
value tilt (long VLUE / short VTI) with SMA100 timing is implementable and
passes the strict gate. However, it requires short selling (or equivalent),
which limits the practical audience.

---

## Phase 4: Calendar Effects — 0/4 PASS (Sharpe Gate) but Significant CAPM Alpha

### Corrected After Codex R2: Off-by-one calendar bug fixed, beta-adjusted alpha added.

### Primary Test: Sharpe-of-Excess Gate (0/4 PASS)

| Strategy | Sharpe | CAGR | MaxDD | ExSharpe | Holm p | Gate |
|---|---|---|---|---|---|---|
| sell_in_may | 0.899 | 10.1% | -35.1% | -0.113 | 1.000 | FAIL |
| turn_of_month | 1.188 | 10.2% | -20.0% | -0.106 | 1.000 | FAIL |
| year_end_rally | 1.077 | 6.6% | -17.3% | -0.332 | 1.000 | FAIL |
| january_effect | 1.232 | 5.5% | -18.8% | -0.393 | 1.000 | FAIL |
| buy_and_hold | 0.712 | 10.8% | -54.6% | — | — | — |

### Beta-Adjusted Alpha: ALL SIGNIFICANT

Codex correctly identified that Sharpe-of-excess penalizes low-exposure strategies
when the market risk premium is positive. CAPM regression reveals genuine alpha:

| Strategy | CAPM Alpha (ann) | Beta | t(alpha) |
|---|---|---|---|
| turn_of_month | **+6.9%** | 0.274 | **+7.56** |
| january_effect | +4.6% | 0.073 | +8.54 |
| year_end_rally | +4.9% | 0.141 | +6.90 |
| sell_in_may | +4.5% | 0.494 | +4.43 |

All 4 have annualized alpha of +4.5% to +6.9% with t-statistics 4.4 to 8.5.
Turn-of-month has the largest alpha at +6.9% with the highest Sharpe ratio (1.188).

**Interpretation:** Calendar strategies DO select favorable return periods (genuine alpha
vs CAPM). But because they're in cash 50-90% of the time, the excess-return series
(strategy minus buy-and-hold) has high variance, making Sharpe-of-excess negative.
The strategies sacrifice total return for dramatically better risk-adjusted return.

### Per-Decade Stability: Calendar Alpha is Declining

| Strategy | 1963-72 | 1973-82 | 1983-92 | 1993-02 | 2003-12 | 2013-26 |
|---|---|---|---|---|---|---|
| sell_in_may | +0.11 | +0.10 | +0.07 | -0.01 | -0.11 | -0.64 |
| turn_of_month | +0.36 | +0.05 | +0.10 | -0.05 | -0.04 | -0.62 |

Even corrected, the per-decade excess Sharpe shows monotonic decline into 2013-2026.
The alpha may be genuine historically but is deteriorating, consistent with arbitrage.

### SMA100 vs Calendar Correlation: Near Zero

Calendar signals have zero correlation with SMA100 (0.00 to 0.08). SMA timing alpha
is NOT a calendar effect — it is a genuine trend-following signal.

---

---

## Phase 6: Rebalancing Frequency Sensitivity

### Bug Fix: Hedged Return PIT Violation (Pre-Phase 6)

Codex R3 identified that `compute_hedged_returns()` used same-day rolling beta to
hedge same-day return. Fixed: `rolling_beta.shift(1)`. Rerun Phase 3 Stage C:
hedged VLUE ExSharpe IMPROVED from +0.739 to **+0.795** (Holm p 0.045 → **0.030**).
The PIT fix strengthened the result.

### Test 1: Paper Factor Signal-Frequency Sensitivity

**Pre-committed hypothesis REJECTED:** monthly-checked ExSharpe is NOT within +/-0.10
of daily-checked. The degradation is 0.16-0.46.

| Frequency | Avg ExSharpe (SMA100, 4 factors) | Delta vs Daily |
|---|---|---|
| Daily | +0.611 | baseline |
| Weekly | +0.541 | -0.070 |
| Monthly | +0.404 | **-0.207** |

Key patterns:
- Degradation is monotonic: daily > weekly > monthly
- Shorter SMA windows degrade more (SMA50: -0.26 avg, SMA100: -0.19 avg)
- **All 24 variants remain positive** — monthly still produces meaningful ExSharpe (+0.37 to +0.57)
- Signal concordance: weekly agrees with daily 93-95%, monthly 79-87%

### Test 2: Hedged VLUE Spread Frequency Sensitivity

| SMA | Freq | ExSharpe | MaxDD | DD Reduction | Sw/Yr |
|---|---|---|---|---|---|
| 100 | daily | **+0.795** | -6.3% | 84% | 10.5 |
| 100 | weekly | +0.690 | -13.0% | 66% | 4.3 |
| 100 | monthly | +0.531 | -12.5% | 67% | 3.3 |
| 50 | daily | +0.609 | -10.5% | 73% | 17.5 |
| 50 | weekly | +0.632 | -10.6% | 72% | 8.5 |
| 50 | monthly | +0.383 | -20.7% | 46% | 4.5 |

All 6 hedged VLUE variants are positive. Weekly SMA50 (+0.632) beats daily SMA50 (+0.609),
suggesting shorter windows benefit from whipsaw reduction at weekly frequency.

### Test 3: Cost Impact

Transaction costs (3 bps one-way) are negligible relative to timing alpha:
- Daily SMA100: ~9 switches/yr = 0.54% annual cost = ~0.07 Sharpe drag
- Weekly SMA100: ~4.5 switches/yr = 0.27% annual cost
- Monthly SMA100: ~2.4 switches/yr = 0.14% annual cost

**Costs are NOT the binding constraint — signal freshness is.** The net-of-cost
ranking is the same as gross: daily > weekly > monthly.

### Key Takeaway

**Weekly is the implementation sweet spot.** It captures 85-90% of the daily signal
quality with half the turnover and slightly better whipsaw management for shorter SMAs.
Monthly checking loses 20-40% of the timing alpha — more than the literature predicted.

---

---

## Phase 7: International Out-of-Sample Replication

### The Critical Test

If SMA value timing only works in the US, it's likely a data-mining artifact.
If it replicates internationally, the mechanism (crash avoidance on factor drawdowns)
is structural and market-independent.

### HML (Value) Timing Across 5 Regions

| Region | Folds | Years | B&H Sharpe | SMA Sharpe | ExSharpe | Raw p | DD Reduction |
|---|---|---|---|---|---|---|---|
| US | 60 | 59.6 | 0.380 | 1.009 | **+0.535** | 0.0005 | 55% |
| Developed ex-US | 33 | 33.5 | 0.859 | 1.582 | **+0.600** | 0.0095 | 72% |
| Europe | 33 | 33.5 | 0.528 | 1.251 | **+0.567** | 0.0060 | 79% |
| Japan | 33 | 33.5 | 0.528 | 0.916 | +0.241 | 0.0915 | 60% |
| Asia-Pacific ex-Japan | 33 | 33.5 | 0.746 | 0.766 | -0.212 | 0.8931 | 31% |

**4 of 5 regions show positive excess Sharpe.** 3 of 4 international regions positive.
Developed ex-US and Europe both statistically significant (p < 0.01).
Japan positive but borderline (p = 0.09). Asia-Pacific is the sole failure.

### All 4 Factors Across Regions

| Factor | US | Dev ex-US | Europe | Japan | Asia-Pac | Positive |
|---|---|---|---|---|---|---|
| HML | +0.535 | +0.600 | +0.567 | +0.241 | -0.212 | 4/5 |
| SMB | +0.641 | +0.084 | +0.055 | +0.317 | +0.689 | **5/5** |
| RMW | +0.582 | +0.433 | +0.304 | +0.512 | -0.167 | 4/5 |
| CMA | +0.687 | +0.841 | +0.801 | +0.482 | -0.049 | 4/5 |

**SMB timing is positive in ALL 5 regions** — the most internationally robust finding.
HML, RMW, and CMA are positive in 4/5 (Asia-Pacific is the consistent exception).

### Drawdown Reduction: Universal

SMA100 reduces MaxDD in ALL 5 regions, ALL 4 factors, regardless of timing alpha sign:

| Region | HML DD Red | SMB DD Red | RMW DD Red | CMA DD Red |
|---|---|---|---|---|
| US | 55% | 58% | 68% | 61% |
| Dev ex-US | 72% | 33% | 52% | 56% |
| Europe | 79% | 31% | 47% | 61% |
| Japan | 60% | 48% | 38% | 56% |
| Asia-Pac | 31% | 34% | 4% | 2% |

### Interpretation

The finding replicates. SMA factor timing is NOT a US-specific artifact. The mechanism
(avoiding prolonged factor drawdowns) operates in developed markets globally. The
exception — Asia-Pacific ex-Japan — is the smallest and most heterogeneous region,
covering Australia, Hong Kong, Singapore, New Zealand, and others with diverse market
microstructures.

Key nuances:
- Europe shows the strongest replication: ExSharpe +0.567 with 79% drawdown reduction on HML
- Developed ex-US is the broadest test and shows ExSharpe +0.600 (EXCEEDS the US result)
- Japan shows positive direction but weaker magnitude and borderline significance
- CMA timing is surprisingly strong internationally (+0.801 Europe, +0.841 Dev ex-US)

---

---

## Phase 8: Regime-Conditional International Analysis

### VIX Regime Decomposition (HML, all regions)

| Region | ExSharpe | Low VIX | Normal VIX | High VIX | Crisis VIX |
|---|---|---|---|---|---|
| US | +0.535 | +0.7% | **+2.1%** | -0.8% | +0.9% |
| Dev ex-US | +0.600 | +0.5% | +1.1% | +0.2% | +0.5% |
| Europe | +0.567 | +0.4% | **+1.5%** | +0.2% | +0.8% |
| Japan | +0.241 | -0.0% | +1.2% | +0.2% | +0.2% |
| Asia-Pac | -0.212 | +0.2% | -1.6% | +0.3% | -0.0% |

**Timing alpha is NOT crisis-concentrated.** Normal VIX (15-25) contributes the most
for US and Europe. The effect is distributed across market regimes.

### Time Period Decomposition (HML)

| Region | Pre-GFC | GFC | Post-GFC | COVID+ |
|---|---|---|---|---|
| US | **+2.6%** | +0.2% | +0.5% | +0.1% |
| Dev ex-US | +1.2% | +0.1% | +0.7% | +0.3% |
| Europe | +1.1% | +0.0% | **+1.5%** | +0.4% |

**Not era-dependent.** Positive contributions across all periods for US, Dev ex-US, Europe.
Europe's strongest era is post-GFC, not pre-GFC — the effect is not just historical.

### Bear/Bull Regime (All factors, all regions)

**Bear-driven mechanism confirmed internationally.** 16 of 20 factor×region combinations
are bear-driven. The 4 exceptions are all Asia-Pacific (HML, RMW, CMA) where bull-regime
drag (-2.1 to -2.6%) exceeds bear-regime benefit (+1.0 to +2.0%).

---

## Phase 9: Asia-Pacific Exception + Multi-Region Diversification

### Part A: Why Asia-Pacific Fails

| Metric | US | Dev ex-US | Europe | Japan | Asia-Pac |
|---|---|---|---|---|---|
| HML avg DD duration | 76d | 53d | 90d | 56d | **31d** |
| HML signal switches/yr | 8.3 | 7.4 | 6.6 | 10.4 | **12.8** |
| HML mean-reversion corr | +0.075 | +0.203 | +0.251 | +0.104 | +0.069 |

Asia-Pacific HML drawdowns are **shortest** (31 days vs 76-90 for US/Europe), causing
SMA100 to whipsaw (12.8 switches/yr vs 6.6-8.3 elsewhere). The factor recovers before
SMA can profitably exit and re-enter. This explains the high bull-regime drag.

### Part B: Multi-Region Diversification

Equal-weight SMA100 timing across US + Europe + Japan:

| Factor | US ExSh | Europe ExSh | Japan ExSh | **Combined ExSh** | Cross-Region Corr |
|---|---|---|---|---|---|
| CMA | +0.448 | +0.765 | +0.452 | **+0.847** | 0.099 |
| RMW | +0.197 | +0.299 | +0.549 | **+0.569** | 0.055 |
| HML | +0.310 | +0.532 | +0.200 | **+0.493** | 0.181 |
| SMB | +0.312 | +0.035 | +0.365 | **+0.405** | 0.044 |

Cross-region correlations are very low (0.04-0.18), meaning regional factor timing
signals are genuinely independent. CMA combined ExSharpe (+0.847) exceeds any
individual region. Diversification is real and substantial.

---

## Phase 10: Implementation Cost Analysis

### Cost Breakdown (Base Case: 35 bps borrow, 3 bps trading, 5% margin)

Corrected after Codex R5: switch cost fixed (2x not 4x one-way), borrow/hedge
costs now exposure-weighted (only charged when position is active).

| Cost Component | Daily | Weekly |
|---|---|---|
| VTI borrow (exposure-weighted) | 0.14% | 0.14% |
| Signal switching (2 one-way/switch) | 0.63% | 0.26% |
| Hedge rebalancing (exposure-weighted) | 0.01% | 0.01% |
| Margin drag (exposure-weighted) | **0.99%** | **0.99%** |
| **Total (ex-tax)** | **1.77%** | **1.40%** |

Dominant cost: margin drag (0.99%). Switching is secondary (0.26-0.63%).

### After-Cost Performance

| Frequency | Gross ExSharpe | Annual Cost | Net ExSharpe | Break-Even Cost |
|---|---|---|---|---|
| Daily | +0.795 | 1.77% | **+0.498** | 4.74% |
| Weekly | +0.690 | 1.40% | **+0.453** | 4.07% |

Under pessimistic assumptions (50 bps borrow, 5 bps trading, 6% margin):
- Daily: net ExSharpe = **+0.189** (positive)
- Weekly: net ExSharpe = **+0.178** (positive)

**Both frequencies survive even pessimistic cost assumptions.** The strategy has
~4% annual cost headroom before breaking even.

NOTE: These costs use a flat daily drag approximation. Real costs are lumpy
(concentrated on switch/rebalance days) and path-dependent (margin requirements
increase during drawdowns). The flat approximation may understate peak drag.

---

---

## Phase 12: Composite Defensive Signal (Exploratory)

Tested combining 4 signals (SMA100, vol spike, drawdown threshold, momentum) via
vote system. VOTE-3of4 matches SMA ExSharpe while cutting switches 20-30%.
SMB VOTE-3 shows +0.025 ExSharpe improvement — but per Codex R6, this is ~0.2 SE
and does not survive the 22-strategy search space. The composite's value is whipsaw
reduction (~5-8 bps/yr savings), not alpha improvement. Factor-tailored 2-signal
subsets underperform generic 4-signal VOTE-3.

Individual signal performance (ExSharpe on US factors):
- SMA100: strongest on all 4 factors (+0.535 to +0.687)
- Momentum: second (+0.403 to +0.484)
- VolSpike: modest (+0.163 to +0.248, negative on RMW)
- Drawdown: factor-dependent (+0.124 to +0.391)

Signal correlations are low (SMA-Vol: 0.05, SMA-DD: 0.25) except SMA-Momentum (0.55).

**Verdict:** Useful mechanism study. VOTE-3 is a defensible production rule for
reducing whipsaw, but does not produce statistically significant improvement over
SMA100 alone.

---

## Phase 13: Cross-Factor Rotation

### The Central Question
Does rotating among trending factors beat independent timing to cash?
Codex R6 proposed this as the highest-priority research direction.

### US Results (62 years)

| Strategy | Sharpe | CAGR | MaxDD | ExSharpe vs B&H |
|---|---|---|---|---|
| EW Buy-and-Hold | 0.649 | 2.8% | -25.6% | baseline |
| **Independent SMA Timing** | **1.709** | 6.1% | **-9.1%** | **+1.039** |
| SMA Rotation | 1.135 | 6.7% | -30.1% | +0.837 |
| Trend+Momentum (top-2) | 1.111 | 7.0% | -34.0% | +0.838 |
| InvVol Rotation | 1.181 | 6.7% | -29.0% | +0.826 |

**Independent timing has the highest Sharpe (1.709) and lowest MaxDD (-9.1%).**
Rotation produces higher CAGR (6.7-7.0% vs 6.1%) but at dramatically worse drawdowns
(-29 to -34% vs -9.1%). The Sharpe ratio correctly penalizes this.

### Timing vs Selection Decomposition

| Component | ExSharpe | Interpretation |
|---|---|---|
| Timing (indep vs B&H) | **+1.039** | Cash optionality — the dominant value source |
| Selection (rotation vs indep) | +0.212 | Concentrating in winners — positive but small |
| Total (rotation vs B&H) | +0.837 | Less than timing alone due to added drawdown |

**The cash option IS the mechanism.** Factors are only all trending 7.3% of the time.
Going to cash when your factor is below trend is more valuable than rotating to another
factor, because factor drawdowns are often correlated.

### International Confirmation

| Region | Indep SMA Sharpe | Rotation Sharpe | Indep Better? |
|---|---|---|---|
| US | **1.709** | 1.135 | YES |
| Developed ex-US | **1.729** | 1.335 | YES |
| Europe | **1.495** | 1.106 | YES |
| Japan | **1.108** | 0.675 | YES |

Universally confirmed: independent timing beats rotation on risk-adjusted basis.

### Practical Implication
Don't try to be clever about where to reallocate during factor weakness. Just go to
cash. The simplest approach (SMA100 independently on each factor, exit to RF) produces
the best risk-adjusted result across all regions tested.

---

## Cross-Workflow Implications (Final, After Phase 13)

1. **Paper-factor SMA timing is well-established** (US gate-passing, Holm-corrected).
   8/18 pass on US data with 60 walk-forward folds. Mechanism: bear-driven crash avoidance.
   Parameter-robust (all SMA 50-250 positive). Random null rejected (p<0.002).

2. **Partial international transportability.** HML positive in 4/5 regions, SMB 5/5.
   Descriptive transport test, not cross-region Holm-corrected.

3. **Cash is the mechanism, not factor selection.** Phase 13 definitively showed that
   independent factor-vs-cash timing (Sharpe 1.709) beats every rotation variant tested
   (Sharpe 1.10-1.18). The decomposition: timing alpha ExSharpe +1.039 vs selection
   alpha +0.212. Confirmed in all 3 international regions. Rotation produces higher
   CAGR but dramatically worse MaxDD (-30% vs -9%).

4. **Composite signals reduce whipsaw but don't improve alpha.** Phase 12 VOTE-3of4
   cuts switches 20-30% with comparable ExSharpe. SMB improvement (+0.025) is noise
   per Codex R6. The composite's value is operational (fewer trades) not statistical.

5. **Implementation costs (corrected R5):** Weekly hedged VLUE: 1.40%/yr base case,
   net ExSharpe +0.453 (base) / +0.178 (pessimistic). Break-even ~4%.

6. **Timing alpha oscillates, not declining monotonically.** Rolling 10yr ExSharpe on
   HML: 1.7 (1980) → 0.1 (2010) → 0.5 (2020) → 0.2 (2025). Regime-dependent, not
   being arbitraged away like calendar effects.

7. **Alpha source varies by factor.** SMB: 81% from deep drawdowns (fragile). CMA: 87%
   from normal periods (robust). This should inform factor weighting decisions.

8. **Market SMA and factor SMA signals are independent** (correlation -0.068). The VTI
   SMA overlay and hedged factor timing are genuinely diversifying — the blend has
   real portfolio construction value.

9. **The honest bottom line (calibrated per Codex R5+R6 + Phase 13):**
   - Paper-factor independent timing: STRONG (Sharpe 1.709, beats all alternatives)
   - The mechanism is cash optionality during factor drawdowns
   - Cross-factor rotation is dominated — don't rotate, just exit to cash
   - Composite signals are operationally useful but don't improve alpha
   - Implementation via hedged VLUE: positive net-of-cost but uses approximations
   - VTI SMA drawdown overlay: most broadly applicable practical finding
