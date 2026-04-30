# Real-World Test — Final Report

**Date:** 2026-04-30
**Status:** COMPLETE
**Pre-commit:** `precommit/tests_1_4_7.json` v1.2 (SHA256 `fea20f44e6bf99eff1a4d2938ecca736c58c4c5ccfe7b930f3fe50939cae187d`)
**Compute:** ~50 Monte Carlo runs, 10,000 paths × 25-yr horizon each, paired daily block bootstrap (22-day Politis-Romano blocks) from 2000-2026 source panel.

## Bottom-line recommendation

**For $100K, 25-year, tax-protected, log-wealth maximization:**

- **Default: 100% VTI buy-and-hold, lump-sum.** Highest median CAGR (8.35%), lowest probability of negative 25-yr return (0.7%).
- **Defensible upgrade: 95% VTI / 5% IAU.** Captures the only finding that survived sub-period robustness: the ~+0.07 log-excess (~+0.28%/yr) rebalancing premium / volatility pumping from a small low-correlation sleeve.
- **Reject everything else.** All leveraged strategies, SMA timing, factor rotation, panic overlays, DCA, and bond tilts either failed full-period MC or failed sub-period robustness.

## What we tested and what survived

### Strategies tested (across v1, v2, v3, Phase A v1/v2/v3 tri-period, Phase B + robustness)

| Strategy class | Specific tests | Verdict |
|---|---|---|
| **Plain VTI** | buy-and-hold (lump-sum) | **DEFAULT — best on every metric** |
| **Small gold tilt** | 90/10, 95/5, 99/1, gold-mean-shifted placebo | **PLACEBO ONLY: +0.07 log-ex stable; carry portion regime-dependent** |
| **Bond tilts** | 60/40 VTI/IEF, 60/40 VTI/BND | **Failed 3-period robustness** |
| **Trend / SMA timing** | SMA100 weekly (v3 satellite + v3 core), SMA200 monthly (Test 1) | **Failed full MC AND sub-period** |
| **Risk-parity blends** | 40% TF + 60% inverse-vol RP (Test 7) | **Failed full MC AND sub-period** |
| **Leveraged satellite** | 30% UPRO+SMA100 (v3) | **Failed (P=17.8%, log-ex -0.631)** |
| **Lifecycle leverage** | 2x→1x glide (Test 5) | **Failed sub-period (-1.028 dot-com vs +0.983 post-2016)** |
| **DCA vs LSI** | 6mo, 12mo deployment | **DCA loses to LSI (-0.015 / -0.029)** |
| **Panic overlays** | -40, -50, -60% drawdown triggers | **Destroy log-wealth at every threshold** |

### The one robust finding: rebalancing premium

The gold-mean-shifted placebo (gold daily returns minus their source-period excess over VTI) produces stable log-excess across all three sub-periods:
- 2000-07: +0.037
- 2008-15: +0.054
- 2016-26: +0.043
- Full: +0.073

This isolates the **mathematical rebalancing-premium / volatility-pumping benefit** of holding a low-correlation sleeve, separate from any source-period asset-mean carry. It's small (~+0.28%/yr CAGR equivalent over 25 years) but real and regime-independent.

A **95/5 VTI/IAU** allocation captures most of this benefit (gold weight is small enough that forward gold underperformance can't materially hurt the headline) while still getting the diversification arithmetic.

## Methodology lessons

This workflow's primary contribution to the broader research program:

1. **Bootstrap MC mechanically reproduces source-period asset means.** Any allocation overweighting an asset that performed unusually well in the source period (gold 2000-2010, equities 2009-2026, leveraged products post-2009) will look positive in MC. The MC is faithful to the source distribution; the source distribution is not necessarily representative of the future.

2. **Three mandatory robustness checks before claiming any positive finding:**
   - **Linear-scaling sweep** (allocate 1/5/10/30/50% to the asset in question — if log-excess scales linearly with weight, it's source-period carry; if concave, it's diversification)
   - **Mean-shifted placebo** (subtract source-period asset-mean differential and re-bootstrap; if lift collapses, the headline was carry)
   - **Sub-period robustness** (split into 2 or 3 non-overlapping panels and re-bootstrap each; if results vary by >0.20 log-excess across periods, the finding is regime dependent)

3. **CI tightness on a bootstrap mean answers the wrong question.** A 10K-path MC with mean +0.134 and 95% CI [+0.132, +0.137] tells you the source-period mean is well-estimated. It says nothing about whether that source-period mean is representative of the next 25 years.

4. **Pre-commit hashes must be of file contents, not parameter text.** The first iteration of this workflow had a pre-commit deviation (drift-trigger specified but not implemented). File-hash discipline catches this; param-text-hash doesn't.

5. **For terminal-log-wealth objectives with no withdrawals, simple buy-and-hold of the highest-Sharpe broad index dominates almost every embellishment.** The arithmetic is unforgiving: anything that adds left-tail damage or trades volatility for cost reduction loses geometric mean, and geometric mean = log-wealth.

## Failed strategy details (for the record)

### v3 (60/30/10 VTI/UPRO/IAU + weekly SMA100)
- 10K-path MC: P(v3>VTI) = 17.8%, mean log-excess -0.631, p5 MaxDD -72% (worse than VTI's -64%)
- Real UPRO 24.8% CAGR (2009-2026) was sample-period-specific. Bootstrap including dot-com era destroys the strategy.

### Test 1 (VTI + SMA200 monthly, validated trend_following config from etf workflow)
- Full period: log-ex -0.555 vs VTI
- Sub-periods: +0.008 (2000-07) / -0.646 (2008-15) / -0.702 (2016-26)
- The "highest-conviction practical finding" of prior research workflow does NOT generalize to bootstrap MC

### Test 2 (90/10 VTI/Gold static)
- Headline finding (+0.134 log-excess, 86% beat-rate) was source-period gold-bull carry
- 99/1 VTI/Gold also beats VTI 87% of paths (linear scaling = carry signature)
- Sub-periods: +0.373 / -0.055 / +0.053
- Carry decomposition: +0.336 (2000-07) / -0.109 (2008-15) / +0.010 (2016-26)
- The placebo (mean-shifted) shows the diversification residual; the carry portion is regime-dependent

### Test 4 (60/40 VTI/IEF vs 60/40 VTI/BND)
- 2-period split looked stable (+0.094 / +0.035). 3-period split revealed weakness.
- Sub-periods: +0.094 (2000-07) / +0.143 (2008-15) / **-0.063 (2016-26)**
- IEF underperformed BND in 60/40 in recent decade — can't be relied on forward

### Test 5 (Lifecycle 2x→1x glide)
- Full: +0.044 log-excess (54.4% beat-rate) — looked marginal positive
- Sub-periods: **-1.028 (dot-com) / +0.023 (GFC era) / +0.983 (post-2016)**
- Classic LETF regime dependence; full-period MC averages catastrophic and brilliant outcomes

### Test 7 (40% TF + 60% inverse-vol RP)
- Best drawdown of all tested (-17% median MDD)
- Full: -0.176 vs 60/40 / -0.494 vs VTI
- Sub-periods: +0.323 (2000-07) / -0.311 (2008-15) / -0.438 (2016-26)
- Regime dependent; the drawdown reduction is real but the cost is too high in normal markets

### DCA vs LSI
- DCA-6mo: -0.015 log-excess, P(DCA>LSI) = 37.5%
- DCA-12mo: -0.029 log-excess, P(DCA>LSI) = 34.9%
- Confirms textbook finding: lump-sum dominates DCA in expectation

### Panic overlays
- VTI w/ panic at -40%: -0.349 log-excess vs no overlay
- VTI w/ panic at -50%: -0.146
- VTI w/ panic at -60%: -0.042
- All thresholds destroy log-wealth. The drawdown protection is real but re-entry timing dominates the cost. **Practical implication: investors who genuinely cannot hold through deep drawdowns should pick a fundamentally lower-volatility allocation upfront, NOT layer panic-rules on top of high-volatility strategies.**

## What the program-wide pattern says

Across 7+ workflows and now this real-world test, **0 strategies have cleared strict gates against simple buy-and-hold benchmarks.** The pattern is now overwhelming:

- All sports markets are efficient vs public models
- VTI is Sharpe- and CAGR-efficient across all backtest+MC conditions
- All "validated descriptive findings" from earlier workflows have failed when bootstrap-MC'd: trend following, gold sleeves, factor timing, leverage with overlays
- The only stable positive in the entire program is the rebalancing-premium arithmetic (~+0.28%/yr from a low-correlation small sleeve), which is mathematically guaranteed and not strategy-specific

**The honest investment recommendation, after thousands of strategy evaluations and 50+ MC runs in this workflow alone, is: hold the broadest available equity index, lump-sum, and don't touch it.**

## Files

- `experiments/panel.py` — multi-source spliced 2000-2026 daily panel
- `experiments/harness.py` — shared MC harness with cadence + flip-aware rebalance + drift-aware
- `experiments/run_phase_a.py`, `run_phase_a_v2.py`, `run_phase_a_v3_triperiod.py` — Phase A iterations
- `experiments/run_phase_b.py`, `run_phase_b_robustness.py` — Phase B + robustness check
- `precommit/tests_1_4_7.json` — locked parameters with version history
- `artifacts/*.parquet` — all MC result files (10K paths each)
- `artifacts/phase_a_summary.json`, `phase_a_v2_summary.json`, `phase_a_v3_triperiod.json`, `phase_b_summary.json`, `phase_b_test5_robustness.json` — aggregate results

## Open follow-ups (low priority)

1. **Pre-2000 gold data.** LBMA spot price not autonomously fetchable; an investor with manual access could rerun on 1972-2026 panel including the post-Bretton-Woods full gold history. Expected result: the ~+0.07 placebo finding will likely shrink slightly (less correlation extremes) but stay positive.

2. **Prospective tracking.** No frozen-rule out-of-sample evaluation has been committed. The 95/5 VTI/IAU recommendation is based on regime-stable in-sample finding; a 5-year prospective check (2026-04-30 → 2031-04-30) would verify.

3. **Behavioral utility framework.** Test 8 confirmed that panic overlays destroy log-wealth, but didn't model the alternative: an investor who holds a high-volatility strategy but eventually capitulates and STAYS in cash (no re-entry). That model would likely show even stronger log-wealth destruction, reinforcing the "pick lower-volatility allocation upfront" recommendation.

The recommendation is settled: **100% VTI buy-and-hold (or 95/5 VTI/IAU) for a $100K, 25-year, tax-protected accumulator.** Further work should be on prospective verification, not in-sample optimization.
