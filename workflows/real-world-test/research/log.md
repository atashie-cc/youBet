# Real-World Test — Research Log

## Objective

Test the v3 strategy (60% VTI + 30% UPRO + 10% IAU, weekly SMA100 overlays on the
first two sleeves, annual rebalance, lump-sum entry) against historical data and
10,000 Monte Carlo paths. The user's account profile: $100K, 25-year horizon,
tax-protected, weekly trading allowed.

The objective function for evaluation is terminal log-wealth E[log(W₂₅)], per
the user's framing.

## Strategy Specification (LOCKED)

```
weights:        VTI 60% / UPRO 30% / IAU 10%
overlays:       VTI sleeve gated by SMA100 of VTI (weekly Friday close)
                UPRO sleeve gated by SMA100 of SPY (weekly Friday close)
                IAU static, no overlay
cadence:        weekly (Friday-anchored)
rebalance:      annual to targets, OR drift > 7pp
costs:          10 bps one-way per overlay flip; 91 bps UPRO ER; 25 bps IAU ER
synthetic 3x:   E2-calibrated 25 bps borrow spread for pre-UPRO history
entry:          lump-sum
benchmark:      VTI buy-and-hold (primary), 60/40 VTI/BIL (secondary)
```

## Experiment 1: Historical Backtest (1998-2026)

Source data:
- VTI: 2001-06 inception; pre-VTI period uses SPY as proxy
- SPY: 1998-2026 (full)
- UPRO: 2009-06 inception (real); pre-2009 uses synthetic 3x SPY (E2-calibrated)
- IAU: 2005-01 inception; pre-IAU run as "no-gold" variant
- BIL: 2007-05 inception; pre-2007 uses daily T-bill rate from FRED

### Results

| Strategy | Years | CAGR | Vol | Sharpe | MaxDD | Calmar | Final |
|---|---|---|---|---|---|---|---|
| v3 (1998-2026, no gold) | 28.2 | **+7.3%** | 17.5% | 0.262 | -56.4% | 0.13 | 7.30x |
| VTI B&H (1998-2026) | 28.2 | **+9.4%** | 19.5% | 0.352 | -55.5% | 0.17 | 12.52x |
| 60/40 VTI/BIL (1998-2026) | 28.2 | +6.9% | 11.7% | 0.285 | -36.3% | 0.19 | 6.52x |
| v3 (2005-2026, with IAU) | 21.2 | +11.1% | 17.1% | 0.469 | **-31.7%** | 0.35 | 9.34x |
| VTI B&H (2005-2026) | 21.2 | +10.8% | 19.2% | 0.424 | -55.5% | 0.20 | 8.87x |
| v3 (2009-2026, real UPRO) | 16.8 | +11.7% | 17.0% | 0.499 | -31.7% | 0.37 | 6.37x |
| VTI B&H (2009-2026) | 16.8 | +14.9% | 17.5% | 0.652 | -35.0% | 0.43 | 10.30x |

### Crisis Period Breakdown (1998-2026)

| Crisis | v3 return | v3 MaxDD | VTI return | VTI MaxDD |
|---|---|---|---|---|
| Dot-com bust (2000-03 to 2003-03) | -39.5% | -47.1% | -33.9% | -46.4% |
| GFC (2007-10 to 2009-06) | **-5.9%** | **-24.4%** | -36.4% | -55.5% |
| COVID crash (2020-02 to 2020-04) | -11.6% | -18.2% | -10.4% | -35.0% |
| 2022 rate hike (full year) | -30.0% | -30.8% | -19.5% | -25.4% |
| Full 2000-2013 lost decade | +23.2% | -51.0% | +37.3% | -55.5% |

### Overlay diagnostics (full sample)
- VTI overlay in-market: 69.9% (143 flips over 28 years)
- SPY overlay in-market: 70.5% (143 flips)

### Findings from historical test

1. **v3 underperforms VTI by 2.1% CAGR over 1998-2026.** The synthetic 3x SPY +
   SMA100 satellite gets destroyed in dot-com (-39.5% vs VTI's -33.9%). The 2022
   rate hike chop also kills v3 (-30.0% vs -19.5%) — UPRO grinds down through
   repeated SMA flips.

2. **v3 only beats VTI in the post-2005 with-IAU window** (+0.3% CAGR), and only
   on Sharpe (0.469 vs 0.424). The gold sleeve and the absence of dot-com data
   are doing the heavy lifting.

3. **GFC is the showcase win.** v3 lost only -5.9% with -24% MaxDD vs VTI's -36%
   loss / -55% MaxDD. SMA100 weekly handles fast crashes well — this confirms
   prior research findings.

4. **2022 is a microcosm of the dot-com failure mode.** Sideways grinding decline
   with multiple SMA flips wrecks the leveraged satellite. v3 lost 1.5x what VTI
   lost in 2022.

5. **In the 2009-2026 post-recovery period, v3 underperforms VTI by 3.2% CAGR.**
   This is the regime where UPRO-SMA100-weekly looked best in isolation
   (24.8% standalone CAGR per cagr-max E15), but inside a portfolio with the VTI
   overlay drag and IAU drag, the satellite's gross gain doesn't pay back its own
   friction during a strong VTI run.

## Experiment 2: Monte Carlo (10,000 paths, 25-year horizon)

Method: paired daily block bootstrap (Politis-Romano, 22-day blocks) from the
1998-2026 historical panel. Cross-sectional joint structure preserved within
blocks. Each path runs the full v3 strategy and benchmark comparators.

Source panel: 7,114 trading days × 5 assets (VTI, SPY, UPRO/synth, IAU, BIL).

Total compute: 10,000 paths in 22.3 seconds (~448 paths/sec).

### Terminal wealth distribution ($1 lump-sum → $X at 25 years)

| Strategy | p1 | p5 | p25 | **p50 (median)** | p75 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|
| v3 | 0.47x | 0.90x | 2.31x | **4.60x** | 9.66x | 28.74x | 59.77x | 8.61x |
| VTI B&H | 1.16x | 2.17x | 5.03x | **9.08x** | 16.22x | 35.92x | 61.85x | 12.89x |
| 60/40 VTI/BIL | 1.51x | 2.21x | 3.63x | **5.16x** | 7.27x | 11.71x | 16.14x | 5.83x |

### CAGR distribution (annualized over 25 years)

| Strategy | p1 | p5 | p25 | **p50** | p75 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|
| v3 | -3.0% | -0.4% | +3.4% | **+6.3%** | +9.5% | +14.4% | +17.8% | +6.6% |
| VTI B&H | +0.6% | +3.2% | +6.7% | **+9.2%** | +11.8% | +15.4% | +17.9% | +9.2% |
| 60/40 VTI/BIL | +1.7% | +3.2% | +5.3% | **+6.8%** | +8.3% | +10.3% | +11.8% | +6.8% |

### Maximum drawdown distribution

| Strategy | p1 | p5 | p25 | **p50** | p75 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|
| v3 | -80.5% | -72.3% | -59.5% | **-51.2%** | -44.2% | -36.4% | -32.2% | -52.4% |
| VTI B&H | -73.1% | -63.6% | -51.3% | **-43.6%** | -37.2% | -30.2% | -26.0% | -44.8% |
| 60/40 VTI/BIL | -49.6% | -41.4% | -32.1% | **-26.9%** | -22.7% | -18.3% | -15.7% | -28.0% |

### Head-to-head outcomes

| Comparison | Result |
|---|---|
| **P(v3 terminal > VTI B&H)** | **17.8%** |
| P(v3 terminal > 60/40) | 45.6% |
| **Mean log-wealth excess vs VTI** | **−0.631** |
| Mean log-wealth excess vs 60/40 | −0.070 |
| Median CAGR excess vs VTI | **−2.75%** |
| Median CAGR excess vs 60/40 | −0.37% |
| P(v3 CAGR < 0 over 25 yr) | 6.5% |
| P(VTI CAGR < 0 over 25 yr) | 0.7% |

## Verdict

**v3 fails the Monte Carlo test under its own stated objective.**

1. **v3 loses to plain VTI buy-and-hold in 82.2% of 25-year paths.**
2. **Median CAGR shortfall vs VTI = −2.75% annualized.** Compounded over 25 years,
   this is the difference between $4.60x and $9.08x terminal wealth on $1 invested
   — roughly half the wealth in median.
3. **Mean log-wealth excess vs VTI = −0.631.** The user's stated objective is
   maximizing E[log(W)]. v3 has substantially negative log-wealth excess. By the
   user's own chosen objective function, v3 is dominated by buy-and-hold.
4. **Worst-case drawdowns are WORSE for v3 than for VTI.** v3 p5 MaxDD = −72%;
   VTI p5 MaxDD = −64%. The leveraged satellite's contribution to portfolio MaxDD
   in adverse sequences exceeds whatever protection the SMA overlay provides on
   the VTI sleeve. v3 is not even drawdown-superior, let alone return-superior.
5. **vs 60/40 VTI/BIL**, v3 is roughly a coin flip (45.6%) on terminal wealth and
   modestly worse on log-excess. 60/40 has dramatically lower worst-case
   drawdowns (p5 = −41% vs −72%).

### Why it fails

Three compounding problems, all consistent with the prior research and reviewer
critiques:

1. **The UPRO satellite has high friction in choppy / grinding bear markets.**
   The 2022 historical loss (-30% v3 vs -19.5% VTI) and the dot-com simulation
   drag (-39.5% v3 vs -33.9% VTI) come from repeated SMA100-weekly whipsaws on
   the leveraged sleeve. Block-bootstrapped paths inherit this regime risk
   proportionally.

2. **The 2009-2026 UPRO+SMA100 results were sample-specific.** Real UPRO weekly
   SMA100 = 24.8% CAGR was measured on a window with no dot-com and a regime
   where the SMA signal was very effective (one fast crash, mostly trending).
   Once the sample distribution includes pre-2009 history, the strategy's
   median CAGR falls to ~6.3% — well below VTI's ~9.2%.

3. **The VTI overlay alone provides modest insurance, but the cost is real.**
   In 70% of days the VTI overlay is in-market (matching the historical
   in-market rate), but the 30% out-of-market days during whipsaws cost ~3%/yr
   in bull markets. Combined with the UPRO sleeve drag, the cumulative friction
   is too large.

### What we should have inferred from prior research

- cagr-max rolling stress test (1998-2026 synthetic 3x SPY SMA100): median 20-yr
  CAGR 12.7%, full-sample 8.0% (BELOW SPY's 9.1%), MaxDD -90.8%, dot-com -84%.
  This was already a clear signal that the strategy is below-VTI in expectation
  once the full distribution is sampled.
- E[log(W)] = log of geometric mean. The geometric mean of a leveraged strategy
  with high left-tail risk is dragged down by the worst draws. UPRO+SMA100
  weekly's geometric mean over 1998-2026 is BELOW SPY's despite higher arithmetic
  mean. We knew this and chose to overweight the post-2009 sample anyway.

The Monte Carlo confirms the rolling stress test analytically: under repeated
sampling of the historical distribution, the leveraged-satellite strategy's
geometric mean is lower than VTI's across nearly all percentiles.

## What this means for the recommendation

The honest investment recommendation for the user, given:
- $100K, 25-year horizon, tax-protected, weekly trading allowed, lump-sum, log-wealth objective

Should be something close to one of:

1. **Plain VTI buy-and-hold** — best Monte Carlo median CAGR (9.2%), highest mean
   terminal wealth (12.89x), only 0.7% probability of negative 25-yr CAGR. The
   trade-off is -44% median MaxDD that the user must endure without panicking.

2. **VTI buy-and-hold with VTI SMA200-monthly drawdown overlay** (the strict
   `trend_following` config validated in the etf workflow). This trades ~1-2% of
   median CAGR for materially lower MaxDD (-22% rather than -44%). NOT tested in
   this workflow's Monte Carlo — would need a separate run to confirm.

3. **60/40 VTI/BND** — for the conservative case. p5 MaxDD only -41%, but median
   CAGR drops to ~6.8%. With BND replacing BIL in the actual implementation,
   would gain modest duration return that BIL doesn't capture.

The leveraged-satellite v3 is dominated by all three of these alternatives in
Monte Carlo. The optimistic 11-14% CAGR projection in the v3 write-up was based
on selective historical samples; the honest distribution is 6.3% median.

## Limitations of this Monte Carlo

1. **Block bootstrap with 22-day blocks understates regime persistence.** Real
   dot-com and lost decades persist for years; bootstrap interleaves them with
   bull periods. This likely UNDERSTATES tail risk for v3, not overstates.

2. **No transaction-level model of bid-ask spread.** Switching cost is set at
   10 bps per flip per the cagr-max conventions. Real-world execution might be
   slightly worse, especially on UPRO during volatile periods.

3. **Lump-sum entry assumed** at the start of each path. The historical entry-time
   sensitivity is folded into the bootstrap (each path gets a randomly-shuffled
   sample), so entry timing is appropriately diversified.

4. **No tax effects** — appropriate given the user's tax-protected account.

5. **Drift-threshold rebalance approximated** as annual-only in MC for vectorization
   speed. The threshold rarely fires under our sleeve volatilities; the historical
   backtest (which does enforce it) gives the same qualitative result.

6. **The 1998-2026 source distribution may not be representative of the next 25
   years.** All Monte Carlos are conditioned on history. The user's actual
   forward returns are drawn from a future distribution we don't observe.

## Recommendation

Drop v3 as constructed. The strategy's only winning sample is 2005-2026 (favorable
window without dot-com), and even there it beats VTI by only 0.3% CAGR. Under
honest resampling it loses to VTI 82% of the time.

Two paths forward worth testing in a v4 workflow:
- **v4a — pure VTI** (no overlay, no satellite). Simplest baseline. Highest expected
  log-wealth in the MC.
- **v4b — VTI core + SMA200-monthly drawdown overlay** (the validated trend_following
  config), no leveraged satellite. Should give modest log-wealth cost in exchange
  for materially lower drawdowns.

Both should be Monte Carlo'd against this same panel before any commitment to a
real $100K allocation.
