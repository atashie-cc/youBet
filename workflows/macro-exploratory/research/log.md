# Macro-Exploratory Workflow — Research Log

**Status**: exploratory (hypothesis generation, NOT confirmatory)
**Created**: 2026-04-14
**Plan**: `C:\Users\18033\.claude\plans\smooth-hatching-micali.md`
**Codex pre-implementation review**: incorporated 2026-04-14 (see plan)

## Purpose

Four completed youBet macro workflows (`etf/`, `etflab-max/`, `commodity/`, `factor-timing/`) established that only factor-timing — with 62.6 yr of Ken French daily data — has enough power to pass strict confirmatory gates. The other three workflows are power-starved: realistic post-decay effects (0.08–0.18 ExSharpe) fall below detection thresholds (~0.40 for commodity, ~0.30 for ETF).

This workflow runs **exploratory** experiments targeting max CAGR and Sharpe, explicitly accepting that most will fail strict statistical gates. Goal: generate hypotheses that can later be elevated to a confirmatory run on a fresh holdout, using the elevation rule:
  - ExSharpe > 0.60 on primary benchmark
  - 90% block-bootstrap CI lower > 0
  - Same-sign consistency across 3/3 sub-periods

No Holm correction is applied across experiments. All findings labeled EXPLORATORY.

## Experiment Roster

Runs in this order (E10 first as pipeline smoke test):

| Run | Exp | Focus | Status | Summary |
|---|---|---|---|---|
| 1 | E10 | Sanity | pending | Pure Antonacci GEM baseline using existing DualMomentum scaffold |
| 2 | E2  | Sharpe | pending | 1.5x conditional leverage on CMA/SMB/HML/RMW SMA100 (paper) |
| 3 | E4  | Sharpe | pending | 12-sleeve international pool: {US, Dev ex-US, Europe, Japan} × {CMA, HML, RMW} |
| 4 | E6a | CAGR   | pending | GEM clean replication (duplicate of E10 for report isolation) |
| 5 | E6b | CAGR   | pending | GEM + macro circuit breaker (yield curve + HY OAS) |
| 6 | E9  | Sharpe | pending | Canonical Barroso-Santa-Clara vol-managed factor timing on CMA/HML/RMW/SMB |
| 7 | E1  | CAGR   | pending | Stacked signals (VTI SMA + CMA SMA + yield curve) → synthetic 3x VTI, binary vs ladder |
| 8 | E3  | CAGR   | dropped | Merged into E1merged |
| 9 | E11 | Sharpe/DD | dropped | Rotation mechanism fights cash-optionality finding |
| 10 | E7 | CAGR | dropped | Minor smoothing of E2's failed signal |
| 11 | E12 | Sharpe | **FAIL (0.54)** | Canonical monthly BSC UMD — misses ExSh 0.60 gate (Codex R1 corrected) |
| 12 | E13 | robustness | **PASS** | E4 frozen quasi-holdout — ExSh +0.845 on post-2016 |
| 13 | E14 | investable | FAIL | E4 ETF bridge — costs destroy: net Sharpe **-0.316** (Codex R1 corrected) |
| 14 | E15 | robustness | FAIL (0.54) | E4 weekly cadence — 19.4% collapse, fails gate + 15% threshold |
| 15 | E16a | UMD pool | FAIL (-0.07) | UMD-only 4-sleeve pool — correlated signals, no diversification |
| 16 | E16b | expanded pool | FAIL (0.45) | 16-sleeve with UMD — dilutes E4, 39% degradation |
| 17 | E17 | investable | FAIL (-0.31) | Unhedged factor-signal ETF — reduces DD but misses upside |
| 18 | E18a | monthly OOS | FAIL (0.55) | Monthly 12-sleeve — transfers but 24% degradation |
| 19 | E18b | Asia-Pac OOS | FAIL (0.50) | 15-sleeve with Asia-Pac — absorbs partially, 27% degradation |
| 20 | E21 | CAGR frontier | diagnostic | Leverage sweep 1-6x: no peak, 6x = 12.3% CAGR, Kelly negative |
| 21 | E19 | leveraged pool | FAIL (gate) | 4-6x pool: timing works (ExS +0.7 vs true lev B&H) but factor CAGR base too small |
| 22 | E20 | breadth VTI | **FAIL (-0.10)** | Breadth-gated VTI: -85% MaxDD (GFC), CAGR 12.4% < SMA 3x 16.3%. Factor signal bad for equity timing |

## Guiding Principles

1. **Cash optionality** (factor-timing Phase 13/15): timing works by choosing *in-market vs cash*, not by rotation between risk assets.
2. **Independent signals stack** (factor-timing Phase 15): VTI SMA, factor SMA, and macro regime are orthogonal.
3. **Leverage dominates stock selection for CAGR** (etflab-max): MGK @ 2.3x SMA100 = 20.6% CAGR beats every active momentum strategy.
4. **PIT discipline**: publication lag via `PITFeatureSeries.as_of()`; threshold calibration via fixed literature values OR train-only expanding-window percentiles (`etf.transforms.Normalizer(method="percentile")`).
5. **Leverage > 1 expression**: must use synthetic leveraged returns (pattern in `workflows/etf/experiments/leveraged_strategies.py`), NOT weights > 1 (backtester enforces `weight.sum() <= 1`).

## Entries

### 2026-04-14 — E4 Pooled regional 12-sleeve (SMA100 equal-weight, paper)

Equal-weight composite of {US, Dev ex-US, Europe, Japan} × {CMA, HML, RMW} = 12
sleeves, each running unlevered SMA100 timing, annual rebalance to equal weights.
Common window 1993-01-01 to 2026-02-27, walk-forward test days 2003-01 to
2026-02 (n = 6043). Paper portfolio (Ken French factors, no transaction costs).

**Headline result**:
- Pool Sharpe **+1.570** vs pool-benchmark buy-hold +0.935, CAGR 3.7% vs 2.4%,
  MaxDD **-5.7%** vs -15.4%
- `pool_vs_pool_benchmark`: ExS +0.716 [+0.339, +1.093], Sharpe-diff +0.635,
  3/3 sub-periods positive
- **Elevation gate v2: PASS on all four criteria**

Sub-period breakdown (diff / ExS):
- 2003-2012 (n=2609): +0.603 / +0.714
- 2013-2019 (n=1826): +0.689 / +0.617
- 2020-2026 (n=1608): +0.714 / +0.853

**Secondary comparison — does international diversification lift beyond US CMA SMA?**
- `pool_vs_us_cma_sma`: ExS +0.191 [-0.174, +0.562]
- CI straddles zero, so the bootstrap cannot rule out that the pool's Sharpe
  advantage over single-factor US CMA timing is noise. The underlying magnitudes
  (pool 1.570 vs us_cma 0.645) suggest real diversification benefit, but the
  bootstrap CI is wide on a 24-year sample.

**Per-sleeve pattern**: every one of the 12 sleeves has positive Sharpe-delta
vs its buy-and-hold counterpart. Biggest lifts: japan_RMW +0.681, europe_HML
+0.597, us_CMA +0.592. Smallest lifts: us_RMW +0.134, japan_HML +0.179,
us_HML +0.251. There is no bad sleeve.

**Caveats flagged for Codex review**:
1. Sharpe 1.570 on a paper portfolio is striking. Plausibility checks needed.
2. Zero transaction cost, zero borrow, zero tracking error. Factor-timing Phase
   10 showed costs destroy most apparent lift in real-ETF implementations.
3. Fold schedules: 12 sleeves each generate their own walk-forward folds via
   `simulate_factor_timing`. The common-window slice makes indices align but the
   fold boundaries may still differ per-sleeve. Needs a look for rebalance
   calendar interaction and any implicit cross-sleeve look-ahead.
4. Annual rebalance with drift-within-year is implemented in
   `simulate_pooled_regional`. Codex should check the weights-reset semantics.
5. International factors start 1990-11; 120-month warmup brings first test day
   to ~2003 in intl but earlier for US. The common-window slice intentionally
   cuts US to match intl — verify this is what actually happens in the fold
   generator, not just in the index.

### 2026-04-14 — E1-merged Multi-signal leverage gate (VTI 3x + MGK 2x)

Merger of original E1 (stacked signals → 3x VTI) and E3 (macro-gated 2x MGK)
per the post-E2 roster prune. Three binary signals, frozen pre-run:
  1. VTI > SMA100 (price trend)
  2. cum(CMA) > SMA100(cum(CMA)) (factor trend)
  3. Yield curve spread > 0 (not inverted)

Agreement count k ∈ {0,1,2,3}. Two mapping variants, both frozen:
  - **BINARY**: exposure = leverage if k==3 else 0
  - **LADDER**: leverage if k==3, 1.0 if k==2, else 0

Two arms: VTI @ 3.0x (2003-2026, n=5856), MGK @ 2.0x (2008-2026, n=4601).
Synthetic leveraged returns pattern, 95bps expense ratio (UPRO/TQQQ-style),
50bps borrow spread on the leverage increment. Leverage locked at 3.0/2.0
(NOT 2.3 for MGK — avoided post-hoc tuning to the locked benchmark's exact
leverage).

**k distribution** (VTI arm): k=0 2%, k=1 20%, k=2 **51%**, k=3 **28%**.
The modal case is k=2, meaning the three signals are only in full
agreement 28% of the time. Binary variant holds leverage only 28% of days;
ladder holds leverage 28% at full + 51% at 1x + 21% at 0.

**Results — all four variant×arm cells fail v2 gate**:

| Arm | Variant | Sharpe | CAGR | MaxDD | vs single-SMA ExS | [CI] | Sharpe-diff |
|---|---|---|---|---|---|---|---|
| VTI 3x | binary | 0.719 | 13.3% | −46.4% | −0.413 | [−0.716, −0.109] | **−0.046** |
| VTI 3x | ladder | 0.710 | 15.3% | −49.8% | −0.377 | [−0.670, −0.076] | **−0.055** |
| MGK 2x | binary | 0.669 | 9.1% | −29.2% | −0.533 | [−0.895, −0.162] | **−0.148** |
| MGK 2x | ladder | 0.684 | 13.0% | −44.3% | −0.366 | [−0.700, −0.011] | **−0.133** |

Reference benchmarks (single-SMA leveraged, no multi-signal gating):
- **VTI 3x single-SMA**: Sharpe **0.765**, CAGR **21.96%**, MaxDD −53.2%
- **MGK 2x single-SMA**: Sharpe **0.817**, CAGR **19.49%**, MaxDD −41.5%
- VTI buyhold: Sharpe 0.667; MGK buyhold: Sharpe 0.691

**Clean refutation**: multi-signal agreement gating is **strictly worse** than
single-SMA gating on both arms. CIs on `binary_vs_single_sma_lev` and
`ladder_vs_single_sma_lev` exclude zero *on the negative side* in every cell.
All four v2-gate criteria fail for all four variant-arm cells.

**Substantive side-finding — the single-SMA leveraged benchmark is genuinely
positive**:
- VTI 3x single-SMA vs VTI buyhold: **ExS +0.480 [+0.176, +0.781]**,
  Sharpe-diff +0.098. Three of four gate criteria pass (fails only on the
  0.60 magnitude threshold — the 0.480 point estimate is large but not large
  enough).
- MGK 2x single-SMA vs MGK buyhold: ExS +0.310 [−0.014, +0.640],
  Sharpe-diff +0.126. Barely straddles zero on CI lower.

So "3x VTI long/cash on SMA100" is a real positive phenomenon on 2003-2026
(consistent with the locked benchmark's 21.6% CAGR / 0.649 Sharpe claim).
Multi-signal gating takes that real signal and degrades it by over-gating.

**Mechanism interpretation**: the three signals are *positively correlated*,
not independent. All three tend to be on/off together during risk-on / risk-off
regimes. Requiring all three to agree (k==3) for full leverage is equivalent
to requiring the strictest of the three, which for VTI SMA + CMA SMA + YC
positive turns out to be a very high bar (only 28% of days). The original
E1 hypothesis — "independent signals stack" — fails because the signals are
not independent enough for the AND gate to leave enough days at full
leverage.

The ladder variant partially mitigates this by taking 1x exposure at k==2
(51% of days) instead of 0. Its mean exposure 1.35x is much higher than
binary's 0.84x. But the single-SMA benchmark runs at 3x full time when VTI
is above SMA100 (~75% of days in our sample) — mean exposure roughly
0.75 × 3 = 2.25x. The ladder still under-leverages.

**Variant comparison** (ladder vs binary, isolated):
- VTI: ladder Sharpe 0.710 vs binary 0.719 — binary slightly better. Ladder
  adds CAGR (15.3 vs 13.3%) by taking 1x on k==2 days, but the extra
  leverage-to-1x transition in k==2 periods contributes roughly equal
  upside and downside in expectation.
- MGK: ladder Sharpe 0.684 vs binary 0.669 — ladder slightly better.

Neither dominates the other; both fail vs the benchmark.

Result JSON: `results/e1merged_multi_signal_gate.json`. Zero elevations.

### 2026-04-14 — E6b Antonacci GEM + macro circuit breaker overlay

Base: Antonacci Dual Momentum on VTI/VXUS/BND (12mo lookback), same as E10.
Overlay: force risk-off (BND) when EITHER (a) HY OAS > 80th percentile of
*expanding-history* OAS (all PIT-safe observations up to the fold boundary,
refit each fold via `Normalizer(method="percentile")`), OR (b) yield curve
continuously inverted for 90+ calendar days (approximation of "3 months")
as of the signal date. Thresholds locked in
`config.yaml.pit_protocol.e6b_gem_circuit_breaker`. See Codex R1 note below
on expanding-vs-rolling choice.

ETF backtester pipeline, 2003-2026 window, same Backtester instance runs both
strategies head-to-head for apples-to-apples isolation. All data from fresh
`fetch_prices` pull on 2026-04-15 (may differ slightly from E10's 2026-04-14
vintage — see note below).

**Results**:

| Strategy | Sharpe | CAGR | Max DD |
|---|---|---|---|
| **gem_cb** (circuit breaker) | 0.501 | 6.78% | **−37.4%** |
| pure_gem (this run) | 0.541 | 7.92% | **−37.4%** |
| VTI | 0.615 | 10.63% | −55.5% |

**The circuit breaker REDUCES Sharpe vs pure GEM** (−0.04 Sharpe points) while
**preserving the same max DD**. The overlay is strictly harmful on this sample.

Comparisons under v2 gate:
- `gem_cb_vs_vti`: ExS −0.295 [−0.649, +0.022], Sharpe-diff −0.114 → FAIL all
- `gem_cb_vs_pure_gem` (isolation): ExS −0.181 [−0.495, +0.122], Sharpe-diff
  −0.040 → CI straddles zero but point estimate negative
- `pure_gem_vs_vti_replication`: ExS −0.227 [−0.578, +0.076] — consistent
  with E10's pure-GEM result; GEM itself doesn't beat VTI on this sample

**Circuit breaker activity** (244 decisions):
- GEM was in equity on 207 days (85% of decisions)
- CB overrode 32 times (13% of equity decisions, 15% of all decisions)
- HY OAS triggered 32 times — all 32 coincided with GEM being in equity
- Yield curve triggered 26 times — all 26 coincided with GEM already in bonds
- Zero decisions where both HY and YC fired simultaneously

The HY/YC split pattern is mechanistically interesting: **yield curve
inversions fire before GEM's 12-mo momentum flips**, so YC signals hit only
when the portfolio is already in bonds (no-op). **HY OAS spikes are
coincident with stress**, firing during moments GEM's momentum is still
positive. So on this sample the yield-curve leg of the circuit breaker
provides zero additional value (strict subset of GEM's own risk-off signal),
and the HY OAS leg fires too late to prevent max DD.

**Most damning**: the max DD of −37.4% is IDENTICAL for pure GEM and
GEM-CB. All 32 HY-triggered overrides happened outside the single
drawdown event that defines the sample's worst loss — the overlay was
blind to the event that matters most. The 32 overrides are almost
entirely cost (whipsaw into/out of bonds) with no protective benefit.

**Clean refutation under v2 gate**:
- ExSharpe > 0.60: FAIL (−0.295)
- CI lower > 0: FAIL (−0.649)
- Sub-periods same-sign positive: FAIL
- Sharpe-diff > 0: FAIL (−0.114)

Zero elevations.

### 2026-04-14 — E6b Codex adversarial review

Codex R1 round on E6b. No bugs, six findings — four are reframings, one
noop, one is a suggested rerun for mechanism proof but not blocking.

1. **HY OAS percentile fit uses expanding history, not the 36-month rolling
   train window** (experiment_e6b.py:103, 112-120). The fold's actual train
   window is 36 months, but `hy_oas.as_of(as_of_date)` returns ALL history up
   to the fold boundary. `Normalizer.fit()` uses everything unless `lookback`
   is set explicitly. This is PIT-safe (no future leakage), but it changes
   the estimand from "rolling-train percentile" to "expanding-history
   percentile." By 2020 the normalizer has seen 17 years of OAS including
   2008-09, so the 80th percentile is anchored to the GFC peak; by contrast
   a strict 36-month rolling window in 2020 would have only seen 2017-2020
   and would fire COVID-driven triggers much earlier. Relabeled as
   expanding-history above. **A sensitivity rerun with strict 36-month
   rolling is an open option** but the max-DD-preserved conclusion is
   insensitive to this choice: what matters is whether HY OAS spikes *during
   or before* the worst drawdown, not where the percentile threshold sits.

2. **No PIT leakage in the trigger path** — noop. The OAS and yield-curve
   fetchers wrap series as zero-lag PIT features and the trigger path calls
   `.as_of()` before using the latest value. Normalizer state is overwritten
   each fold → no cross-fold leakage.

3. **Yield curve streak is a 90-calendar-day approximation, not strict "3
   months"**. The walk-back loop correctly rejects broken streaks, but the
   threshold is calendar days rather than monthly-calendar-end. Immaterial at
   90-day scale. Relabeled in the description above.

4. **Max-DD-preserved claim needs persisted override dates for a rigorous
   "fired too late" argument** (experiment_e6b.py:94, 174-196, 320-328). The
   aggregate equal-DD numbers strongly suggest the overrides happened outside
   the drawdown event, but `_cb_trigger_log` is only aggregated to counts —
   the actual dates of the 32 overrides are not in the JSON. A rerun
   persisting those dates would strengthen "the overlay was blind to the
   drawdown event" from qualitative to quantitative. Flagged as an optional
   follow-up; not blocking the E6b-fails verdict.

5. **E10 vs E6b pure-GEM drift is data-vintage, not "one extra day"**. The
   0.055 Sharpe gap over 23 years is larger than one trading day's worth of
   returns. Plausibly explained by fresh `fetch_prices` and
   `fetch_tbill_rates` calls pulling updated adjusted closes and T-bill
   vintages; "one extra day" is too narrow an attribution. Reframed above as
   "data-vintage drift from fresh fetches."

6. **Framing narrowed**: "the overlay is strictly harmful on this sample"
   → "**this specific circuit-breaker construction, with expanding-history
   HY OAS percentile thresholds and 90-day continuous yield-curve inversion,
   added no value on 2003-2026.**" The failure is not a general refutation of
   macro circuit breakers; it's a refutation of this particular overlay at
   these locked thresholds on this calibration protocol.

**Substantive conclusion unchanged by Codex**: E6b fails under v2 gate on
all four criteria, max DD is identical for pure GEM and gem_cb, and the
HY-fires-after-GEM-flips-first pattern means the YC leg has zero marginal
value (strict subset of GEM's own risk-off signal). The overlay is cost
without protection. A follow-up with strict 36-month rolling percentile
and persisted override dates would tighten the mechanism story; neither is
blocking the negative verdict.

Result JSON: `results/e6b_gem_circuit_breaker.json`. Zero elevations.

### 2026-04-14 — E9 Daily BSC-style vol-managed factor timing

**Framing correction (Codex R2)**: despite the original experiment label, E9
is NOT a canonical Barroso-Santa-Clara replication. Canonical BSC 2015 is a
*monthly*-rebalanced risk-managed *momentum* strategy (UMD specifically) that
estimates variance from trailing 6-month daily returns and sizes the position
for the next *month*. E9 is a *daily* inverse-vol adaptation with BSC-style
parameters, applied across CMA/HML/RMW/SMB. A true canonical replication
would be a separate monthly-UMD experiment on a longer history. Relabeled
throughout below.

Constant-vol targeting with BSC-style defaults: target_vol=12% annualized,
126-day (6mo) trailing realized vol, max_leverage=2.0. Applied to
CMA/HML/RMW/SMB on the same 2003+ locked window as E2 (n=5824 days). Borrow
spread 50bps on leverage increment.

Three comparisons per factor (mirroring E2 structure):
1. `full_vs_buyhold`: vol_target lev2.0 vs buy-and-hold factor (primary gate)
2. `capped_vs_buyhold`: vol_target lev1.0 vs buy-and-hold (no-leverage replication)
3. `full_vs_capped`: lev2.0 vs lev1.0 (isolates the leverage contribution)

**Clean refutation — all four factors fail on every comparison**:

| Factor | Full Sharpe | Bench Sharpe | ExS (full vs bh) | Sharpe-diff | Max DD |
|---|---|---|---|---|---|
| CMA | −0.124 | +0.053 | −0.356 [−0.730, +0.020] | −0.178 | **−56.6%** (vs −28.5% bh) |
| HML | −0.119 | +0.017 | −0.275 [−0.632, +0.076] | −0.136 | **−71.9%** (vs −56.3% bh) |
| RMW | +0.236 | +0.360 | −0.016 [−0.369, +0.333] | −0.124 | −27.2% (vs −20.7% bh) |
| SMB | −0.044 | +0.078 | −0.326 [−0.674, +0.029] | −0.123 | **−57.3%** (vs −37.3% bh) |

**Zero elevations**. Every factor's Sharpe-diff is negative; most CIs exclude
positive territory. Sub-periods show the failure is not regime-specific —
Sharpe-diff is negative in all 12 (factor × window) cells.

**The max-DD inversion is the interesting signal**: CMA/HML/SMB show *larger*
max drawdowns under vol targeting than under simple buy-and-hold. This is the
opposite of what vol targeting is supposed to deliver. Most likely explanation:
the 2.0x leverage cap amplifies exposure precisely when realized vol is low
but the regime is about to shift (long quiet periods followed by abrupt
drawdowns). Trailing 126-day realized vol is slow to react; by the time it
flags the new regime, positions have already been scaled to 2x and taken the
brunt of the transition.

**Mechanism isolation**:
- `capped_vs_buyhold` (vol_target at max_leverage=1.0): ExS near zero for
  every factor (CMA −0.003, HML −0.091, RMW +0.009, SMB −0.153). Pure
  inverse-vol scaling without leverage amplification is essentially
  equivalent to buy-and-hold on this sample — no signal, no destruction.
- `full_vs_capped`: leverage uniformly destroys value (CMA −0.357, HML
  −0.316, RMW −0.016, SMB −0.291).

**Consistent with E2**: leverage on the 2003-2026 factor sample
does not add Sharpe — it subtracts. E9 (continuous inverse-vol scaling) and
E2 (binary SMA-gated conditional leverage) fail for different mechanistic
reasons but produce the same qualitative outcome: **leverage > 1 destroys
risk-adjusted return on our factor sample, regardless of the gating rule**.

**Why canonical BSC might work and daily E9 not (Codex R2 clarification)**:
BSC 2015 tested UMD 1927-2011 — 85 years with multiple fat-tail crashes
(1929-33, 1973-74, 2000-02, 2008-09) where vol had a strong crash-prediction
signal. Canonical BSC rebalances monthly (not daily), which smooths both the
signal and transaction costs. Our sample is 2003-2026 with one brief vol
spike (Mar 2020), the factors tested are CMA/HML/RMW/SMB (not UMD), and the
adaptation is daily-rebalanced. Any of those three departures from the
original could account for the failure. It is a refutation of the
*adaptation*, not a refutation of BSC in its original domain.

**Codex R2 adversarial review — no bugs in `VolTargeting`**:
- PIT path verified clean: `.rolling(126).std()` at t uses [t-125, t],
  `.shift(1)` pushes to t+1, so exposure on day t+1 uses only data through t.
- Max-DD inversion is a *real* outcome of slow 126-day vol estimation plus
  2x cap: during long quiet periods exposure rises to 2x, then the regime
  shift arrives before trailing vol catches up, and the position takes the
  full brunt.
- `capped_vs_buyhold` being ~flat is not suspicious — with target_vol=12%
  and factor vols often near or below 12%, the max_leverage=1.0 variant
  spends most of its time pinned at full exposure, making it close to
  buy-and-hold. The damage is entirely from allowing exposure above 1.0.
- Minor bias: `full_vs_capped` is downward-biased by ~0.036 Sharpe points
  because only the `full` variant pays borrow cost on the leverage
  increment. Back-of-envelope: 25 bps/yr borrow drag on ~7% strategy vol →
  ~0.036 Sharpe bias. Removing this bias does not flip any sign; RMW's
  −0.016 becomes roughly +0.02, still near zero, and CMA's −0.357 becomes
  roughly −0.32. Conclusion unchanged.

**Corrected framing (Codex R5)**: "E9 does not replicate BSC-style risk
management on CMA/HML/RMW/SMB in the 2003-2026 locked window under *daily*
rebalance and a 2x leverage cap. Leverage is what destroys Sharpe and DD;
pure inverse-vol scaling without leverage (capped at 1.0x) is essentially
inert on this sample because target vol exceeds the factors' typical vol."

**Open follow-up**: a canonical monthly-UMD BSC replication on longer US
history is a distinct experiment that would actually test the original
mechanism. Flagged but not run in this workflow.

### 2026-04-14 — E9 Codex R2 round 2: exposure distribution diagnostic

Codex second-round review flagged one unresolved risk: *was E9 actually
testing vol management, or was the full variant mostly pinned at the 2x cap
because target_vol=12% exceeded the factors' typical realized vol?* Ran the
exposure distribution diagnostic on a fresh E9 execution:

| Factor | Full median | % at 2x cap | % at ≤1x | [p05, p95] | Capped @1x |
|---|---|---|---|---|---|
| CMA | **2.00** | **66%** | 4% | [1.09, 2.00] | 100% |
| HML | 1.40 | 29% | 25% | [0.47, 2.00] | 100% |
| RMW | **1.98** | **49%** | 6% | [0.98, 2.00] | 100% |
| SMB | 1.40 | 3% | 16% | [0.73, 1.93] | 100% |

**E9 did not actually test vol management for CMA and RMW**. Both had median
exposure ~2.00 and spent half their time or more pinned at the 2x cap. The
strategy was effectively running "capped 2x leverage with a small vol-scaling
correction for ~35-50% of days." For CMA, only 4% of days had exposure ≤1x;
for RMW, 6%. HML is the only factor where the full variant had meaningful
below-1x exposure (25% of days). SMB is the only factor where the cap rarely
bound (3%) — genuine inverse-vol modulation throughout.

**Why**: BSC's target_vol=12% was calibrated for UMD (1927-2011), which had
~15-20% annualized vol in the original sample. Our factors on 2003-2026 have
much lower realized vol — CMA ~4-5%, RMW ~5-6%, HML ~7-8%, SMB ~7-8%. With
target 12% and factor vol ~5%, `target/vol ≈ 2.4`, immediately clipping to
the 2.0 cap. The capped variant at max_leverage=1.0 gives the same story in
reverse: its unclipped value is usually 1.5-2.5, clipping to 1.0, so it sits
at 1x ~100% of the time for CMA/HML/RMW.

**Reframed conclusion (post-diagnostic)**:
- For **CMA and RMW**: E9 is effectively a test of "capped daily 2x leverage,"
  not vol management. Result: capped 2x leverage fails (consistent with E2's
  finding that leverage doesn't add Sharpe on our factor sample).
- For **HML**: E9 is a mix — some days at cap, some below 1x, some in
  between. Fails.
- For **SMB**: E9 genuinely tests vol modulation (3% at cap, active
  modulation across [0.73, 1.93]). Still fails.

**What E9 actually refuted**: daily BSC-style inverse-vol scaling with the
*12% target parameter* on CMA/HML/RMW/SMB in 2003-2026. The 12% target is
too high for the factors' realized vol on this window, so for CMA/RMW the
experiment collapsed into "capped 2x leverage." A follow-up with
target_vol=6% or 8% would exercise the modulation mechanism on all four
factors — but picking a different target after seeing the distribution is
parameter tuning, not a literature-anchored test. That follow-up belongs in
a separate pre-committed experiment, not as a rerun here.

**Final E9 verdict**: zero factors elevated under v2 gate. The negative
result is real but narrower than "vol management fails" — it's more like
"BSC-12% target behaves as capped 2x leverage on these low-vol factors, and
capped 2x leverage fails, consistent with E2." SMB is the only factor where
the vol-scaling mechanism was actually exercised; it still fails.

Experiment description string in `results/e9_vol_managed_factor.json` and
`scripts/experiment_e9.py:208` updated from "Canonical Barroso-Santa-Clara"
to "Daily BSC-style inverse-vol scaling" per Codex R2-4.

**Round 2 open items** (Codex noop or deferred):
- Benchmark leakage: none, path is clean (Codex R2-1).
- Excess-vs-total return vol basis: second-order, not pursued (Codex R2-3).
- Canonical monthly-UMD replication: optional side-car, not blocking E6b (Codex R2-5).

Result JSON: `results/e9_vol_managed_factor.json`. Zero factors elevated.

### 2026-04-14 — E4 Codex adversarial review + fixes

Codex rescue round. No bugs in the rebalance loop — weights reset / drift
semantics are mechanically correct, no cross-fold look-ahead, SMA PIT shift is
clean. Three real findings, all actioned:

1. **Defensive `dropna()` can mask upstream alignment bugs**. Replaced with
   a loud `assert not .isna().any()` on both pool_returns and pool_benchmark.
   On rerun the assertion passed — confirming `simulate_pooled_regional` is
   actually producing aligned sleeve indices, and the original `dropna()` was
   unnecessary. Fix is structural (fails loudly on any future regression) not
   cosmetic.

2. **`pool_vs_us_cma_sma` secondary comparison was sample-misaligned**. Pool
   had 6043 days, US CMA had 5824 days, bootstrap was internally intersecting
   while reported metrics were on different samples. Fixed by reindexing pool
   to `us_cma_returns.index` before computing both metrics and the CI. Rerun
   secondary numbers: pool-on-uscma-idx Sharpe 1.590 vs us_cma 0.645, ExS
   +0.191 [-0.174, +0.562] — CI still straddles zero, so international
   diversification lift beyond single-factor US CMA timing remains *not*
   statistically distinguishable on a ~23yr sample. The underlying magnitude
   gap (1.590 vs 0.645) suggests real benefit, but bootstrap variance is wide.

3. **Framing correction — paper mechanism study, not investable strategy**.
   Codex's strongest point: pure Ken French factor returns zero out market
   beta by construction, giving artificially low cross-sleeve correlation.
   The 12-sleeve diversification is *real on paper* but over-counted relative
   to any implementable version (VLUE/QUAL/MTUM ETFs have tracking error,
   borrow drag, capacity constraints, non-trivial correlation). Factor-timing
   Phase 10 already showed realistic cost models destroy most single-factor
   timing alpha; applying that at 12-sleeve scale with within-year rebalance
   crossings would almost certainly eliminate most of the +0.635 Sharpe-diff.

**Primary result survives all fixes unchanged**: ExS +0.716 [+0.339, +1.093],
Sharpe-diff +0.635, 3/3 sub-periods positive, v2 gate PASS on all four
criteria. This is the corrected headline.

**Reframed interpretation**: E4 demonstrates that **pooling independent
factor-vs-cash timing signals works cleanly on paper**. It is a mechanism
study, not an investable-strategy recommendation. The elevation verdict should
be read as "candidate mechanism for a confirmatory paper-data holdout," not
"candidate portfolio construction for real money." Real-money confirmation
would require either (a) hedged ETF bridges per Phase 10's methodology applied
at scale, or (b) a dedicated factor-ETF cost/capacity model, neither of which
is in scope for this exploratory track.

**Minor hygiene deferred**: Codex flagged RF forward-fill in
`simulate_factor_timing` (line 317, 340) as too permissive, and `'A'`
rebalance frequency as deprecated. Neither affects the E4 headline and
neither is in scope for this round. Logged for potential follow-up.

### 2026-04-14 — Roster prune and merge (post-E2 reassessment)

E2 falsified the "leverage a timing signal" hypothesis (negative diff-of-Sharpes on
every factor). This updates priors for the remaining roster — experiments whose
mechanism is "add leverage on top of a trend/SMA signal" have dropped expected
value; experiments whose mechanism is distinct (pooled diversification, macro
overlay on momentum, continuous vol management, multi-signal gate of orthogonal
signals) are unaffected. Roster trimmed from 8 remaining experiments to 4.

**Dropped**:
- **e6a** (pure Antonacci GEM replication): duplicate of e10. Already run. Dropping to
  avoid a bookkeeping artifact.
- **e11** (macro-tilted inverse-vol risk parity on VTI/TLT/IAU/DBC): mechanism is
  rotation-based with ±10% macro tilts. Factor-timing Phase 13/15 showed cash
  optionality beats rotation by ~5x in selection alpha. A small tilt on a rotation
  baseline cannot overcome that. Low expected value, mechanism fights the strongest
  finding from prior work.
- **e7** (multi-horizon SMA voting on synthetic 2x MGK): SMA voting is a smoothing
  of the same timing signal E2 just failed to leverage productively. Minimal mechanism
  novelty given E2; the voting construction is a parameter tweak, not a new lever.

**Merged**:
- **e1 + e3** → **e1merged** (multi-signal agreement gate with two arms). Both
  experiments tested the same core question: can multi-signal agreement between
  orthogonal signals (VTI SMA + CMA SMA + yield curve) justify holding synthetic
  leverage? E1 tests 3x VTI, E3 tests 2x MGK. Running both as arms of one experiment
  with the same gate, comparisons, and bootstrap. Saves one round without losing any
  evidence.

**Kept**:
- **e4** (pooled regional 12-sleeve SMA100): tests pooling diversification on known-good
  underlying factor-SMA signals. E2 doesn't touch this mechanism.
- **e9** (canonical Barroso-Santa-Clara vol-managed factor timing on CMA/HML/RMW/SMB):
  continuous inverse-vol scaling, distinct from E2's binary conditional leverage.
  Strongest published theoretical basis (Barroso & Santa-Clara 2015).
- **e6b** (GEM + macro circuit breaker overlay): novel macro gating on momentum,
  different mechanism from both E2 and factor-SMA.
- **e1merged**: multi-signal gate as described above.

**Important note on direction**: This prune updates priors based on a *mechanism
refutation*, not based on the *outcomes* we saw. The dropped experiments are dropped
because their mechanism shares with E2's failed mechanism, not because we have
forecasts that they would specifically fail. The kept experiments test distinct
mechanisms that E2 did not touch. This is a prior update, not a cherry-pick.

**New order**: `e10 → e2 → e4 → e9 → e6b → e1merged`. E10 and E2 already complete.
E4 next.

### 2026-04-14 — Elevation gate v2 (methodology correction)

The v1 elevation rule evaluated only `Sharpe(strat - bench)` (Sharpe-of-excess).
E2 demonstrated that this lets through strategies whose
`Sharpe(strat) - Sharpe(bench)` (difference-of-Sharpes) is negative — a case
where the portfolio is objectively worse risk-adjusted than its benchmark but
the residual series happens to have a positive-mean/high-vol shape that gives
it ExSharpe > 0. Every E2 factor showed this pattern on the leverage-increment
comparison: CMA +0.230 ExS / −0.138 diff, HML +0.040 / −0.076, RMW +0.167 /
−0.109, SMB −0.083 / −0.097.

**Rule v2**: add `Sharpe(strat) - Sharpe(bench) > 0` as a fourth required
criterion. Existing criteria unchanged (ExSharpe > 0.60, CI lower > 0, 3/3
sub-periods same-sign positive on Sharpe-of-excess).

This is a **rejection-of-spurious-pass** correction, not a
make-my-failing-result-pass correction. The direction is opposite to p-hacking:
we are tightening in the direction that rejects results, informed by a
structural bug in the gate, not by the outcomes we saw.

Patched `scripts/_common.py:check_elevation` to take `sharpe_diff_point` as a
required argument and enforce p4. `config.yaml` `exploratory_gate.version`
bumped to 2 with `elevation_require_positive_sharpe_diff: true`.
`experiment_e2.py` and `experiment_e10.py` updated to pass
`ci["point_estimate"]` (which is the Sharpe-difference point estimate from
`excess_sharpe_ci`).

**Re-evaluation (no reruns)**: `scripts/reevaluate_gate.py` re-applies v2 to
saved JSONs. Results:
- E10 VTI comparison: still FAIL (all 4 criteria failing, unchanged outcome).
- **E2 CMA lev_vs_buyhold: still PASS** under v2 — ExS +0.633, CI [+0.298,
  +0.978], 3/3 positive sub-periods, Sharpe-diff +0.454. CMA's elevation
  survives on the correct grounds: the **combined** position (1.5x leveraged
  CMA SMA100) beats naive buy-and-hold CMA by a positive Sharpe difference
  AND a positive Sharpe-of-excess. This is the factor-timing Phase 6 result
  carried forward on the 2003+ window with leverage on top.
- E2 SMB/HML/RMW lev_vs_buyhold: still FAIL (same as v1, different reasons).
- E2 lev_vs_unlev_sma (leverage increment isolation): these comparisons were
  not originally evaluated by the gate (the script only gated `lev_vs_buyhold`),
  so they carry no sub-period block and the re-eval script marks them
  `no_subperiods`. Substantively, all four would still fail v2 since the
  diff-of-Sharpes is negative for every factor, which was the motivating
  finding for v2 in the first place.

Both result JSONs now carry `elevation_version: 2` and the updated per-comparison
reasons. The `elevation_v2_all_comparisons` table is appended to each for
transparency.

### 2026-04-14 — Infra fix: excess_sharpe_ci memory blowup

Previous session crashed (OOM) running E2 due to `excess_sharpe_ci` materializing full
`(n_bootstrap, n)` matrices at float64/int64 — ~7.7 GB peak on 16k-row Ken French daily.
Patched `src/youbet/etf/stats.py` to batch the bootstrap in chunks (~250 reps when n=16k)
with int32/float32 index arrays. Peak now < 150 MB per call. All 15 existing stats tests
still pass. RNG stream shifts (not bit-exact) but statistically equivalent.

### 2026-04-14 — E10 GEM baseline (pipeline smoke test)

Pure Antonacci GEM on VTI/VXUS/BND, 12-month relative + absolute momentum.
- GEM Sharpe 0.596 vs VTI 0.612 → ExSharpe -0.187, CI [-0.541, +0.108]
- Elevation: FAIL (all 3 criteria)
- Sub-periods all negative after 2012 (Sharpe +0.081 / -0.185 / -0.109)
- **Interpretation**: pipeline is green, GEM does not add Sharpe in our sample. Expected — this is the sanity check, not a novel claim.

### 2026-04-14 — E2 Codex adversarial review + re-run

Two real findings from Codex that materially changed E2:

1. **Estimand mismatch** (`scripts/_common.py:subperiod_consistency`). Elevation
   gate combined `excess_sharpe_ci["excess_sharpe_point"]` (Sharpe of excess =
   Sharpe(strat - bench)) with a sub-period sign check using
   `Sharpe(strat) - Sharpe(bench)`. These are not the same statistic and can
   disagree. **Fix**: sub-period check now computes Sharpe of the excess series
   to match the CI estimand; difference-of-Sharpes retained as diagnostic.

2. **Sample scope mismatch** (`scripts/experiment_e2.py`). The workflow's locked
   `start_date` is 2003-01-01, but E2 scored on ~53 yr of Ken French history
   (n_days = 13277, going back to 1973 after warmup). Other experiments in this
   workflow run on the 2003+ window, so E2's headline was not comparable and was
   silently inflated by regime diversity not available to the other experiments.
   **Fix**: slice factors to `start_date - factor_train_months` before
   simulation, preserving the walk-forward warmup within the locked window.

Non-findings: Codex confirmed no PIT, fold-boundary, or financing-cost bug in
`ConditionallyLeveragedSMA` / `simulate_factor_timing`. Batched
`excess_sharpe_ci` is numerically sound (paired correlation preserved,
concatenation unbiased).

### 2026-04-14 — E2 Conditionally-leveraged factor SMA (corrected run)

1.5x leverage during SMA100 on-state, 0 off-state, 50bps borrow spread on the increment.
Ken French daily 2003-2026 (5824 days after 10yr warmup).

**CMA ELEVATED** (3/3 pass) — but with a significant caveat (see below):
  - Lev vs buy-and-hold ExSharpe +0.633, CI [+0.298, +0.978]
  - Sub-periods all positive on Sharpe-of-excess: +0.933 / +0.519 / +0.566
  - Strategy Sharpe 0.507 vs buyhold 0.053

Other factors — none elevated after sample correction:
  - HML ExSharpe +0.221 [-0.128, +0.577], 3/3 sub-periods positive but CI crosses zero
  - RMW ExSharpe +0.129 [-0.216, +0.474], 2013-19 negative (-0.520)
  - SMB ExSharpe +0.036 [-0.280, +0.363], ~zero

**Critical nuance — leverage contribution is Sharpe-DESTROYING, not just inconclusive**:

The `lev_vs_unlev_sma` comparisons isolate the leverage contribution from the
underlying SMA timing effect (factor-timing Phase 6 replication):

| Factor | ExS (Sharpe of excess) | Diff of Sharpes (primary) |
|---|---|---|
| CMA | +0.230 [-0.169, +0.620] | **-0.138** |
| HML | +0.040 [-0.318, +0.391] | **-0.076** |
| RMW | +0.167 [-0.162, +0.490] | **-0.109** |
| SMB | -0.083 [-0.444, +0.278] | **-0.097** |

**The difference-of-Sharpes is NEGATIVE for every factor.** Adding 1.5x leverage
to an unlevered SMA100 factor strategy *lowers* the Sharpe ratio in every case.
The ExSharpe (Sharpe-of-excess) being weakly positive for CMA/HML/RMW is a
misleading artifact — the excess series (leveraged minus unlevered) has a small
positive mean but huge vol, so its Sharpe can be positive even when the
underlying portfolio's Sharpe is strictly lower.

**Honest summary**: E2's "CMA elevation" is *entirely* carried by the unlevered
CMA SMA100 replication (unlev_vs_buyhold ExS +0.593, CI [+0.260, +0.913]), which
is just factor-timing Phase 6 restated on the 2003+ window. The 1.5x conditional
leverage adds CAGR at proportionally higher vol and makes risk-adjusted returns
*worse* in every factor tested. **E2 delivers zero novel evidence.** CMA stays
on the "known to work unlevered" list; the conditional-leverage hypothesis is
falsified on this sample.

**Open question for the roster**: the gate uses Sharpe-of-excess only, which
produced a PASS on a strategy whose Sharpe is objectively lower than the
benchmark. Worth revisiting whether future experiments should require BOTH
estimands to pass (Sharpe-of-excess AND Sharpe-difference > 0). Not changing
mid-roster, but flagging for the final report.

### 2026-04-16 — E12 Canonical monthly BSC vol-managed UMD (1927-2011)

Barroso & Santa-Clara (2015) canonical construction: monthly UMD, trailing
6-month daily realized variance (monthly variance fallback for pre-1963 months
when daily data unavailable), target_vol=12%, max_leverage=2.0, monthly
rebalance. All parameters literature-anchored. Walk-forward 120mo train /
12mo test / 12mo step, 75 folds, 894 test months on 1927-2011 window.

**Codex R1 corrections applied**: (1) RF from the 3-factor monthly file
(1926-07+) instead of filling zeros pre-1963 — RF=0 had undercharged
leveraged months and removed cash yield in off-periods. (2) Variance regime
diagnostic added: pre-1963 uses monthly variance (x sqrt(12)), post-1963
uses daily variance (x sqrt(252)); the estimator switches at 1963-07.

**Results (corrected)**:
- BSC Sharpe **0.724**, CAGR 11.3%, MaxDD -43.9%
- UMD buy-and-hold Sharpe 0.562, CAGR 7.1%, MaxDD -57.8%
- ExSharpe **+0.539 [+0.360, +0.717]**, Sharpe-diff +0.162

**Elevation: FAIL** — misses ExSharpe > 0.60 (point estimate +0.539).
Passes the other three v2 criteria: CI lower > 0 (+0.360), 3/3
sub-periods positive, Sharpe-diff > 0 (+0.162).

**Exposure diagnostic (regime-split)**:
- All: median 1.55, [0.58, 2.00], 29% at cap, 18% at/below 1x
- Pre-1963 (monthly vol): median 1.39, [0.65, 2.00], 20% at cap, n=312
- Post-1963 (daily vol): median 1.68, [0.56, 2.00], 35% at cap, n=582

The regime break is visible: post-1963 exposure is systematically higher
(median 1.68 vs 1.39, more time at cap 35% vs 20%). Likely because
6-month monthly variance is a noisier estimator that produces more
conservative (lower) exposure than the 126-day daily variance equivalent.
This means the pre-1963 period contributes less leverage and therefore
less to the overall result than a uniform daily-variance estimator would.

**Extended window (1927-2026, diagnostic only)**: BSC Sharpe 0.680, ExSharpe
+0.519, consistent with post-publication decay in UMD.

**Interpretation (corrected per Codex R1)**: canonical-style monthly UMD
vol-targeting is positive and the mechanism appears real, but needs RF and
variance-regime validation before strong mechanism claims. The ExSharpe
0.539 is moderate — below the 0.60 exploratory threshold. E9's failure was
the adaptation (daily cadence on low-vol non-momentum factors), not a
failure of the BSC mechanism itself.

Result JSON: `results/e12_bsc_canonical_umd.json`. Zero elevations.

### 2026-04-16 — E13 E4 frozen quasi-holdout (post-2016 slice)

Exact E4 construction frozen verbatim (12 sleeves, SMA100, equal weight,
annual rebalance). Identity check: full-sample Sharpe 1.570414 and CAGR
0.037093 match E4's saved result to 6 decimal places.

Gate-v2 metrics evaluated on holdout 2016-01-01 to 2026-02-27 (2651 days).

**IMPORTANT CAVEAT**: quasi-holdout, NOT true holdout. E4 already saw the
full 2003-2026 window.

**Results**:
- Pool holdout Sharpe **1.531**, CAGR 4.3%, MaxDD -5.7%
- Benchmark holdout Sharpe 0.796, CAGR 2.4%, MaxDD -15.4%
- ExSharpe **+0.845 [+0.239, +1.456]**, Sharpe-diff **+0.736**

**Elevation: PASS** on all four v2 criteria. The holdout result is actually
*stronger* than the full-sample E4 result (ExSharpe 0.845 vs 0.716,
Sharpe-diff 0.736 vs 0.635). The quasi-holdout supports the E4 paper
mechanism on post-2016 data — it does not depend on the early-sample
regime (2003-2012 / GFC). Caveat: this is a quasi-holdout, not a true
prospective holdout (Codex R1 framing correction).

Secondary (pool vs US CMA SMA on holdout): ExSharpe +0.272 [-0.280, +0.848]
— CI straddles zero, consistent with E4's finding that the international
diversification lift is not statistically separable.

Result JSON: `results/e13_e4_holdout.json`. One elevation (primary).

### 2026-04-16 — E14 E4 investable ETF bridge (2-sleeve US, weekly, Phase 10 costs)

2-sleeve US subset of E4: VLUE (HML bridge) and QUAL (RMW bridge), each
hedged with VTI (lagged rolling 60-day beta). Weekly SMA100 signal
on Ken French paper factors. Phase 10 base-case costs: 35 bps borrow, 3 bps
trading, 5% margin (Reg T). Common window 2013-07 to 2026-02 (3171 days).

**Codex R1 corrections applied**: (1) Cost model fixed — original version
double-counted exposure weighting (cost already avg_exposure-weighted, then
multiplied net_returns by exposure again). Corrected: calendar-spread daily
cost subtracted from gross timed returns. (2) Benchmark relabeled: "gross
hedged-factor B&H" (not unhedged VLUE+QUAL, not cost-adjusted). (3) Removed
spurious `gross_vs_net` diagnostic (near-constant series, ExSharpe artifact).

**Results (corrected)**:

| Metric | Pool net | Pool gross | Pool hedged B&H |
|---|---|---|---|
| Sharpe | **-0.316** | 0.332 | -0.051 |
| CAGR | -0.9% | 0.8% | -0.3% |
| MaxDD | -17.4% | -9.7% | -20.8% |

**Costs destroy everything and then some**: gross Sharpe 0.332 falls to net
**-0.316**. The corrected cost model shows the net strategy is WORSE than
holding the hedged factor passively.

Primary (net vs hedged B&H): ExSharpe **-0.197 [-0.629, +0.233]** — negative
point estimate, CI straddles zero. Gate fails on all four criteria.

**Per-sleeve costs** (annual, unchanged from pre-correction):
- VLUE/HML: borrow 0.14%, switch 0.32%, rebal 0.03%, margin 0.99%, total **1.48%**
- QUAL/RMW: borrow 0.19%, switch 0.39%, rebal 0.02%, margin 1.33%, total **1.92%**

**Interpretation**: E4's paper mechanism survives on paper (E13 quasi-holdout)
but weekly cadence degrades it (E15 fails gate), and when mapped onto real ETF
instruments with realistic costs, the 2-sleeve investable subset has negative
net alpha. The dominant cost (margin on the short VTI hedge) is structural.
The corrected cost model makes the negative verdict stronger than the original
run (Codex R1 found the original undercharged costs).

This is **NOT a direct test of E4** (2/12 sleeves, ETF tracking error, no
international). But it confirms the general pattern from factor-timing Phase 10:
costs destroy most single-factor timing alpha when implemented via hedged ETFs.

Result JSON: `results/e14_e4_etf_bridge.json`. Zero elevations.

### 2026-04-16 — E15 E4 weekly-cadence robustness test

Identical E4 construction with CheckedFactorStrategy(check_period="W") — each
sleeve's SMA100 signal evaluated only at weekly boundaries, forward-filled
between. Same 12 sleeves, same annual rebalance.

**Results**:
- Pool weekly Sharpe **1.447**, CAGR 3.4%, MaxDD -6.3%
- Benchmark Sharpe 0.935, CAGR 2.4%, MaxDD -15.4%
- ExSharpe **+0.538 [+0.160, +0.919]**, Sharpe-diff +0.512

**Elevation: FAIL** — ExSharpe 0.538, below 0.60 threshold. Passes other three.

**Cadence collapse diagnostic**: E4 daily Sharpe-diff +0.635 -> E15 weekly
+0.512 = **19.4% collapse**. This is above the 15% benchmark from factor-timing
Phase 6 (which found weekly captures 85-90% of daily alpha for single-factor
hedged VLUE). The pooled construction is more cadence-sensitive than single-
factor timing, likely because the 12-sleeve annual rebalance interacts with
the weekly checkpoint schedule.

**Signal validation** (persisted in JSON per Codex R1): 88% of signal changes
occur on Monday (first trading day of week); the remaining 12% are
after-holiday first-of-week adjustments. The CheckedFactorStrategy wrapper
appears to be working correctly; validation persisted for auditability.

Secondary (weekly pool vs weekly US CMA SMA): ExSharpe +0.055 [-0.300, +0.424]
— essentially zero, same as E4's secondary finding.

Result JSON: `results/e15_e4_weekly_cadence.json`. Zero elevations.

### 2026-04-16 — E16 Pooled UMD momentum timing (4-region + expanded 16-sleeve)

Two arms testing whether UMD (momentum) adds value to E4's pooled construction.

**Arm A: UMD-only pool (4 sleeves, {US, Dev ex-US, Europe, Japan} x {UMD})**:
- Pool Sharpe 0.680, CAGR 4.6%, MaxDD -17.9%
- Benchmark Sharpe 0.549, CAGR 4.8%, MaxDD -45.5%
- ExSharpe **-0.074 [-0.558, +0.392]** — essentially zero, CI massively straddles zero
- **Elevation: FAIL** on all criteria except Sharpe-diff > 0

**Cross-region signal correlation: 0.471.** UMD on/off signals across regions
are moderately correlated — less diversification than E4's cross-factor pool
(E4 had mean signal correlation 0.046). Momentum crashes are correlated across
geographies, as expected. This explains why the 4-sleeve UMD pool doesn't work:
the sleeves don't provide independent timing signals.

**Arm B: Expanded 16-sleeve pool ({CMA, HML, RMW, UMD} x 4 regions)**:
- Pool Sharpe 1.422, CAGR 4.0%, MaxDD -5.6%
- ExSharpe **+0.448 [+0.006, +0.871]**, Sharpe-diff +0.387
- **Elevation: FAIL** — ExSharpe 0.448 < 0.60 threshold
- Signal correlation 0.046 (same as E4 — adding UMD doesn't change the cross-signal independence)

**Adding UMD to E4 HURTS the pool**: E4's Sharpe-diff was +0.635; the expanded
16-sleeve pool's Sharpe-diff is +0.387 (39% degradation). UMD's weak individual
timing signal dilutes the strong CMA/HML/RMW signals.

**Interpretation**: momentum's cross-region correlation makes UMD pooling ineffective
(Arm A). Adding UMD to E4's pool dilutes it (Arm B). The mechanism that makes E4
work — independent signals across factors — is absent when using a single factor
across regions. This is the opposite of what we hypothesized: E4's diversification
comes from cross-factor independence, not from cross-regional independence.

Result JSON: `results/e16_pooled_umd.json`. Zero elevations.

### 2026-04-16 — E17 Unhedged factor-signal ETF timing

Paper-factor SMA100 (weekly) signal applied to unhedged VLUE/QUAL ETFs. When
HML/RMW SMA is on, hold the ETF; when off, hold VGSH. 2-sleeve equal-weight
pool. Costs: 3 bps one-way switching only (no borrow, no margin).

**Signal concordance: 52%** — confirming this is a genuinely different signal
path from Phase 3 Stage B (which used SMA on the ETF price). The paper-factor
signal and ETF-price signal agree only at chance level.

**Results**:
- Timed net Sharpe **0.928**, CAGR 9.1%, MaxDD -18.2%
- ETF B&H Sharpe 0.758, CAGR 12.4%, MaxDD -36.8%
- ExSharpe **-0.306 [-0.839, +0.099]** — negative, CI straddles zero

**Elevation: FAIL** on all criteria except Sharpe-diff > 0.

**Interesting side-finding**: the timed strategy has higher Sharpe (0.928 vs 0.758)
but lower CAGR (9.1% vs 12.4%). Timing with the paper-factor signal successfully
reduces drawdowns (-18.2% vs -36.8% MaxDD) but at the cost of missing upside
during the ~50% of days the signal is off. The CAGR drag from being in VGSH
rather than the equity ETF dominates. The signal works for risk management but
not for return enhancement.

The timed strategy beats VGSH B&H (ExSharpe +0.763) and the Sharpe-diff vs VTI
is only slightly negative (-0.370), but both comparisons are secondary.

Result JSON: `results/e17_unhedged_factor_etf.json`. Zero elevations.

### 2026-04-16 — E18 True OOS: monthly frequency + Asia-Pac absorption

Two independent OOS dimensions for E4's mechanism.

**Arm A: Monthly 12-sleeve pool (SMA5 on monthly, 60mo train)**:
- Pool Sharpe **1.197**, CAGR 4.3%
- Benchmark Sharpe 0.632, CAGR 2.7%
- ExSharpe **+0.545 [+0.237, +0.838]**, Sharpe-diff +0.566
- **Elevation: FAIL** — ExSharpe 0.545 < 0.60 threshold
- Sub-periods: 3/3 positive, CI lower > 0

The monthly OOS is positive and passes 3 of 4 gate criteria. The ExSharpe
(0.545) is 24% lower than E4's daily result (0.716), consistent with the
signal degradation pattern from E15 (daily→weekly lost 19%, daily→monthly
loses 24%). The mechanism transfers to monthly data but at reduced strength.

**Arm B: 15-sleeve pool with Asia-Pac (daily, SMA100)**:
- Pool15 Sharpe **1.527**, CAGR 3.5%, MaxDD -5.7%
- Benchmark Sharpe 1.062, CAGR 2.7%, MaxDD -12.8%
- ExSharpe **+0.495 [+0.091, +0.903]**, Sharpe-diff +0.465
- **Elevation: FAIL** — ExSharpe 0.495 < 0.60 threshold
- Sub-periods: 3/3 positive, CI lower > 0

**Degradation: 26.7%** (above the 20% proportional-weight threshold). Asia-Pac's
3 known-bad sleeves (Phase 7: CMA -0.049, HML -0.212, RMW -0.167) drag the pool
more than their 1/5 weight would predict. The pooling mechanism partially absorbs
the bad region (Sharpe-diff is still positive at +0.465, and the pool still passes
3/4 gate criteria) but the absorption is not clean — it degrades more than
proportionally.

**Cross-arm synthesis**: E4's mechanism transfers to both monthly cadence and a
broader geographic scope, but with 24-27% degradation in each case. The daily,
4-region, CMA/HML/RMW construction appears to be the strongest variant — not
because the mechanism is fragile, but because the specific construction is near
the optimum on this data.

Result JSON: `results/e18_oos_monthly_asiapac.json`. Zero elevations.

### 2026-04-16 — E21 CAGR frontier diagnostic (leverage sweep 1-6x on E4's pool)

Sweep of synthetic leverage on E4's 12-sleeve pool using
`ConditionallyLeveragedSMA` with 50bps borrow + 95bps expense per level.

| Lev | CAGR | Sharpe | Vol | MaxDD | Calmar |
|-----|------|--------|-----|-------|--------|
| 1.0x | 2.7% | 1.164 | 2.3% | -6.2% | 0.44 |
| 1.5x | 3.7% | 1.027 | 3.6% | -9.7% | 0.38 |
| 2.0x | 4.6% | 0.957 | 4.8% | -13.3% | 0.34 |
| 3.0x | 6.5% | 0.884 | 7.4% | -21.3% | 0.31 |
| 4.0x | 8.4% | 0.844 | 10.2% | -29.0% | 0.29 |
| 5.0x | 10.3% | 0.818 | 13.1% | -36.3% | 0.29 |
| 6.0x | **12.3%** | 0.800 | 16.1% | -43.1% | 0.28 |

**No CAGR peak within tested range** — monotonically increasing through 6x.
The pool's Sharpe degrades slowly (1.16→0.80) because the underlying factor
vol is very low (2.3% at 1x). At 6x, vol is only 16% (comparable to
unleveraged VTI), so vol-decay hasn't yet kicked in meaningfully.

**Kelly diagnostic**: negative. Unlevered pool CAGR (2.7%) is below approximate
RF (~3%), so Kelly says "don't leverage the pool." The leverage works only because
SMA timing avoids drawdowns (Sharpe contribution), not because the absolute return
is high. This is a risk-management strategy being force-leveraged for CAGR.

**Key finding**: at 6x leverage, the paper factor pool (12.3% CAGR) barely beats
VTI buy-and-hold (11.1%) and is far below 3x VTI SMA (21.6%). Leveraging a
low-return/high-Sharpe strategy cannot compete with leveraging a high-return/
moderate-Sharpe strategy for CAGR maximization.

Result JSON: `results/e21_cagr_frontier.json`. Diagnostic only.

### 2026-04-16 — E19 Leveraged factor pool (4x/5x/6x with full CIs)

Pre-committed levels from E21's peak neighborhood. Full bootstrap CIs and
gate-v2 evaluation. Codex R1 fix: true leveraged buy-and-hold benchmark
constructed via `ConditionallyLeveragedSMA(on_leverage=lev, off_exposure=lev)`.

| Level | CAGR | Sharpe | MaxDD | ExS vs 1x | Beats VTI? |
|-------|------|--------|-------|-----------|------------|
| 4x | 8.4% | 0.844 | -29.0% | +0.627 [+0.212, +1.029] | N |
| 5x | 10.3% | 0.818 | -36.3% | +0.653 [+0.240, +1.055] | N |
| 6x | **12.3%** | 0.800 | -43.1% | **+0.667** [+0.254, +1.066] | **Y** |

All three levels FAIL gate v2 (Sharpe-diff is negative because leveraged
Sharpe < unlevered Sharpe 1.57 in every case). But all three PASS on the
other criteria — ExSharpe > 0.60, CI lower > 0, sub-periods positive. The
failure is purely on the Sharpe-difference leg: leverage trades Sharpe for CAGR.

**6x is the only level that beats VTI CAGR** (12.3% > 11.1%). None beats 3x
VTI SMA (21.6%).

**Timing genuinely works under leverage** (Codex R1 corrected comparison vs
TRUE leveraged B&H): ExSharpe +0.74/+0.70/+0.66 at 4x/5x/6x — all CIs
exclude zero. Leveraged B&H has terrible risk-adjusted performance (Sharpe
0.37-0.40, MaxDD -56% to -69%). Timing preserves Sharpe at 0.80-0.84 and
cuts MaxDD by roughly half. The issue is not that timing fails under leverage
— it's that paper factor returns are too small as a CAGR base.

Result JSON: `results/e19_leveraged_pool.json`. Zero formal elevations.

### 2026-04-16 — E20 Breadth-gated VTI leverage (12-signal ladder)

E4's 12-sleeve factor breadth mapped to a VTI leverage ladder:
breadth >= 10 -> 3x, >= 7 -> 2x, >= 4 -> 1x, < 4 -> cash.

**Breadth distribution**: mean 6.6, median 7. Ladder allocation: 3x on 8.7%
of days, 2x on 47.4%, 1x on 34.7%, cash on 9.2%. Mean exposure 1.56x.

**Codex R1 fix**: extended window from 2011-2026 (cache artifact, missing GFC)
to 2003-2026 (full locked window, 5783 days). The original 2011-start hid a
catastrophic drawdown.

**Results (corrected, 2003-2026)**:
- Breadth-gated: Sharpe **0.519**, CAGR **12.4%**, MaxDD **-85.1%**
- Single-SMA 3x VTI: Sharpe 0.621, CAGR 16.3%, MaxDD -54.2%
- VTI B&H: Sharpe 0.641, CAGR 10.7%, MaxDD -59.1%

**Clean refutation**: the breadth-gated strategy is WORSE than single-SMA 3x VTI
on both CAGR (12.4% vs 16.3%) and Sharpe (0.519 vs 0.621). The -85.1% MaxDD is
a near-total wipeout — far worse than both SMA 3x (-54.2%) and VTI B&H (-59.1%).

**Mechanism failure diagnosis**: factor breadth is orthogonal to market returns.
During the 2008 GFC, factor signals did NOT go to zero (breadth stayed moderate
at ~5-7 because paper factor returns behaved differently from VTI). The ladder
kept the strategy in 1-2x VTI exposure through the crash, while single-SMA 3x
correctly exited to cash (VTI fell below SMA). The factor signal is a good signal
for paper factor timing but a BAD signal for market-equity timing.

**Pre-correction vs post-correction**: the 2011-2026 window showed 20.5% CAGR and
0.822 Sharpe (seemingly the best in the workflow). The 2003-2026 window shows 12.4%
CAGR and 0.519 Sharpe with -85% MaxDD. This is the strongest example in the
workflow of why including the GFC stress period is essential for any leveraged
strategy evaluation.

CAGR gate: beats VTI (12.4% > 11.1%), does NOT beat 3x VTI SMA (12.4% < 21.6%).

Result JSON: `results/e20_breadth_vti.json`. Zero formal elevations.

<!-- Each experiment gets a dated entry below with raw numbers and qualitative findings. -->
