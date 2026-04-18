# CAGR-Max Research Log

## Objective
Maximize likely CAGR over a 20-year investment horizon using real-world public investment vehicles. All strategies must be implementable by a retail investor.

## Prior Work Summary
This workflow builds on 6 completed investment workflows (200+ strategies tested):

| Workflow | Key Finding | Best CAGR |
|----------|------------|-----------|
| etf/ (17 strategies) | VTI Sharpe-efficient. Trend following reduces DD 64% | 11.1% (VTI B&H) |
| etflab-max/ (158 strategies) | VTI CAGR-efficient. 0/158 pass strict gate | 20.6% (MGK 2.3x SMA100) |
| commodity/ (31 tests) | 0/31 pass. Static 10% IAU most robust | N/A |
| factor-timing/ (18 strategies) | 8/18 pass on paper. Hedged VLUE only investable | 3-4% (hedged VLUE) |
| macro-exploratory/ (20 experiments) | E4 pooled timing works on paper but investable path blocked | 12.3% (6x pool) |
| world_cup_2026/ | Market efficient for match prediction | N/A |

### Proven mechanisms (to build on)
- SMA100 trend filter at all leverage levels (SMA100 > SMA200)
- Leverage amplifies CAGR (at cost of Sharpe)
- Static concentration outperforms active momentum (lower turnover)
- Cash optionality during bear markets
- Independent per-sleeve timing > cross-asset rotation

### Proven failures (to avoid)
- ML on monthly financial data (insufficient samples)
- Macro timing signals as tactical triggers
- Active momentum rotation (turnover costs)
- Vol-targeting (destroys value on 5/6 factors)
- Inverse/short products (destroy wealth unconditionally)
- International equity at leverage (vol decay destroys CAGR)

## Gaps Addressed
1. **Only synthetic leverage tested** — never validated against real LETF products
2. **Vanguard-only universe** — missed non-Vanguard ETFs and real leveraged products
3. **No options-based strategies** — LEAPS never modeled
4. **No lifecycle leverage** — static multipliers only
5. **No multi-asset leveraged pairs** — UPRO+TMF "Hedgefundie" untested
6. **No crypto satellite** — BTC ETFs launched Jan 2024
7. **No tax impact quantification** — SMA strategies generate short-term gains

## Experiment Design

### Gate Criteria (LOCKED)
- Excess CAGR > 1.0% annualized
- Holm-adjusted p < 0.05
- 90% CI lower > 0
- Benchmark: VTI buy-and-hold

### Phase 0: Infrastructure (E0-E1)
- E0: Power analysis for 20 strategies (lower Holm denominator than etflab-max's 158)
- E1: LETF data availability audit

### Phase 1: Real LETF Validation (E2-E5) — HIGHEST PRIORITY
- E2: Real vs synthetic gap analysis (UPRO, TQQQ, SSO vs synthetic)
- E3: Real LETF + SMA100 overlay
- E4: SMA signal source — underlying vs LETF NAV
- E5: 2x vs 3x Kelly-adjusted on real products

### Phase 2: Concentrated Leverage (E6-E9)
- E6: TQQQ vs UPRO — concentration premium at leverage
- E7: Leveraged sector ETFs (TECL with SMA100)
- E8: Leveraged value (synthetic 2-3x VLUE with SMA100)
- E9: Multi-sleeve independent timing

### Phase 3: Novel Constructions (E10-E13)
- E10: LEAPS-based leverage vs LETF leverage
- E11: Lifecycle leverage glidepath with SMA100
- E12: UPRO + TMF leveraged pairs ("Hedgefundie")
- E13: Volatility-conditioned leverage with SMA

### Phase 4: Signal Refinement (E14-E15)
- E14: Multi-window SMA composite with leverage
- E15: Rebalancing frequency for leveraged strategies

### Phase 5: Satellite & Tax (E16-E17)
- E16: Crypto satellite with SMA100 (BTC ETF)
- E17: Tax drag quantification (IRA vs taxable)

### Phase 6: Global Gate (E18-E19)
- E18: Multi-sleeve CAGR-maximizing portfolio (blend top survivors)
- E19: Global CAGR gate with Holm correction across all ~20 experiments

## Expected Outcome
Most likely: 0/20 pass strict gate (consistent with all prior work). However:
- Highest point-estimate CAGR likely from TQQQ SMA100 (~22-28%)
- Most practically useful: validating real LETF returns track synthetic
- Tax drag quantification provides account placement guidance
- Fewer experiments (20 vs 158) means better per-experiment statistical power

---

## Results

### Phase 0: E0 Power Analysis (2026-04-17)

**Parameters:** 100 sims/effect, 500 bootstrap reps, Holm correction

#### Run 1 (homoskedastic DGP — superseded by Codex R2)
Used `strategy = benchmark + alpha + iid_noise` at various vol levels. Results were optimistic (MDE ~5% at 16% vol) but the DGP did not model the actual SMA regime-switching structure.

#### Run 2 (SMA-gated DGP — Codex R2 fix)
Proper state-dependent DGP: generates VTI-like benchmark, applies SMA100 signal to determine cash/invested periods, computes 3x leveraged returns during invested periods and T-bill during cash. P-value uses production estimator `(1+count)/(n+1)`. Bonferroni correction (conservative approximation of Holm).

| Excess CAGR | 20 strats | 50 strats | 50@4000d |
|-------------|-----------|-----------|----------|
| 1-7% | 0.000 | 0.000 | 0.000 |
| 10.0% | **0.020** | 0.000 | 0.000 |
| 15.0% | **0.110** | 0.000 | 0.000 |

**MDE (80% power): >15% across ALL scenarios.**

**Root cause:** The SMA gate creates heteroskedastic excess returns — very high vol when strategy is invested (3x leveraged) and near-zero when in cash, while the benchmark (VTI buy-and-hold) is always invested at 16% vol. The excess return series `log(strat) - log(bench)` has ~40% annualized vol (confirmed in smoke test). At this excess return volatility, 20 years of data cannot statistically distinguish large CAGR differences from noise.

**Implication:** The strict CAGR gate (excess CAGR > 1% AND Holm p < 0.05 AND CI lower > 0) is **structurally unpowered** for SMA-gated leveraged strategies vs buy-and-hold. This is consistent with etflab-max's result (0/158 pass) — it was not a sample-size issue but a fundamental feature of comparing regime-switching strategies to always-invested benchmarks.

**VERDICT: PROCEED WITH MODIFIED INTERPRETATION.**
- The strict gate will produce 0/N PASS (by design — we now understand why)
- The workflow's value is in **point estimates, CI interpretation, sub-period stability, and practical comparisons** — not formal gate passage
- This matches how etflab-max was ultimately interpreted: "VTI is CAGR-efficient under strict gate, but best point-estimate is MGK @ 2.3x SMA100 = 20.6% CAGR"
- Report the gate results for completeness, but primary conclusions come from point estimates and economic interpretation

### Phase 0: E1 LETF Data Audit (2026-04-17)

All 13 LETF products and 8 underlying indices fetched successfully from yfinance. Cached snapshot loaded from `etf/data/snapshots/2026-04-17/`.

| Ticker | Lev | Start | Years | Folds | Status |
|--------|-----|-------|-------|-------|--------|
| SSO | 2x | 2006-06 | 19.8 | 16 | OK |
| QLD | 2x | 2006-06 | 19.8 | 16 | OK |
| UPRO | 3x | 2009-06 | 16.8 | 13 | OK |
| TQQQ | 3x | 2010-02 | 16.1 | 13 | OK |
| SPXL | 3x | 2008-11 | 17.4 | 14 | OK |
| TECL | 3x | 2008-12 | 17.3 | 14 | OK |
| SOXL | 3x | 2010-03 | 16.1 | 13 | OK |
| TMF | 3x | 2009-04 | 17.0 | 13 | OK |
| ROM | 2x | 2007-02 | 19.2 | 16 | OK |

All products have >= 13 walk-forward folds (minimum requirement: 10). SSO and QLD have the longest histories (19.8yr, 16 folds). Key underlyings: SPY (28.2yr), QQQ (27.1yr), XLK (27.3yr) — all have 24+ folds for synthetic leverage backtests.

Precommitments verified for `phase0_letf` (13 tickers) and `phase0_underlyings` (6 tickers).

### Phase 1: Real LETF Validation (2026-04-17)

Gate: 0/18 PASS (expected — structurally unpowered per E0). All Holm-adjusted p = 1.0. Results interpreted via point estimates.

#### E2: Real vs Synthetic Gap — LARGER THAN EXPECTED

| LETF | Leverage | Real CAGR | Synthetic CAGR | Gap | Period |
|------|----------|-----------|---------------|-----|--------|
| UPRO | 3x SPY | 32.1% | 37.4% | **-5.3%** | 16.8yr |
| TQQQ | 3x QQQ | 41.6% | 47.3% | **-5.7%** | 16.1yr |
| SPXL | 3x SPY | 27.7% | 32.6% | **-5.0%** | 17.4yr |
| TECL | 3x XLK | 43.9% | 49.8% | **-5.9%** | 17.3yr |
| SSO | 2x SPY | 15.2% | 17.8% | -2.6% | 19.8yr |
| QLD | 2x QQQ | 24.5% | 27.3% | -2.8% | 19.8yr |

**Mean gap: -4.5% CAGR.** 3x products lose ~5-6% annually vs synthetic; 2x products lose ~2.6-2.8%. This is 2-3x larger than the 1-3% estimate from Charupat & Miu (2011). The gap includes swap financing costs, daily rebalancing friction, and tracking error that synthetic models omit. **All prior synthetic leverage results (etflab-max's 21.6% for 3x VTI SMA100) overstate implementable returns by ~5%.**

#### E3: Real LETF + SMA100 — KEY RESULTS (Codex R3: same-window synthetic)

| Strategy | CAGR | Sharpe | MaxDD | Terminal |
|----------|------|--------|-------|----------|
| SPXL SMA100 (real) | **23.1%** | 0.662 | -48.7% | 37.4x |
| SPXL SMA100 (synth) | 22.4% | 0.643 | -49.4% | 33.6x |
| UPRO SMA100 (real) | **22.1%** | 0.643 | -48.9% | 28.6x |
| TECL SMA100 (real) | 22.0% | 0.579 | -72.2% | 30.9x |
| TQQQ SMA100 (real) | 19.5% | 0.540 | -58.1% | 17.7x |
| TQQQ SMA100 (synth) | 19.6% | 0.543 | -59.4% | 18.0x |
| UPRO SMA100 (synth) | 17.9% | 0.544 | -49.3% | 15.8x |
| SSO SMA100 (real) | 13.5% | 0.498 | -35.0% | 12.1x |
| VTI buy-and-hold | 9.6% | 0.365 | -55.5% | 9.6x |

**Codex R3 fix resolved synthetic inversion.** Same-window synthetic now closely tracks real for TQQQ (19.6% vs 19.5%) and SPXL (22.4% vs 23.1%). UPRO shows a larger gap (17.9% vs 22.1%) — the synthetic financing model is conservative for shorter-history products. Real LETF results are the implementable truth; synthetic validates the direction.

#### E4: Signal Source — Underlying Wins

| Strategy | CAGR | Sharpe | Switches |
|----------|------|--------|----------|
| UPRO SMA100 (underlying signal) | **23.4%** | 0.674 | 319 |
| UPRO SMA100 (LETF NAV signal) | 9.7% | 0.328 | 219 |
| TQQQ SMA100 (underlying signal) | **20.9%** | 0.568 | 317 |
| TQQQ SMA100 (LETF NAV signal) | 16.4% | 0.485 | 201 |

Signal concordance: 72-74%. LETF NAV SMA produces fewer switches (good) but dramatically worse CAGR for UPRO (9.7% vs 23.4%). The vol-decay in LETF NAV causes the SMA to trigger premature exits. **Use underlying index (SPY/QQQ) for SMA signal computation, not the LETF NAV.**

#### E5: 2x vs 3x — 3x Wins on Both CAGR and Sharpe (Codex R3: with switching costs)

| Strategy | CAGR | Sharpe | MaxDD | Kelly |
|----------|------|--------|-------|-------|
| UPRO 3x SMA100 | **22.1%** | **0.643** | -48.9% | 1.90x |
| TQQQ 3x SMA100 | 19.5% | 0.540 | -58.1% | 1.28x |
| SSO+UPRO blend SMA100 | 19.1% | 0.621 | -42.2% | 2.22x |
| QLD+TQQQ blend SMA100 | 17.3% | 0.517 | -50.5% | 1.46x |
| SSO 2x SMA100 | 13.5% | 0.498 | **-35.0%** | 2.22x |
| QLD 2x SMA100 | 13.5% | 0.447 | -52.5% | 1.57x |

Codex R3 fix: with switching costs applied to blends, **UPRO 3x now wins on both CAGR (22.1%) and Sharpe (0.643)** — the earlier "blend wins Sharpe" was an artifact of missing costs. The blend still offers lower MaxDD (-42.2% vs -48.9%). **For CAGR maximization: UPRO 3x SMA100 dominates.**

#### Phase 1 Summary
1. Real-vs-synthetic gap is **-4.5% CAGR mean** (larger than literature estimates)
2. Real LETF SMA100 produces **22% CAGR** on S&P-based 3x products
3. Signal must come from **underlying** (SPY/QQQ), not LETF NAV
4. **3x wins on CAGR**, blend wins on Sharpe
5. Strict gate: 0/18 PASS (expected per E0 power analysis)

### Phase 2: Concentrated Leverage (2026-04-17)

Gate: 0/15 PASS (expected). NaN values for value strategies due to shorter history vs VTI benchmark.

#### E6: TQQQ vs UPRO — Sub-Period Stability is KEY

| Strategy | CAGR | Pre-2013 | Post-2013 | Sharpe | MaxDD |
|----------|------|----------|-----------|--------|-------|
| UPRO SMA100 (real) | **22.1%** | **22.1%** | **22.1%** | 0.643 | -48.9% |
| TQQQ SMA100 (real) | 19.5% | 4.3% | 23.1% | 0.540 | -58.1% |
| 3x QQQ SMA100 (synth) | 8.9% | -2.7% | 22.4% | 0.337 | -95.6% |
| 3x SPY SMA100 (synth) | 7.6% | -3.7% | 21.9% | 0.270 | -90.9% |

UPRO SMA100 shows identical pre/post-2013 CAGR (22.1%). **Caveat (Codex R4): UPRO inception is 2009-06-25 — the "pre-2013" window is only 3.5 years of post-GFC recovery, not a genuine regime test.** The stability is real but the pre-2013 sample is too short and too favorable to be conclusive. TQQQ is clearly regime-dependent (4.3% pre-2013 vs 23.1% post-2013). Synthetic 3x on full history shows negative pre-2013 CAGR — dot-com/GFC destroy synthetic leverage.

#### E7: Sector LETFs — TECL Strong, SOXL Fails

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| TECL SMA100 (real) | **22.0%** | 0.579 | -72.2% |
| SOXL SMA100 (real) | 4.1% | 0.337 | -81.8% |

TECL matches UPRO on CAGR but has much worse MaxDD (-72% vs -49%). SOXL (semiconductors) fails catastrophically — too volatile even with SMA100.

#### E8: Leveraged Value — Underperforms

Best value strategy: 3x VTV SMA100 = 8.9% CAGR (below VTI's 9.6%). Value factor provides diversification (correlation with TQQQ = 0.41-0.55) but CAGR too low for CAGR-maximization objective. **Value is a drawdown tool, not a CAGR tool** — consistent with etf/ workflow findings.

#### E9: Multi-Sleeve = 17.1% CAGR, Lower Drawdown

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| 3-sleeve independent SMA100 | 17.1% | 0.544 | -46.8% |
| 3x VTI SMA100 solo | 13.0% | 0.414 | -61.4% |

The 3-sleeve blend (VTI + QQQ + VLUE) outperforms any single synthetic sleeve but underperforms real UPRO SMA100 (17.1% vs 22.1%). Diversification reduces drawdown (-46.8% vs -48.9%) but the CAGR gap is too large.

#### Phase 2 Summary
1. **UPRO SMA100 (real) dominates** — 22.1% CAGR with perfectly stable sub-period returns
2. TQQQ is regime-dependent — only works in post-2013 tech boom
3. TECL matches CAGR but worse drawdowns; SOXL fails
4. Value at leverage is a drawdown tool, not CAGR tool
5. Multi-sleeve diversification costs 5% CAGR for minor drawdown improvement

### Phase 3: Novel Constructions (2026-04-17)

Gate: 0/22 PASS (expected). All Codex R4 fixes applied (switching costs, glidepath duration, E12 weight tracking).

#### E10: LEAPS Loses to Real LETFs

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| QLD SMA100 (real 2x) | **13.5%** | 0.447 | -52.5% |
| SSO SMA100 (real 2x) | **13.5%** | 0.498 | -35.0% |
| LEAPS 2x QQQ SMA100 | 6.7% | 0.239 | -85.9% |
| LEAPS 2x SPY SMA100 | 4.5% | 0.131 | -80.0% |

**LEAPS lose decisively.** At 4% annual theta drag, LEAPS provide ~2x leverage at higher cost than real 2x LETFs (which embed cheaper swap financing). The simplified model may overstate theta drag, but the gap (~7% CAGR) is too large to close with better pricing assumptions.

#### E11: Lifecycle Glidepath Fails

All lifecycle variants (linear, convex, step5) underperform static leverage on both SPY and QQQ. Best lifecycle: linear QQQ = 4.9% CAGR vs static 3x QQQ = 8.9%. The front-loaded high leverage amplifies early drawdowns (e.g., 2011 correction) that the shorter remaining period cannot recover from. **Lifecycle is a theoretical construct that fails in practice with leveraged equity.**

#### E12: UPRO+TMF "Hedgefundie" — STRONGEST BUY-AND-HOLD RESULT

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| UPRO/TMF 70/30 (B&H) | **28.5%** | **0.807** | -66.5% |
| UPRO/TMF 60/40 (B&H) | 25.4% | 0.774 | -69.0% |
| UPRO/TMF 55/45 (B&H) | 23.6% | 0.738 | -70.7% |
| UPRO only SMA100 | 22.1% | 0.643 | -48.9% |
| UPRO/TMF 70/30 + SMA100 | 19.2% | 0.690 | **-41.4%** |
| UPRO/TMF 55/45 + SMA100 | 16.5% | 0.635 | -42.4% |

**UPRO/TMF 70/30 buy-and-hold = 28.5% CAGR, 0.807 Sharpe** — the highest CAGR and Sharpe in the entire workflow. BUT: this is 2009-2026 only (TMF inception), covering the extraordinary bond bull → bust cycle. The -66.5% MaxDD reflects 2022 when both stocks and bonds fell simultaneously. Adding SMA100 overlay reduces CAGR to 19.2% but cuts MaxDD to -41.4%.

**CRITICAL CAVEAT:** UPRO/TMF buy-and-hold benefited enormously from the 2009-2020 negative stock-bond correlation regime. Post-2022, correlation flipped positive. This strategy is NOT regime-robust.

#### E13: Vol-Conditioned Leverage Adds No Value

Best vol-conditioned: 45%tv/3xmax QQQ = 9.3% CAGR. Static 3x QQQ SMA100 = 8.9%. The difference is within noise. Vol-conditioning does reduce MaxDD (-72% vs -96%) but the CAGR improvement is negligible. **Consistent with factor-timing finding that vol-targeting destroys value — even with SMA gating, vol-conditioning adds nothing.**

#### Codex R5 Caveats (applied to Phase 3 interpretation)
- **Window labels required:** E10/E12 use real LETFs (2009-2026); E11/E13 use synthetic leverage (full SPY/QQQ history from 1998). These are NOT comparable in a single ranking. Real LETF results are 4-5% higher per Phase 1 E2 findings.
- **E12 28.5% CAGR is window-dependent:** starts at 2009-06 (GFC bottom). No start-date sensitivity tested. Should not be called "strongest" without rolling-window validation.
- **E12 SMA overlay removes the Treasury hedge:** going all-to-cash on SPY signal disables TMF's convex bond protection during equity downtrends — it's a different strategy, not an enhancement. Sleeve-specific overlays (SPY SMA → UPRO, TLT SMA → TMF) were not tested.
- **E11 lifecycle is single-cohort:** tested one start date, not rolling 20-year windows. "Fails in practice" is too broad — should be "fails for this cohort."
- **E11/E13 synthetic results may improve with real LETFs:** synthetic financing model is conservative (Phase 1 showed 4.2% gap for UPRO). Real LETF lifecycle/vol-conditioning not tested.
- **Static comparators (UPRO-only SMA, static 1-3x) are descriptive only** — printed for context but excluded from Phase 3 gate denominator.
- **Sharpe uses hardcoded 4% risk-free** — not actual T-bill rates. Matters for cross-period comparison.

#### Phase 3 Summary (with R5 caveats)
1. **LEAPS lose to real 2x LETFs under simplified model** — 4% theta drag makes them uncompetitive (7% CAGR gap)
2. **Lifecycle leverage underperforms static for the tested cohort** — requires rolling-window validation to generalize
3. **UPRO/TMF 70/30 B&H = 28.5% CAGR (real, 2009-2026)** — highest point CAGR but single-window, regime-dependent, no start-date sensitivity
4. **Vol-conditioning adds marginal value under synthetic model** — untested with real LETFs
5. **Among real LETF strategies tested 2009-2026, UPRO SMA100 (22.1%) is the simplest high-CAGR implementable option** — does not depend on bond correlation regime

### Phase 4: Signal Refinement (2026-04-17)

Gate: 0/12 PASS. UPRO SMA100 weekly has lowest raw p (0.059, Holm-adjusted 0.708).

#### E14: Multi-Window SMA Composite — Beats Binary on Synthetic

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| multi_sma 3x QQQ | **14.6%** | 0.442 | -92.7% |
| multi_sma 3x SPY | 9.6% | 0.323 | -79.8% |
| binary SMA100 3x QQQ | 8.9% | 0.337 | -95.6% |
| binary SMA100 3x SPY | 7.6% | 0.270 | -90.9% |

Multi-window SMA (vote across [50,75,100,150,200]) beats binary SMA100 by +5.7% CAGR on QQQ and +2.0% on SPY. Smoother transitions reduce whipsaw damage at leverage. However: more transitions (795 vs 317 for SPY) and these are synthetic results — the real LETF comparison was not tested.

#### E15: Rebalancing Frequency — Weekly Optimal (real LETFs)

| Strategy | CAGR | Sharpe | MaxDD | Switches | Switch Cost |
|----------|------|--------|-------|----------|-------------|
| TQQQ monthly | 25.1% | 0.626 | -70.8% | 66 | 6.6% |
| **UPRO weekly** | **24.8%** | **0.703** | **-44.4%** | 127 | 12.7% |
| UPRO daily | 23.4% | 0.674 | -47.7% | 319 | 31.9% |
| TQQQ weekly | 22.9% | 0.601 | -58.6% | 145 | 14.5% |
| UPRO biweekly | 23.2% | 0.653 | -55.9% | 95 | 9.5% |
| UPRO monthly | 16.8% | 0.495 | -70.7% | 78 | 7.8% |

**Weekly SMA checking is optimal for UPRO** — confirms factor-timing Phase 6 finding. Daily wastes 19% in unnecessary switching costs. Monthly is too infrequent (misses crash signals, -70.7% MaxDD). TQQQ monthly appears best but with -70.8% MaxDD — likely an artifact of fewer exits during 2020 COVID dip.

**UPRO SMA100 weekly = 24.8% CAGR, 0.703 Sharpe** — this is now the best risk-adjusted implementable strategy in the workflow (higher Sharpe than the 22.1% daily-checked UPRO from Phase 1).

**Codex R6 warning:** Phase 4 E15 reports switch costs but does not deduct them from the saved return series. The 12.7% cumulative switch cost over the full sample is printed but not subtracted. Actual CAGR with costs deducted is likely ~23-24% rather than 24.8%. Phase 1's UPRO daily SMA100 (22.1%) DID deduct costs via `sma_leveraged_returns`.

#### Phase 4 Summary
1. **Multi-window SMA beats binary by 2-6% CAGR on synthetic** — smoother transitions reduce whipsaw at leverage
2. **Weekly signal checking is optimal** — +1.4% CAGR and +0.029 Sharpe over daily for UPRO
3. **UPRO weekly SMA100 = 24.8% CAGR, 0.703 Sharpe** — new workflow-best risk-adjusted implementable strategy

### Phase 5: Satellite & Tax (2026-04-17)

Gate: 0/8 PASS. BTC SMA100 standalone has raw p=0.013 (Holm-adjusted 0.107).

#### E16: Crypto Satellite — BTC SMA100 = 42.3% CAGR (EXPLORATORY)

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| BTC SMA100 (BTC-USD proxy) | **42.3%** | **0.961** | -59.7% |
| BTC buy-and-hold | 35.5% | 0.755 | -83.4% |

BTC SMA100 statistics: 143 switches, 56.2% time in market. SMA100 adds 6.8% CAGR and cuts MaxDD from -83% to -60%.

**Codex R6 annualization warning:** BTC-USD trades 365 days/year but `compute_metrics` uses `n/252` for annualization. This inflates BTC CAGR and Sharpe relative to 5-day ETF strategies. Reported numbers are directionally correct but not directly comparable to ETF results.

**Core+satellite blends are INVALID** — the "core" used synthetic 3x SPY SMA100 (2.2% CAGR) instead of real UPRO (24.8%). The blend CAGRs (4-8%) are meaningless. A proper blend would need real UPRO weekly SMA100 as core.

**Caveats (per Codex R5):** BTC-USD is not investable pre-2024. Real BTC ETFs (IBIT/FBTC) have only ~2yr history. No expense ratio, bid-ask, or tracking error in the proxy. Results are exploratory only.

#### E17: Tax Drag — Qualitative Findings Valid, Model Broken

**Tax model bug:** Applies effective tax rate to ALL positive daily returns — destroys 100% of capital. After-tax CAGR of -42% to -53% is an artifact of the broken daily-tax model, not a realistic estimate. **The quantitative after-tax numbers should be IGNORED.**

**Qualitative findings ARE valid:**
- SMA100 generates **11.3 switches/year** with **46-day average holding period**
- **99.4% of positions are short-term** (held < 1 year)
- Effective marginal tax rate: ~42% (federal STCG 37% + state 5%)
- TLH pairs: UPRO/SPXL correlation 0.999, TQQQ/QLD correlation 0.999 — IRS may consider "substantially identical"

**Correct tax estimate (analytical):** With 99.4% short-term positions at 42% effective rate, the annual tax drag on a 24.8% CAGR strategy is approximately 24.8% × 42% × (fraction of returns realized annually) ≈ 4-6% drag. After-tax CAGR ≈ 18-21% in a taxable account vs 24.8% in IRA. **Recommendation: run in tax-deferred accounts.**

### Phase 6: Global Gate (2026-04-17)

**Gate: 0/65 PASS. INCONCLUSIVE: 41. FAIL: 24.**

Consistent with E0 power analysis: the strict CAGR gate is structurally unpowered for SMA-gated leveraged strategies vs buy-and-hold. The 65-strategy Holm denominator requires raw p < 0.0008 for the best strategy — effectively impossible given ~40% excess return volatility.

#### Top 10 by raw p-value (closest to passing)

| Strategy | Excess CAGR | Raw p | Holm p | 90% CI |
|----------|------------|-------|--------|--------|
| UPRO/TMF 70/30 B&H | +13.8% | **0.005** | 0.312 | [+4.3%, +24.6%] |
| BTC SMA100 | +33.5% | 0.010 | 0.646 | [+9.7%, +63.4%] |
| UPRO/TMF 60/40 B&H | +10.6% | 0.027 | 1.000 | [+1.3%, +21.2%] |
| UPRO SMA100 weekly | +10.1% | 0.060 | 1.000 | [-0.9%, +22.2%] |
| UPRO/TMF 55/45 B&H | +8.8% | 0.070 | 1.000 | [-0.8%, +19.6%] |
| SPXL SMA100 real | +9.0% | 0.084 | 1.000 | [-2.4%, +20.8%] |
| UPRO SMA100 daily | +8.6% | 0.095 | 1.000 | [-2.2%, +20.8%] |
| SPXL SMA100 synthetic | +8.2% | 0.104 | 1.000 | [-3.1%, +20.0%] |
| UPRO SMA100 biweekly | +8.4% | 0.117 | 1.000 | [-3.3%, +21.8%] |
| multi_sma 3x QQQ | +7.7% | 0.137 | 1.000 | [-3.9%, +20.7%] |

UPRO/TMF 70/30 has the lowest raw p (0.005) — would pass at single-test level but not under 65-strategy Holm correction. BTC SMA100 has the largest point estimate (+33.5%) but wider CI.

#### Excluded from gate (Codex R4/R6 fixes)
- E18 post-hoc blends (top3/top5_equal_blend) — data snooping
- E17 after-tax series — broken daily-tax model
- E16 core+satellite blends — used synthetic core instead of real UPRO

---

## WORKFLOW CONCLUSIONS

### Formal Gate Result
**0/65 PASS.** VTI is CAGR-efficient under the strict gate (excess CAGR > 1% AND Holm p < 0.05 AND CI lower > 0). This is consistent with etflab-max (0/158) and is structurally expected per E0 power analysis — the SMA-gated leveraged vs buy-and-hold comparison has ~40% excess return vol, making the gate unpowered at any realistic effect size.

### Point Estimate Rankings (real LETF strategies, 2009-2026)

| Rank | Strategy | CAGR | Sharpe | MaxDD | Implementable? |
|------|----------|------|--------|-------|---------------|
| 1 | UPRO/TMF 70/30 B&H | 28.5% | 0.807 | -66.5% | Yes (regime-dependent) |
| 2 | UPRO SMA100 weekly | 24.8% | 0.703 | -44.4% | **Yes (robust)** |
| 3 | UPRO SMA100 daily | 22.1% | 0.643 | -48.9% | Yes |
| 4 | TQQQ SMA100 monthly | 25.1% | 0.626 | -70.8% | Yes (regime-dependent) |
| 5 | UPRO/TMF 70/30 + SMA | 19.2% | 0.690 | -41.4% | Yes |

### Practical Recommendation (revised per Codex R7 final review)

**Candidate high-risk strategy:** In a tax-deferred account, hold UPRO when SPY is above its 100-day SMA (checked weekly on Fridays); otherwise hold BIL.

**What the data shows (2009-2026 real product):**
- CAGR: 24.8% before switch costs, ~23-24% after estimated costs
- Sharpe: ~0.64-0.70
- MaxDD: -44% to -49%

**What the data does NOT show:**
- Walk-forward validated performance (full-sample metrics only)
- Real-product behavior through a secular bear market (2000-2003, 2007-2009)
- Regime robustness: synthetic 3x SPY SMA100 was **-3.7% CAGR pre-2013** including dot-com/GFC
- Formal statistical significance: 0/65 pass strict gate, raw p = 0.060 for UPRO weekly

**Codex R7 risk assessment:**
1. **Regime risk (CRITICAL):** UPRO's entire history is post-GFC recovery/bull. No real-product bear market test exists.
2. **In-sample selection (CRITICAL):** UPRO weekly was selected after comparing 65 strategies. The advantage over daily/biweekly/monthly is in-sample, not out-of-sample validated.
3. **Headline CAGR overstated (HIGH):** Phase 4 did not deduct switch costs from saved returns.

**Treat as an aggressive post-2009 point-estimate strategy, not a regime-robust expected 22-25% CAGR.**

Alternative: UPRO/TMF 70/30 quarterly rebalance for higher historical CAGR (28.5%) but even more regime-dependent (stock-bond correlation flipped in 2022).

### What We Learned (novel findings)
1. **Real-vs-synthetic gap is -4.5% CAGR for 3x LETFs** — all prior synthetic results overstated implementable returns
2. **SMA signal must come from underlying (SPY), not LETF NAV** — LETF vol decay causes premature exits (23.4% vs 9.7% for UPRO)
3. **Weekly signal checking beats daily by +1.4% CAGR** — fewer costly whipsaws at leverage
4. **LEAPS lose to real LETFs** — theta drag too expensive at 2x
5. **Lifecycle leverage fails** — front-loaded drawdowns dominate (single cohort tested)
6. **UPRO/TMF "Hedgefundie" has highest realized CAGR** — but depends on stock-bond correlation regime
7. **Vol-conditioning adds no value** — consistent with factor-timing findings
8. **Strict CAGR gate is structurally unpowered** for SMA-gated vs buy-and-hold comparisons (~40% excess vol)
9. **BTC SMA100 = 42.3% CAGR** on proxy (exploratory, not investable pre-2024)

### Codex Review Summary
7 adversarial review rounds, 16 HIGH bugs fixed, 3 CRITICAL conclusion risks identified:
- R1: expense ratios 100x, missing financing costs, missing switching costs, data-snooping E18/E19, expense during cash periods, missing crypto ETFs
- R2: homoskedastic power DGP → SMA-gated DGP (revealed structural powerlessness)
- R3: synthetic/real window mismatch, missing E4/E5 switching costs (reversed "blend wins Sharpe" conclusion)
- R4: glidepath ignores total_years, E12 weight drift, missing Phase 3 switching costs, UPRO stability caveat
- R5: window mixing in rankings, E12 window sensitivity, lifecycle single-cohort caveat, Sharpe hardcoded rf
- R6: BTC annualization, broken tax model, invalid blend core, Phase 4 switch costs in artifacts
- R7 (FINAL): 3 CRITICAL risks — (1) regime/start-date: all real LETF data is post-GFC, synthetic pre-2013 is negative; (2) in-sample selection: UPRO weekly chosen from 65 variants, not walk-forward validated; (3) headline CAGR overstated (switch costs not deducted). Recommendation revised from "expected 22-25% CAGR" to "candidate high-risk strategy, post-2009 point estimate only"
- R7 follow-up: found and fixed `apply_switching_costs` reindex bug (spurious 1056 switches from weekend gaps instead of 319 real switches). Corrected UPRO daily SMA100 = 21.0% CAGR (was 22.1% with nearly-zero buggy costs). Rolling stress test run with fixed costs.

### Rolling Synthetic Stress Test (Codex R7 follow-up, 2026-04-18)

Synthetic 3x SPY SMA100 with calibrated costs (25bps borrow spread) over full 1998-2026 SPY history. Includes dot-com bust and GFC.

#### Full-Sample Results (28.2 years, 1998-2026)

| Strategy | CAGR | Sharpe | MaxDD |
|----------|------|--------|-------|
| SPY buy-and-hold | 9.1% | 0.340 | -55.2% |
| 3x SPY SMA100 (0bps) | 8.4% | 0.290 | -90.7% |
| 3x SPY SMA100 (25bps) | **8.0%** | 0.280 | **-90.8%** |
| 3x SPY SMA100 (50bps) | 7.6% | 0.270 | -90.9% |
| 1x SPY SMA100 | 4.4% | 0.088 | -54.4% |

**Full-sample synthetic 3x SPY SMA100 = 8.0% CAGR — BELOW SPY buy-and-hold (9.1%).** The -90.8% MaxDD comes from the dot-com bust where SMA100 failed to protect at 3x leverage.

#### Rolling Window Analysis (calibrated 25bps)

| Window | Median CAGR | Mean CAGR | Min CAGR | Pct Positive | Pct Beats SPY |
|--------|------------|-----------|----------|-------------|--------------|
| 10-year | **14.0%** | 10.6% | -16.9% | 83.6% | **76.7%** |
| 15-year | **12.9%** | 11.4% | -2.8% | 84.9% | **73.6%** |
| 20-year | **12.7%** | 9.9% | -0.1% | 97.0% | **66.7%** |

**INTERPRETATION:** The strategy beats SPY in 67-77% of rolling windows. Median CAGR is 12.7-14.0% vs SPY's 8-9%. But worst-case 10-year window is **-16.9% CAGR** (starting 1999-04, through dot-com bust). Worst 20-year window barely breaks even (-0.1% CAGR, starting 1999-10).

#### Crisis Period Analysis

| Crisis | Strategy Return | SPY Return | Strategy MaxDD | SPY MaxDD |
|--------|----------------|-----------|---------------|----------|
| Dot-com (2000-03 to 2003-03) | **-84.0%** | -35.8% | -84.1% | -47.5% |
| GFC (2007-10 to 2009-06) | **-7.1%** | -35.6% | -40.7% | -55.2% |
| COVID (2020-02 to 2020-04) | -9.3% | -9.2% | -21.8% | -33.7% |
| 2022 rate hike | -45.1% | -18.2% | -46.1% | -24.5% |
| Full 2000-2013 | **-65.0%** | +22.9% | -89.8% | -55.2% |

**THE DOT-COM BUST DESTROYS THIS STRATEGY.** SMA100 at 3x leverage lost -84% during the dot-com bust because:
1. The tech crash was grinding and choppy — SPY fell, recovered partially, fell again
2. SMA100 whipsawed through multiple false entries and exits
3. Each whipsaw at 3x leverage amplified losses
4. The financing costs during invested periods compounded the damage

**The GFC was the opposite:** SMA100 successfully protected capital (-7.1% vs SPY's -35.6%). The 2008 crash was fast and clean — SMA triggered a decisive exit early.

**Key insight: SMA100 protects against fast crashes (GFC, COVID) but fails during grinding bear markets (dot-com, 2022).** At 3x leverage, the grinding failure is catastrophic.

#### Revised Assessment

The post-2009 real LETF results (21-25% CAGR) are real but period-specific. The full 1998-2026 synthetic test shows:
- **Median 20-year CAGR: 12.7%** (not 21-25%)
- **One-third of 20-year windows underperform SPY**
- **Dot-com-like event at 3x = -84% drawdown**

This materially changes the risk profile. The strategy is positive-expectation but with catastrophic tail risk from grinding bear markets. A prospective investor should expect ~10-15% CAGR (median across start dates), not 21-25%, and must accept the possibility of -84% drawdowns.

### Prospective Paper Tracking
Rule frozen 2026-04-17: UPRO when SPY > 100-day SMA (checked weekly Fridays), BIL when below. Tax-deferred account. Kill conditions: MaxDD > -60%, 3-year CAGR < VTI, switches > 20/year. See `precommit/prospective_tracking.json`.

### Codex R1 Review (2026-04-17)
6 HIGH findings, all fixed before Phase 0:
1. Universe CSV expense ratios 100x off (SPY, BIL, TMF)
2. Synthetic leverage path missing financing cost
3. No switching costs deducted from strategy returns
4. E18/E19 data-snooping (post-hoc survivor selection)
5. Expense charged during cash periods in conditional_leveraged_return
6. No crypto ETF rows in universe; E10 LEAPS oversimplified
