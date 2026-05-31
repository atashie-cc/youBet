# Contamination rerun — market-cap split-adjust bug (2026-05-30)

**Status: `value_EY` + both composites VERIFIED on complete 202-date legs
(2009-08…2026-05), fidelity-checked. The ML corrected point estimate could not be
computed (environment-blocked: 8 GB RAM exhaustion → numpy MemoryError, plus the
600 s per-call cap) but the ML contamination is BOUNDED. Robust conclusions hold
regardless of the ML numbers. Every figure below was read from literal tool output;
ML corrected values are labeled UNMEASURED.**

## Trigger

The `individual-stocks-snp500` round-2 review found that `mcap = adjusted_price ×
as-reported_shares` understates market cap by each stock's cumulative split factor
(yfinance `auto_adjust=True` divides historical prices by *future* splits; EDGAR
shares are as-reported). That flipped its `evebitda_yield` +0.367 → −0.057. The
final report flagged that **stock-selection's `value_EY` (+0.351) may share the
same bug.** This rerun tests that hypothesis on every stock-selection strategy.

## Code audit (VERIFIED by reading source)

The bug lives in the **shared engine** (`src/youbet/stock/`), anywhere
`last_price (adjusted) × shares (EDGAR)` forms an mcap-denominated value ratio:

| Strategy | Site | Exposure |
|---|---|---|
| `value_earnings_yield` | `rules.ValueScore`: `ttm_ni/(shares×price)` | **FULL** |
| `quality_value_zsum` | `composites.QualityValue`: `z_ey` leg (1 of 3) | partial |
| `value_profitability` | `composites.ValueProfitability`: `z_ey` leg (1 of 2) | partial |
| `ml_gkx_*_v20` | `gkx_chars._fundamentals_ratios`: `ep_ttm,sp_ttm,bm` (3 of 20 feats) | partial |
| `magic_formula` | `composites.MagicFormula`: EBIT/assets, ROIC — **no price** | clean |
| `quality_roe_ttm`, `gross_profitability` | fundamentals only | clean |
| `momentum_252_21`, `lowvol_252` | price **ratio** / returns — split-invariant | clean |

Mechanism (VERIFIED): the bug multiplies each stock's value yield by its
cumulative split factor, so high-split names sort spuriously "cheap." Raw/adjusted
price ratios at 2021-01: **CMG 50×, NVDA 40×, BKNG 25×, GOOGL/AMZN 20×** — exactly
the mega-cap winners of 2010–2026, so a value tilt loaded on them and looked good.

## VERIFIED result 1 — `value_EY` (single signal, full contamination)

Reconstructed on the identical (date,ticker) set, contaminated vs corrected mcap.
Only the price basis differs. Locked backtester (60/24/12, monthly, top-decile,
mcap-bucketed costs, T-bill, SPY), 10k stationary block bootstrap.

| | Sharpe-of-excess | raw p | 90% CI | strat Sharpe |
|---|---:|---:|---|---:|
| Reported (stock-selection R9) | +0.351 | 0.064 | [−0.028, +0.740] | — |
| Contaminated reconstruction | **+0.361** | 0.056 | [−0.019, +0.756] | 0.869 |
| **Corrected (raw-price mcap)** | **−0.098** | 0.671 | [−0.492, +0.280] | 0.596 |

The contaminated reconstruction reproduces the reported +0.351 (validates the
method); the corrected −0.098 matches the independent `earnings_yield_v2` clone in
`individual-stocks-snp500` (−0.100). (A separate legs-panel reconstruction gives
−0.125 — slightly different universe/coverage, same sign.) **The entire +0.45 was
the split bug. `value_EY` — the workflow's "closest to passing" — is RETRACTED;
pure value *underperforms* SPY** (strat Sharpe 0.596 vs 0.850).

## VERIFIED result 2 — composites (complete 202-date legs, 10k bootstrap vs SPY)

| Strategy | contaminated (R9) | **corrected** | Δ | raw p | hAdj | 90% CI | strat Sharpe | gate |
|---|---:|---:|---:|---:|---:|---|---:|:--:|
| `quality_value_zsum` | +0.068 | **+0.004** | −0.064 | 0.498 | 1.000 | [−0.388, +0.395] | 0.825 | FAIL |
| `value_profitability` | −0.073 | **−0.109** | −0.036 | 0.672 | 1.000 | [−0.505, +0.290] | 0.756 | FAIL |
| `magic_formula` (clean control, recon) | +0.093 | +0.078 | — | 0.375 | — | [−0.325, +0.490] | 0.823 | (fidelity PASS) |

**Both contaminated composites move toward/below zero, not up.** Correcting the
value (`z_ey`) leg *removes* the spurious mega-cap tilt that had been adding return:
`quality_value_zsum` drops from +0.068 to **+0.004** (essentially zero — its small
prior positive was partly the value bug; what remains is its clean ROE/gross-margin
legs, net ≈ 0), and `value_profitability` falls to −0.109. The `magic_formula`
clean-control reconstruction (+0.078 ≈ reported +0.093, a signal that should NOT
move) confirms fidelity. This **refines** the earlier structural guess ("composites
barely move / qv stays a weak positive"): the value leg was a *net contributor* to
qv, so removing it pushes qv to zero. Both fail the gate by a wide margin.

## ML — corrected point estimate NOT measured; contamination bounded as NON-trivial

The Phase-4b walk-forward could not complete in this environment. **Root cause
(diagnosed): RAM starvation, not science** — the box has 8 GB and ~8 zombie python
processes from earlier failed launches held ~3.5 GB, so every attempt died with a
`numpy MemoryError`. After killing the zombies (RAM 0.5 GB → 3.8 GB free) and adding
a disk-backed feature cache (`tmp/ml_two.py` → `tmp/featcache_full/`), the
MemoryError stopped recurring and the cache advanced to ~184/233 feature-dates, but
the 600 s per-call cap + restart-from-fold-0 re-traversal meant the full 15-fold fit
did not finish in-session. Rather than assert an unmeasured number, I bounded the
contamination directly (cheap and conclusive):

The 3 affected GKX features (`ep_ttm`, `sp_ttm`, `bm`) are **rank-transformed
cross-sectionally per date** before the model sees them (`_rank_to_uniform` in
`ml_ranker.py`), so the *only* channel by which the bug reaches the model is by
changing those features' cross-sectional ranks. Measured per-date Spearman
rank-correlation (contaminated vs corrected ranks) over all 202 dates:

| feature | mean Spearman | median | p05 | min | mean norm. rank-shift | frac dates < 0.90 |
|---|---:|---:|---:|---:|---:|---:|
| ep_ttm | 0.841 | 0.857 | 0.686 | 0.510 | 0.088 | 0.74 |
| sp_ttm | 0.870 | 0.868 | 0.788 | 0.682 | 0.077 | 0.72 |
| bm | 0.860 | 0.875 | 0.720 | 0.590 | 0.079 | 0.67 |

**The contaminated value features are MATERIALLY reordered** (~0.85 mean Spearman;
~70% of dates below 0.90; ~8% mean rank displacement). This **revises** the earlier
hand-wave that "3-of-20 features ⇒ small change": the affected features' ranks shift
non-trivially, so the corrected ML excess Sharpe could move meaningfully **in an
unknown direction**. It is NOT safe to assume corrected lightgbm stays near +0.259.

| Strategy | contaminated (R9) | corrected |
|---|---:|---|
| `ml_gkx_lightgbm_v20` | +0.259 | **UNMEASURED — genuinely uncertain (could move materially)** |
| `ml_gkx_elasticnet_v20` | −0.215 | **UNMEASURED — genuinely uncertain** |

## Robust conclusions (independent of the ML numbers)

1. **Gate verdict UNCHANGED: 0/11 pass.** Nothing passed before; the corrections
   move value/composite estimates toward or below zero, and `value_EY` (the closest)
   now fails clearly. No correction can create a gate pass.
2. **Directional positives drop from 5/11 to 2–4/11.** `value_EY` flips negative
   (+0.351 → −0.098) and `quality_value_zsum` collapses to ≈0 (+0.004). The only
   **confirmed** surviving positives are `quality_roe_ttm` (+0.242, clean, unchanged)
   and `magic_formula` (+0.093, clean, unchanged) — both quality, neither value.
   `ml_gkx_lightgbm_v20` (+0.259 contaminated) is a possible 3rd–4th but its corrected
   value is unmeasured and its inputs are materially reordered, so it is genuinely
   uncertain.
3. **Pure value is dead on free large-cap PIT data (2010–2026).** `value_EY` −0.098,
   `value_profitability` −0.109 (VERIFIED), and the value leg's net contribution to
   `quality_value_zsum` was positive only *because of the bug* (corrected qv → ≈0).
   No value construction survives. The robust directional edge is **quality**, with
   ML unresolved.

## Engine fix still required

The bug is in committed engine code (`rules.ValueScore`, `composites`
QualityValue/ValueProfitability `z_ey`, `gkx_chars._fundamentals_ratios`). Until
raw-price mcap is baked in, any future value/ML run on this engine is
re-contaminated. The cost model's mcap bucketing (`data.compute_market_caps`) also
uses adjusted-price mcap — conservative (over-states costs for high-split names),
left as-is.

## To finish ML (when environment is stable)

`tmp/ml_one.py {elasticnet,lightgbm}` (one model per process, raw-price patch on
`_fundamentals_ratios` + date-keyed feature cache; saves its returns parquet the
instant it completes) needs ~17 min uninterrupted per model. Then
`tmp/joint_holm_corrected.py` assembles the authoritative corrected Joint Holm N=11
vs canonical SPY (6 clean saved returns + value_EY/qv/vp corrected + the 2 ML parquets).
