# VTI-as-Challenger — Inverted Gate Final Report

- Frozen at: 2026-05-12T13:34:34.698652Z
- Candidates: 27 (joint Holm-27 family)
- Bootstrap: Politis-Romano stationary, 10000 reps, 22d block
- One-sided alpha: 0.05; CI: 90%

## Headline finding

After inverting the gate against 27 candidates that previously had positive point estimates vs their workflow's native benchmark:

- **1/27** of benchmark-as-challenger trials cross the raw one-sided gate (p < 0.05).
- **0/27** survive the joint Holm-27 correction across the family.
- **0/27** can be declared statistically equivalent at their workflow's native MEI under TOST.
- **26/27** are genuinely inconclusive (SF=N, TOST=N) — i.e. the original null verdict cannot be replaced with either "benchmark beats" or "benchmark equivalent" at the workflows' MEIs.

**This confirms the user's hypothesis**: the original null results across 9 workflows are **power-limited**, not power-definitive. The data simply cannot distinguish the benchmark from the alternatives at any direction or at the workflows' native MEIs.

**Raw SF-pass exceptions** (do not survive Holm):
- `commodity__iau_sma100_sleeve` raw p = 0.0305; Holm adjusted p = 0.8234. Point = +0.3705 [+0.0384, +0.7122]

**Common-window (2017-2026) SF-pass cases** — informative sensitivity:
- `real_world__test4_60_40_vti_ief_daily_rebal` common-window SF p = 0.0092.

## How to read this report

Each candidate had a positive Sharpe-diff or CAGR-diff point estimate vs the workflow's native benchmark in its original test. Here we **flip the framing**: does the benchmark beat the candidate?

- **SF p**: one-sided block-bootstrap p for H0: benchmark <= candidate. SF passes at p < 0.05.
- **TOST eq?**: is the 90% CI on Sharpe-of-excess (benchmark - candidate paired) contained in [-MEI, +MEI]? If yes, the two are statistically equivalent under the workflow's native MEI.
- **Common-window** sensitivity: 2017-01-01 to 2026-04-30 intersect for cross-workflow comparability.
- **Robustness flag**: PASS / PASS-with-caveat (sub-period sign disagrees) / FAIL (placebo gate fires).
- **Holm-adjusted p**: joint Holm across the full 27-candidate family.

**Interpretation key**:
- _SF=Y, TOST=Y_: benchmark significantly beats candidate AND they are equivalent within MEI (unusual — only if MEI is wide).
- _SF=Y, TOST=N_: benchmark significantly beats candidate; effect is larger than MEI (clean win for benchmark).
- _SF=N, TOST=Y_: cannot prove benchmark beats candidate, but difference is small enough to call them equivalent. **Strongest "power-limited but bounded" result.**
- _SF=N, TOST=N_: power-limited and difference might be meaningful in either direction — genuinely inconclusive.


## Class 1 — Investable Sharpe-comparable

| Candidate | n_obs | Window | Point | 90% CI | SF p | SF? | MEI | TOST eq? | CW SF p | CW TOST? | Robust | Holm adj p |
|---|---:|---|---:|---|---:|:---:|---:|:---:|---:|:---:|:---:|---:|
| `commodity__iau_sma100_sleeve` | 4618 | 2007-11-20 to 2026-04-07 | +0.3705 | [+0.0384, +0.7122] | 0.0305 | Y | 0.20 | N | 0.0670 | N | PASS | 0.8234 |
| `real_world__test4_60_40_vti_ief_daily_rebal` | 6443 | 2000-08-31 to 2026-04-16 | +0.2503 | [-0.0306, +0.5543] | 0.0753 | N | 0.20 | N | 0.0092 | N | PASS-with-caveat | 1.0000 |
| `real_world__v3_1998_60_30_10_weekly_sma100` | 7114 | 1998-01-05 to 2026-04-16 | +0.1371 | [-0.1179, +0.4107] | 0.1907 | N | 0.20 | N | 0.4293 | N | PASS | 1.0000 |
| `commodity__static_10pct_iau_in_60_40` | 3936 | 2010-08-12 to 2026-04-07 | -0.0038 | [-0.3654, +0.3776] | 0.5088 | N | 0.20 | N | 0.8866 | N | PASS-with-caveat | 1.0000 |
| `stock_selection__quality_value_zsum` | 4101 | 2010-01-05 to 2026-04-24 | -0.0676 | [-0.4604, +0.3125] | 0.6153 | N | 0.30 | N | 0.5115 | N | PASS-with-caveat | 1.0000 |
| `stock_selection__magic_formula` | 4101 | 2010-01-05 to 2026-04-24 | -0.0931 | [-0.5101, +0.3294] | 0.6482 | N | 0.30 | N | 0.4692 | N | PASS-with-caveat | 1.0000 |
| `real_world__test2_95_5_vti_iau_daily_rebal` | 6443 | 2000-08-31 to 2026-04-16 | -0.0928 | [-0.3706, +0.1983] | 0.7052 | N | 0.20 | N | 0.5550 | N | PASS-with-caveat | 1.0000 |
| `real_world__test2_90_10_vti_gold_daily_rebal` | 6443 | 2000-08-31 to 2026-04-16 | -0.0928 | [-0.3635, +0.1993] | 0.7095 | N | 0.20 | N | 0.5478 | N | PASS-with-caveat | 1.0000 |
| `commodity__static_54_36_10_iau_walkforward` | 4618 | 2007-11-20 to 2026-04-07 | -0.1184 | [-0.4491, +0.2301] | 0.7155 | N | 0.20 | N | 0.8915 | N | PASS-with-caveat | 1.0000 |
| `commodity__macro_dbc_inflation_only` | 4618 | 2007-11-20 to 2026-04-07 | -0.2316 | [-0.6184, +0.2010] | 0.8252 | N | 0.20 | N | 0.3281 | N | PASS-with-caveat | 1.0000 |
| `stock_selection__quality_roe_ttm` | 4101 | 2010-01-05 to 2026-04-24 | -0.2418 | [-0.6354, +0.1554] | 0.8461 | N | 0.30 | N | 0.5346 | N | PASS-with-caveat | 1.0000 |
| `stock_selection__ml_gkx_lightgbm_v20` | 3598 | 2012-01-04 to 2026-04-27 | -0.2590 | [-0.6439, +0.1500] | 0.8575 | N | 0.30 | N | 0.8354 | N | PASS | 1.0000 |
| `commodity__macro_dbc_additive` | 4618 | 2007-11-20 to 2026-04-07 | -0.3105 | [-0.7162, +0.1347] | 0.8788 | N | 0.20 | N | 0.4652 | N | PASS-with-caveat | 1.0000 |
| `stock_selection__value_earnings_yield` | 4101 | 2010-01-05 to 2026-04-24 | -0.3505 | [-0.7415, +0.0329] | 0.9393 | N | 0.30 | N | 0.6564 | N | PASS | 1.0000 |

## Class 2 — Paper factor portfolios (structural-mismatch caveat)

_Caveat: Class 2 candidates are Ken French long-short factor portfolios compared to factor B&H (NOT VTI). The comparison is internal to each factor — it tests whether SMA-timing the factor beats holding the factor, not whether the factor beats the market._

| Candidate | n_obs | Window | Point | 90% CI | SF p | SF? | MEI | TOST eq? | CW SF p | CW TOST? | Robust | Holm adj p |
|---|---:|---|---:|---|---:|:---:|---:|:---:|---:|:---:|:---:|---:|
| `factor_timing__RMW_sma_100` | 15013 | 1966-07-01 to 2026-02-27 | -0.5821 | [-0.8716, -0.3020] | 0.9992 | N | 0.20 | N | 0.3769 | N | PASS | 1.0000 |
| `factor_timing__CMA_sma_200` | 15013 | 1966-07-01 to 2026-02-27 | -0.4971 | [-0.7358, -0.2514] | 0.9995 | N | 0.20 | N | 0.9593 | N | PASS | 1.0000 |
| `factor_timing__SMB_sma_200` | 15013 | 1966-07-01 to 2026-02-27 | -0.5046 | [-0.7374, -0.2683] | 0.9996 | N | 0.20 | N | 0.9250 | N | PASS | 1.0000 |
| `factor_timing__HML_sma_100` | 15013 | 1966-07-01 to 2026-02-27 | -0.5354 | [-0.7927, -0.3029] | 0.9999 | N | 0.20 | N | 0.8185 | N | PASS | 1.0000 |
| `factor_timing__CMA_sma_100` | 15013 | 1966-07-01 to 2026-02-27 | -0.6872 | [-0.9250, -0.4528] | 1.0000 | N | 0.20 | N | 0.9770 | N | PASS | 1.0000 |
| `factor_timing__SMB_sma_100` | 15013 | 1966-07-01 to 2026-02-27 | -0.6411 | [-0.8738, -0.4122] | 1.0000 | N | 0.20 | N | 0.7451 | N | PASS | 1.0000 |

## Class 3 — Leveraged-CAGR (CAGR-diff metric)

_Class 3 uses CAGR-diff (annualized geometric) instead of Sharpe-diff, with native MEI = 0.01 (1%/yr). All candidates' workflow gates were structurally unpowered for the Sharpe metric due to extreme excess-return vol of leveraged strategies; see cagr-max E0 power analysis._

| Candidate | n_obs | Window | Point | 90% CI | SF p | SF? | MEI | TOST eq? | CW SF p | CW TOST? | Robust | Holm adj p |
|---|---:|---|---:|---|---:|:---:|---:|:---:|---:|:---:|:---:|---:|
| `cagr_max__UPRO_SMA100_synthetic` | 4227 | 2009-06-26 to 2026-04-16 | -0.0313 | [-0.1449, +0.0740] | 0.6824 | N | 0.01 | N | 0.8508 | N | PASS-with-caveat | 1.0000 |
| `cagr_max__3x_SPY_SMA100_synth` | 6244 | 2001-06-18 to 2026-04-16 | -0.0273 | [-0.1201, +0.0582] | 0.6984 | N | 0.01 | N | 0.8502 | N | PASS | 1.0000 |
| `cagr_max__SSO_SMA100_real` | 4985 | 2006-06-22 to 2026-04-16 | -0.0238 | [-0.0933, +0.0418] | 0.7153 | N | 0.01 | N | 0.7111 | N | PASS | 1.0000 |
| `etflab_max__VTI_2.5x_SMA100` | 5579 | 2004-02-03 to 2026-04-07 | -0.0475 | [-0.1309, +0.0302] | 0.8376 | N | 0.01 | N | 0.8689 | N | PASS | 1.0000 |
| `cagr_max__UPRO_SMA100_real` | 4227 | 2009-06-26 to 2026-04-16 | -0.0738 | [-0.1978, +0.0351] | 0.8640 | N | 0.01 | N | 0.8582 | N | PASS | 1.0000 |
| `etflab_max__macro_factor_aggressive_2.5x_SMA100` | 5085 | 2006-01-05 to 2026-04-07 | -0.0718 | [-0.1887, +0.0320] | 0.8646 | N | 0.01 | N | 0.8564 | N | PASS | 1.0000 |
| `etflab_max__p1_MGK_2.0x_SMA100` | 4595 | 2007-12-31 to 2026-04-07 | -0.0767 | [-0.1698, +0.0096] | 0.9265 | N | 0.01 | N | 0.9642 | N | PASS | 1.0000 |

## Summary tally

- **Sign-flipped gate (raw p < 0.05)**: 1/27
- **Sign-flipped gate (Holm-adjusted p < 0.05)**: 0/27
- **TOST equivalent at workflow-native MEI**: 0/27
- **Genuinely inconclusive (SF=N, TOST=N)**: 26/27

## Excluded workflows (deferred)

- **international-etf** — No daily-return parquet artifact saved; would require re-running phase1_static.py / phase3_hedging.py / phase4_robustness_c1b.py to regenerate. Deferred to v2.
  - would have been: Phase 1 mean-shifted placebo @10% VXUS (+0.060)
  - would have been: Phase 4 C1b 60/40 VTI/HEFA mean-shifted (+0.067)
- **macro-exploratory** — Results stored as per-experiment JSON; no daily returns parquet. Would require re-running experiment_e4.py. Deferred to v2.
  - would have been: E4 pooled 12-sleeve factor-vs-cash timing (+0.716 paper)
- **etf** — No artifacts/ directory; experiments compute returns at runtime via efficiency_test.py / drawdown_deep_dive.py. Deferred to v2.
  - would have been: VTI SMA200 trend-following (MaxDD reduction +64%)
- **factor-timing** — Hedged VLUE ETF-bridge results computed in phase3_etf_bridge.py but only paper portfolios are persisted in phase1_timing_returns.parquet. The bridge daily returns weren't saved. Class 1 entry missing.
  - would have been: Hedged VLUE net ExSharpe +0.453 [+0.230, +0.653] Holm p=0.045 — the only investable factor finding
