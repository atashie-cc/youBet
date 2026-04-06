# ETF Market Efficiency: Comprehensive Findings

## Summary

Tested 17 systematic strategies against buy-and-hold VTI using 52 Vanguard ETFs
from 2003-2026 with walk-forward validation (36mo train / 12mo test / 12mo step,
21 folds), block bootstrap with Holm correction, and realistic transaction costs.

**Result: No strategy passes the strict gate. VTI buy-and-hold is efficient against
all tested systematic approaches.**

## Strategy Categories Tested

### Rule-Based (no fitting)
- Vol targeting, trend following (200-day SMA), dual momentum (Antonacci GEM),
  sentiment extremes (VIX contrarian), vol risk premium (VIX - realized vol)

### Macro-Signal Driven (no fitting, pre-specified rules)
- Macro risk composite (yield curve + credit spread + VIX + PMI z-scores),
  asset class rotation (5 classes), factor timing (growth/value pairs)

### Cross-Sectional Momentum (no fitting)
- Momentum rotation (top 5 from 40 ETFs), sector rotation (top 3 from 10 sectors),
  full universe momentum (top 10 from 52 ETFs, class-constrained)

### Multi-Asset Allocation (minimal fitting)
- Risk parity (unleveraged inverse-vol across 5 class reps)

### ML Models (fitted, 10-fold rolling CV with 24-month train window)
- Logistic regression, Ridge regression, XGBoost, LightGBM, equal-weight ensemble,
  hierarchical ML (Ridge on class returns + within-class momentum)

## Final Results (ranked by excess Sharpe)

| # | Strategy | ExSharpe | 90% CI | Diagnostic |
|---|---|---|---|---|
| 1 | **trend_following** | **+0.210** | [-0.080, +0.510] | INCONCLUSIVE-POS |
| 2 | risk_parity | +0.113 | [-0.109, +0.341] | INCONCLUSIVE-POS |
| 3 | full_universe_momentum | +0.072 | [-0.219, +0.374] | INCONCLUSIVE-POS |
| 4 | asset_class_rotation | +0.069 | [-0.016, +0.145] | INCONCLUSIVE-POS |
| 5 | factor_timing | +0.016 | [-0.056, +0.085] | INCONCLUSIVE |
| 6 | sector_rotation | +0.007 | [-0.220, +0.235] | INCONCLUSIVE-POS |
| 7 | dual_momentum | -0.019 | [-0.240, +0.195] | INCONCLUSIVE |
| 8 | momentum_rotation | -0.026 | [-0.289, +0.228] | INCONCLUSIVE-POS |
| 9 | sentiment_extremes | -0.029 | [-0.076, +0.027] | INCONCLUSIVE |
| 10 | vol_risk_premium | -0.041 | [-0.106, +0.033] | INCONCLUSIVE |
| 11 | macro_risk_composite | -0.048 | [-0.093, +0.001] | INCONCLUSIVE |
| 12 | vol_targeting | -0.086 | [-0.157, -0.018] | NEGATIVE |
| 13 | hierarchical_ml | -0.090 | [-0.254, +0.068] | INCONCLUSIVE |
| 14 | ml_ridge | -0.111 | [-0.212, -0.005] | NEGATIVE |
| 15 | ml_lightgbm | -0.114 | [-0.188, -0.029] | NEGATIVE |
| 16 | ml_xgboost | -0.116 | [-0.199, -0.022] | NEGATIVE |
| 17 | ml_ensemble | -0.128 | [-0.213, -0.033] | NEGATIVE |
| 18 | ml_logistic | -0.186 | [-0.313, -0.060] | NEGATIVE |

## Key Findings

### 1. Trend-following is the only strategy to exceed the 0.20 magnitude threshold
Excess Sharpe +0.210 with 200-day SMA (Faber GTAA-style). CI spans zero so it fails
the full strict gate, but point estimate is meaningful. Consistent with literature
showing trend-following has the strongest post-publication persistence.

### 2. ML models consistently destroy value
All 6 ML strategies have negative excess Sharpe. More thorough HP search (10-fold
rolling CV, expanded grids) made them WORSE, not better. With ~36 monthly training
samples per walk-forward fold, the models overfit to CV folds.

| Model | Simple CV | 10-Fold Rolling CV | Degradation |
|---|---|---|---|
| ml_logistic | -0.030 | -0.186 | -0.156 |
| ml_ridge | -0.083 | -0.111 | -0.028 |
| ml_xgboost | -0.097 | -0.116 | -0.019 |
| ml_lightgbm | -0.083 | -0.114 | -0.031 |
| ml_ensemble | -0.068 | -0.128 | -0.060 |

### 3. Multi-asset diversification helps but doesn't beat VTI
Risk parity (+0.113), full universe momentum (+0.072), and asset class rotation
(+0.069) all show positive point estimates. Pattern: spreading across asset
classes reduces risk relative to VTI, but not enough to compensate for VTI's
equity premium in a 20-year bull market.

### 4. Macro signals have no predictive power for tactical allocation
Three macro-driven strategies (macro risk composite, asset class rotation, factor
timing) all produce small effect sizes. The z-score thresholds rarely trigger,
and when they do, the signal is as likely wrong as right.

### 5. Survivorship bias was a real issue (caught by Codex review)
Dual momentum dropped from +0.127 to -0.019 after fixing survivorship handling
for early folds when VXUS/BND didn't exist. Factor timing NaN issue was caused
by allocating to pre-inception ETFs.

## Global Holdout CV Results (85% train / 15% test)

Separate from walk-forward: full dataset with last 15% held out (267 monthly
samples → 226 train, 41 test from Nov 2022 to Mar 2026).

| Model | Selection | Val MSE | Test MSE | Val-Test Gap | Consistency |
|---|---|---|---|---|---|
| ridge | BEST | 0.0021 | 0.0014 | 0.0007 | 1.60 |
| ridge | CONSISTENT | 0.0022 | 0.0014 | 0.0008 | 1.51 |
| xgboost | CONSISTENT | 0.0032 | 0.0014 | 0.0019 | 1.26 |
| lightgbm | BEST | 0.0017 | 0.0013 | 0.0004 | 1.74 |

All regression models converge to similar test MSE (~0.0013-0.0015). No model
has a meaningful edge over the others. Consistency ratios are all >1.0,
indicating high variance relative to signal.

## Codex Review Fixes Applied

1. **ML fold boundary leak**: target shift(-1) leaked test-period returns
2. **NaN return corruption**: backtester NaN→cash fallback + benchmark survivorship
3. **Factor timing NaN**: tradability checks for pre-inception ETFs
4. **Early-fold survivorship**: cascading fallback lists for unavailable ETFs
5. **Missing macro z-scores**: explicit logging, None sentinel instead of silent 0

## Parallels to Sports Betting

| Domain | Finding |
|---|---|
| NBA moneylines | Closing lines efficient (model LL 0.6219 vs market 0.5918) |
| MLB moneylines | Market approximately efficient (model ties market, p=0.869) |
| MMA/UFC | Both opening and closing lines efficient |
| **Vanguard ETFs** | **VTI efficient vs all 17 systematic strategies on Sharpe** |

The pattern holds across all domains: well-studied, liquid markets are efficient
against systematic strategies using public data.
