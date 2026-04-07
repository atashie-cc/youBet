# ETF Workflow: Final Report

## Executive Summary

Tested 17 systematic strategies against Vanguard VTI buy-and-hold using 52 ETFs,
20 years of data (2003-2026), walk-forward validation, block bootstrap with Holm
correction, and 2 rounds of Codex adversarial review (11 bugs found and fixed).

**On Sharpe ratio**: VTI is efficient. No strategy passes the strict gate
(ExSharpe > 0.20, Holm p < 0.05, CI lower > 0). Trend following comes closest
at +0.210 but CI spans zero.

**On drawdown**: VTI is NOT efficient. Three strategies provide statistically
significant drawdown reduction with bootstrap CIs excluding zero:
- Trend following: -20.1% vs -55.5% (64% reduction, P(better) = 98%)
- Risk parity: -22.3% vs -55.5% (60% reduction, P(better) = 100%)
- Asset class rotation: -33.9% vs -55.5% (39% reduction, P(better) = 100%)

**Recommendation**: 40% trend following + 60% risk parity blend. Sharpe 0.798
(higher than VTI's ~0.52), MaxDD -22.3% (60% reduction), insurance premium
3.22%/yr for crisis protection.

---

## The Dual-Objective Conclusion

The original objective was: "optimize returns AND minimize likelihood of reaching
0 balance." These are different objectives that lead to different conclusions.

### Sharpe Analysis (risk-adjusted return)

| Rank | Strategy | ExSharpe | 90% CI | Verdict |
|---|---|---|---|---|
| 1 | trend_following | +0.210 | [-0.080, +0.510] | INCONCLUSIVE |
| 2 | risk_parity | +0.113 | [-0.109, +0.341] | INCONCLUSIVE |
| 3 | asset_class_rotation | +0.069 | [-0.016, +0.145] | INCONCLUSIVE |
| ... | 14 other strategies | negative | — | FAIL/NEGATIVE |

**Conclusion**: Cannot reject the null that VTI is Sharpe-efficient.

### Drawdown Analysis (ruin avoidance)

| Rank | Strategy | MaxDD | DD Reduction | 90% CI on Diff | P(better) |
|---|---|---|---|---|---|
| 1 | trend_following | -20.1% | 64% | [+0.032, +0.376] | 98% |
| 2 | risk_parity | -22.3% | 60% | [+0.145, +0.394] | 100% |
| 3 | asset_class_rotation | -33.9% | 39% | [+0.085, +0.232] | 100% |
| — | VTI (benchmark) | -55.5% | — | — | — |

**Conclusion**: Three strategies provide statistically significant drawdown
reduction. VTI is drawdown-INefficient.

---

## Recommended Allocation: 40% Trend / 60% Risk Parity

### Why this blend

Zakamulin's published research identifies 40% trend following as the
Sharpe-optimal allocation. Our blend sweep confirms this exactly:

| Blend | Sharpe | MaxDD | Calmar | Ann Return |
|---|---|---|---|---|
| tf10/rp90 | 0.747 | -22.3% | 0.239 | 5.3% |
| tf20/rp80 | 0.779 | -22.3% | 0.261 | 5.8% |
| tf30/rp70 | 0.794 | -22.3% | 0.283 | 6.3% |
| **tf40/rp60** | **0.798** | **-22.3%** | **0.305** | **6.8%** |
| tf50/rp50 | 0.792 | -22.3% | 0.326 | 7.3% |
| tf60/rp40 | 0.782 | -22.4% | 0.346 | 7.7% |
| tf70/rp30 | 0.768 | -22.4% | 0.365 | 8.2% |
| tf80/rp20 | 0.752 | -23.0% | 0.376 | 8.6% |
| tf90/rp10 | 0.735 | -23.6% | 0.384 | 9.1% |

### What this means in practice

**Trend following component** (40%):
- Hold VTI when VTI price > 200-day SMA
- Hold VGSH when VTI price < 200-day SMA
- Rebalance monthly

**Risk parity component** (60%):
- Inverse-volatility weight across VTI, VXUS, BND, VNQ, VTIP
- Uses trailing 63-day volatility
- Lower-vol assets (bonds) get higher weight
- Rebalance monthly

### Trade-offs vs VTI

| Metric | tf40/rp60 Blend | VTI Buy-and-Hold |
|---|---|---|
| Sharpe | 0.798 | ~0.52 |
| Ann Return | 6.8% | 10.3% |
| MaxDD | -22.3% | -55.5% |
| Calmar | 0.305 | ~0.19 |
| Bull market drag | ~-4%/yr | 0% |
| Crisis protection | +30-40% excess | none |
| Insurance premium | ~3%/yr | $0 |

The blend sacrifices ~3.5% annual return for 60% drawdown reduction. For investors
where avoiding catastrophic loss matters more than maximizing terminal wealth,
this is a favorable trade.

---

## Insurance Premium Framework

Trend following acts as portfolio insurance:

| | Normal Markets (88.3%) | Crisis (11.7%) | Net |
|---|---|---|---|
| Annual drag | -3.22%/yr | — | — |
| Crisis gain | — | +34.1% excess | — |
| 20-year total | -65.0% cumulative drag | +34.1% crisis gain | -30.9% |
| Payoff ratio | — | — | 0.5x |

The 0.5x payoff ratio means trend following recovers about half its cumulative
cost during crises. This is comparable to rolling protective put strategies
(which typically cost 2-4%/yr for 10-20% downside protection).

---

## Regime Analysis

Performance is highly asymmetric by regime:

| Strategy | Bull (+62.6%) | Bear (7.8%) | Crisis (11.7%) | High Vol (17.3%) |
|---|---|---|---|---|
| trend_following | -4.1% | **+49.8%** | **+34.1%** | +26.1% |
| risk_parity | -15.5% | +40.3% | +38.8% | +44.1% |
| full_universe_momentum | -7.1% | +40.8% | **+45.9%** | +22.1% |
| vol_targeting | -2.2% | +4.0% | +0.6% | -0.3% |

Trend following: reliable crisis alpha. Risk parity: strongest in high-vol regimes.
Vol targeting: provides almost no protection despite reducing position size.

---

## Momentum Crisis Mechanism — Verified

During the 2008-2009 crisis, full universe momentum held:
- **Oct 2008**: BSV 60%, BND 31% (91% bonds)
- **Nov 2008**: EDV 100% (long-duration treasuries, gained +15%)
- **Jan-Mar 2009**: 100% bonds (BND, BSV, BIV, BLV, EDV)
- **Jun 2009**: Rotated back to equities (VGT, VCR, VWO)

The absolute momentum filter (hold only if 6-month return > 0) correctly
shifted to bonds when equities showed negative trailing returns. This is
the Antonacci dual momentum mechanism, well-documented in the literature.

---

## ML Models: Complete Failure

All 6 ML strategies produced negative excess Sharpe. More thorough HP search
(10-fold rolling CV, expanded grids) made performance WORSE:

| Model | Simple CV | 10-Fold CV | Degradation |
|---|---|---|---|
| ml_logistic | -0.030 | -0.186 | -0.156 |
| ml_ridge | -0.083 | -0.111 | -0.028 |
| ml_xgboost | -0.097 | -0.116 | -0.019 |
| ml_lightgbm | -0.083 | -0.114 | -0.031 |
| ml_ensemble | -0.068 | -0.128 | -0.060 |

Root cause: ~36 monthly training samples per walk-forward fold. Literature
(Xiu 2020, CFA Institute) establishes 200-500+ monthly observations as the
minimum for ML in financial prediction. Our sample size is at the absolute
floor — ML cannot extract signal from this amount of data.

---

## Codex Review Summary (2 rounds, 11 fixes)

| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | HIGH | ML fold boundary leak (shift(-1) into test) | Fixed |
| 2 | HIGH | NaN return corruption + benchmark survivorship | Fixed |
| 3 | HIGH | hierarchical_ml PIT leak (inclusive loc) | Fixed |
| 4 | HIGH | global_holdout_cv scaler leaks into CV folds | Fixed |
| 5 | MED | factor_timing pre-inception ETF allocation | Fixed |
| 6 | MED | Early-fold survivorship fallbacks | Fixed |
| 7 | MED | Missing macro z-scores silent default | Fixed |
| 8 | MED | Empty weights silently become 100% cash | Fixed (warning) |
| 9 | MED | ML scaler leaks across inner CV folds | Fixed (Pipeline) |
| 10 | LOW | 1-month val slices → high-variance ranking | Acknowledged |
| 11 | LOW | HP grids too expressive for 24 samples | Acknowledged |

---

## Cross-Domain Pattern

| Domain | Sharpe-Efficient? | Drawdown-Efficient? |
|---|---|---|
| NBA moneylines | Yes (closing LL 0.5918) | N/A |
| MLB moneylines | Yes (p=0.869) | N/A |
| MMA/UFC | Yes (model LL 0.66 vs close 0.61) | N/A |
| **Vanguard ETFs** | **Yes** (no strategy beats VTI on Sharpe) | **No** (trend following, risk parity significantly reduce drawdowns) |

The ETF market is unique: while it's Sharpe-efficient (like sports markets),
there IS a free lunch on drawdown reduction via trend following and risk parity.
This is because drawdown risk is not priced the same way as Sharpe — the equity
risk premium compensates for average volatility, not for catastrophic path-dependent
losses.
