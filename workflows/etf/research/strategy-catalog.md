# Predictive Feature Framework for ETF Strategies

## Context

Phase 0 tested two price-only strategies (vol targeting, momentum rotation) against
VTI. Both FAILED the strict gate. This document catalogs a broader feature universe
— macro, fundamental, sentiment, and cross-asset signals — ranked by evidence quality,
data availability, and implementation feasibility.

The fundamental question changes from "can a simple allocation rule beat VTI?" to
"can a feature-driven signal, consuming information beyond price history alone,
produce excess risk-adjusted returns?"

---

## Feature Taxonomy

### Tier 1: Strong Evidence, Free Data, Feasible Implementation

These features have peer-reviewed academic backing, are available from free public
APIs (primarily FRED), and can be incorporated into the existing walk-forward framework
without major architectural changes.

#### 1A. Yield Curve Slope (10Y − 2Y Treasury Spread)
- **Predicts**: Recession probability (every recession since 1955 except one); equity drawdown timing
- **Horizon**: 6-18 months (S&P 500 peaks 6-9 months post-inversion)
- **Evidence**: Very high (Chicago Fed, Boston Fed research)
- **Data**: FRED series T10Y2Y (daily, real-time, free)
- **Signal type**: Risk-off when inverted + duration > 3 months; risk-on when steepening
- **Caveat**: Poor standalone equity timing signal. Best combined with credit spreads. QE distorts interpretation.

#### 1B. High-Yield Credit Spread (OAS)
- **Predicts**: Market stress, equity drawdowns, bond relative value
- **Horizon**: Weeks to months (credit markets price deterioration before equities)
- **Evidence**: Very high ("most reliable spread" per practitioner literature)
- **Data**: FRED series BAMLH0A0HYM2 (ICE BofA US HY OAS, daily, free)
- **Signal type**: Risk-off when OAS > historical 75th percentile; risk-on when < 25th. When OAS ≥ 500bps, HY bonds outperform Treasuries 83% of the time.
- **Caveat**: Fed policy can suppress spreads artificially. Mean reversion timing uncertain.

#### 1C. OECD Composite Leading Indicator (CLI)
- **Predicts**: Equity returns cross-sectionally across global markets
- **Horizon**: 1-6 months (based on monthly changes)
- **Evidence**: Strong (peer-reviewed, 60 years data, 39 countries; "1.43% monthly alpha in top quintile")
- **Data**: OECD data explorer (monthly, free)
- **Signal type**: Momentum — overweight countries/regions with rising CLI
- **Caveat**: Declining predictive power in recent periods. Works better for developed markets.

#### 1D. CAPE Ratio (Cyclically-Adjusted P/E)
- **Predicts**: 10-year forward equity returns (R² = 0.78-0.85)
- **Horizon**: 10-20 years (near zero correlation with 1-year returns)
- **Evidence**: Very high (Shiller, decades of out-of-sample validation)
- **Data**: Shiller website, multpl.com, FRED (monthly, free)
- **Signal type**: Strategic allocation tilts — reduce equity weight when CAPE > 30, increase when < 15
- **Caveat**: Useless for short-term tactical allocation. Distorted by accounting changes. Current CAPE ~35 has persisted for years without mean-reverting.

#### 1E. ISM Manufacturing PMI
- **Predicts**: Economic expansion/contraction; equity drawdown risk
- **Horizon**: 12 months
- **Evidence**: Moderate (backtested: 7.3% annual return vs 8.5% buy-and-hold, but better drawdown profile)
- **Data**: FRED (monthly, free)
- **Signal type**: Risk-on when PMI > 50; risk-off below. Reduces drawdowns but underperforms in bull markets.
- **Caveat**: Market typically discounts before release. Similar to vol targeting in effect.

### Tier 2: Moderate Evidence, Free/Cheap Data

#### 2A. AAII Sentiment Survey (Contrarian at Extremes)
- **Predicts**: 6-12 month equity returns AT EXTREMES ONLY
- **Evidence**: Moderate-strong at extremes; weak at normal levels
- **Data**: AAII.com (weekly, free)
- **Signal type**: Contrarian — buy when bullish% < 20% (100% hit rate, avg 14-20% 6-month gains); reduce when bullish% > 60% (only 48% hit rate)
- **Caveat**: Only works at 2+ standard deviation extremes. Useless in normal range. Must be combined with other signals.

#### 2B. VIX Level and Term Structure
- **Predicts**: 3-12 month equity returns (contrarian); realized volatility
- **Evidence**: Moderate-strong
- **Data**: CBOE (daily, free via Yahoo Finance)
- **Signal type**: Contrarian — VIX > 45 predicts strong 3-month and 1-year returns. VIX term structure slope adds information.
- **Caveat**: Complex relationship. Not a simple direct predictor. Overlaps conceptually with vol-targeting.

#### 2C. Forward P/E Ratio (Market-Wide)
- **Predicts**: 10-year forward returns (R² = 0.39 broad market, 0.48 growth)
- **Horizon**: 10 years (weak short-term)
- **Evidence**: Moderate
- **Data**: FactSet (paid), or proxied via free Shiller/multpl.com
- **Signal type**: Strategic tilt — same logic as CAPE but forward-looking
- **Caveat**: Limited small-cap predictive power (R² = 0.02). Already incorporates analyst expectations.

#### 2D. Conference Board LEI
- **Predicts**: Recession turning points (~7 months lead)
- **Evidence**: High for recession prediction; moderate for equity timing
- **Data**: Conference Board (monthly, free)
- **Signal type**: Risk-off when YoY decline > 2%
- **Caveat**: Includes S&P 500 as component (circular). False signals occur.

#### 2E. Monetary Policy Stance (Fed Funds Rate vs. Neutral)
- **Predicts**: Equity risk premium direction
- **Evidence**: Moderate (NBER, Boston Fed)
- **Data**: FRED (Fed Funds Rate, r-star estimates from Cleveland Fed)
- **Signal type**: Risk premium rises when policy is far from neutral (either direction). U-shaped relationship.
- **Caveat**: Neutral rate (r-star) is unobservable and estimated with wide confidence bands.

### Tier 3: Emerging/Weaker Evidence, Higher Cost

#### 3A. Earnings Revision Momentum
- **Predicts**: 1-12 month relative sector/stock returns
- **Evidence**: Strong (83% direction persistence)
- **Data**: FactSet or Bloomberg (paid subscriptions, ~$300-500/mo minimum)
- **Signal type**: Momentum — follow revision direction. 12-month lookback most effective.
- **Caveat**: Requires paid data. Best for stock-level; sector aggregation reduces signal.

#### 3B. Cross-Asset Momentum
- **Predicts**: 1-3 month equity returns using bond, commodity, FX signals
- **Evidence**: Strong (45% higher Sharpe than single-asset momentum in academic studies)
- **Data**: Free (Yahoo Finance for commodity ETFs, bond ETFs, currency ETFs)
- **Signal type**: If bonds are trending up and commodities down → risk-off for equities
- **Caveat**: Regime-dependent. Correlation structure changes in crises.

#### 3C. Volatility Risk Premium (VRP)
- **Predicts**: Future realized volatility; modestly predicts returns
- **Evidence**: Moderate (BIS research; commodity VRP predicts 3-4 quarters ahead)
- **Data**: Computed from VIX vs. realized vol (free)
- **Signal type**: High VRP → market complacent → increase caution. Low VRP → market stressed → contrarian buy.
- **Caveat**: Hard to compute cleanly. Overlaps with VIX-level signals.

#### 3D. Social Media / NLP Sentiment
- **Predicts**: Short-term volatility (days); modestly predicts returns when combined with other data
- **Evidence**: Emerging; standalone unreliable
- **Data**: Twitter/X API (paid), Reddit (free but noisy), StockTwits (free)
- **Signal type**: Momentum (extreme positive sentiment → short-term continuation) or contrarian (extreme → reversal)
- **Caveat**: High noise, requires NLP pipeline, non-stationary. Best as supplementary, not primary.

---

## Implementation Strategy: Feature-Enhanced Strategies

### Architecture Change: From Rule-Based to Signal-Driven

Current strategies (vol_targeting, momentum_rotation) use hard-coded rules on price data.
Feature-enhanced strategies consume a FEATURE VECTOR at each rebalancing date and produce
allocation weights. Two approaches:

**Approach A: Rule-Based Feature Composites**
Combine Tier 1 features into a composite risk score using pre-specified rules:
```
risk_score = w1 * yield_curve_signal + w2 * credit_spread_signal + w3 * cli_signal + ...
equity_weight = f(risk_score)  # monotonic mapping, pre-specified
```
Weights (w1, w2, ...) pre-specified from literature evidence, NOT fitted to data.
This avoids overfitting but limits the model's ability to find non-obvious interactions.

**Approach B: Walk-Forward ML on Features**
Train a model (gradient-boosted trees, like youBet) on feature vectors to predict
next-month excess return, then allocate based on predicted return.
- Walk-forward: train on expanding window, predict next fold
- Feature vector: Tier 1 + Tier 2 features at each month
- Target: next-month VTI excess return (or quintile rank of sector returns)
- This is more flexible but risks overfitting with ~240 monthly observations (20 years)

**Recommendation: Start with Approach A, test Approach B if A fails.**
With 240 monthly data points and 5-8 features, ML overfitting risk is very high.
Rule-based composites can be tested with the existing walk-forward backtester
without any architectural changes.

### Data Pipeline Design

```
data/
  macro/
    yield_curve.py      # FRED T10Y2Y
    credit_spreads.py   # FRED BAMLH0A0HYM2
    pmi.py              # FRED ISM PMI
    lei.py              # Conference Board LEI
    cli.py              # OECD CLI
    fed_funds.py        # FRED FEDFUNDS + r-star
  sentiment/
    aaii.py             # AAII.com scraper
    vix.py              # Yahoo Finance VIX
  valuation/
    cape.py             # Shiller CAPE from multpl.com
    forward_pe.py       # Proxied or FactSet
  cross_asset/
    bond_momentum.py    # BND/AGG trailing returns
    commodity_signal.py # GLD/DBC trailing returns
    dollar_signal.py    # UUP/DXY trailing returns
```

Each module:
1. Fetches data from source
2. Caches locally with date-stamped snapshots (same as price data)
3. Returns a pd.Series indexed by date with the signal value
4. Explicitly handles publication lag (e.g., PMI published first business day of
   month for PRIOR month → signal available T+1 month)

### PIT Concerns for Economic Features

Economic data has SEVERE point-in-time risks that differ from price data:

| Feature | Publication Lag | Revision Risk | PIT Strategy |
|---------|----------------|---------------|--------------|
| Yield curve | Real-time (T+0) | None | Use directly |
| Credit spreads | Real-time (T+0) | None | Use directly |
| PMI | T+1 month | Minor revisions | Lag 1 month in features |
| CLI | T+2 months | Moderate | Lag 2 months |
| CAPE | T+1 month | Earnings revisions | Lag 1 month; use Shiller's methodology |
| LEI | T+1 month | Moderate | Lag 1 month |
| AAII | T+1 week | None | Lag 1 week |
| GDP | T+1 quarter, then revised 3x | SEVERE (0.5-1.5% revision) | Use vintage data or lag 3 months |

**Critical**: GDP and employment data undergo massive revisions. Using "final" revised
data in backtests creates lookahead bias. Either use real-time vintage data (from
ALFRED at the St. Louis Fed) or lag by the full revision window.

### Feature Normalization

All features must be normalized using ONLY data available at signal time:
- Z-score: (current - trailing_mean) / trailing_std, using expanding or rolling window
- Percentile rank: current value's rank in the trailing distribution
- Both methods use the existing `core/transforms.py` fit/transform pattern

### Proposed New Strategies

#### Strategy: Macro Risk Composite
Combine yield curve + credit spread + PMI + CLI into a single risk score.
Pre-specified weights from literature (e.g., equal weight across z-scored signals).
When composite < −1σ → defensive (60% VTI / 40% VGSH).
When composite > +1σ → aggressive (100% VTI).
Otherwise → neutral (80% VTI / 20% VGSH).

#### Strategy: Sentiment Extremes
Only trade at AAII + VIX extremes (occurs ~5-10x per year).
When AAII bullish < 20% AND VIX > 30 → go 100% VTI (contrarian buy).
When AAII bullish > 55% AND VIX < 15 → reduce to 60% VTI (defensive).
Otherwise → default allocation (hold current).
Very low turnover, targets only the highest-conviction moments.

#### Strategy: Cross-Asset Momentum Enhanced
Extend existing momentum_rotation with cross-asset signals:
Original: rank ETFs by trailing 6-month return.
Enhanced: also incorporate bond momentum (BND trending up → risk-off tilt),
commodity signal (GLD strong → inflation hedge tilt), dollar signal (UUP strong →
reduce international weight).

#### Strategy: Valuation-Adjusted Allocation
Monthly equity allocation = baseline − k × (CAPE_zscore).
When CAPE is elevated (>1σ above 20yr mean), reduce equity exposure.
When CAPE is depressed (<1σ below), increase.
Very slow-moving (CAPE changes monthly). Long-horizon signal.

---

## Evidence Summary Table

| Feature | Predicts | Horizon | Evidence | Free? | PIT Risk |
|---------|----------|---------|----------|-------|----------|
| **Yield curve** | Recession/drawdown | 6-18mo | Very high | Yes | Low |
| **Credit spread** | Market stress | Weeks-months | Very high | Yes | Low |
| **OECD CLI** | Cross-country equity | 1-6mo | Strong | Yes | Medium |
| **CAPE** | Long-term returns | 10-20yr | Very high | Yes | Low |
| **PMI** | Economic cycle | 12mo | Moderate | Yes | Medium |
| **AAII sentiment** | Contrarian extremes | 6-12mo | Moderate (at extremes) | Yes | Low |
| **VIX** | Contrarian returns | 3-12mo | Moderate | Yes | Low |
| **Earnings revisions** | Sector rotation | 1-12mo | Strong | Paid | Medium |
| **Cross-asset momentum** | Tactical allocation | 1-3mo | Strong | Yes | Low |
| **Forward P/E** | Long-term returns | 10yr | Moderate | Paid/proxy | Low |
| **VRP** | Vol timing | 1-3mo | Moderate | Computed | Medium |
| **Social media NLP** | Short-term vol | Days | Weak standalone | Mixed | High |

## Recommended Phased Approach

1. **Phase 1 (immediate)**: Implement Tier 1 free-data features (yield curve, credit spread,
   PMI, CLI, CAPE) with proper PIT lag handling. Test Macro Risk Composite strategy.
2. **Phase 2**: Add sentiment features (AAII, VIX). Test Sentiment Extremes strategy.
3. **Phase 3**: Add cross-asset momentum. Test enhanced momentum rotation.
4. **Phase 4**: If signal exists, explore ML approach (Approach B) on feature vector.
5. **Phase 5**: Add paid data (earnings revisions) only if free features show promise.
