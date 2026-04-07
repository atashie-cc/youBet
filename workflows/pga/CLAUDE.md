# PGA Golf Tournament Prediction

## Domain Context
Predict outcomes of PGA Tour golf tournaments. Individual sport — no teams, no seasons in the team-sport sense, neutral venues. ~45-50 PGA Tour events per year with fields of 120-156 players. Top players compete in 20-25 events per year, producing rich rolling statistics. Tournaments are 4 rounds (72 holes) with a cut after round 2 (typically top 65 + ties). Strokes Gained decomposition provides uniquely granular skill measurement. Course-player fit is a major factor — player skill profiles interact non-linearly with course demands.

**Primary target market**: Head-to-head matchups (binary: did player A finish ahead of player B?). Maps directly to the existing binary classification framework. Secondary markets: top-5/10/20 finishes, outright winner (via simulation).

## Pipeline

```bash
python scripts/collect_data.py                # Download tournament results + round scores
python scripts/collect_data.py --process-only # Process raw data only
python scripts/collect_odds.py                # Download historical H2H + outright odds
python scripts/collect_weather.py             # Download historical weather for course locations
python scripts/compute_elo.py                 # Player Elo ratings (multi-player field adaptation)
python scripts/tune_elo.py                    # Grid search Elo parameters
python scripts/build_features.py              # 16 differential features + 2 context
python scripts/train_v2.py                    # Walk-forward CV via core Experiment runner
python scripts/evaluate_market.py             # Model LL vs market H2H LL
python scripts/predict.py                     # Generate predictions for upcoming tournaments
```

## Data Sources
1. **Data Golf API** (primary, Scratch Plus ~$30/mo): Round-level scoring, strokes gained by category, tee times, skill ratings, model predictions. 22 tours, 2004-present.
2. **Data Golf Odds Archives** (included with Scratch Plus): H2H matchup + outright odds from DraftKings, Pinnacle, FanDuel, Caesars, others. 2019-present.
3. **Kaggle PGA datasets** (fallback, free): ~10 years of PGA results with hundreds of variables.
4. **Open-Meteo / Visual Crossing** (free): Historical weather data matched to tournament dates and course locations.

## Features (16 differentials + 2 context, player_a - player_b)
Configured explicitly in `config.yaml` — `train_v2.py` reads the declared list and fails on mismatches.

### Strokes Gained (core predictors)
- **SG Total**: Overall strokes gained (EWMA span=8, shift(1))
- **SG Off-the-Tee**: Driving skill (distance + accuracy)
- **SG Approach**: Iron play — most predictive per literature (Broadie)
- **SG Around-the-Green**: Short game within 30 yards
- **SG Putting**: On-green performance — most volatile (~50% noise)
- **SG Tee-to-Green**: OTT + APP + ARG combined (less noisy than SG Total)

### Course Fit
- **Course fit score**: Dot product of player SG profile x course SG demand profile
- **Course history**: Weighted average past performance at this course (shift(1), EWMA with decay)

### Skill Rating
- **Elo**: Sequential rating, updated tournament-by-tournament (multi-player field adaptation)

### Recent Form
- **Form (5 events)**: Average adjusted finish position over last 5 tournaments (shift(1))
- **Form (10 events)**: Same over last 10 tournaments
- **Cuts made (5 events)**: Proportion of cuts made in last 5
- **SG trend**: Slope of SG_total over last 10 events (improving/declining form)

### Context
- **Days since last event**: Rest/rust indicator
- **Events in 90 days**: Workload/fatigue proxy
- **Major flag** (non-differential): Binary, major championship indicator
- **Field strength** (non-differential): Average Elo of tournament field

## Golf-Specific Elo Adaptations
- **Multi-player field**: Update each player's Elo based on expected vs actual finish position relative to field
- **No season resets** — continuous with annual mean reversion toward 1500
- **No home advantage** — neutral venues
- **Low K-factor** (~15-20) — golfers play 20-25 events/year (vs MMA's K=80 for 2-3 fights/year)
- **Win bonus** (1.2-1.5x K) — tournament wins carry extra signal
- **Time decay** (~0.05-0.10 annually) — inactivity erodes rating
- **K-factor schedule** — higher K for players with fewer career events

## Key Domain Considerations
- **Individual sport with large fields**: 120-156 players per event; outright prediction is extremely high variance
- **Strokes Gained hierarchy**: Approach > OTT > ARG > Putting for predictive value
- **Putting volatility**: ~50% noise; bettors/books may overweight recent results driven by putting luck
- **Course-player fit**: Non-linear interaction between player skill profile and course demands (Connolly & Rendleman, 2008)
- **Cut line**: Top ~65 players after round 2 continue; missed cut = no finishing position data for rounds 3-4
- **Weather effects**: Non-linear — wind >15mph accelerates scoring increase; morning/afternoon wave assignments create condition asymmetry
- **No true home advantage**: All tournaments at neutral venues (some players may have geographic familiarity)
- **Field composition varies**: Major championships vs regular events vs alternate-field events have different competitive dynamics
- **Data richness**: Unlike MMA (2-3 fights/year), golfers produce 20-25 tournaments of data per year, enabling richer rolling statistics

## Feature Safety Rules
**CRITICAL: All features must be point-in-time. No exceptions.**
- Rolling SG stats -> EWMA with `.shift(1)` — shift(1) is non-negotiable, excludes current tournament
- Course history -> Only prior visits, shift(1), EWMA with recency decay
- Elo -> Sequential by construction (pre-event rating used as feature, updated after event)
- Form metrics -> Rolling with shift(1) over prior tournaments only
- Weather -> Actuals acceptable for backtesting; forecasts required for live prediction
- Season-end statistics -> NEVER used mid-season; rolling/cumulative only
- Feature selection -> Explicit list in config.yaml, train_v2.py fails on mismatches
- **NEVER** use tournament-level aggregates that include the current tournament

## Current Performance (Phase 1 — IN PROGRESS)
Data acquisition and market benchmarking phase. No model results yet.

## Research Log
See `research/log.md` — read at start of every session.
