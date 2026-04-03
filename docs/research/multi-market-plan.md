# Multi-Market Expansion Plan

## Framework Recap

For each market, follow the validated 5-stage pipeline:
1. **Screen** (1-2 weeks): Data availability, historical odds, market LL benchmark
2. **Baseline** (2-4 weeks): Simple model, compare LL vs market. **Kill if model LL > market LL.**
3. **Validate** (2-4 weeks): Full pipeline, LOO-CV, walk-forward P&L with realistic vig
4. **Paper Trade** (4-8 weeks): Live predictions, track CLV over 200+ bets
5. **Live** (ongoing): Eighth-Kelly → quarter-Kelly, rolling 500-bet monitoring

## Completed Markets

| Market | Phases | Result | Key Finding |
|--------|--------|--------|-------------|
| **NBA moneylines** | 12 phases | Market wins by 0.030 LL | Closing lines unbeatable; 0/18 seasons profitable |
| **MLB moneylines** | 6 phases | Market wins by 0.000 LL | Model ties market but p=0.869; 2022-2025 = -0.3% ROI |

Both confirm: **major-sport moneyline closing lines are approximately efficient against public statistical models.** Ensemble disagreement helps bet selection in NBA but not MLB.

## Revised Market Prioritization

Based on lessons learned: avoid deep-liquidity moneyline markets. Target markets where books model less carefully (props, niche leagues) or where the closing line is less efficient.

| Priority | Market | Data Volume | Efficiency | Rationale |
|----------|--------|-------------|------------|-----------|
| **1** | **NBA/MLB Player Props** | High | Low-moderate | Books use simpler models; combinatorial explosion limits attention |
| **2** | **MLB Totals/Run Lines** | Very high | Moderate | Same data, different market — totals may be less efficient than ML |
| **3** | **NCAA Basketball** | High (5,500/yr) | Moderate | 350+ teams = information asymmetry; existing workflow |
| **4** | **NHL** | High (1,312/yr) | Moderate | Goaltender-driven variance; less liquid than NBA/MLB |
| **5** | **Soccer (niche/national)** | Low-moderate | Low | Less sharp money; data scarcity is the challenge |
| **6** | **Esports** | Moderate | Low | Immature market but rapid meta-changes |
| ~~7~~ | ~~NBA moneylines~~ | ~~High~~ | ~~Very high~~ | ~~KILLED: market too efficient (Phase 12)~~ |
| ~~8~~ | ~~MLB moneylines~~ | ~~Very high~~ | ~~Very high~~ | ~~KILLED: market too efficient (Phase 6)~~ |
| ~~9~~ | ~~Soccer (top leagues)~~ | ~~High~~ | ~~Very high~~ | ~~Deprioritized: massive global liquidity = very sharp~~ |

## Per-Market Implementation Plans

### Market 1: NBA Player Props (IMMEDIATE — data exists)
**What**: Predict player stat lines (PTS, REB, AST, 3PM, etc.) and compare vs prop lines
**Data needed**: Player traditional box scores (PTS, REB, AST per player per game) — fetch via `PlayerGameLogs(measure_type='Base')`, same bulk endpoint as advanced stats
**Historical odds**: Player prop lines harder to find historically; may need to scrape or start collecting prospectively
**Key features**: Player rolling averages, opponent defensive ratings, pace, minutes projection, home/away splits, rest days, matchup history
**Initial scope**: Focus on 5 representative prop types × 20-30 high-volume players:
- PTS (scoring), AST (passing), REB (rebounding), 3PM (shooting), PTS+REB+AST (combo)
- Stars (Jokic, Luka), high-volume (Haliburton), role players (bench scorers), rookies

### Market 2: MLB
**What**: Game outcomes (ML, run line, totals) + player props (strikeouts, hits, HRs)
**Data sources**:
- Statcast (pitch-level, publicly available via Baseball Savant)
- Retrosheet (play-by-play back to 1921)
- `pybaseball` Python package (wraps Statcast + FanGraphs)
- Historical odds: multiple Kaggle datasets available
**Key features**: Pitcher matchups (ERA, WHIP, K/9, pitch mix), park factors, weather, umpire tendencies, platoon splits, bullpen usage
**Advantage**: Pitching matchups create game-specific variance that generic team ratings miss

### Market 3: NCAA Basketball
**What**: Game outcomes for March Madness and regular season
**Status**: Existing workflow at `workflows/ncaa_march_madness/`
**Key advantage**: 350+ teams means oddsmakers can't deeply model every matchup
**Data**: KenPom, Barttorvik, NCAA stats API

### Market 4: NHL
**What**: Game outcomes (ML, puck line, totals) + goaltender props
**Data**: NHL API (similar to NBA API), MoneyPuck (expected goals), Natural Stat Trick
**Key features**: Goaltender matchups (save %, xGA), shot quality, special teams, back-to-back

### Market 5: Soccer (Top Leagues)
**What**: Match outcomes (1X2, Asian handicap, totals) for EPL, La Liga, Bundesliga, Serie A, Ligue 1
**Data**: FBref.com (StatsBomb xG data), Transfermarkt, Understat
**Key features**: xG, xGA, squad rotation, fixture congestion, managerial changes
**Challenge**: Very efficient market due to global liquidity

### Market 6: Soccer (Niche/National)
**What**: Same as top leagues but for lower divisions, national cups, smaller leagues
**Data**: More limited; FBref covers many leagues but depth varies
**Opportunity**: Less sharp money = more inefficiency; but data scarcity = harder to model

### Market 7: Esports
**What**: Match outcomes for LoL, CS2, Valorant, Dota 2
**Data**: oracles.gg, liquipedia, game-specific APIs, HLTV (CS2)
**Challenge**: Rapid meta-changes, roster instability, patch updates invalidate historical patterns
**Opportunity**: Market still maturing; fewer sophisticated models in play

### Market 8: MMA
**What**: Fight outcomes for UFC and major promotions
**Data**: UFCStats.com (official stats), Tapology, Sherdog
**Key features**: Style matchups (striker vs grappler), reach/height differentials, weight cuts, age curves
**Challenge**: Small sample (~500 UFC fights/yr), high variance, very matchup-dependent

### Market 9: Boxing
**What**: Fight outcomes for major bouts
**Data**: BoxRec
**Challenge**: Very low volume (~100 significant fights/yr), highly matchup-dependent, promotional politics
**Assessment**: Likely not viable for statistical modeling due to sample size

## Immediate Actions

### Action 1: Download MLB Data (background)
- Install `pybaseball` package
- Fetch Statcast pitch-level data (2015-present)
- Fetch team game logs and standings
- Fetch historical MLB odds from Kaggle
- Store in `workflows/mlb/data/raw/`

### Action 2: Download NBA Player Traditional Stats (background)
- Use `PlayerGameLogs(measure_type='Base')` — same bulk endpoint pattern as advanced
- Gets PTS, REB, AST, STL, BLK, TOV, FGM, FGA, FG3M, FTM, FTA, MIN per player per game
- Store in `workflows/nba/data/raw/all_player_traditional.csv`
- This is the foundation for player props modeling

### Action 3: Build NBA Player Props Baseline (while data downloads)
- Select 5 prop types × 20-30 players for initial scope
- Build rolling player averages as baseline predictions
- Compare vs available prop lines (need to source historical props)
- Assess market efficiency for props specifically
