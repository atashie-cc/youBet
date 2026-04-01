# NBA Prediction Research Log

Track experiments, findings, and decisions. Most recent entries at top. Read at start of every session.

---

## 2026-03-26 — Phase 13: Historical NBA Player Prop Lines — Data Source Research

### Motivation
Phase 12 proved NBA moneylines are too efficient for our model to beat. Player props are a plausible next market: higher vig (more margin to overcome), less sharp money, more variance in line quality, and our player-level features may provide an edge that moneyline markets already price in. To backtest a player props strategy, we need historical over/under lines for individual player stats (points, rebounds, assists, threes, etc.).

### Research Summary

Comprehensive web research was conducted across free datasets, paid APIs, scrapeable sources, DFS proxies, and prospective collection methods. **The core finding: there is no free, comprehensive dataset of historical NBA player prop lines. This data is commercially valuable and gatekept.** But several viable paths exist.

---

### Category 1: Free Datasets (Kaggle, GitHub)

**Kaggle — NBA Historical Stats and Betting Data** (ehallmar)
- Coverage: Large number of historical NBA games
- Content: Match stats + betting odds
- Player props: **NO** — game-level spreads/totals/moneylines only
- Format: ZIP (~38 MB), free (CC license)
- URL: https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data

**Kaggle — NBA Betting Data Oct 2007–Jun 2025** (cviaxmiwnptr)
- Coverage: 2007-2025, 27 columns
- Content: Spreads, totals, moneylines, quarter scores
- Player props: **NO** — game-level only
- Format: ZIP (~633 KB), CC0 Public Domain
- URL: https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024

**GitHub repos** (parlayparlor, ejjlittle, chevyphillip, etc.)
- Multiple NBA player prop prediction models exist on GitHub
- **None contain historical prop line datasets** — they fetch lines in real-time via APIs
- Most use The Odds API or scrape PrizePicks for current-day lines
- No repo found with archived historical prop line CSVs

**Verdict: No free historical player prop line datasets exist on Kaggle or GitHub.**

---

### Category 2: APIs with Historical Player Props

#### The Odds API (the-odds-api.com) — BEST OPTION FOR HISTORICAL PROPS
- **Player props available since**: May 3, 2023 (snapshots every 5 min)
- **NBA markets**: player_points, player_rebounds, player_assists, player_threes, player_blocks, player_steals, player_turnovers, player_blocks_steals, player_points_rebounds_assists, player_points_rebounds, player_points_assists, player_rebounds_assists, player_field_goals, player_frees_made, player_first_basket, player_double_double, player_triple_double, player_fantasy_points, plus alternate lines for most markets
- **Quarter markets**: player_points_q1, player_rebounds_q1, player_assists_q1
- **Data format**: JSON via REST API (one event at a time for props)
- **Includes actual O/U line numbers**: YES (e.g., player_points Over 24.5 at -110)
- **Bookmakers**: US bookmakers (DraftKings, FanDuel, BetMGM, etc.)
- **Pricing**:
  - Free: 500 credits/month
  - $30/mo: 20,000 credits
  - $59/mo: 100,000 credits
  - $119/mo: 5,000,000 credits
  - $249/mo: 15,000,000 credits
- **Historical quota cost**: 10 credits per region per market **per event**
- **Cost estimate for backfill**: ~1,230 regular season games/year x ~6 prop markets x 10 credits = ~73,800 credits per season. The $59/mo tier (100K credits) covers roughly 1 season of 6-market backfill. ~3 seasons of data exist (May 2023 to present).
- **Verdict**: Most practical source for historical props backtesting. ~$60-120 to backfill all available history.

#### SportsDataIO (sportsdata.io)
- **Coverage**: Props and futures from 2020 onwards
- **Bookmakers**: DraftKings, FanDuel, BetMGM, Caesars, BetRivers, Circa, BetOnline, Fanatics, ESPN BET
- **Data delivery**: API, S3 bucket, or custom
- **Player props**: YES, included in historical data
- **Pricing**: Custom quotes only (contact sales@sportsdata.io). Estimated $300+/month for historical props.
- **Verdict**: Broader historical coverage than The Odds API (2020 vs 2023), but pricing is opaque and likely expensive.

#### BALLDONTLIE (balldontlie.io)
- **Player props endpoint**: YES (`/v2/odds/player_props`)
- **Historical data**: **NO** — explicitly stated: "Player prop data is LIVE and updated in real-time. We do not store historical data."
- **Pricing**: Free ($0, 5 req/min), ALL-STAR ($9.99/mo, 60 req/min), GOAT ($39.99/mo, 600 req/min)
- **Markets**: Points, rebounds, assists, blocks, steals, threes, combined categories
- **Verdict**: Useful for prospective collection only, not historical backfill.

#### SportsGameOdds (sportsgameodds.com)
- **Historical data**: YES — closing odds, historical pre-match and in-play odds
- **Player props**: YES
- **Pricing**: Free tier (limited), $75/mo (Rookie), $299/mo (Pro)
- **Query granularity**: Per event, per player, or per bookmaker
- **Verdict**: Worth evaluating as an alternative to The Odds API, especially if they have deeper history.

#### OpticOdds (opticodds.com)
- **Historical data**: YES — full price history with every movement tracked
- **Player props**: YES, comprehensive player-level markets
- **Bookmakers**: 200+ operators
- **Pricing**: Custom quotes only, no free tier. Enterprise-level (estimated $hundreds to low $thousands/month).
- **Verdict**: Too expensive for individual research. Enterprise product.

#### OddsJam (oddsjam.com)
- **Historical data**: YES — closing lines, opening odds, live line changes
- **Player props**: YES, real-time props and alternate markets
- **Pricing**: Custom (premium tier)
- **Verdict**: Interesting for backtesting but pricing unclear. Worth inquiring.

#### Sportradar (developer.sportradar.com)
- **Player props**: YES (Odds Comparison Player Props product)
- **Historical data**: Available within NBA API via season_year parameter
- **Pricing**: $500-$1,000+/month (enterprise)
- **Verdict**: Way too expensive for individual research.

#### Goalserve (goalserve.com)
- **Player props**: YES
- **Historical data**: Since 2010
- **Pricing**: $150-$250/month for basketball, 14-day free trial
- **Data format**: XML, JSON
- **Verdict**: Could be viable if their historical props go back far enough. Worth testing free trial.

#### OddsPapi (oddspapi.io)
- **Historical data**: YES, included at no penalty at all tiers
- **Player props**: YES
- **Bookmakers**: 350+ including Pinnacle, sharp books
- **Pricing**: Free (250 req/month), $49/mo (Pro)
- **Verdict**: Potentially the best value if their historical props data is comprehensive. The $49/mo tier with historical included is cheaper than The Odds API for historical queries. Needs verification of NBA props depth.

---

### Category 3: Scrapeable Sources

**OddsPortal** (oddsportal.com)
- Has NBA game odds (spreads, moneylines, totals)
- Player props: **NOT available** on OddsPortal
- Verdict: Not usable for player props

**BettingPros** (bettingpros.com)
- Has current-day prop analyzer with historical player stats
- Player props: Shows current lines, no historical line archive accessible
- Verdict: Not scrapeable for historical lines

**Covers.com** (covers.com)
- Has current player props pages
- No accessible historical prop line archive
- Verdict: Not usable for historical data

**RotoWire** (rotowire.com)
- Has betting archive tool for 2025-26 season
- Has current player props comparisons across sportsbooks
- No deep historical prop line data
- Verdict: Limited to current season

**Action Network** (actionnetwork.com)
- Reported to have closing odds of all player props going back "quite a ways"
- Sources include Pinnacle, BetCris, and others
- **No public API** — data is locked in their app/website
- Verdict: Potentially rich data but no programmatic access

**PrizePicks** (prizepicks.com)
- No public API
- Can be scraped with Selenium (see github.com/yomerosho/WebScraping-PrizePicks-NBA)
- Apify has a PrizePicks scraper actor
- Data: Current-day player prop lines only
- Verdict: Usable for prospective collection, not historical

**Underdog Fantasy**
- Can be scraped (see github.com/aidanhall21/underdog-fantasy-pickem-scraper)
- Saves to CSV format
- Current-day only
- Verdict: Usable for prospective collection, not historical

---

### Category 4: DFS as a Proxy for Prop Lines

**Core idea**: DraftKings/FanDuel daily fantasy salaries and projected fantasy points encode market expectations for player performance. A player priced at $8,500 on DraftKings is implicitly projected for ~40 fantasy points, which implies certain stat lines.

**Analytics by Adam — NBA DFS ULTIMATE Dataset**
- Seasons: 2018-19 and 2019-20 only
- Content: DK/FD salaries, ownership, Vegas game totals, actual performance
- Price: $49.95
- Player props: Not directly, but DFS salaries could be reverse-engineered into implied stat projections
- Verdict: Too narrow (2 seasons). DFS salary is a weak proxy for specific prop lines.

**NBAstuffer / BigDataBall**
- DFS game logs with DK/FD/Yahoo salaries and positions
- Historical data back to 2006-07
- Pricing: $25-$50/season
- Player props: **NO** — game box scores and DFS salaries, not prop lines
- Verdict: DFS salaries give directional signal but cannot substitute for actual O/U lines

**Why DFS is a poor proxy**:
1. DFS salary reflects *relative* player value, not absolute stat projections
2. No direct mapping from salary to specific stat categories (pts/reb/ast are combined into fantasy points)
3. DFS ownership and salary are influenced by game scripts, matchups, and slate composition — not the same as prop line dynamics
4. You cannot backtest "beat the prop market" using DFS data because you don't know what the actual prop lines were

**Verdict: DFS data cannot substitute for actual prop lines in backtesting.**

---

### Category 5: Paid Data Sources (Premium)

| Provider | Player Props | History | Pricing | Notes |
|----------|-------------|---------|---------|-------|
| The Odds API | YES | May 2023+ | $30-$249/mo | Best value for research |
| SportsDataIO | YES | 2020+ | Custom (est. $300+/mo) | Broadest history |
| OddsPapi | YES | Included | Free-$49/mo | Needs verification |
| SportsGameOdds | YES | YES | $75-$299/mo | Closing odds included |
| Goalserve | YES | 2010+ | $150-$250/mo | 14-day trial |
| Sportradar | YES | YES | $500-$1000+/mo | Enterprise only |
| OpticOdds | YES | YES | Custom ($$$$) | Enterprise only |
| OddsJam | YES | YES | Custom | Closing + opening lines |
| Sports Insights | Game only | 2003+ | Unknown | No player props |
| Betfair Historical | Limited | 2016+ | Per download | Exchange odds, weak NBA props coverage |

---

### Category 6: Prospective Collection (Start Now)

If no affordable historical source works, we can start archiving lines daily. Best options:

**The Odds API (free tier)**
- 500 credits/month free
- At 1 credit per market per region for live odds, this covers: ~500 single-market queries/month
- Strategy: Fetch player_points + player_rebounds + player_assists + player_threes for each game day (~15 games/day during season = ~60 queries/day). Would need paid tier at ~$30/mo.
- Data: JSON with actual O/U lines and odds from multiple sportsbooks

**BALLDONTLIE (free tier)**
- Free at 5 requests/minute
- Player props endpoint returns all props for a game in one call
- Strategy: Poll before each game, archive to local DB
- Risk: They explicitly don't store historical data, so if we miss a day, it's gone

**PrizePicks scraping**
- Free (scraping with Selenium)
- Risk: Fragile, can break with site changes, possibly against ToS

**Recommended prospective collection stack**:
1. Daily cron job at 10am ET (before games)
2. Fetch The Odds API player props for all NBA games that day
3. Store raw JSON + parsed CSV in `betting/data/props/`
4. After game, fetch actual stats from nba_api
5. Build up dataset over remainder of 2025-26 season + 2026-27

---

### Recommendations (Ranked)

1. **The Odds API historical backfill** ($59-119 for ~3 seasons of data, May 2023 to present) — most practical, verified data, clear API, includes actual O/U line numbers from major US sportsbooks. Start here.

2. **OddsPapi** ($49/mo, historical included) — potentially better value if their NBA player props history is deep. Verify with free tier first.

3. **Start prospective collection now** using The Odds API free tier ($0) or $30/mo tier — even if we get historical data, we'll want fresh lines going forward.

4. **Goalserve 14-day trial** — test whether their historical player props data is real and usable before committing $150/mo.

5. **SportsDataIO** — if budget allows, their 2020+ coverage is the deepest available. Contact for pricing.

### Next Steps
- [ ] Sign up for The Odds API free key, test historical player props endpoint for a sample NBA game from 2024
- [ ] Sign up for OddsPapi free tier, verify NBA player props history depth
- [ ] Build prospective collection script (`betting/scripts/collect_props.py`) using The Odds API
- [ ] Once data acquired: build player stat prediction model, compare predicted O/U vs actual lines, backtest

---

## 2026-03-31 — Phase 13: Player Props Model Quality Assessment

### Motivation
NBA moneylines proven unbeatable (Phase 12). Player props have higher vig (~8-15%) but weaker oddsmaker modeling — a potential edge. Before investing in historical prop line data ($59-119/mo), establish whether our prediction quality warrants it.

### Data
- 664,622 player-game rows, 25 seasons (2001-2025), 2,508 unique players
- Traditional box scores: PTS, REB, AST, FG3M, STL, BLK, MIN per player per game
- Downloaded via bulk `PlayerGameLogs(measure_type='Base')` endpoint

### Method
Walk-forward evaluation: for each season 2015-2025, predict player stats using only prior data. Features: expanding mean, rolling-5, rolling-10, rolling-20, opponent defense adjustment, home/away. Simple weighted blend model (35% expanding + 30% roll10 + 20% roll5 + 15% opponent-adjusted).

### Results

| Prop | Avg Value | Std | Expanding MAE | Model MAE | Improvement |
|------|-----------|-----|---------------|-----------|-------------|
| PTS | 10.6 | 8.6 | 5.22 | **4.70** | +10.0% |
| REB | 4.2 | 3.5 | 2.09 | **1.95** | +6.6% |
| AST | 2.4 | 2.6 | 1.41 | **1.31** | +7.0% |
| 3PM | 1.1 | 1.4 | 0.85 | **0.82** | +3.5% |
| PRA | 17.2 | 11.9 | 7.14 | **6.25** | +12.5% |

### Key Findings

1. **Model improves 7-12% over naive expanding mean** consistently across all 11 test seasons.
2. **Rolling-10 is surprisingly competitive** — nearly matches the full model. Simple recency is a strong baseline for props.
3. **Opponent adjustment adds 1-3%** — real signal but small.
4. **PTS MAE of 4.7 points** — cannot determine profitability without actual prop lines. Need directional accuracy against market lines, not just MAE.

### Decision
Model quality is sufficient to warrant acquiring historical prop lines. Set up automated prop line collection via The Odds API. Circle back for market efficiency test once sufficient data accumulated.

### Files
- `prediction/scripts/player_props_model.py` — props quality assessment
- `prediction/output/reports/player_props_quality.json` — full results
- `data/raw/all_player_traditional.csv` — 664K player rows (PTS, REB, AST, etc.)

---

## 2026-03-29 — Phase 12: Retroactive P&L Simulation & Market Efficiency Analysis

### Motivation
The prediction model converged at LOO-CV 0.6087. The H4 ensemble disagreement signal showed 70.1% accuracy on high-agreement games vs 63.0% on low-agreement. The question: can this translate to actual betting profit?

### Data
- **Closing moneylines**: 23,612 games (2007-2025) from SportsBookReview via kyleskom GitHub repo
- **Opening spreads**: 13,903 games (2011-2021) from FinnedAI SBR scraper. Converted to approximate opening moneylines using empirically-derived spread-to-ML table
- Walk-forward simulation: for each season S, train ensemble on all prior seasons, predict S, bet against actual lines

### Phase 12a: Simulation Against Closing Lines

**Result: All 6 strategies lose 100% of bankroll.**

| Strategy | Bets | Win Rate | Final Bankroll |
|----------|------|----------|---------------|
| Flat 0.25 Kelly | 16,107 | 41.7% | $0 |
| Q1/Q2 only | 6,350 | 42.0% | $0 |
| Disagreement-scaled | 16,107 | 41.7% | $0 |
| Threshold conservative | 4,403 | 41.2% | $0 |

**Root cause — Model vs Market Log Loss:**

| Predictor | Log Loss | vs Model |
|-----------|----------|----------|
| Our model | 0.6120 | — |
| Market (closing, vig-free) | 0.5918 | **-0.0202 better** |

The market beats our model in **every single season** (0/18 seasons won). The gap (0.020 LL) represents the market's information advantage: real-time injuries, lineup announcements, sharp money, and collective bettor intelligence.

**Model calibration**: Slightly overconfident at every probability level (1-3% gap). When the model says 65%, actual is ~62.4%.

### Phase 12b: Opening Line vs Closing Line Comparison

**Key question**: The closing line has same-day information we lack. Does our model beat OPENING lines (similar ~24-hour lead time)?

**Log Loss Comparison (walk-forward, 2012-2021):**

| Predictor | Log Loss |
|-----------|----------|
| Our model | 0.6219 |
| Opening lines (approx) | 0.6076 |
| Closing lines | 0.6008 |

**Model loses to BOTH opening and closing lines.** The market is 0.014 LL better even at opening time.

**Information flow**: Opening → Closing gains 0.007 LL from same-day information. But even before that information arrives, the opening line is already 0.014 LL better than our model.

### Key Findings

1. **The NBA moneyline market is efficient at ALL lead times.** Our model cannot beat opening lines (24h before) or closing lines (0h before). Professional oddsmakers use the same historical information our model uses, plus proprietary data.

2. **The 0.020 LL gap decomposes as:**
   - 0.014 LL: Structural oddsmaker advantage (better base models, proprietary data)
   - 0.007 LL: Same-day information flow (injuries, lineups, sharp money)

3. **Win rate is 41.7%** when betting perceived edge games — we're betting the wrong side more often than not. The ensemble disagreement signal doesn't help because the model's probability estimates are systematically less accurate than the market's.

4. **NBA moneylines are among the MOST efficient sports betting markets** — deep liquidity, sophisticated bettors, accurate odds. This is the hardest market to beat.

### Implications for Strategy

- **The prediction model is excellent for ANALYSIS but not for BETTING** against NBA moneylines
- Potential paths to profitability require:
  - Real-time information (injuries, lineups) — requires changing operational assumptions
  - Less efficient markets (player props, live betting, smaller leagues)
  - Opening line timing exploitation at less sharp books
  - Alternative sports with larger market inefficiencies

### Files
- `betting/scripts/retroactive_simulation.py` — Closing line simulation
- `betting/scripts/retroactive_opening_lines.py` — Opening vs closing comparison
- `betting/data/lines/nba_moneylines_2007_2025.csv` — 23,612 games closing lines
- `betting/data/lines/nba_archive_10Y.json` — 13,903 games with opening spreads
- `betting/output/reports/retroactive_simulation.json` — Closing line results
- `betting/output/reports/retroactive_opening_lines.json` — Opening line results

---

## 2026-03-29 — Phase 11: Advanced Stats Integration & Model Convergence

### Context
Advanced team and player data fully downloaded (23.7 min via bulk endpoints — see log entry from 2026-03-28). All 25 seasons at 100% coverage: 63,878 team rows, 664,125 player rows. Ran the systematic test suite against the Phase 10 ensemble baseline (LOO-CV 0.6087).

### Phase 11a: Advanced Team Stats Ablation (N=10,000)

**Baselines:** game-log+elo = 0.6113, advanced-only = 0.6757, best combined = **0.6065** (single val-split).

**Per-feature verdicts:**

*Features that HELP (KEEP):*
- diff_DEF_RATING (-0.0059), diff_OREB_PCT (-0.0058) — strong
- diff_scoring_margin (-0.0021), diff_PACE (-0.0006), diff_AST_PCT (-0.0007)
- diff_elo (-0.0154), rest_days_home (-0.0052), rest_days_away (-0.0025)

*Features that HURT (DROP):*
- diff_OFF_RATING (+0.0066), diff_DREB_PCT (+0.0058) — strong DROP
- diff_win_pct_last10 (+0.0053) — becomes harmful with advanced stats available

*Observation:* OFF_RATING hurts but DEF_RATING helps. DREB_PCT hurts but OREB_PCT helps. Striking offensive/defensive asymmetry.

### Phase 11b: Player Features Ablation (N=10,000)

**All 6 player-aggregated features are MARGINAL** (delta ~0.0000, top-50 inclusion ~50% = random):
- top3_usg, top3_pie, bench_pie_gap, pie_std, usg_hhi, min_weighted_net_rtg

Player-level aggregates provide zero signal beyond what team-level stats already capture. The improvement in the experiment (0.6116 → 0.6067) came entirely from the toggled marginal TEAM features (fg3_pct, win_pct_last10, rest_days), not from player features.

### Phase 11c: Multi-Architecture LOO-CV with Advanced Stats (600 fits)

**LOO-CV Mean LL (sorted by cross-architecture mean):**

| Architecture | ph10_best_10a | ph10_production | ph11_advanced | ph11_kitchen | ph11_adv_full |
|-------------|--------------|----------------|---------------|-------------|--------------|
| xgboost | 0.6094 | **0.6093** | 0.6095 | 0.6096 | 0.6097 |
| catboost | 0.6096 | **0.6094** | 0.6095 | **0.6094** | 0.6099 |
| lightgbm | **0.6096** | 0.6097 | **0.6096** | 0.6097 | 0.6097 |
| logistic | **0.6097** | 0.6107 | 0.6110 | 0.6108 | 0.6111 |
| random_forest | **0.6121** | 0.6125 | 0.6125 | 0.6132 | 0.6126 |
| **MEAN** | **0.6101** | 0.6103 | 0.6104 | 0.6105 | 0.6106 |

**Critical finding: Advanced stats do NOT improve LOO-CV for any architecture.** The Phase 11a val-split improvement was another artifact. ph10_best_10a remains the best cross-architecture config.

### Phase 11d: Ensemble with Advanced Stats (8 ensembles × 24 folds)

| Ensemble | #M | LOO-CV | Test LL |
|----------|-----|--------|---------|
| ph11_lgb_adv | 3 | **0.6087** | 0.6103 |
| ph10_best3 | 3 | **0.6087** | 0.6099 |
| ph10_xgb_lr | 2 | **0.6087** | 0.6086 |
| ph11_all_adv | 3 | 0.6089 | 0.6102 |

Three ensembles tie at 0.6087. Advanced stats neither help nor hurt the ensemble. The simplest option — XGBoost(production) + Logistic(best_10a) — matches the 3-model ensemble.

### Model Convergence Declaration

**The prediction model has converged at LOO-CV = 0.6087.** We have exhaustively tested:
- 30+ feature engineering approaches (game-log, advanced, player, SOS, schedule, travel, altitude, pace)
- 5 model architectures (XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression)
- 31 ensemble combinations with equal/optimized/stacked/calibrated weighting
- 8 ensemble refinement hypotheses
- Temporal variants, decay weighting, cross-season memory
- Training window optimization (start year 2001-2016)

Nothing moves LOO-CV below 0.6087. The remaining gap to the theoretical NBA prediction ceiling (~0.58-0.60) requires qualitatively new information: real-time injury/lineup data, market-derived signals, or information our operational assumptions exclude (predictions made ≥1 day before tip-off).

### Decisions
1. **Production prediction model**: XGBoost(production) + Logistic(best_10a), equal-weight average. LOO-CV = 0.6087.
2. **Advanced stats and player features are not included** in production — they don't improve LOO-CV.
3. **Phase 10a context features (schedule/travel) are included only in the Logistic component** (best_10a config) — they help linear models specifically.
4. **Focus shifts to betting routine optimization**: the H4 ensemble disagreement signal (70.1% accuracy on high-agreement games vs 63.0% on low-agreement) is the highest remaining practical value.

### Files
- `prediction/scripts/experiment_phase11_loocv.py` — Phase 11c multi-arch LOO-CV
- `prediction/scripts/experiment_phase11_ensemble.py` — Phase 11d ensemble experiment
- `prediction/scripts/fetch_bulk_advanced.py` — Bulk data download (320x faster)
- `prediction/output/reports/experiment_advanced_team.json` — 11a results
- `prediction/output/reports/experiment_player_features.json` — 11b results
- `prediction/output/reports/experiment_phase11_loocv.json` — 11c results
- `prediction/output/reports/experiment_phase11_ensemble.json` — 11d results

---

## 2026-03-28 — Advanced Data Download: Bulk Endpoint Discovery

### Problem
The original approach (`BoxScoreAdvancedV3`, one HTTP request per game) required ~32,000 API calls at 1.5s each, plus batch pauses and rate-limit cooldowns. After 3+ days of running, only 18% of team advanced and 18% of player advanced data had been downloaded. Subprocess timeouts (even at 4 hours) couldn't keep up with the per-game fetch rate.

### Solution
Discovered that `nba_api` has bulk endpoints that return an entire season of per-game advanced stats in a single API call:
- **`TeamGameLogs(measure_type='Advanced')`**: Returns ~2,460 rows per season (30 teams × ~82 games), all advanced stats (OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT, EFG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TM_TOV_PCT, POSS, PIE).
- **`PlayerGameLogs(measure_type='Advanced')`**: Returns ~26,000 rows per season (all players × all games), with OFF_RATING, DEF_RATING, NET_RATING, USG_PCT, PIE, TS_PCT, EFG_PCT, MINUTES, etc.

These endpoints are marked "deprecated" in NBA docs but remain fully functional as of March 2028. Data is numerically identical to BoxScoreAdvancedV3.

### Performance Comparison

| Metric | Old: BoxScoreAdvancedV3 | New: TeamGameLogs/PlayerGameLogs |
|--------|------------------------|----------------------------------|
| API calls | ~32,000 (1 per game) | ~100 (1 per season per type) |
| Time to complete | 3+ days (only 18% done) | **23.7 minutes (100% done)** |
| Speedup | — | **~320x** |
| Rate-limit risk | High (frequent cooldowns) | Minimal (100 calls total) |
| Resume needed | Yes (complex) | No (fast enough to restart) |

### Result
All 25 seasons (2001-2025) now have complete advanced data:
- **Team advanced**: 63,878 rows across 31,939 games (100% coverage)
- **Player advanced**: 664,125 rows across 31,939 games (100% coverage)

### Key Lesson
Always search for bulk/season-level endpoints before implementing per-record fetching. The NBA Stats API has many undocumented or "deprecated" endpoints that still work and are far more efficient.

### Files
- `prediction/scripts/fetch_bulk_advanced.py` — New bulk fetch script (replaces `fetch_all_advanced.py` for backfill)
- `data/raw/all_advanced.csv` — 63,878 team-advanced rows, all 25 seasons
- `data/raw/all_player_advanced.csv` — 664,125 player-advanced rows, all 25 seasons

### Next Steps
- Run `aggregate_player_stats.py` on complete player data
- Rebuild `matchup_features.csv` with full advanced coverage
- Run `experiment_advanced_team.py` and `experiment_player_features.py` (scripts already written, waiting on this data)

---

## 2026-03-27 — Phase 10: Feature Engineering, LOO-CV Validation, Multi-Architecture & Ensemble Experiments

### Overview

Phase 10 explored three directions: (a) new features derived from existing data (schedule, travel, altitude, SOS), (b) rigorous validation of findings via LOO-CV across multiple model architectures, and (c) ensemble modeling. The headline result: **ensembling provides the largest validated improvement of the entire project**, while the new features turned out to be largely redundant with existing ones under LOO-CV.

---

### Phase 10a: Schedule, Travel, and Altitude Features (N=10,000)

**New features added (14 context features, zero API calls):**
- Schedule: `is_b2b_home/away`, `is_3in4_home/away`, `games_last_7_home/away`, `consecutive_away_home/away`
- Travel: `travel_km_home/away`, `tz_change_home/away` (Haversine from `arenas.csv` with 35 entries — 30 current + 5 historical teams)
- Altitude: `altitude_ft` (venue), `altitude_diff` (home arena minus away team home arena)

**Also fixed a latent bug** in the rest_days merge — was joining on GAME_ID alone with `drop_duplicates`, which could match the wrong team's rest value. Now all per-team features merge on (GAME_ID, TEAM_ABBREVIATION).

**Per-feature verdicts (10K random search, single val-split):**

| Feature | Delta LL | Verdict | Top 50 Rate |
|---------|----------|---------|-------------|
| rest_days_away | -0.0010 | KEEP | 90% |
| is_b2b_away | -0.0011 | KEEP | 64% |
| rest_days_home | -0.0005 | MARGINAL | 68% |
| is_b2b_home | -0.0005 | MARGINAL | 56% |
| altitude_ft | -0.0001 | MARGINAL | 76% |
| tz_change_home | -0.0002 | MARGINAL | 66% |
| All others | ~0 | MARGINAL | 0-62% |

**Finding:** Away-team fatigue (rest_days_away, is_b2b_away) is the dominant context signal. is_b2b does NOT replace rest_days — they're complementary. Travel distance and altitude show signal through top-50 inclusion but don't move the median reliably.

---

### Phase 10b: SOS-Adjusted Efficiency & Pace Interaction (N=10,000)

**New features added (4 features):**
- `diff_sos`: expanding mean of opponent pre-game Elo (schedule difficulty)
- `diff_sos_adj_margin`: expanding mean of `PLUS_MINUS * (opp_elo / 1500)` — gives more credit for beating strong teams
- `diff_avg_PTS`: scoring rate proxy (pace from game logs)
- `pace_interaction`: `home_avg_PTS * away_avg_PTS / 10000`

**Key finding:** `diff_sos_adj_margin` correlates with home_win at 0.326 (vs raw diff_scoring_margin at 0.294) — the SOS adjustment adds real signal.

**Per-feature verdicts (10K random search):**

| Feature | Delta LL | Verdict | Top 50 Rate |
|---------|----------|---------|-------------|
| diff_sos_adj_margin | -0.0013 | KEEP | 98% |
| diff_scoring_margin | -0.0009 | KEEP | 36% (largely replaced by SOS) |
| diff_sos | -0.0000 | MARGINAL | 38% |
| diff_avg_PTS | +0.0002 | MARGINAL | 16% |
| pace_interaction | +0.0003 | MARGINAL | 4% |

**Redundancy:** SOS-adjusted margin alone (LL=0.6084) beats raw margin alone (0.6090). Including both (0.6085) gives negligible benefit. The SOS adjustment subsumes the raw signal.

**Baselines:** Production val LL = 0.6079. Best 10b search val LL = 0.6057 (-0.0022). Promising on single val-split.

---

### Phase 10c: LOO-CV Validation (24-fold)

**Motivation:** Phase 7 showed single val-split results can be artifacts. This experiment validates 10a/10b findings with leave-one-season-out CV.

**Configs tested:**
- `production`: 14 features (Phase 8 baseline)
- `best_10a`: 19 features (+ schedule/travel context)
- `best_10b`: 12 features (sos_adj_margin replaces scoring_margin)
- `combined`: 15 features (SOS + schedule/travel)
- `minimal`: 7 features (only consistently KEEP features)

**LOO-CV Results (XGBoost only):**

| Config | LOO Mean LL | Test LL | Test Acc | #Features |
|--------|-------------|---------|----------|-----------|
| **production** | **0.6093** | 0.6172 | 66.1% | 14 |
| best_10a | 0.6094 | 0.6137 | 66.1% | 19 |
| combined | 0.6098 | 0.6177 | 64.5% | 15 |
| best_10b | 0.6099 | 0.6158 | 65.3% | 12 |
| minimal | 0.6105 | 0.6257 | 64.6% | 7 |

**Critical finding: The 10a/10b improvements were val-split artifacts.** Production wins LOO-CV. No config beats production on more than 12/24 seasons. Season-level variance (std~0.018) is 30x larger than config differences (~0.0005).

**Decision:** Do not change production feature set based on 10a/10b alone. The game-log feature space is saturated for gradient boosting.

---

### Phase 10d: Multi-Architecture LOO-CV (5 architectures × 5 configs × 24 folds = 600 fits)

**Motivation:** Test whether the LOO-CV result is specific to XGBoost or holds across model families.

**Architectures:** XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression.

**LOO-CV Mean LL (rows=architecture, cols=config):**

| Architecture | best_10a | combined | production | best_10b | minimal |
|-------------|----------|----------|------------|----------|---------|
| xgboost | 0.6094 | 0.6098 | **0.6093** | 0.6099 | 0.6105 |
| catboost | 0.6096 | 0.6097 | **0.6094** | 0.6100 | 0.6105 |
| lightgbm | **0.6096** | 0.6100 | 0.6097 | 0.6104 | 0.6109 |
| logistic | **0.6097** | 0.6101 | 0.6107 | 0.6104 | 0.6111 |
| random_forest | 0.6121 | **0.6114** | 0.6125 | 0.6118 | 0.6119 |
| **MEAN** | **0.6101** | 0.6102 | 0.6103 | 0.6105 | 0.6110 |

**Key findings:**
1. Production wins with gradient boosting (XGBoost, CatBoost), but **best_10a wins with LightGBM and Logistic Regression**, and combined wins with Random Forest.
2. **best_10a has the best MEAN across all architectures** (0.6101 vs production's 0.6103).
3. **Logistic regression is surprisingly competitive** (0.6097 LOO-CV; best test LL of any architecture at 0.6059).
4. Different architectures prefer different feature sets — this implies **model diversity** is exploitable.

---

### Phase 10e: Ensemble Experiment (31 combinations × 24 folds)

**Motivation:** If different architectures prefer different features and capture different patterns, ensembling should improve predictions.

**Design:** Each architecture uses its LOO-CV-optimal feature config. All 31 non-empty subsets of {XGBoost, LightGBM, CatBoost, Random Forest, Logistic} tested. Ensemble via simple probability averaging. Pre-computed 120 arch-fold predictions, then assembled cheaply.

**Results — Single Models (LOO-CV):**

| Model | Config Used | LOO Mean | Test LL | Test Acc |
|-------|------------|----------|---------|----------|
| xgboost | production | 0.6094 | 0.6168 | 66.5% |
| catboost | production | 0.6097 | 0.6111 | 65.9% |
| lightgbm | best_10a | 0.6097 | 0.6140 | 65.9% |
| logistic | best_10a | 0.6098 | 0.6058 | 66.5% |
| random_forest | combined | 0.6114 | 0.6125 | 65.7% |

**Results — Best Ensembles:**

| Ensemble | #Models | LOO Mean | Test LL | Test Acc |
|----------|---------|----------|---------|----------|
| xgboost+lightgbm+logistic | 3 | **0.6088** | 0.6091 | 66.4% |
| xgboost+logistic | 2 | 0.6089 | 0.6083 | 66.4% |
| xgboost+catboost+logistic | 3 | 0.6089 | 0.6088 | **66.9%** |
| All 5 models | 5 | 0.6089 | 0.6089 | 66.3% |

**Best by test LL:** logistic alone (0.6058), then catboost+logistic (0.6074), then lightgbm+logistic (0.6075).

**Headline result:** Best ensemble LOO-CV = **0.6088** vs best single = 0.6094. LOO improvement = **-0.0005**. Test improvement = **-0.0077** (0.6091 vs 0.6168).

**Key findings:**
1. **Ensembling is the largest validated improvement in the project.** LOO-CV confirms it's real (not val-split luck).
2. **Logistic regression appears in every top-10 ensemble** — its linear boundary is complementary to tree models.
3. **Sweet spot is 2-3 models.** Beyond that, marginal returns are negligible.
4. **Diversity > accuracy** for ensembles: logistic regression is the weakest single model by LOO-CV (0.6098) but the most valuable ensemble component.

### Files
- `prediction/scripts/experiment_schedule_travel.py` — Phase 10a ablation
- `prediction/scripts/experiment_sos_pace.py` — Phase 10b ablation
- `prediction/scripts/experiment_loocv_validation.py` — LOO-CV validation (single architecture)
- `prediction/scripts/experiment_loocv_architectures.py` — Multi-architecture LOO-CV
- `prediction/scripts/experiment_ensemble.py` — Ensemble experiment
- `prediction/output/reports/experiment_schedule_travel.json` — 10a results
- `prediction/output/reports/experiment_sos_pace.json` — 10b results
- `prediction/output/reports/experiment_loocv_validation.json` — LOO-CV results
- `prediction/output/reports/experiment_loocv_architectures.json` — Multi-arch results
- `prediction/output/reports/experiment_ensemble.json` — Ensemble results
- `data/reference/arenas.csv` — Arena coordinates/elevation/timezone (35 entries)

---

### Phase 10f: Ensemble Hypothesis Testing (H1-H8)

Tested 8 hypotheses about improving the ensemble beyond simple averaging. All share the same LOO-CV base predictions (120 arch-folds + 24 MLP folds).

**Baseline:** best3 equal-weight average (XGBoost + LightGBM + Logistic) LOO-CV = 0.6088

| Hypothesis | Description | LOO-CV LL | vs Baseline | Verdict |
|-----------|-------------|-----------|-------------|---------|
| **H1** | Optimized weights (scipy, nested CV) | 0.6088 | +0.0000 | NEUTRAL |
| **H2** | Stacking meta-learner (logistic) | 0.6095 | +0.0007 | HURTS |
| **H3** | Add MLP to best3 | 0.6088 | -0.0000 | NEUTRAL |
| **H5** | Feature-diverse vs shared features | 0.6089 vs 0.6094 | -0.0005 | **IMPROVES** |
| **H6** | Caruana ensemble selection (30 iters) | 0.6090 | +0.0001 | NEUTRAL |
| **H7** | Advanced stats in ensemble | SKIPPED | — | NO DATA |
| **H8a** | Pre-calibration then average | 0.6091 | +0.0002 | NEUTRAL |
| **H8b** | Average then post-calibration | 0.6093 | +0.0004 | HURTS |

**H1 result:** scipy.optimize discovers weights of exactly 1/3 each — equal weighting is already optimal. The three models are equally valuable in the ensemble.

**H2 result:** Stacking HURTS (+0.0007). The meta-learner overfits with only 24 folds of meta-training data. With ~1,200 games per fold and only 3 meta-features, there isn't enough signal for the meta-learner to learn season-dependent weighting that generalizes.

**H3 result:** MLP adds nothing to the best-3 ensemble. As a single model, MLP LOO-CV is comparable to the other models, but it doesn't contribute additional diversity beyond what logistic regression already provides. The linear model is sufficient for the "non-tree" perspective.

**H4 result — the most actionable finding:**

| Disagreement Bin | N Games | Mean Std | Log Loss | Accuracy | Brier |
|-----------------|---------|----------|----------|----------|-------|
| Q1 (agree) | 5,714 | 0.008 | 0.5816 | **70.1%** | 0.1981 |
| Q2 | 5,713 | 0.013 | 0.6019 | 67.7% | 0.2071 |
| Q3 | 5,714 | 0.017 | 0.6092 | 66.9% | 0.2109 |
| Q4 | 5,713 | 0.022 | 0.6191 | 64.9% | 0.2154 |
| Q5 (disagree) | 5,714 | 0.034 | 0.6334 | **63.0%** | 0.2220 |

**Disagreement is strongly predictive of accuracy.** When models agree (Q1, std<0.01), accuracy is 70.1% — 7 points higher than when they disagree (Q5, 63.0%). Log loss improves by 0.05 from Q5 to Q1. This directly maps to bet sizing: high-agreement games deserve larger Kelly fractions.

**H5 result:** Feature-diverse ensemble (each model on its own optimal features) beats shared-feature ensemble by -0.0005 LOO-CV. This validates the design choice — diversity in feature views contributes meaningfully to ensemble quality.

**H6 result:** Caruana selection converges to equal 1/3 weights, same as H1. The three models are symmetrically valuable.

**H8 result:** Both calibration strategies HURT. The ensemble is already well-calibrated (averaging has a known calibration-improving effect). Adding isotonic calibration introduces overfitting given the limited calibration training data per fold.

### Key Findings

1. **Simple equal-weight averaging is optimal.** No weighting scheme, meta-learner, or calibration strategy improves over 1/3-1/3-1/3 averaging. The ensemble is already at its ceiling for the current base models.

2. **The only validated improvement from this series is H5 (feature diversity)**, confirming that each model should use its own optimal feature set.

3. **H4 (disagreement → accuracy) is the highest-value finding for the betting routine.** Ensemble std provides a reliable confidence signal: bet more on high-agreement games.

4. **MLP doesn't add diversity beyond logistic regression.** The linear/nonlinear diversity is already captured by the logistic + tree combination.

5. **Stacking and calibration both overfit** with 24 seasons of data. These techniques would need 50+ seasons to learn reliably.

### Decisions
1. **Production ensemble: equal-weight average of XGBoost + LightGBM + Logistic.** No weights to tune, no meta-learner to maintain.
2. **Each model uses its own optimal feature config** (H5 confirmed).
3. **Integrate ensemble std into betting routine** as a confidence multiplier on Kelly fraction (H4).
4. **Do not use stacking or calibration** on the ensemble (H2, H8 — they hurt).
5. **Advanced stats (H7) remain the highest-ceiling opportunity** — blocked on data download.

---

## 2026-03-25 — Phase 9: Per-Feature Memory Experiment

### Motivation
Phase 8 identified which features help/hurt but couldn't explain *how* each feature uses temporal information. All features use expanding season-mean with no cross-season memory. This experiment systematically tests how each useful feature responds to different memory configurations — window shape, decay weighting, and cross-season bleed — while holding all other features at production defaults. The goal is understanding, not optimization.

### Method
For each of 8 features (scoring_margin, three_pt_rate, fg_pct, ft_rate, tov_rate, fg3_pct, ast_rate, win_pct), test all combinations of:
- **Window size** (6): expanding, rolling_5, rolling_10, rolling_20, rolling_40, rolling_82
- **Decay** (7): 0.0 (uniform), 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 — exponential weighting within window
- **Cross-season** (2): reset (per-season, production default) vs bleed (carry across seasons)

One feature varied at a time, all others at production defaults. Fixed model hyperparams. Total: 672 model fits in 130 min.

### Results

**Per-feature best config vs production baseline (expanding, decay=0, reset = 0.6112):**

| Feature | Best LL | Delta | Window | Decay | Cross-Season |
|---------|---------|-------|--------|-------|-------------|
| **ast_rate** | **0.6100** | **-0.0012** | rolling_10 | 0.3 | reset |
| fg_pct | 0.6105 | -0.0007 | expanding | 0.5 | reset |
| win_pct | 0.6105 | -0.0007 | rolling_10 | 0.1 | bleed |
| ft_rate | 0.6106 | -0.0005 | rolling_10 | 0.5 | bleed |
| three_pt_rate | 0.6107 | -0.0005 | rolling_10 | 0.1 | reset |
| tov_rate | 0.6107 | -0.0005 | rolling_20 | 0.0 | bleed |
| fg3_pct | 0.6108 | -0.0004 | rolling_10 | 0.3 | reset |
| scoring_margin | 0.6111 | -0.0001 | rolling_40 | 0.0 | reset |

**Cross-season bleed effect (median LL: reset vs bleed):**

| Feature | Reset | Bleed | Delta |
|---------|-------|-------|-------|
| scoring_margin | 0.6122 | 0.6123 | +0.0001 |
| three_pt_rate | 0.6113 | 0.6114 | +0.0001 |
| fg_pct | 0.6111 | 0.6110 | -0.0000 |
| ft_rate | 0.6115 | 0.6113 | -0.0001 |
| tov_rate | 0.6111 | 0.6113 | +0.0002 |
| fg3_pct | 0.6111 | 0.6112 | +0.0001 |
| ast_rate | 0.6107 | 0.6108 | +0.0001 |
| win_pct | 0.6111 | 0.6111 | +0.0001 |

**Marginal effects (median LL across all features):**
- Window: all within 0.6111-0.6114 — essentially flat
- Decay: all within 0.6111-0.6113 — essentially flat
- Cross-season: reset and bleed within 0.0002 for every feature

### Key Findings

1. **The memory landscape is remarkably flat.** All 672 configs fall within a 0.003 LL range (0.6100-0.6131). No feature shows a dramatic response to any memory configuration. The production default (expanding, no decay, season reset) is near-optimal for every feature.

2. **ast_rate is the most memory-sensitive feature** (-0.0012 from baseline). It clearly prefers rolling_10 with moderate decay (0.3) — recent assist patterns are more predictive than season-long averages. This is the only feature where memory tuning yields a meaningful (though still small) improvement.

3. **rolling_10 is the most common optimal window** (5 of 8 features). ~10-game recent form is a sweet spot, suggesting team performance over the last 2-3 weeks is the most informative lookback for most stats.

4. **Cross-season bleed has no effect.** Median deltas are all within ±0.0002. NBA teams change enough between seasons (roster turnover, coaching changes) that carrying stats across season boundaries provides no benefit. Elo's 75% mean reversion already captures the right amount of cross-season memory.

5. **scoring_margin is the most temporally stable feature** — barely responds to any memory config (best is only 0.0001 better than baseline). Its predictive signal is robust regardless of how you compute it.

6. **High decay (1.0) uniformly hurts.** For every feature, decay=1.0 (only most recent game matters) produces worse results than moderate decay. Team quality requires at least a few games to estimate.

7. **The production expanding mean is a safe, near-optimal default** for all features. The marginal improvements from tuning memory are smaller than the noise between validation splits, making them unreliable to act on.

### Decision
- **No production changes.** The improvements are too small (max -0.0012 for ast_rate) and within val-split noise. The expanding season-mean default is confirmed as a sound choice.
- **ast_rate is the only candidate** for a memory tweak (rolling_10, decay=0.3) if we later need marginal gains — but only after validation with LOO-CV.
- **Cross-season bleed is not worth implementing** in production — zero benefit for added complexity.

### Files
- `prediction/scripts/experiment_memory.py` — experiment script
- `prediction/output/reports/experiment_memory_raw.csv` — 672 raw results
- `prediction/output/reports/experiment_memory.json` — summary

---

## 2026-03-25 — Phase 8: Feature Ablation with Temporal Variant Search (10K)

### Motivation
Previous experiments tested features one-at-a-time (Phase 3 EWMA sweep). This misses interactions — e.g., a feature might only help when combined with a specific temporal window for another feature. Additionally, the Phase 3 EWMA experiment held all other features at expanding mean, so it couldn't detect cases where a feature is redundant given other features' optimal representations.

### Method
10,000-iteration random search over feature inclusion × temporal variant selection. For each of the 9 stat features, randomly select one of 8 options: {excluded, expanding, rolling_5, rolling_10, rolling_20, ewma_5, ewma_10, ewma_20}. For each of 4 non-temporal features (elo, rest_home, rest_away, is_playoff), randomly include/exclude. Model hyperparameters fixed to Phase 6/7 best.

Pre-built "mega feature matrix" with all 63 differential columns (9 features × 7 variants) + elo + rest + is_playoff. Each iteration is just column selection from this pre-cached matrix — no feature rebuilding needed.

### Results

**Per-feature importance (median val LL: included vs excluded):**

| Feature | Excluded | Included | Delta | Verdict |
|---------|----------|----------|-------|---------|
| **diff_elo** | 0.6308 | 0.6137 | **-0.0171** | Essential |
| **scoring_margin** | 0.6298 | 0.6253 | **-0.0045** | Strong keep |
| **three_pt_rate** | 0.6272 | 0.6252 | **-0.0020** | Keep |
| fg_pct | 0.6260 | 0.6252 | -0.0009 | Keep |
| ft_rate | 0.6260 | 0.6252 | -0.0008 | Keep |
| tov_rate | 0.6258 | 0.6252 | -0.0006 | Keep |
| fg3_pct | 0.6254 | 0.6253 | -0.0001 | Marginal |
| is_playoff | 0.6254 | 0.6252 | -0.0002 | Marginal |
| rest_days_away | 0.6254 | 0.6253 | -0.0001 | Marginal |
| ast_rate | 0.6250 | 0.6254 | +0.0003 | Marginal |
| oreb_rate | 0.6248 | 0.6254 | **+0.0006** | Drop |
| **win_pct** | **0.6152** | **0.6253** | **+0.0101** | **Drop** |
| **rest_days_home** | **0.6152** | **0.6257** | **+0.0105** | **Drop** |

**Per-variant marginal (across all features):**

| Variant | Median val LL |
|---------|---------------|
| expanding | **0.6248** |
| ewma_10 | 0.6249 |
| ewma_5 | 0.6253 |
| rolling_20 | 0.6253 |
| ewma_20 | 0.6254 |
| rolling_5 | 0.6254 |
| rolling_10 | 0.6256 |

**Feature inclusion rate in top 50 configs:**
- 100%: diff_elo, scoring_margin, three_pt_rate
- 90-98%: ft_rate (98%), oreb_rate (94%), fg_pct (92%), win_pct (90%)
- 82-88%: ast_rate (88%), fg3_pct (82%), tov_rate (84%)
- 54-64%: rest_days_home (64%), is_playoff (60%), rest_days_away (54%)

**Best config** (val LL=0.6104, test LL=0.6256): 8 features — win_pct:rolling_10, scoring_margin:expanding, fg_pct:ewma_5, fg3_pct:rolling_20, three_pt_rate:rolling_10, ft_rate:ewma_20, oreb_rate:rolling_10, diff_elo

**Best test LL** (rank 15, val LL=0.6113, test LL=0.6182, acc=65.8%): 11 features

### Key Findings

1. **win_pct hurts the model (+0.0101 LL when included).** This is the most surprising result. Elo already captures team quality (and is 10x more important). Adding win_pct on top introduces noise — likely because win_pct is a noisier, less informative version of the same signal Elo provides. The top 50 configs still include it 90% of the time, but the marginal analysis shows it's net negative.

2. **rest_days_home hurts (+0.0105 LL when included).** Home rest days add noise. This contradicts the common belief that rest matters — Elo and other features may already capture rest effects indirectly. Notably, rest_days_away is neutral (-0.0001), suggesting asymmetric noise: home team rest is less predictive than away team rest.

3. **oreb_rate is droppable (+0.0006).** Offensive rebounding rate adds no value. The model already captures shooting efficiency through fg_pct and scoring_margin.

4. **Expanding mean is marginally best (0.6248 median)** across all features, confirming Phase 3. But the differences are tiny (~0.0008 across variants). The temporal representation matters far less than which features are included.

5. **Three core features dominate**: elo (-0.0171), scoring_margin (-0.0045), three_pt_rate (-0.0020). These three appear in 100% of top 50 configs. Everything else contributes < 0.001 LL.

6. **No feature × variant interaction detected.** The heatmap shows no feature that dramatically benefits from a non-expanding variant. Some features show slightly better results with specific variants (e.g., three_pt_rate with rolling_10), but the effect sizes are within noise.

### Decision
- **Do not change production features yet** — the improvements are very small (0.6104 val vs 0.6108 production) and the test LL story is mixed (best test=0.6182 but best val's test=0.6256). The findings inform future feature engineering priorities:
  - Consider dropping win_pct, rest_days_home, oreb_rate in a targeted follow-up
  - Prioritize advanced stats (ORTG, DRTG, NET_RATING) over refining temporal windows
- **Expanding mean confirmed as default** — no compelling reason to switch temporal variant for any feature

### Files
- `prediction/scripts/build_mega_features.py` — builds the 63-variant mega feature matrix
- `prediction/scripts/experiment_feature_ablation.py` — 10K ablation experiment
- `prediction/output/reports/experiment_feature_ablation_raw.csv` — 10K raw results
- `prediction/output/reports/experiment_feature_ablation.json` — summary
- `data/processed/mega_features.csv` — mega feature matrix (30,787 rows × 73 columns)

---

## 2026-03-25 — Phase 7: Cross-Validated Start Year Analysis

### Motivation
Phase 6's 10K search showed start_year=2007 dramatically outperforming all others (val LL 0.5889 median vs 0.5916 for 2008). All top 50 configs used 2007. However, the experimental design had a confound: each start_year used a **different random val split** of a different-length season list. This meant apparent start_year effects could actually reflect val-set difficulty rather than training window quality.

### Method
Leave-one-season-out cross-validation per start_year with **fixed hyperparameters** (from Phase 6 best: depth=3, lr=0.08, n_est=750, mcw=3, ss=0.7, cs=1.0). For each start_year (2001-2016), every available non-test season takes a turn as val. Reports mean val LL across all folds — an unbiased estimate of each training window's value.

### Results

| Start | Folds | Mean LL | Median LL | Test LL | Test Acc |
|-------|-------|---------|-----------|---------|----------|
| **2002** | 23 | **0.6101** | 0.6115 | 0.6216 | 65.3% |
| 2007 | 18 | 0.6102 | 0.6118 | 0.6235 | 64.6% |
| 2004 | 21 | 0.6105 | 0.6131 | 0.6203 | 66.2% |
| 2003 | 22 | 0.6108 | 0.6126 | 0.6221 | 65.7% |
| **2001** | **24** | **0.6108** | 0.6134 | **0.6192** | **66.2%** |
| 2006 | 19 | 0.6108 | 0.6138 | 0.6185 | 66.0% |
| 2008 | 17 | 0.6119 | 0.6109 | 0.6207 | 65.4% |
| 2009 | 16 | 0.6134 | 0.6137 | 0.6230 | 65.6% |
| 2012 | 13 | 0.6183 | 0.6175 | 0.6300 | 65.0% |
| 2016 | 9 | 0.6288 | 0.6274 | 0.6263 | 66.0% |

Per-val-season difficulty (mean LL when used as val, across all start_years):
- Easiest: 2007 (0.5825), 2008 (0.5848), 2015 (0.5874)
- Hardest: 2022 (0.6492), 2020 (0.6400), 2021 (0.6379)

### Key Findings

1. **The Phase 6 result was an artifact.** 2007 appeared dominant because the 2007 _season_ is the easiest to predict (LL=0.5825 when used as val), and the random split for start_year=2007 happened to place easy seasons in val. With proper CV, start_years 2001-2008 are essentially tied (mean LL 0.6101-0.6119).

2. **More data is monotonically better** for CV mean LL. Clear degradation as start_year increases beyond 2008 — fewer training seasons = worse performance. The "sweet spot at 2013" from Phase 4 was also a val-split artifact.

3. **Test LL confirms: use all data.** Start_year=2001 has the best test LL (0.6192) and accuracy (66.2%), outperforming 2007 (0.6235, 64.6%) and 2008 (0.6207, 65.4%). Old NBA data helps, not hurts.

4. **Season difficulty varies wildly** — 2022 (LL=0.6492) is nearly 0.07 worse than 2007 (LL=0.5825). This ~0.07 range in val-season difficulty dwarfs the ~0.002 differences between start_years in Phase 6, explaining how a single unlucky val assignment can create phantom effects.

5. **Model hyperparameters are insensitive** (confirmed from Phase 6): depths 2-5 essentially tied, learning rate and other params have minimal marginal effect.

### Decision
- Update `train_start_year` from 2008 to **2001** — use all available data
- Update model params to Phase 6 best: depth=3, lr=0.08, n_est=750, mcw=3, ss=0.7, cs=1.0
- **Methodological lesson**: any experiment comparing training windows must use leave-one-season-out CV or equivalent, never a single random val split

### Files
- `prediction/scripts/experiment_large_search.py` — Phase 6 10K search (confounded)
- `prediction/scripts/experiment_start_year_cv.py` — Phase 7 LOO-CV (corrected)
- `prediction/output/reports/experiment_large_search_raw.csv` — 10K raw results
- `prediction/output/reports/experiment_start_year_cv_raw.csv` — LOO-CV raw results
- `prediction/output/reports/experiment_large_search_params.png` — Phase 6 hyperparameter plots

---

## 2026-03-25 — Phase 6: 10K Large-Scale Hyperparameter Search

### Motivation
Resolve the Phase 4/5 conflict (2013 vs 2008 optimal start year) with a definitive 10,000-iteration random search across 16 start years × 6 model hyperparameters.

### Method
10,000 random search iterations. Parameter space: start_year [2001-2016], max_depth [2-8], learning_rate [0.005-0.2], n_estimators [100-1000], min_child_weight [1-10], subsample [0.5-1.0], colsample_bytree [0.4-1.0]. Pre-cached train/val splits per start_year (random 85/15, seed=42). Early stopping at 50 rounds.

### Results
- **Best config**: start_year=2007, depth=3, lr=0.08, n_est=750, mcw=3, ss=0.7, cs=1.0
- **Best val LL**: 0.5844 | **Test LL**: 0.6109 | **Test Acc**: 66.1%
- All top 50 configs used start_year=2007
- Depth 2-5 essentially tied (median LL 0.6129-0.6153)

### Key Finding — CONFOUNDED (see Phase 7)
The dramatic start_year=2007 advantage was an artifact of the experimental design: different start_years used different random val splits, and 2007 happened to get easy val seasons. **Phase 7's cross-validated analysis overturned this result.** The hyperparameter findings (depth, lr, etc.) remain valid as those were marginalized across start_years.

### Validated Hyperparameters
- max_depth=3 (marginal best, but 2-5 essentially tied)
- learning_rate=0.08 (mid-range values optimal)
- n_estimators=750 (converges via early stopping)
- min_child_weight=3, subsample=0.7, colsample_bytree=1.0

---

## 2026-03-24 — Phase 3: Baseline Experiments (Splits, Normalization, Decay)

### Split Redesign
Changed from random 70/15/15 to: **2024-25 held out as test**, remaining 2001-2023 randomly split (seed=42) into ~85% train / ~15% val. Each season is an independent unit — no data bleed. Random assignment avoids temporal bias while the 2024-25 holdout confirms current-era generalization.

### Re-Baselined Model
| Metric | Val | Test (2024-25) |
|--------|-----|---------------|
| Log Loss | 0.6041 | 0.6131 |
| Accuracy | 67.3% | 65.4% |

Val-test gap is 0.009 LL (much tighter than previous 0.026 with the old split). Tuned params: md=5, lr=0.08, ne=750, mcw=1, ss=0.6, cs=0.6.

### Normalization Decision: None
XGBoost tree splits are scale-invariant. Literature confirms: "Decision trees are not sensitive to the scale of features" (XGBoost docs, GitHub issue #357). No normalization applied.

### Categorical Treatment: Correct as-is
All 13 features are numeric (10 differentials + 2 rest days + 1 binary playoff flag). XGBoost handles all natively via threshold splits. No encoding needed.

### Experiment 2a: Sample-Level Decay Sweep

| Decay | Raw LL | Cal LL | Accuracy |
|-------|--------|--------|----------|
| 0.0 | 0.6115 | 0.6058 | 66.9% |
| 0.1 | 0.6113 | 0.6061 | 66.2% |
| **0.3** | **0.6101** | **0.6041** | **66.8%** |
| 0.5 | 0.6118 | 0.6063 | 66.9% |
| 1.0 | 0.6104 | 0.6053 | 67.2% |

**Finding**: Decay=0.3 is optimal (raw LL 0.6101). Range is tight (0.002 LL) — sample decay has modest impact. Default of 0.3 confirmed.

### Experiment 2b: Per-Feature EWMA Span Sweep

Tested 9 features × 4 spans (5, 10, 20, expanding). All features tested independently while others remain at expanding mean. Used sample decay=0.3 and baseline hyperparams.

| Feature | Span=5 | Span=10 | Span=20 | Expanding |
|---------|--------|---------|---------|-----------|
| win_pct | +0.0019 | +0.0032 | +0.0013 | **baseline** |
| scoring_margin | +0.0030 | +0.0018 | +0.0017 | **baseline** |
| fg_pct | +0.0009 | +0.0004 | +0.0023 | **baseline** |
| fg3_pct | +0.0039 | +0.0039 | +0.0022 | **baseline** |
| three_pt_rate | +0.0031 | +0.0038 | +0.0022 | **baseline** |
| ft_rate | +0.0028 | +0.0017 | +0.0022 | **baseline** |
| oreb_rate | +0.0020 | +0.0021 | +0.0015 | **baseline** |
| tov_rate | +0.0028 | +0.0024 | +0.0018 | **baseline** |
| ast_rate | +0.0020 | +0.0010 | +0.0026 | **baseline** |

*All values are delta LL vs baseline (positive = worse). Baseline = 0.6101.*

**Finding**: **No feature benefits from EWMA.** Expanding mean wins for every feature. The signal in NBA game prediction comes from stable season-level team quality, not recent form. Short EWMA windows introduce noise. The existing `win_pct_last10` and Elo K-factor updating already capture recency at the appropriate level.

### Experiment 3: Cross-Season Prior Initialization

Tested 4 approaches for carrying prior-season stats into new seasons (16 configs total):

| Config | Overall LL | Early LL | Mid LL | d_Overall | d_Early | d_Mid |
|--------|-----------|---------|--------|-----------|---------|-------|
| **BASELINE (no priors)** | **0.6126** | **0.6252** | **0.6099** | — | — | — |
| D: Zero noisy (Oct-Nov) | 0.6125 | 0.6253 | 0.6098 | -0.0001 | +0.0001 | -0.0001 |
| **A: Bayesian k=3** | **0.6119** | 0.6276 | **0.6085** | **-0.0007** | +0.0024 | **-0.0014** |
| A: Bayesian k=5 | 0.6142 | 0.6292 | 0.6110 | +0.0016 | +0.0040 | +0.0011 |
| B: Regress s=0.9 k=5 | 0.6129 | 0.6283 | 0.6096 | +0.0003 | +0.0031 | -0.0003 |
| B: Regress s=0.9 k=10 | 0.6134 | 0.6264 | 0.6105 | +0.0008 | +0.0012 | +0.0006 |
| C: Multi (all configs) | 0.6143-56 | 0.6308-23 | 0.6104-22 | +0.0017-30 | +0.0056-71 | +0.0005-23 |

*Early = Oct-Nov (660 games, 18%). Mid = Dec+ (3,028 games, 82%).*

**Key findings:**

1. **Best overall: Bayesian k=3** (LL 0.6119, -0.0007 vs baseline). This is a very fast-decaying prior (equivalent to only 3 games). The improvement comes entirely from mid-season (+0.0014 better), NOT early season.

2. **All priors HURT early-season predictions.** Every single prior configuration made Oct-Nov LL worse (by +0.0012 to +0.0071). The baseline's noisy-but-unbiased expanding mean outperforms biased prior-season values for early games.

3. **Stronger priors = worse performance.** k=3 is the only configuration that helps overall. k=5, 10, 20 all hurt. Regression-to-mean and multi-season approaches all hurt. This means prior-season stats are poor predictors of current-season performance — likely due to roster changes, coaching changes, and year-over-year variance in the NBA.

4. **The ablation (D) shows almost zero effect.** Zeroing noisy features for Oct-Nov barely changes anything (d=-0.0001 overall). XGBoost is already handling noisy early-season features reasonably well — it's learning to downweight them through tree splits.

5. **Elo already captures the cross-season signal.** The 75% Elo carryover provides the right amount of prior information. Adding more prior data on top is redundant or harmful.

**Interpretation:** NBA rosters and team dynamics change too much between seasons for last year's shooting percentages, scoring margins, etc. to be informative. Elo — which adapts through actual game results with margin-of-victory scaling — is a better cross-season bridge than any stat-level prior. The k=3 Bayesian result (-0.0007 overall) may be noise within the tuning variance.

**Decision:** Do not adopt cross-season priors. Keep the baseline (expanding mean, Elo-only cross-season carryover). The infrastructure is built and available if we want to revisit with advanced stats.

### Experiment 4: Training Window Size — How Much Historical Data Helps?

Tested 22 configurations: start year from 2001 to 2022. Test always 2024-25 season.

| Start | Seasons | Train | Val LL | Test LL | Test Acc |
|-------|---------|-------|--------|---------|----------|
| 2001 | 24 | 24,884 | 0.6101 | 0.6138 | 65.7% |
| 2005 | 20 | 19,999 | 0.6376 | 0.6150 | 65.5% |
| 2008 | 17 | 17,631 | 0.5909 | 0.6136 | 66.1% |
| 2010 | 15 | 15,313 | 0.6335 | **0.6128** | 65.8% |
| **2013** | **12** | **13,005** | **0.6097** | **0.6118** | **66.0%** |
| 2015 | 10 | 10,608 | 0.6083 | 0.6151 | 66.6% |
| 2018 | 7 | 6,951 | 0.6222 | 0.6137 | 65.4% |
| 2020 | 5 | 4,633 | 0.6453 | 0.6200 | 65.8% |
| 2022 | 3 | 2,307 | 0.6533 | 0.6278 | 65.0% |

**Correlation (start_year vs test_LL): +0.592** — more data is generally better.

**Key findings:**

1. **Best test LL: start=2013** (0.6118, -0.0020 vs baseline). Using 12 seasons (2013-2023) outperforms using all 24 seasons. The last decade of NBA data is more relevant than 2001-era data.

2. **Removing the oldest 12 seasons (2001-2012) helps by 0.002 LL.** This is a modest but consistent improvement — the early 2000s NBA (pre-analytics, different pace/style) slightly dilutes the model.

3. **Below ~7 seasons (start=2018+), performance degrades sharply.** Test LL jumps from 0.6137 to 0.6200-0.6278. Too little data = underfitting.

4. **The sweet spot is 10-15 seasons (start=2010-2015).** Enough data to train well, recent enough to be relevant. The 2013 optimum coincides with the start of the modern "analytics era" in the NBA.

5. **Val LL is noisy and not well correlated with test LL** — the random val split puts different seasons in val for each config, making val LL unreliable as a comparison metric across configs. Test LL (fixed 2024-25 season) is the reliable metric.

**Decision:** ~~Update default training window to start at 2013.~~ **SUPERSEDED by Phase 7**: the 2013 optimum was a val-split artifact. LOO-CV shows more data is monotonically better — use all available data (start_year=2001).

### Conclusion
The baseline feature engineering is sound. Improvements will come from:
1. **Better features** (advanced box scores with possession-adjusted stats — in progress)
2. **Injury data** (player availability — not yet started)
3. **Not from** normalization, categorical encoding, per-feature EWMA, cross-season priors, or training window restriction

---

## 2026-03-24 — Phase 5: Joint Hyperparameter + Training Window Tuning (SUPERSEDED by Phase 7)

> **Note**: The start_year=2008 finding was overturned by Phase 7's LOO-CV analysis, which showed the result was a val-split artifact. The hyperparameter findings (depth=3, shallow models preferred) remain valid and were confirmed by Phase 6's 10K search.

### Setup
100-iteration random search over 7 dimensions: start_year [2008-2015] + 6 XGBoost hyperparams. Val LL as objective, test on 2024-25 season.

### Best Config
- **start_year: 2008** | max_depth: 3, lr: 0.02, n_est: 750, mcw: 2, ss: 1.0, cs: 0.9
- Val LL: **0.5888** | Test LL: **0.6123** | Test Acc: 65.6%

### Comparison to Previous Baseline
| Metric | Phase 3 Baseline | Phase 5 Tuned | Delta |
|--------|-----------------|---------------|-------|
| Val LL | 0.6041 | 0.5888 | -0.0153 |
| Test LL | 0.6131 | 0.6123 | -0.0008 |
| Test Acc | 65.4% | 65.6% | +0.2% |

### Key Findings

1. **Start year 2008-2009 dominates top 20 configs** (20/20). Joint optimizer prefers ~15-17 seasons with a shallower model (depth=3) over fewer seasons with deeper model. More data + less complexity = better generalization.

2. **Feature importance**: diff_elo 50.9%, diff_scoring_margin 11.0%, diff_win_pct 6.3%. Shooting/rate stats each ~3%. Elo carries half the signal.

3. **Bottom features near zero**: rest_days_home (2.9%), diff_oreb_rate (2.9%), diff_fg3_pct (2.7%). Ablation shows dropping any has <0.0005 LL impact. Dropping fg3_pct actually improves test by 0.0005.

4. **Calibration hurts**: Isotonic calibration worsened LL from 0.6123 to 0.6168. With only 2,230 val games, isotonic overfits. May need Platt or temperature scaling for smaller val sets.

5. **Test improvement is modest** (-0.0008 LL). We're likely near the ceiling for game-log-based features. The next meaningful improvement requires **possession-adjusted advanced stats** (ORTG, DRTG, NET_RATING) which capture efficiency information that Elo and scoring margin only approximate.

### Decision
Adopt start_year=2008 + depth=3 as the new production config. The shallower model with more data is a better bias-variance tradeoff than the previous depth=5 on fewer seasons.

## 2026-03-24 — Phase 2: Enhanced Data Collection

### Motivation
Data provenance review identified that `collect_data.py` was only fetching basic box scores (PTS, FGM, FGA, etc.) via `LeagueGameFinder`, then `build_features.py` was approximating advanced stats (eFG%, TS%, etc.) from those raw numbers. The NBA's official `BoxScoreAdvancedV2` endpoint provides possession-adjusted ORTG, DRTG, NET_RATING, PACE, etc. — more accurate than our approximations because they use the NBA's actual possession calculation.

### Changes
1. **Added advanced box score fetching** to `collect_data.py` via `boxscoreadvancedv2`. One API call per game returns: OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT, EFG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TM_TOV_PCT.
2. **Simplified `build_features.py`** — loads official advanced stats directly instead of computing approximations from raw box scores. Removed all hand-rolled eFG%, TS%, etc. calculations.
3. **Added `--advanced-only` flag** to `collect_data.py` for incremental fetching after game logs already exist.
4. **Updated feature names** in train.py, predict.py, backtest.py to match official NBA column names (e.g., `diff_NET_RATING` not `diff_net_rating`).
5. **Documented data provenance** in CLAUDE.md: nba_api is unofficial, endpoints can break, historical data reliable, current-season updates after games final.

### Operational Assumptions (Simplified)
- Predictions generated >= 1 day before tip-off
- Batch data fetch (overnight refresh), no real-time polling
- Last-minute injuries out of scope
- Rate limiting is not a concern for batch historical fetch

### Data Budget
- Game logs: ~30 API calls per season (one per team) × 11 seasons = ~330 calls, ~4 min
- Advanced box scores: ~1,300 calls per season × 11 seasons = ~14,300 calls, ~2.8 hours initial backfill
- Total initial backfill: ~3 hours. Subsequent daily updates: ~5-10 calls

## 2026-03-24 — Phase 1: Initial Workflow Setup

### Motivation
Second prediction domain for youBet, building on lessons from NCAA March Madness workflow (12 phases, 0.496 val LL, 75.4% accuracy, +16% ROI on conviction bets).

### Architecture: Two Independent Routines
1. **Prediction** (`prediction/`): Optimize model quality (log loss, calibration). Pipeline: collect → elo → features → train → predict → backtest. Never sees betting lines.
2. **Betting** (`betting/`): Optimize wagering strategy. Consumes prediction outputs + market lines. Never influences model training.

The boundary is the model's predicted probabilities. This separation was motivated by the NCAA finding that calibration artifacts (Platt compression) created phantom betting edges — conflating prediction quality with betting strategy led to confusion about where the real issues were.

### Design Decisions
1. **Isotonic calibration** (not Platt) — NBA provides >10K calibration samples; isotonic is more flexible and avoids the Platt compression problem discovered in NCAA Phase 12
2. **15 stat differentials + 3 context features** — follows the 17-25 feature sweet spot from literature
3. **Rest days as non-differential features** — rest advantage is asymmetric
4. **No hardcoded model_min** — the 0.55 threshold used in NCAA was a Platt-specific workaround, not a general principle. Betting strategy parameters should be tuned independently of prediction quality.
5. **Shared betting logic in `src/youbet/core/betting.py`** — extracted from NCAA workflow to avoid cross-workflow imports

### Expected Performance Targets
- Val log loss: < 0.60 (NBA games are harder to predict due to parity)
- Accuracy: > 65% (NBA home team wins ~60%, so baseline is higher)
- Beating closing lines by 1-2% would be a strong result

### Next Steps
- Implement `collect_data.py` and verify NBA API data quality
- Compute Elo ratings and validate against known power rankings
- Build features and train initial baseline model
