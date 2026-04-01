# Data Sources Catalog

## NCAA Men's Basketball

| Source | Type | Cost | Coverage | Notes |
|--------|------|------|----------|-------|
| Kaggle March ML Mania | CSV download | Free | 2003-2017 | Detailed box scores 2003-2017; extended to 2025 via ESPN |
| ESPN Scoreboard API | API (unofficial) | Free | 2018-2025 | Compact + detailed results for the Kaggle gap period. Via `scrape_game_results.py` |
| KenPom | Web/API | $24.95/yr | 2002-present | Adjusted efficiency, tempo, SOS. Gold standard |
| Bart Torvik T-Rank | Web | Free | 2008-present | Similar to KenPom, free alternative |
| Sports Reference (CBB) | Web scraping | Free | Historical | Box scores, advanced stats. Rate limit: 1 req/3s |
| NCAA.com | Web | Free | Current season | Official stats, brackets |

### Kaggle March ML Mania Files
- `MRegularSeasonDetailedResults.csv` — Game-level box scores (2003-2017)
- `MNCAATourneyDetailedResults.csv` — Tournament game results (2003-2017)
- `MNCAATourneySeeds.csv` — Tournament seedings
- `MMasseyOrdinals.csv` — 100+ ranking systems over time
- `MTeams.csv` — Team ID mappings

### Scraped Data Files (ESPN)
- `data/raw/scraped/regular_season_results.csv` — Regular season game results (2018-2025)
- `data/raw/scraped/tournament_results.csv` — Tournament game results (2018-2025)
- `data/raw/scraped/unmapped_teams.txt` — ESPN team names that failed to map to Kaggle IDs

### ESPN API Endpoints
- **Scoreboard**: `/scoreboard?dates=YYYYMMDD&limit=400&groups=50` — daily game results with scores, team names, neutral site flag, OT info. Returns `location` (school name) and `displayName` (with mascot).
- **Summary**: `/summary?event={id}` — full team box scores per game. Stats in combined format (e.g., `"fieldGoalsMade-fieldGoalsAttempted": "30-62"`). Includes: FGM/FGA, 3PT, FT, OR/DR, Ast, TO, Stl, Blk, PF.
- **Rate limit**: No official limit; use 0.5s delay between calls. ~40K games at 0.5s = ~5.5 hours for detailed mode.
- **Coverage**: Historical data available back to at least 2017-18 season.

### KenPom Key Metrics
- AdjOE (Adjusted Offensive Efficiency)
- AdjDE (Adjusted Defensive Efficiency)
- AdjEM (Adjusted Efficiency Margin = AdjOE - AdjDE)
- AdjTempo
- SOS (Strength of Schedule)
- Luck rating

## NBA

| Source | Type | Cost | Coverage | Notes |
|--------|------|------|----------|-------|
| NBA API (nba.com/stats) | API (official) | Free | 1996-present | Team/player game logs, advanced stats. Use `nba_api` Python package. Soft rate limit ~1 req/0.6s |
| Basketball Reference | Web | Free | 1947-present | Historical box scores, advanced stats, per-100-poss. Rate limit 1 req/3s |
| Cleaning the Glass | Web | $10/mo | 2010-present | Shot quality, luck-adjusted ratings, lineup data. NBA-specific advanced analytics |
| ESPN API | API (unofficial) | Free | 2017-present | Game results, same scraping pattern as NCAA (`scrape_game_results.py`) |
| Covers.com | Web | Free | Historical | Historical closing lines, spreads, totals. For CLV tracking |
| SportsOddsHistory | Web | Free | 2007-present | Archived betting odds for backtesting |

### NBA API Key Endpoints (`nba_api` Python package)
- `leaguegamefinder.LeagueGameFinder` — All games for a season with box score stats
- `teamgamelog.TeamGameLog(team_id, season)` — Per-team game logs
- `teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits` — Team advanced stats splits
- `commonteamroster.CommonTeamRoster` — Roster and player info
- `boxscoreadvancedv2.BoxScoreAdvancedV2` — Per-game advanced box score (ORTG, DRTG, pace, eFG%, TS%)

### NBA Key Metrics
- ORTG / DRTG (Offensive / Defensive Rating — points per 100 possessions)
- Net Rating (ORTG - DRTG)
- Pace (possessions per 48 minutes)
- eFG% (Effective FG% — weights 3-pointers at 1.5x)
- TS% (True Shooting % — accounts for FT)
- AST% / TOV% / OREB% / DREB% (possession-based rates)
