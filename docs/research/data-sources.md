# Data Sources Catalog

## NCAA Men's Basketball

| Source | Type | Cost | Coverage | Notes |
|--------|------|------|----------|-------|
| Kaggle March ML Mania | CSV download | Free | 2003-present | Primary. Season/tournament results, seeds, ordinals |
| KenPom | Web/API | $24.95/yr | 2002-present | Adjusted efficiency, tempo, SOS. Gold standard |
| Bart Torvik T-Rank | Web | Free | 2008-present | Similar to KenPom, free alternative |
| Sports Reference (CBB) | Web scraping | Free | Historical | Box scores, advanced stats. Rate limit: 1 req/3s |
| ESPN | API (unofficial) | Free | Current season | Schedules, scores, rosters |
| NCAA.com | Web | Free | Current season | Official stats, brackets |

### Kaggle March ML Mania Files
- `MRegularSeasonDetailedResults.csv` — Game-level box scores
- `MNCAATourneyDetailedResults.csv` — Tournament game results
- `MNCAATourneySeeds.csv` — Tournament seedings
- `MMasseyOrdinals.csv` — 100+ ranking systems over time
- `MTeams.csv` — Team ID mappings

### KenPom Key Metrics
- AdjOE (Adjusted Offensive Efficiency)
- AdjDE (Adjusted Defensive Efficiency)
- AdjEM (Adjusted Efficiency Margin = AdjOE - AdjDE)
- AdjTempo
- SOS (Strength of Schedule)
- Luck rating
