# NCAA Men's Basketball March Madness

## Domain Context
Predict outcomes of NCAA Men's Basketball Tournament games and generate bracket picks. 68 teams, single elimination, seeded 1-16 in 4 regions.

## Pipeline

```bash
python scripts/collect_data.py       # Download Kaggle data (2003-2017)
python scripts/scrape_game_results.py --force  # Scrape ESPN 2018-2025 (~15 min compact, ~8 hrs --detailed)
python scripts/compute_elo.py        # Pre-tournament Elo from game data (2003-2025)
python scripts/build_features.py     # Build 16 stat differentials → matchup_features.csv
python scripts/train.py --tune --production  # Train XGBoost + Platt calibration (random 80/20 split)
python scripts/predict.py            # Predict tournament matchups
python scripts/generate_bracket.py   # Monte Carlo bracket simulation (East vs South, Midwest vs West)
python scripts/backtest.py           # Validate on 2005-2026 tournaments

# Other train.py modes:
python scripts/train.py --tune --random-split  # 60/20/20 random split (with test holdout)
python scripts/train.py --tune                 # Config temporal split (train 2003-2022, val 2023-2025)
```

## Data Sources
1. **Kaggle March ML Mania** (primary): Free CSV download. Detailed box scores 2003-2017, tournament results, seeds, ordinals.
2. **ESPN Scoreboard API** (free, unofficial): Game results 2018-2025 via `scrape_game_results.py`. Use `--detailed` for full box scores (26 columns). Uses `location` field for team name matching.
3. **KenPom** (subscription $24.95/yr): Adjusted efficiency metrics (AdjOE, AdjDE, AdjEM, AdjTempo), SOS, 2002-2026. Gold standard for college basketball analytics.
4. **CBB dataset** (Kaggle): Supplementary team stats (W/L, conference) for 2013-2025.
5. **Bart Torvik T-Rank** (free): Similar to KenPom. Good free alternative.

### Data Coverage
| Source | Seasons | Games | Notes |
|--------|---------|-------|-------|
| Kaggle detailed box scores | 2003-2017 | ~78K | Regular season + tournament |
| ESPN scraped (compact) | 2018-2025 | ~47K | Regular season + tournament (includes NIT/CBI) |
| KenPom features | 2002-2026 | — | 8,680 team-seasons |
| Elo (computed) | 2003-2026 | — | 14,439 team-seasons, real game data through 2025 |

## Features (16 differentials)
All computed as Team A stat - Team B stat:
1. adj_oe — Adjusted offensive efficiency (KenPom)
2. adj_de — Adjusted defensive efficiency (KenPom)
3. adj_em — Net efficiency margin (KenPom)
4. adj_tempo — Adjusted tempo (KenPom)
5. kenpom_rank — KenPom ranking
6. seed_num — Tournament seed
7. elo — Pre-tournament Elo rating (computed from game data 2003-2025)
8. win_pct — Season win percentage (game-level W/L for all seasons)
9. to_rate — Turnover rate
10. oreb_rate — Offensive rebound rate
11. ft_rate — Free throw rate
12. three_pt_rate — Three-point shooting rate
13. ast_rate — Assist rate
14. blk_rate — Block rate
15. stl_rate — Steal rate
16. experience — Team experience/continuity (KenPom)

## Key Considerations
- Tournament games are different from regular season (single elimination, neutral courts, pressure)
- Seed matchups have historical patterns (12-5 upsets are common, ~35%)
- First Four games and play-in teams behave differently
- Conference tournament performance is a recency signal
- Coach tournament experience matters but is hard to quantify
- Scraped tournament data uses DayNum >= 134 cutoff which includes NIT/CBI — filter to NCAA-only for clean evaluation

## Current Performance (Phase 11)
- Validation: Log Loss **0.496**, Accuracy **75.4%** (16,314 games across 5 random seasons)
- Model: XGBoost (md=7, lr=0.01, ne=1000, mcw=7, ss=0.8, cs=1.0, decay=0.3, early_stop=50) + Platt calibration
- Production split: Random 80/20 (18 train seasons / 5 val seasons), no test holdout
- Tuning: 100-iter random search with early stopping
- Feature importance: kenpom_rank 32%, elo 28%, adj_em 24% (balanced)
- 2026 prediction: Florida chalk, Duke balanced MC #1 (29.7%). See `output/bracket_2026.md`
- **Phase 10 conclusion**: All 5 experiments clustered within 0.005 LL — near ceiling for this feature set

## Research Log
See `research/log.md` — read at start of every session.
