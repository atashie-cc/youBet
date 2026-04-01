# NBA Game Prediction

## Domain Context
Predict NBA regular season and playoff game outcomes. 30 teams, 82-game regular season (1,230 games/year), 16-team playoffs with best-of-7 series. Season runs October through June.

## Operational Assumptions
- Predictions are generated **at least 1 day before tip-off**
- Data is fetched in batch (historical backfill or overnight refresh)
- No real-time or in-game data needed; no last-minute injury tracking
- Rate limiting is not a concern — we don't repeatedly poll the API

## Two Routines

This workflow is split into two independent routines with a clear boundary:

### Routine 1: Prediction (prediction/)
**Goal**: Minimize log loss. Produce well-calibrated win probabilities.

The prediction routine never sees betting lines. It is evaluated purely on model quality: log loss, accuracy, Brier score, and calibration curves.

```bash
# Data collection
python prediction/scripts/collect_data.py            # Fetch game logs (LeagueGameFinder)
python prediction/scripts/fetch_bulk_advanced.py     # Fetch advanced stats — FAST (~25 min for all 25 seasons)
python prediction/scripts/fetch_bulk_advanced.py --force  # Re-download all seasons

# Feature engineering
python prediction/scripts/compute_elo.py             # Pre-game Elo ratings from results
python prediction/scripts/aggregate_player_stats.py  # Aggregate player stats to team-level features
python prediction/scripts/build_features.py          # Build differentials + context features

# Training and prediction
python prediction/scripts/train.py --tune --production  # Train XGBoost + isotonic calibration
python prediction/scripts/predict.py                 # Generate predictions for upcoming games
python prediction/scripts/backtest.py                # Walk-forward validation across seasons
```

**Note**: `fetch_bulk_advanced.py` uses `TeamGameLogs`/`PlayerGameLogs` bulk endpoints (~100 API calls total). The old `fetch_all_advanced.py` and `collect_data.py --advanced-only` use per-game `BoxScoreAdvancedV3` (~32,000 calls) — avoid for backfill.

**Output**: Predicted win probabilities per game (CSV or trained model + calibrator).

### Routine 2: Betting (betting/)
**Goal**: Maximize ROI. Develop optimal wagering strategy given predictions and market lines.

The betting routine consumes prediction outputs alongside market data. It never influences model training or calibration.

```bash
python betting/scripts/betting_analysis.py --lines betting/data/lines/lines.csv
python betting/scripts/betting_analysis.py --lines betting/data/lines/lines.csv --retroactive
python betting/scripts/retroactive_simulation.py        # Walk-forward P&L vs closing lines
python betting/scripts/retroactive_opening_lines.py     # Opening vs closing line comparison
```

**Input**: Model predictions + betting lines CSV (moneylines for both sides).
**Output**: Bet recommendations with Kelly sizing, edge analysis, P&L reports.

### Market Efficiency Finding (Phase 12)
**The NBA moneyline market is too efficient for this model to beat profitably.**
- Model walk-forward LL: 0.6219 | Market closing LL: 0.5918 | Market opening LL: 0.6076
- Model loses to the market at ALL lead times (opening AND closing)
- 0/18 seasons profitable in retroactive simulation across 6 strategies
- The 0.020 LL gap decomposes as: 0.014 structural (oddsmaker advantage) + 0.007 same-day info (injuries, lineups, sharp money)
- See `docs/research/betting-market-efficiency.md` for general lessons

### The Boundary
```
prediction/  ──produces──>  win probabilities  ──consumed by──>  betting/
                                                                    +
                                                              market lines
```

## Data Sources

### Prediction Routine
1. **NBA API — Game Logs** (`LeagueGameFinder`): Basic box scores (PTS, FGM, FGA, REB, AST, etc.). One row per team per game. Used for Elo computation, win percentages, rest day derivation.
2. **NBA API — Advanced Box Scores** (`BoxScoreAdvancedV3`): Official possession-adjusted stats. One call per game returns both:
   - **Team stats** (index 1): OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT, EFG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TM_TOV_PCT.
   - **Player stats** (index 0): Per-player OFF_RATING, DEF_RATING, NET_RATING, TS_PCT, EFG_PCT, USG_PCT, PIE, MINUTES, etc.
3. **Computed Elo** (from game results): Pre-game Elo snapshots, K=20, home advantage=75.
4. **Static reference** (`data/reference/arenas.csv`): Arena coordinates, elevation, timezone for all 30 current + 5 historical teams. Used for travel distance, altitude, timezone change features.

### Betting Routine
5. **Historical closing lines** ([Kaggle](https://www.kaggle.com/datasets/ehallmar/nba-historical-stats-and-betting-data), [SBRO archives](https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm)): Moneylines, spreads, totals. Lives in `betting/data/lines/`.

### Provenance Note
`nba_api` is an **unofficial** wrapper around nba.com/stats. It is community-maintained, not supported by the NBA. Endpoints can break without notice. We use `BoxScoreAdvancedV3` (V2 is deprecated). Historical data is reliable; current-season data updates after games are final (not real-time).

## Features

### Differential Features (home stat minus away stat)

**Game-Log (11 differentials):**
- **elo** — Pre-game Elo rating (single most predictive feature, -0.017 LL)
- **scoring_margin** — PLUS_MINUS expanding mean
- **sos_adj_margin** — Elo-weighted scoring margin (Phase 10b: r=0.326 with outcome vs raw 0.294)
- **three_pt_rate, fg_pct, fg3_pct, ft_rate** — Shooting rates
- **oreb_rate, tov_rate, ast_rate** — Possession stats
- **win_pct, win_pct_last10, avg_PTS** — Win rates and scoring pace proxy

**Advanced Team Stats (10 differentials, full coverage — all 25 seasons):**
- OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT, EFG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TM_TOV_PCT

**Player-Aggregated (6 differentials, full coverage — all 25 seasons):**
- top3_usg, top3_pie, bench_pie_gap, pie_std, usg_hhi, min_weighted_net_rtg

**SOS (2 differentials):**
- sos — Expanding mean of opponent Elo (schedule difficulty)
- sos_adj_margin — Elo-weighted scoring margin

### Context Features (non-differential)

**Rest/fatigue:** rest_days_home/away, is_b2b_home/away, is_3in4_home/away, games_last_7_home/away, consecutive_away_home/away
**Travel/altitude:** travel_km_home/away, tz_change_home/away, altitude_ft, altitude_diff
**Other:** pace_interaction, is_playoff

## Current Production Model — Ensemble (Phase 11, Converged)

**Architecture:** Equal-weight average of 2 models, each with its own optimal feature set:
- **XGBoost** → production features (14 features: game-log differentials + elo + rest_days + is_playoff)
- **Logistic Regression** → best_10a features (19 features: game-log + schedule/travel context)

**Performance:**
- **LOO-CV LL**: 0.6087 (24-fold leave-one-season-out, all seasons 2001-2023)
- **Test LL**: 0.6086 (2024-25 season holdout)
- **Test Acc**: 66.2%

**Convergence verified across:** 30+ feature sets, 5 architectures, 31 ensemble combinations, 8 ensemble refinement techniques, 10K+ ablation iterations. Nothing improves LOO-CV below 0.6087. Advanced team stats and player-aggregated features do not improve the ensemble.

**XGBoost hyperparameters:** depth=3, lr=0.08, n_est=750, mcw=3, ss=0.7, cs=1.0
**Sample decay:** 0.3 (exponential within-season recency weighting)

### Ensemble Confidence Signal (Phase 10f-H4)
When models agree (prediction std < 0.01): **70.1% accuracy, LL=0.5816**
When models disagree (prediction std > 0.03): **63.0% accuracy, LL=0.6334**
→ Use ensemble std to scale Kelly fraction in betting routine.

## Research & Experiment Status
- **Research log**: `research/log.md` — read at start of every session. Contains all experiment findings, decisions, and methodological lessons (Phases 1-10f).
- **Experiment outputs**: `prediction/output/reports/` — raw CSVs and JSON summaries from all experiments.
- **Model converged** at LOO-CV = 0.6087. Advanced stats and player features tested (Phases 11a-d) but do not improve the ensemble.
- **Market efficiency proven** (Phase 12): NBA moneylines are unbeatable — model loses to both opening and closing lines. See `docs/research/betting-market-efficiency.md`.
- **Moneyline betting is not viable** — model loses to market at all lead times. See Phase 12.
- **Player props: model quality assessed (Phase 13)** — PTS MAE 4.70 (10% better than naive). Quality sufficient to warrant market efficiency test.
- **Props data collection automated** — `betting/scripts/collect_prop_lines.py` archives daily prop lines from The Odds API. Needs `ODDS_API_KEY` env var. Run daily to accumulate data for future market efficiency test.
- **Next**: Accumulate ~2 months of prop line data, then test directional accuracy against actual lines. Also: apply framework to MLB, NCAA, other markets.

## Guardrails
- Never compare training windows with a single random val split — use leave-one-season-out CV (Phase 7 lesson)
- Test feature importance via marginal ablation (include vs exclude across many configs), not XGBoost's built-in `feature_importances_` (Phase 8 lesson)
- Always validate findings across multiple architectures — single-architecture results can be misleading (Phase 10d lesson)
- Simple equal-weight ensemble averaging outperforms stacking and calibration with limited data (Phase 10f lesson)
- Prediction routine must never see betting lines; betting routine must never influence model training
