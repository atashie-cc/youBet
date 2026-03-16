# NCAA March Madness Bracket Prediction

Predict NCAA Men's Basketball Tournament outcomes using XGBoost on 16 stat differentials with Platt scaling calibration.

## Quick Start

```bash
# Data collection
python scripts/collect_data.py                # Download Kaggle data (2003-2017)
python scripts/scrape_game_results.py --force # Scrape ESPN 2018-2025 game results (~15 min)

# Pipeline
python scripts/compute_elo.py        # Pre-tournament Elo from game data
python scripts/build_features.py     # Build 16 stat differentials (93K+ matchups)
python scripts/train.py --tune --production  # Train XGBoost + Platt calibration (random 80/20 split)
python scripts/predict.py            # Predict tournament matchups
python scripts/generate_bracket.py   # Monte Carlo bracket simulation
python scripts/backtest.py           # Validate on 2005-2026 tournaments

# Optional: full box scores for EWMA rate stats
python scripts/scrape_game_results.py --force --detailed  # ~8 hours
```

### Train script options

```bash
python scripts/train.py --tune                 # Tune + train on config temporal split
python scripts/train.py --tune --random-split  # Tune + train on random 60/20/20 split
python scripts/train.py --tune --production    # Tune + train on random 80/20 split (no test holdout)
python scripts/train.py --n-tune-iter 200      # More tuning iterations (default: 100)
python scripts/train.py --early-stopping-rounds 0  # Disable early stopping
```

## Data Sources
- **Kaggle March ML Mania**: Detailed box scores 2003-2017 (~78K games)
- **ESPN Scoreboard API**: Game results 2018-2025 (~47K games) via `scrape_game_results.py`
- **KenPom**: Adjusted efficiency metrics 2002-2026 (8,680 team-seasons)

## Current Performance (Phase 11)
- **Validation:** Log Loss **0.496**, Accuracy **75.4%** (16,314 games across 5 random seasons)
- Model: XGBoost md=7 lr=0.01 ne=1000 mcw=7, Platt calibration, decay=0.3, early stopping 50
- Split: Random 80/20 by season (18 train / 5 val), 100-iter tuning
- Feature importance: KenPom Rank 32%, Elo 28%, Adj Efficiency Margin 24%

## 2026 Predictions
- **Chalk champion: Florida** (1-seed, South)
- Monte Carlo (balanced): Duke 29.7%, Arizona 24.6%, Michigan 20.2%, Florida 15.4%
- Full bracket: [`output/bracket_2026.md`](output/bracket_2026.md)

## Target Performance
- Log loss: < 0.55 (tournament games) -- **achieved** (0.522 on 2025 test)
- Accuracy: > 75% (all games) -- **achieved** (75.4% on validation)
- Calibration: within 5% of diagonal on calibration curve -- **achieved**
