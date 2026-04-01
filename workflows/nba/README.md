# NBA Game Prediction

Predict NBA game outcomes using XGBoost on team stat differentials with isotonic calibration.

## Quick Start — Prediction Routine

```bash
pip install -e ".[dev]"
pip install nba_api

cd workflows/nba
python prediction/scripts/collect_data.py
python prediction/scripts/compute_elo.py
python prediction/scripts/build_features.py
python prediction/scripts/train.py --tune --production
python prediction/scripts/predict.py
python prediction/scripts/backtest.py
```

## Quick Start — Betting Routine

Requires: trained model predictions + betting lines CSV.

```bash
cd workflows/nba
python betting/scripts/betting_analysis.py --lines betting/data/lines/lines.csv
python betting/scripts/betting_analysis.py --lines betting/data/lines/lines.csv --retroactive
```

## Architecture

```
prediction/scripts/  — Model quality optimization (collect → train → predict → backtest)
betting/scripts/     — Wagering strategy (predictions + lines → edges → Kelly sizing → P&L)
data/                — Shared raw and processed data
models/              — Shared trained model artifacts
```

The prediction routine produces win probabilities. The betting routine consumes them alongside market lines. They are independent — the betting routine never influences model training.

## Current Results

Baseline model (Phase 3): **Val 0.604 LL / 67.3% acc** | **Test 0.613 LL / 65.4% acc** (2024-25 holdout)

13 game-log features + Elo + rest days. Experiments confirmed: sample decay=0.3 optimal, no EWMA benefit, no cross-season prior benefit, no normalization needed.

See `CLAUDE.md` for domain context and `research/log.md` for experiment history.
