# youBet

General-purpose prediction and betting framework. Built for iterative research with calibrated probabilistic models.

## Workflows

### NCAA March Madness 2026
Predict tournament game outcomes and generate bracket picks. XGBoost on 16 stat differentials with Platt calibration. **Phase 12**: 79.2% accuracy through R32, tied betting market on picks, +16% ROI with conviction-filtered Kelly betting.

### NBA Game Prediction (New)
Predict NBA regular season and playoff outcomes. XGBoost on 15 stat differentials + rest/home features with isotonic calibration. Uses `nba_api` for official NBA stats.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run NCAA pipeline
cd workflows/ncaa_march_madness
python scripts/collect_data.py
python scripts/train.py --tune --production
python scripts/predict.py
python scripts/generate_bracket.py

# Run NBA pipeline
cd workflows/nba
pip install nba_api
python scripts/collect_data.py
python scripts/compute_elo.py
python scripts/build_features.py
python scripts/train.py --tune --production
python scripts/predict.py
```

## Architecture

```
src/youbet/core/    — Domain-agnostic prediction components
workflows/          — Domain-specific prediction pipelines
docs/               — Research, decisions, runbooks
```

## Key Design Decisions

- **Calibration > Accuracy**: Log loss is the primary metric
- **XGBoost on 16 stat differentials**: Team A stat - Team B stat
- **Platt scaling calibration**: 2-parameter post-processing with probability clipping
- **Random season splits**: Recent years in training for current team dynamics
- **Kelly Criterion**: For bankroll management and bet sizing
