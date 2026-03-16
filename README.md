# youBet

General-purpose prediction and betting framework. Built for iterative research with calibrated probabilistic models.

## First Workflow: NCAA March Madness 2026

Predict tournament game outcomes and generate bracket picks using XGBoost on team stat differentials with Platt calibration.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run NCAA pipeline
cd workflows/ncaa_march_madness
python scripts/collect_data.py
python scripts/scrape_game_results.py --force
python scripts/compute_elo.py
python scripts/build_features.py
python scripts/train.py --tune --production
python scripts/predict.py
python scripts/generate_bracket.py
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
