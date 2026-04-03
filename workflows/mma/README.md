# MMA/UFC Fight Prediction

Predict UFC fight outcomes using fighter statistics and Elo ratings.

## Quick Start

```bash
# 1. Place raw data in data/raw/ (see CLAUDE.md for sources)
# 2. Run pipeline:
python scripts/collect_data.py
python scripts/compute_elo.py
python scripts/tune_elo.py
python scripts/build_features.py
python scripts/train.py
python scripts/evaluate_market.py
```

See `CLAUDE.md` for full documentation.
