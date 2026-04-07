# PGA Golf Tournament Prediction

## Overview
Head-to-head matchup prediction for PGA Tour golf tournaments. Uses Strokes Gained decomposition, course-player fit, Elo ratings, and recent form as features. Targets H2H matchup betting markets as the primary evaluation surface.

## Quick Start
```bash
# Phase 1: Data collection (requires DATAGOLF_API_KEY env var)
python scripts/collect_data.py          # Tournament results + round scores
python scripts/collect_odds.py          # Historical H2H + outright odds

# Phase 2: Feature engineering + modeling
python scripts/compute_elo.py           # Player Elo ratings
python scripts/build_features.py        # 16 differential features + 2 context
python scripts/train_v2.py              # Walk-forward CV via Experiment runner
python scripts/evaluate_market.py       # Model LL vs market LL (kill gate)
```

## Data Sources
- **Data Golf API** (primary): Round-level scoring, SG breakdown, skill ratings (2004-present)
- **Data Golf Odds Archives**: H2H matchup + outright odds (2019-present)
- **Kaggle PGA datasets**: Free fallback for prototyping

## Features
16 differential features (player_a - player_b) + 2 context features:
- Strokes Gained: total, OTT, approach, ARG, putting, tee-to-green
- Course fit: course fit score, course history
- Skill: Elo rating
- Form: 5-event form, 10-event form, cuts made, SG trend
- Context: days since last event, events in 90 days, major flag, field strength
