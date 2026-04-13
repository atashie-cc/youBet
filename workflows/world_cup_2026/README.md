# 2026 FIFA World Cup Bracket Prediction

## Overview
Monte Carlo bracket simulator for the 2026 FIFA World Cup (48-team format).
Primary use case: **bracket pool optimization**, not betting or probability forecasting.

The model captures ~60% of the signal between random and market on WC match prediction. Its genuine value is structural bracket-path simulation and pool-play strategy optimization — neither of which bookmakers publish.

## Status: Complete (Phases 0-6 + Market Efficiency Assessment)

**Champion picks (balanced strategy)**: Spain 22.8%, Argentina 20.9%, France 12.1%.

## Quick Start
```bash
# Phase 0: Download data
python scripts/collect_data.py

# Phase 1: Compute Elo ratings (eloratings-inspired params)
python scripts/compute_elo.py

# Phase 1: Build features
python scripts/build_features.py

# Phase 2: Train walk-forward model (evaluation only — simulator uses Elo-only)
python scripts/train.py

# Phase 4: Generate bracket picks
python scripts/generate_bracket.py

# Phase 6: Squad quality features (optional adjustment)
python scripts/build_squad_features.py
```

## Key Findings

| Finding | Evidence |
|---|---|
| Market beats model on match prediction | Pinnacle ~0.99 LL vs our Elo 0.995 LL on WC 2022 |
| Outright odds dominate Elo on team strength | Market-implied beats Elo by 0.028 LL; blending degrades monotonically |
| XGBoost features add zero lift | +0.004 LL (slightly harmful) on top of recalibrated Elo |
| All knockout models pick same winner | 66.7% accuracy = "always pick higher-Elo team" across all 5 comparators |
| Phase 6 Elo recalibration was the biggest improvement | Morocco #3→#11, Brazil #10→#4, top-5 concentration 32%→62% |

## Pipeline

See `research/final-report.md` for the comprehensive experiment log.
See `research/log.md` for the detailed phase-by-phase research notes.

## Artifacts
- `output/bracket_2026.md` — Human-readable bracket picks
- `output/champion_probabilities.csv` — Per-team P(champion) × 3 strategies
- `output/per_round_probabilities.csv` — Per-team per-round advancement
- `research/final-report.md` — Comprehensive final report with strategic assessment
