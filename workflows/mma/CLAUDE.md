# MMA/UFC Fight Prediction

## Domain Context
Predict outcomes of UFC fights (moneylines). Individual combat sport — no teams, no seasons, no home advantage. ~500-550 UFC fights per year. Fighters compete 2-3 times per year, making per-fighter data sparse. Weight classes segment the competition (flyweight through heavyweight). Fights end by KO/TKO, submission, or decision.

## Pipeline

```bash
python scripts/collect_data.py                # Load and process UFC fight data
python scripts/collect_data.py --process-only # Process raw data only
python scripts/compute_elo.py                 # Fighter Elo ratings (overall + weight-class)
python scripts/tune_elo.py                    # Grid search Elo parameters (~720 combos)
python scripts/build_features.py              # 13 differential features + 2 context
python scripts/train.py                       # Walk-forward CV by year (ad-hoc)
python scripts/train_v2.py                    # Walk-forward CV via core Experiment runner (preferred)
python scripts/train.py --tune                # Hyperparameter tuning
python scripts/evaluate_market.py             # Model LL vs market closing LL
python scripts/scrape_opening_lines.py        # Scrape BFO opening/closing lines
python scripts/evaluate_opening_lines.py      # Model LL vs opening lines
```

## Data Sources
1. **Kaggle UFC dataset** (primary): Fight records with per-fight stats and odds. 7,169 fights (2010-2026).
2. **BestFightOdds.com**: Historical opening/closing odds scraped from fighter profiles. 21,060 fight-odds records.

## Features (13 differentials + 2 context, fighter_a - fighter_b)
Configured explicitly in `config.yaml` — `train.py` reads the declared list and fails on mismatches.
- **Elo (overall)**: All-fights Elo differential (time decay, finish bonus)
- **Elo (weight-class)**: Per-weight-class Elo differential — more indicative of performance
- Significant strikes landed avg differential (EWMA, shift(1))
- Strike accuracy % differential
- Takedown landed avg differential
- Takedown accuracy % differential
- Submission attempts avg differential
- Win rate differential
- Win streak differential
- Reach differential (per-fight PIT, forward-filled)
- Height differential (per-fight PIT, forward-filled)
- Age differential (per-fight from raw data — NOT from profile snapshots)
- Days since last fight differential (ring rust vs freshness)
- Weight class movement (categorical per fighter: -1/0/+1 for moved down/same/up)

## MMA-Specific Elo Adaptations
- **Two Elo systems**: Overall (all fights) + per-weight-class (separate ratings per division)
- **No season resets** — continuous time decay instead (rating drifts toward 1500 during inactivity)
- **No home advantage** — always neutral venue
- **High K-factor** (80) — sparse fights per fighter require each fight to carry more weight
- **Finish bonus** (1.2x) — KO/TKO/SUB wins update Elo more than decisions
- **K-factor schedule** — higher K for first N fights (faster convergence for new fighters)

## Key Domain Considerations
- **Sparse data**: Fighters have 2-30 career UFC fights vs 82+ games/season in team sports
- **No seasons**: Walk-forward by calendar year instead of leave-one-season-out
- **Weight classes**: Flyweight, Bantamweight, Featherweight, Lightweight, Welterweight, Middleweight, Light Heavyweight, Heavyweight (+ Women's divisions)
- **Weight class movement**: Fighters moving up/down perform differently relative to their stats
- **Fighter decline**: Age and accumulated damage reduce performance — age is a key feature
- **Ring rust**: Long layoffs (1+ year) correlate with underperformance
- **Style matchups**: Striker vs grappler dynamics are non-linear (future Phase 3 feature)
- **Survivorship bias**: Fighters who lose badly often leave the UFC, biasing training data

## Feature Safety Rules
**CRITICAL: All features must be point-in-time. No exceptions.**
- Rolling fight stats → EWMA with `.shift(1)` — shift(1) is non-negotiable
- Reach, height → Per-fight from raw data, forward-filled (NOT from profile snapshots — Codex R2 fix)
- Age → Per-fight from raw data (NOT back-computed from profile — Codex R2 fix)
- Days since last fight → Computed from fight dates (safe)
- Weight class movement → Compare current fight weight class to previous fight (safe)
- Feature selection → Explicit list in config.yaml, train.py fails on mismatches (Codex R3 fix)
- **NEVER** use career averages that include the current fight or future fights

## Current Performance (Phase 2 — COMPLETE, Market Efficient)
After 3 rounds of Codex adversarial review and fixes:

| Source | Log Loss | Accuracy | N |
|--------|----------|----------|---|
| **Market closing line** | **0.6130** | **65.6%** | 4,050 |
| **Market opening line** | **0.6318** | **64.0%** | 4,050 |
| XGBoost model | 0.6608 | 60.1% | 5,461 |
| Elo-only (overall) | 0.6754 | 57.6% | 5,461 |
| Elo (weight-class) | 0.6855 | 56.2% | 7,169 |

- **Model vs closing: +0.050 LL** — market is significantly better
- **Model vs opening: +0.033 LL** — even opening lines beat the model (kill threshold 0.02)
- **Opening → Closing gap: +0.019 LL** — MMA lines move more than NBA (~0.007) due to late-breaking info
- **Elo tuning**: K=80, decay=0.10, finish_bonus=1.2 (0.006 LL gain over baseline)
- **Verdict: STOP** — both opening and closing lines are efficient against public statistical models
- Average vig: opening 5.7%, closing 1.8%

### Codex Review Fixes Applied
1. Calibration fold excluded from training → smarter 80/20 temporal split (R1)
2. Age feature PIT leakage → per-fight age from raw data (R2)
3. Synthetic closing line → paired close from same source (R2)
4. BFO scraper name validation → fuzzy match check (R2)
5. Feature selection → explicit config-driven schema (R3)
6. Cal year exclusion too aggressive → 80/20 split within pre-eval window (R3)
7. Reach/height PIT leakage → per-fight forward-fill from raw data (R3)

## Research Log
See `research/log.md` — read at start of every session.
