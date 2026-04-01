# youBet - General-Purpose Prediction & Betting Framework

## Project Principles
1. **Calibration over accuracy** — optimize for log loss, not accuracy. Well-calibrated probabilities yield better betting returns.
2. **Beat the market, not the outcome** — a model must produce better probabilities than the market's implied odds to be profitable. Accuracy alone is insufficient. See `docs/research/betting-market-efficiency.md`.
3. **Config-driven** — all parameters live in `config.yaml` per workflow. No magic numbers in code.
4. **One-way dependencies** — `workflows/` depends on `src/youbet/core/`, never reverse.
5. **Research log is the feedback loop** — `research/log.md` in each workflow is read at start of every session.
6. **Differentials, not absolutes** — features are always Team A stat minus Team B stat.
7. **Iterative refinement** — start simple (XGBoost baseline), measure, then improve.
8. **Validate with LOO-CV** — single val-split results are unreliable. Always confirm with leave-one-season-out cross-validation.
9. **Test market efficiency early** — compare model LL vs market-implied LL at Stage 2 (baseline model), not Stage 12. Kill early if the market is more accurate.

## Architecture
- `src/youbet/core/` — domain-agnostic prediction components (Elo, features, models, calibration, evaluation, bankroll, pipeline)
- `src/youbet/utils/` — I/O and visualization helpers
- `workflows/` — domain-specific prediction workflows (NCAA March Madness, etc.)
- `docs/` — research findings, architectural decisions, runbooks

## Conventions
- Python 3.11+
- Use type hints everywhere
- Use `pathlib.Path` for all file paths
- Use `logging` module, not print statements
- Tests in `tests/` mirroring `src/` structure
- Data files never committed to git (in `.gitignore`)
- Primary metric: log loss. Always report alongside accuracy and Brier score.

## Current Workflows
- `workflows/ncaa_march_madness/` — NCAA Men's Basketball March Madness bracket prediction
- `workflows/nba/` — NBA game prediction (Phase 12 complete — model converged at LOO-CV 0.6087, market proven unbeatable)
- `workflows/mlb/` — MLB game prediction (Phase 3 complete — 2-feat LogReg beats market by 0.0083 LL, 15% ROI, 13/14 seasons)

## Key Research Documents
- `docs/research/betting-market-efficiency.md` — Lessons on model vs market efficiency, market entry framework, and market prioritization
- `docs/research/methodologies.md` — Proven prediction methodologies catalog
- `workflows/nba/research/log.md` — Complete NBA experiment log (Phases 3-12)
- `workflows/mlb/research/log.md` — MLB experiment log (Phase 1: baseline + market screening)
