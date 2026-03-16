# youBet - General-Purpose Prediction & Betting Framework

## Project Principles
1. **Calibration over accuracy** — optimize for log loss, not accuracy. Well-calibrated probabilities yield better betting returns.
2. **Config-driven** — all parameters live in `config.yaml` per workflow. No magic numbers in code.
3. **One-way dependencies** — `workflows/` depends on `src/youbet/core/`, never reverse.
4. **Research log is the feedback loop** — `research/log.md` in each workflow is read at start of every session.
5. **Differentials, not absolutes** — features are always Team A stat minus Team B stat.
6. **Iterative refinement** — start simple (XGBoost baseline), measure, then improve.

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
