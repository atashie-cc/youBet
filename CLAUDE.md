# youBet - General-Purpose Prediction & Betting Framework

## Project Principles
1. **Calibration over accuracy** — optimize for log loss, not accuracy. Well-calibrated probabilities yield better betting returns.
2. **Beat the market, not the outcome** — a model must produce better probabilities than the market's implied odds to be profitable. Accuracy alone is insufficient. See `docs/research/betting-market-efficiency.md`.
3. **Config-driven** — all parameters live in `config.yaml` per workflow. No magic numbers in code.
4. **One-way dependencies** — `workflows/` depends on `src/youbet/` (`core/`, `etf/`, `utils/`), never reverse.
5. **Research log is the feedback loop** — `research/log.md` in each workflow is read at start of every session.
6. **Differentials, not absolutes** — features are always Team A stat minus Team B stat.
7. **Iterative refinement** — start simple (XGBoost baseline), measure, then improve.
8. **Validate with walk-forward CV** — use the core `Experiment` runner which enforces temporal splits and PIT safety structurally. Never use random season splits for validation.
9. **Test market efficiency early** — compare model LL vs market-implied LL at Stage 2 (baseline model), not Stage 12. Kill early if the market is more accurate.
10. **Point-in-time features only** — NEVER use season-level aggregates for mid-season games. All features must use only data available before game time: prior-season stats, rolling/cumulative in-season stats from game logs, or sequential ratings (Elo). The core `pit.py` module provides structural PIT checks that raise `PITViolation` on detected leakage.
11. **Audit before celebrating** — when backtest results seem too good to be true, run three independent checks before trusting them: code audit for lookahead, literature review for plausibility, and forum check for common pitfalls.
12. **Stateful transforms** — feature normalization and imputation must use fit/transform separation via `core/transforms.py`. Fit on training data only, never on the full dataset. The old `normalize_features()` leaked test distribution into training across all three workflows.

## Architecture
- `src/youbet/core/` — domain-agnostic prediction components:
  - `experiment.py` — **Split-aware experiment runner** with walk-forward folds, PIT enforcement, audit metadata, and market comparison. New workflows should use this instead of ad-hoc train scripts.
  - `transforms.py` — **Stateful transforms** (Normalizer, Imputer, FeaturePipeline) with fit/transform separation. Prevents the most common cross-workflow leakage pattern.
  - `pit.py` — **Point-in-time validation** checks. Raises `PITViolation` on temporal overlap, cal/train leakage, or suspicious feature stationarity.
  - `elo.py` — Elo rating system with configurable K-factor, time decay, MOV scaling
  - `features.py` — Differential computation, rolling stats
  - `models.py` — GradientBoostModel wrapper (XGBoost/LightGBM)
  - `calibration.py` — Platt scaling, isotonic regression
  - `evaluation.py` — Log loss, accuracy, Brier score, calibration curves
  - `bankroll.py` — Kelly criterion, vig removal
  - `betting.py` — Edge detection, bet sizing, P&L reporting
  - `pipeline.py` — Legacy abstract pipeline (prefer `experiment.py` for new workflows)
- `src/youbet/etf/` — ETF strategy evaluation engine:
  - `backtester.py` — Walk-forward portfolio backtester with T+1 execution, transaction costs, T-bill cash rate
  - `strategy.py` — BaseStrategy ABC (fit, generate_weights)
  - `stats.py` — Block bootstrap (Politis-Romano), Holm correction, excess Sharpe CIs, Romano-Wolf simultaneous CIs
  - `risk.py` — Sharpe, Sortino, MaxDD, CVaR, information ratio, Calmar, risk of ruin
  - `pit.py` — PIT validation: temporal splits, survivorship guard, publication lag enforcement (PITFeatureSeries)
  - `costs.py` — Per-category transaction cost model with expense ratio drag
  - `transforms.py` — Stateful normalizer with drift monitoring (PSI, KS test)
  - `allocation.py` — Momentum rank, inverse-vol weight, absolute momentum filter
  - `data.py` — yfinance price fetcher, FRED T-bill rates, snapshot caching
  - `macro/fetchers.py` — Tier 1 macro signals: yield curve, credit spread, VIX, PMI, CAPE
- `src/youbet/utils/` — I/O and visualization helpers
- `workflows/` — domain-specific prediction workflows (NCAA March Madness, ETF strategies, etc.)
- `docs/` — research findings, architectural decisions, runbooks

## Conventions
- Python 3.11+
- Use type hints everywhere
- Use `pathlib.Path` for all file paths
- Use `logging` module, not print statements
- Tests in `tests/` mirroring `src/` structure
- Data files never committed to git (in `.gitignore`)
- Primary metric: log loss. Always report alongside accuracy and Brier score.
- Feature columns declared explicitly in `config.yaml` — train scripts must read from config and fail on mismatches (never auto-discover `diff_*` columns)
- New workflows should use `core/experiment.py` Experiment runner, not ad-hoc walk-forward loops

## New Workflow Quickstart
```python
from youbet.core.experiment import Experiment, compare_to_market
from youbet.core.models import GradientBoostModel
from youbet.core.transforms import FeaturePipeline, Imputer, Normalizer

experiment = Experiment(
    data=features_df,
    target_col="team_a_win",
    date_col="event_date",
    fold_col="year",         # Walk-forward by calendar year
    feature_cols=[...],      # Explicit list from config.yaml
    min_train_folds=5,
    cal_fraction=0.2,        # Last 20% of pre-eval window for calibration
)
result = experiment.run(
    model_factory=lambda: GradientBoostModel(backend="xgboost", params=...),
    feature_pipeline=FeaturePipeline(steps=[
        ("impute", Imputer(strategy="median", group_col="weight_class")),
        ("normalize", Normalizer(method="standard")),
    ]),
)
comparison = compare_to_market(result, data, "market_prob_a", "team_a_win")
```

## Current Workflows
- `workflows/ncaa_march_madness/` — NCAA Men's Basketball March Madness bracket prediction
- `workflows/nba/` — NBA game prediction (Phase 12 complete — model converged at LOO-CV 0.6087, market proven unbeatable)
- `workflows/mlb/` — MLB COMPLETE (Phases 1-6) — market efficient, ensemble betting not viable, p=0.869
- `workflows/mma/` — MMA/UFC COMPLETE (Phases 1-2) — model LL 0.6608 vs opening 0.6318, closing 0.6130. Both opening and closing lines efficient. 3 Codex review rounds, 7 fixes applied.
- `workflows/pga/` — PGA Golf H2H matchup prediction (Phase 1 — Screen). Individual sport with SG decomposition, course-player fit, 140+ player fields. Data Golf API primary source.
- `workflows/etf/` — Vanguard ETF strategy evaluation COMPLETE. 17 unleveraged + 8 leveraged experiments. Sharpe: VTI efficient. Drawdown: trend following -20% vs VTI -55%, supports 6% SWR (vs 5% for VTI). SMA100 > SMA200 at all leverage levels. 3x long/cash SMA100: 21.6% CAGR, 0.649 Sharpe. 3x inverse destroys wealth. ML destroyed value. 3 Codex reviews, 16 fixes.
- `workflows/etflab-max/` — Vanguard ETF CAGR maximization COMPLETE. 158 strategies tested across 5 phases. CAGR gate: 0/158 pass (VTI CAGR-efficient). Best unleveraged: VGT 13.8%, MGK 12.8%. Best leveraged: MGK @ 2.3x SMA100 = 20.6% CAGR, 0.645 Sharpe. Active momentum adds no value over static holds. 2 Codex reviews, 26 fixes.

## Key Research Documents
- `docs/research/betting-market-efficiency.md` — Lessons on model vs market efficiency, market entry framework, and market prioritization
- `docs/research/methodologies.md` — Proven prediction methodologies catalog
- `workflows/nba/research/log.md` — Complete NBA experiment log (Phases 3-12)
- `workflows/mlb/research/log.md` — MLB experiment log (Phases 1-6, complete)
- `workflows/mma/research/log.md` — MMA experiment log (Phases 1-2, complete)
- `workflows/pga/research/log.md` — PGA Golf experiment log (Phase 1, active)
- `workflows/etf/research/final-report.md` — Comprehensive ETF findings: dual-objective, SWR, leveraged, regime analysis
- `workflows/etf/research/etf-market-efficiency.md` — Phase 0 ETF market efficiency findings
- `workflows/etf/research/strategy-catalog.md` — Feature taxonomy and Phase 1-5 strategy roadmap
- `workflows/etflab-max/research/log.md` — ETF CAGR maximization experiment log (158 strategies, 5 phases, 2 Codex reviews)
