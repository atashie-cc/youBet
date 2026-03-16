# Hyperparameter Experiments

Track model tuning experiments here.

## Baseline Configuration
```yaml
backend: xgboost
max_depth: 6
learning_rate: 0.05
n_estimators: 500
subsample: 0.8
colsample_bytree: 0.8
min_child_weight: 3
reg_alpha: 0.1
reg_lambda: 1.0
```

## Experiments

| Date | Change | Log Loss | Accuracy | N | Notes |
|------|--------|----------|----------|---|-------|
| 2026-03-15 | Phase 1: XGBoost + isotonic | 1.021 | 79.5% | 1,017 | Overconfident, isotonic overfitting |
| 2026-03-15 | Phase 4: Platt scaling + clip | 0.578 | 71.8% | 1,243 | Calibration fixed, log loss -43% |
| 2026-03-16 | Phase 5: Elo leakage fix | 0.619 | 69.5% | 1,243 | Honest eval, elo importance 43→28% |
| 2026-03-16 | Phase 6: ESPN data gap fill | 0.623 | 68.0% | 2,500 | +58% training data, N includes NIT/CBI |
| 2026-03-16 | Phase 7: Random split decay experiment | 0.559 | 69.5% | 403 | decay=0.5 best, tuned per-decay (random season split, tourney-only test) |
| 2026-03-16 | Phase 8: Robust tuning, decay=0.5 | 0.522 | 69.5% | 311 | Tuned md=5 lr=0.01 ne=750; clean train/val split (2003-2018/2019-2024); tourney-only test 2025 |
| 2026-03-16 | Phase 10 Exp1: Early stop + cal holdout | 0.559 | 69.5% | 403 | Baseline (no ES) best; ES@50 LL=0.562; separate cal LL=0.564 (random split) |
| 2026-03-16 | Phase 10 Exp2: LightGBM comparison | 0.564 | 68.7% | 403 | XGBoost wins (0.562 vs 0.564); LightGBM spreads importance more evenly (random split) |
| 2026-03-16 | Phase 10 Exp3: Feature selection | 0.561 | 68.5% | 403 | no_elo (15 feat) best; no_redundancy 2nd; KenPom critical; rates_only LL=0.605 (random split) |
| 2026-03-16 | Phase 10 Exp4: Fine decay sweep | 0.561 | 69.2% | 403 | decay=0.1 best; sweet spot [0.1-0.4]; flat curve (random split, ES@50) |
| 2026-03-16 | Phase 10 Exp5: Ensemble (XGB+LGB+LR) | 0.559 | 68.7% | 403 | XGBoost solo best; equal-weight avg=0.560; stacking hurts (random split) |

## Phase 10: Model Performance Enhancement Experiments

All experiments use random season split (seed=42), 30-iter tuning, tournament-only test (403 games). Baseline: Phase 7 decay=0.5 (test LL=0.5588).

### Experiment 4: Fine-Grained Decay Sweep (with early stopping @50)

| Decay | Log Loss | Accuracy | Brier  | Best Iter | max_depth | learning_rate | n_estimators | min_child_weight | subsample | colsample_bytree |
|-------|----------|----------|--------|-----------|-----------|---------------|--------------|------------------|-----------|------------------|
| 0.0   | 0.5628   | 69.2%    | 0.1924 | 115       | 6         | 0.05          | 1000         | 7                | 0.7       | 1.0              |
| **0.1** | **0.5612** | **69.2%** | **0.1919** | **480** | **4** | **0.02** | **750** | **5** | **0.6** | **0.5** |
| 0.2   | 0.5615   | 69.5%    | 0.1919 | 435       | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| 0.3   | 0.5629   | 68.7%    | 0.1924 | 449       | 6         | 0.01          | 1000         | 3                | 1.0       | 0.6              |
| 0.4   | 0.5614   | 69.2%    | 0.1919 | 414       | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| 0.5   | 0.5621   | 69.0%    | 0.1922 | 414       | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| 0.6   | 0.5624   | 69.0%    | 0.1923 | 403       | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| 0.7   | 0.5629   | 69.7%    | 0.1925 | 490       | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| 0.8   | 0.5629   | 69.5%    | 0.1925 | 482       | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| 0.9   | 0.5620   | 68.0%    | 0.1922 | 579       | 6         | 0.01          | 1000         | 3                | 1.0       | 0.6              |
| 1.0   | 0.5622   | 69.2%    | 0.1922 | 413       | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |

**Takeaway**: Decay=0.1 is the new optimum, improving 0.0009 over 0.5. Sweet spot [0.1-0.4]. Curve is flat — model is not sensitive to decay in this range.

### Experiment 3: Feature Selection

| Subset | N_feat | Test LL | Test Acc | Features removed |
|--------|--------|---------|----------|-----------------|
| **no_elo** | 15 | **0.5612** | 68.5% | diff_elo |
| no_redundancy | 13 | 0.5618 | 68.2% | diff_adj_oe, diff_adj_de, diff_kenpom_rank |
| all_16 | 16 | 0.5621 | 69.0% | (baseline) |
| top_8 | 8 | 0.5625 | 69.2% | bottom 8 by importance |
| no_kenpom | 11 | 0.5965 | 67.2% | adj_oe, adj_de, adj_em, adj_tempo, kenpom_rank |
| rates_only | 9 | 0.6049 | 67.0% | all except seed_num, win_pct, 7 rate stats |

## Phase 7: Random Split Recency Decay Experiment

Random season assignment (seed=42): 13 train / 5 val / 5 test. Independent 30-iter hyperparameter random search per decay. Test on tournament games only (403 games).

| Decay | Log Loss | Accuracy | Brier  | max_depth | learning_rate | n_estimators | min_child_weight | subsample | colsample_bytree |
|-------|----------|----------|--------|-----------|---------------|--------------|------------------|-----------|------------------|
| 0.0   | 0.5615   | 69.5%    | 0.1920 | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| **0.5** | **0.5588** | **69.5%** | **0.1910** | **4** | **0.02** | **300** | **2** | **1.0** | **0.9** |
| 1.0   | 0.5649   | 68.2%    | 0.1935 | 4         | 0.02          | 750          | 5                | 0.6       | 0.5              |
| 1.5   | 0.5621   | 69.0%    | 0.1921 | 5         | 0.02          | 200          | 3                | 0.6       | 0.5              |
| 2.0   | 0.5626   | 70.0%    | 0.1924 | 5         | 0.02          | 200          | 3                | 0.6       | 0.5              |
| 3.0   | 0.5683   | 69.0%    | 0.1945 | 5         | 0.02          | 200          | 3                | 0.6       | 0.5              |
| 5.0   | 0.5732   | 67.7%    | 0.1966 | 5         | 0.02          | 200          | 3                | 0.6       | 0.5              |

**Takeaway**: Mild decay (0.5) wins. Aggressive decay (≥3.0) hurts. Further refinement of the decay space around 0.5 (e.g., [0.2-0.8] in 0.1 increments) is warranted to find the true optimum.

## Current Production Configuration (Phase 11)
```yaml
backend: xgboost
max_depth: 7
learning_rate: 0.01
n_estimators: 1000
subsample: 0.8
colsample_bytree: 1.0
min_child_weight: 7
reg_alpha: 0.1
reg_lambda: 1.0
decay: 0.3
early_stopping_rounds: 50
```
Split: Random 80/20 by season (18 train / 5 val). Val LL: 0.496, Val Acc: 75.4%.
100-iter tuning with early stopping. Feature importance: kenpom_rank 32%, elo 28%, adj_em 24%.
2026 chalk champion: Florida. See `output/bracket_2026.md`.

## Phase 11 Experiment Row

| Date | Change | Log Loss | Accuracy | N | Notes |
|------|--------|----------|----------|---|-------|
| 2026-03-16 | Phase 11: 100-iter tune, decay=0.3, ES@50, random split | 0.497 (test all) / 0.563 (tourney) | 75.4% / 70.5% | 16,314 / 403 | 60/20/20 random; lr=0.08 found (vs 0.02 at 30-iter) |
| 2026-03-16 | Phase 11 production: 80/20 random, 100-iter | 0.496 (val) | 75.4% | 16,314 | md=7 lr=0.01 ne=1000 mcw=7; kenpom_rank now #1 feature |
