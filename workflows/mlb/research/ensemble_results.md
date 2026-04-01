# MLB Ensemble Experiment Results

**Date**: 2026-04-01
**Walk-forward**: Train on seasons < hold_season, test on hold_season (2012-2025, 14 seasons)
**Market closing LL**: 0.6792 (pooled across 29,600 games with odds)
**Features**: 20 differentials (same for all models), including park_factor

## Tuned Hyperparameters

### XGBoost (baseline -- unchanged)
```
max_depth=3, learning_rate=0.02, n_estimators=700, min_child_weight=15
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0
```

### Logistic Regression: C=0.01

### LightGBM: num_leaves=8, learning_rate=0.05, min_child_samples=50

### Random Forest: max_depth=6, min_samples_leaf=50

## Individual Model Results

| Model | Walk-Forward LL | Accuracy | Gap vs Market | Seasons Beat Market |
|-------|----------------|----------|---------------|---------------------|
| XGBoost | 0.6756 | 0.578 | -0.0036 | 10/14 |
| LogReg | 0.6716 | 0.585 | -0.0076 | 14/14 |
| LightGBM | 0.6724 | 0.579 | -0.0067 | 14/14 |
| RF | 0.6722 | 0.582 | -0.0070 | 13/14 |
| Elo | 0.6807 | 0.566 | +0.0016 | 6/14 |

## Ensemble Results

| Ensemble | Walk-Forward LL | Accuracy | Gap vs Market | Seasons Beat Market |
|----------|----------------|----------|---------------|---------------------|
| XGB+LogReg | 0.6719 | 0.582 | -0.0073 | 14/14 |
| XGB+LightGBM | 0.6730 | 0.580 | -0.0062 | 14/14 |
| XGB+RF | 0.6728 | 0.580 | -0.0064 | 13/14 |
| XGB+Elo | 0.6740 | 0.579 | -0.0052 | 11/14 |
| XGB+LogReg+LightGBM | 0.6716 | 0.581 | -0.0075 | 14/14 |
| All5_EqWt | 0.6719 | 0.582 | -0.0073 | 13/14 |
| All5_InvLL | 0.6718 | 0.581 | -0.0073 | 13/14 |

## Best Overall: LogReg (LL 0.6716)

## XGBoost Per-Season Breakdown (Baseline)

| Season | N | Model LL | Market LL | Gap | Acc | Winner |
|--------|---|----------|-----------|-----|-----|--------|
| 2012 | 2,267 | 0.6920 | 0.6831 | +0.0090 | 0.555 | Market |
| 2013 | 2,245 | 0.6850 | 0.6809 | +0.0041 | 0.559 | Market |
| 2014 | 2,233 | 0.6836 | 0.6856 | -0.0021 | 0.565 | **Model** |
| 2015 | 2,248 | 0.6809 | 0.6837 | -0.0028 | 0.569 | **Model** |
| 2016 | 2,259 | 0.6814 | 0.6805 | +0.0009 | 0.573 | Market |
| 2017 | 2,240 | 0.6733 | 0.6864 | -0.0130 | 0.585 | **Model** |
| 2018 | 2,231 | 0.6675 | 0.6809 | -0.0134 | 0.592 | **Model** |
| 2019 | 2,226 | 0.6673 | 0.6739 | -0.0067 | 0.590 | **Model** |
| 2020 | 727 | 0.6534 | 0.6767 | -0.0233 | 0.615 | **Model** |
| 2021 | 2,178 | 0.6718 | 0.6816 | -0.0098 | 0.577 | **Model** |
| 2022 | 2,397 | 0.6670 | 0.6674 | -0.0004 | 0.597 | **Model** |
| 2023 | 2,423 | 0.6716 | 0.6773 | -0.0057 | 0.578 | **Model** |
| 2024 | 2,410 | 0.6727 | 0.6725 | +0.0002 | 0.596 | Market |
| 2025 | 1,716 | 0.6763 | 0.6769 | -0.0006 | 0.567 | **Model** |

## LogReg Per-Season Breakdown

| Season | N | Model LL | Market LL | Gap | Acc | Winner |
|--------|---|----------|-----------|-----|-----|--------|
| 2012 | 2,267 | 0.6830 | 0.6831 | -0.0000 | 0.583 | **Model** |
| 2013 | 2,245 | 0.6724 | 0.6809 | -0.0086 | 0.587 | **Model** |
| 2014 | 2,233 | 0.6811 | 0.6856 | -0.0045 | 0.558 | **Model** |
| 2015 | 2,248 | 0.6797 | 0.6837 | -0.0039 | 0.570 | **Model** |
| 2016 | 2,259 | 0.6769 | 0.6805 | -0.0036 | 0.572 | **Model** |
| 2017 | 2,240 | 0.6742 | 0.6864 | -0.0122 | 0.589 | **Model** |
| 2018 | 2,231 | 0.6658 | 0.6809 | -0.0151 | 0.605 | **Model** |
| 2019 | 2,226 | 0.6618 | 0.6739 | -0.0122 | 0.596 | **Model** |
| 2020 | 727 | 0.6526 | 0.6767 | -0.0241 | 0.618 | **Model** |
| 2021 | 2,178 | 0.6655 | 0.6816 | -0.0161 | 0.590 | **Model** |
| 2022 | 2,397 | 0.6622 | 0.6674 | -0.0053 | 0.601 | **Model** |
| 2023 | 2,423 | 0.6704 | 0.6773 | -0.0069 | 0.580 | **Model** |
| 2024 | 2,410 | 0.6714 | 0.6725 | -0.0011 | 0.586 | **Model** |
| 2025 | 1,716 | 0.6731 | 0.6769 | -0.0038 | 0.573 | **Model** |

## Full Ranking (All Models + Ensembles)

| Rank | Model | WF LL | Gap vs Market | Beats Market? |
|------|-------|-------|---------------|---------------|
| 1 | LogReg | 0.6716 | -0.0076 | YES |
| 2 | XGB+LogReg+LightGBM | 0.6716 | -0.0075 | YES |
| 3 | All5_InvLL | 0.6718 | -0.0073 | YES |
| 4 | All5_EqWt | 0.6719 | -0.0073 | YES |
| 5 | XGB+LogReg | 0.6719 | -0.0073 | YES |
| 6 | RF | 0.6722 | -0.0070 | YES |
| 7 | LightGBM | 0.6724 | -0.0067 | YES |
| 8 | XGB+RF | 0.6728 | -0.0064 | YES |
| 9 | XGB+LightGBM | 0.6730 | -0.0062 | YES |
| 10 | XGB+Elo | 0.6740 | -0.0052 | YES |
| 11 | XGBoost | 0.6756 | -0.0036 | YES |
| 12 | Elo | 0.6807 | +0.0016 | no |

## Key Findings

### 1. Logistic Regression is the best individual model (LL 0.6716)
Surprising result: a simple L2-regularized logistic regression (C=0.01, heavy regularization) beats
XGBoost by 0.0040 LL and beats the market by 0.0076 LL. It wins **14/14 seasons** against the market,
compared to XGBoost's 10/14. This suggests the signal in MLB differential features is approximately
linear and the main risk is overfitting, which logistic regression with strong regularization avoids.

### 2. All 11 ML models beat the market; only Elo loses
Every model and ensemble except raw Elo beats the market. This confirms the MLB market edge
is real and robust across model architectures. The gap ranges from -0.0036 (XGBoost alone) to
-0.0076 (LogReg alone).

### 3. Ensembles do NOT improve over the best individual model
Unlike NBA, where ensembles helped, MLB ensembles consistently underperform the best individual
model (LogReg). The XGB+LogReg+LightGBM ensemble (0.6716) ties LogReg alone (0.6716). Adding
more models (All5_EqWt: 0.6719) or weighting by inverse-LL (All5_InvLL: 0.6718) both slightly
worsen results. Ensembling with XGBoost (0.6756) dilutes LogReg's edge.

### 4. Regularization is the key theme across all models
- LogReg: best at C=0.01 (heaviest regularization tested)
- LightGBM: best at num_leaves=8 (simplest trees), min_child_samples=50 (most conservative)
- RF: best at max_depth=6, min_samples_leaf=50 (most conservative leaf setting)
- XGBoost: max_depth=3, min_child_weight=15 (already known to need heavy regularization)

Every model does best with maximum regularization. MLB features are noisy enough that preventing
overfitting matters far more than capturing nonlinearities.

### 5. XGBoost is the weakest ML model
XGBoost (0.6756) ranks last among the 4 ML models. LogReg (0.6716), RF (0.6722), and LightGBM
(0.6724) all outperform it. XGBoost's flexibility is a liability when the signal is linear.

### 6. LogReg beats market in EVERY season (14/14)
This is remarkable consistency. Even in 2012 (minimal training data, 1 season), LogReg ties the
market (0.6830 vs 0.6831). By 2018-2021, the edge grows to 0.012-0.024 LL per season.

### Recommendations
1. **Switch primary model from XGBoost to Logistic Regression** (C=0.01, StandardScaler)
2. **Do not ensemble** -- LogReg alone is optimal for MLB
3. **The MLB market edge is larger than initially estimated**: 0.0076 LL (LogReg) vs 0.0043 (XGBoost from Phase 2)
4. Consider even heavier regularization (C=0.001, 0.005) in follow-up experiments
5. Re-run flat-bet P&L simulation with LogReg predictions -- expect improved ROI
