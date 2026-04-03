# MMA Elo Tuning Report

## Grid Search Results

Total configurations tested: 36

### Top 10 Configurations

| Rank | K | Decay | Finish | WC Penalty | Log Loss | Accuracy |
|------|---|-------|--------|------------|----------|----------|
| 1 | 80 | 0.10 | 1.2 | 0.00 | 0.6750 | 0.5794 |
| 2 | 80 | 0.10 | 1.2 | 0.10 | 0.6756 | 0.5759 |
| 3 | 80 | 0.25 | 1.2 | 0.00 | 0.6760 | 0.5783 |
| 4 | 80 | 0.10 | 1.0 | 0.00 | 0.6763 | 0.5713 |
| 5 | 80 | 0.25 | 1.2 | 0.10 | 0.6767 | 0.5772 |
| 6 | 80 | 0.10 | 1.0 | 0.10 | 0.6769 | 0.5695 |
| 7 | 80 | 0.25 | 1.0 | 0.00 | 0.6773 | 0.5792 |
| 8 | 80 | 0.25 | 1.0 | 0.10 | 0.6779 | 0.5743 |
| 9 | 50 | 0.10 | 1.2 | 0.00 | 0.6779 | 0.5755 |
| 10 | 50 | 0.10 | 1.2 | 0.10 | 0.6784 | 0.5744 |

### Parameter Sensitivity

Average log loss by parameter value:


**k_factor**:
  - 30: mean=0.6835, std=0.0007, best=0.6821
  - 50: mean=0.6800, std=0.0012, best=0.6779
  - 80: mean=0.6778, std=0.0021, best=0.6750

**time_decay_rate**:
  - 0.0: mean=0.6817, std=0.0016, best=0.6798
  - 0.1: mean=0.6792, std=0.0030, best=0.6750
  - 0.25: mean=0.6803, std=0.0030, best=0.6760

**finish_bonus**:
  - 1.0: mean=0.6810, std=0.0027, best=0.6763
  - 1.2: mean=0.6799, std=0.0028, best=0.6750

**weight_class_transfer_penalty**:
  - 0.0: mean=0.6803, std=0.0029, best=0.6750
  - 0.1: mean=0.6806, std=0.0027, best=0.6756

## Best Parameters

```yaml
elo:
  k_factor: 80.0
  time_decay_rate: 0.1
  finish_bonus: 1.2
  weight_class_transfer_penalty: 0.0
```

## Ablation Study

| Configuration | Log Loss | Accuracy | Delta vs Baseline |
|--------------|----------|----------|-------------------|
| Baseline (K only) | 0.6809 | 0.5616 | +0.0000 |
| + Time decay | 0.6763 | 0.5713 | -0.0046 |
| + Finish bonus | 0.6801 | 0.5647 | -0.0008 |
| + WC transfer penalty | 0.6809 | 0.5616 | +0.0000 |
| All enhancements | 0.6750 | 0.5794 | -0.0059 |

## K-Factor Schedule Experiment

| Schedule | Log Loss | Accuracy |
|----------|----------|----------|
| None | 0.6750 | 0.5794 |
| {'first_n_fights': 3, 'early_k_multiplier': 1.5} | 0.6749 | 0.5799 |
| {'first_n_fights': 5, 'early_k_multiplier': 1.5} | 0.6754 | 0.5763 |
| {'first_n_fights': 5, 'early_k_multiplier': 2.0} | 0.6807 | 0.5777 |
| {'first_n_fights': 10, 'early_k_multiplier': 1.5} | 0.6750 | 0.5814 |