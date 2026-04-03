# MMA Model Training Report

## Walk-Forward Cross-Validation Results

- **Model Log Loss**: 0.6608
- **Model Accuracy**: 0.6010
- **Model Brier Score**: 0.2343
- **Elo-only Log Loss**: 0.6754
- **Elo-only Accuracy**: 0.5763
- **Model vs Elo**: -0.0146 LL
- **Total fights evaluated**: 5461
- **Features used**: 15

## Per-Year Breakdown

| Year | Model LL | Model Acc | Elo LL | Elo Acc | N | Model vs Elo |
|------|----------|-----------|--------|---------|---|--------------|
| 2015 | 0.6845 | 0.5711 | 0.6778 | 0.5862 | 464 | +0.0067 |
| 2016 | 0.6680 | 0.5921 | 0.6761 | 0.5797 | 483 | -0.0081 |
| 2017 | 0.6729 | 0.5591 | 0.6808 | 0.5591 | 440 | -0.0079 |
| 2018 | 0.6656 | 0.5945 | 0.6737 | 0.5860 | 471 | -0.0081 |
| 2019 | 0.6743 | 0.5700 | 0.6937 | 0.5404 | 507 | -0.0194 |
| 2020 | 0.6617 | 0.6075 | 0.6870 | 0.5565 | 451 | -0.0254 |
| 2021 | 0.6734 | 0.5688 | 0.6824 | 0.5749 | 494 | -0.0090 |
| 2022 | 0.6459 | 0.6383 | 0.6663 | 0.6206 | 506 | -0.0205 |
| 2023 | 0.6488 | 0.6270 | 0.6721 | 0.5675 | 504 | -0.0232 |
| 2024 | 0.6521 | 0.6023 | 0.6661 | 0.5536 | 513 | -0.0140 |
| 2025 | 0.6383 | 0.6569 | 0.6639 | 0.6004 | 513 | -0.0257 |
| 2026 | 0.6106 | 0.6609 | 0.6407 | 0.6261 | 115 | -0.0300 |

## Features (15)

- diff_elo
- diff_wc_elo
- diff_sig_str_landed_avg
- diff_str_accuracy
- diff_td_landed_avg
- diff_td_pct
- diff_sub_att_avg
- diff_win_rate
- diff_win_streak
- diff_reach
- diff_height
- diff_age
- diff_days_since_last
- weight_class_move_a
- weight_class_move_b