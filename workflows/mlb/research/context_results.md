# Experiment 5.6: Market Efficiency by Game Context

**Date**: 2026-04-01
**Model**: LogReg(C=0.005) with 8 features (Elo + prior-season FG)
**Method**: Walk-forward (train on seasons < hold, test on hold), 2012-2025
**Total games**: 27,394
**Overall**: Model LL=0.6793, Market LL=0.6789, Gap=+0.0004

---

## 1. By Month

**Hypothesis**: Early season (April/May) shows more model edge since market relies on stale priors.

| Month | N | Model LL | Market LL | Gap | Beats Market? |
|---|---:|---:|---:|---:|---|
| Mar/Apr | 226 | 0.6898 | 0.6793 | +0.0105 | No |
| April | 4,183 | 0.6820 | 0.6803 | +0.0017 | No |
| May | 4,658 | 0.6806 | 0.6822 | -0.0016 | **Yes** |
| June | 4,529 | 0.6811 | 0.6782 | +0.0029 | No |
| July | 4,294 | 0.6846 | 0.6839 | +0.0007 | No |
| August | 4,857 | 0.6711 | 0.6726 | -0.0015 | **Yes** |
| September | 4,429 | 0.6762 | 0.6769 | -0.0007 | **Yes** |
| Oct | 218 | 0.6932 | 0.6766 | +0.0165 | No |

**Early season (Mar-May)**: Model=0.6815, Market=0.6812, Gap=+0.0002
**Late season (Jul-Sep)**: Model=0.6771, Market=0.6776, Gap=-0.0006

**Verdict**: Model beats market in 3/8 months.

---

## 2. By Line Size

**Hypothesis**: Toss-up games may show more model edge where market has less confidence.

| Bucket | N | Model LL | Market LL | Gap | Beats Market? |
|---|---:|---:|---:|---:|---|
| A_heavy_fav(|ML|>200) | 4,895 | 0.6463 | 0.6470 | -0.0007 | **Yes** |
| B_moderate(150-200) | 10,580 | 0.6785 | 0.6805 | -0.0019 | **Yes** |
| C_tossup(100-150) | 11,919 | 0.6936 | 0.6906 | +0.0029 | No |

**Verdict**: Mixed — toss-up gap=+0.0029, heavy fav gap=-0.0007.

---

## 3. By Odds Source Era

**Hypothesis**: Any model 'edge' in 2017-2021 may be inflated by softer Kaggle-sourced lines. DraftKings 2022-2025 lines are sharper.

| Era | N | Model LL | Market LL | Gap | Beats Market? |
|---|---:|---:|---:|---:|---|
| DraftKings(2022-2025) | 8,807 | 0.6783 | 0.6734 | +0.0049 | No |
| Kaggle(2011-2021) | 18,587 | 0.6798 | 0.6815 | -0.0017 | **Yes** |

**Kaggle gap**: -0.0017 | **DraftKings gap**: +0.0049

**Verdict**: Hypothesis SUPPORTED — model only beats softer Kaggle-era lines, not modern DraftKings lines.

---

## 4. By Day of Week

**Hypothesis**: Weekend lines may be softer (recreational bettors).

| Day Type | N | Model LL | Market LL | Gap | Beats Market? |
|---|---:|---:|---:|---:|---|
| Weekday(Mon-Fri) | 18,408 | 0.6787 | 0.6786 | +0.0001 | No |
| Weekend(Sat-Sun) | 8,986 | 0.6805 | 0.6795 | +0.0010 | No |

**Verdict**: Hypothesis REJECTED — model loses on both.

---

## 5. Cross-Slice: Month x Era

Do early-season edges persist in the DraftKings era?

### Kaggle(2011-2021)

| Month | N | Model LL | Market LL | Gap | Beats Market? |
|---|---:|---:|---:|---:|---|
| Mar/Apr | 98 | 0.6979 | 0.6865 | +0.0114 | No |
| April | 2,723 | 0.6869 | 0.6872 | -0.0003 | **Yes** |
| May | 3,077 | 0.6813 | 0.6838 | -0.0024 | **Yes** |
| June | 2,986 | 0.6813 | 0.6821 | -0.0007 | **Yes** |
| July | 2,861 | 0.6828 | 0.6820 | +0.0008 | No |
| August | 3,440 | 0.6722 | 0.6757 | -0.0036 | **Yes** |
| September | 3,270 | 0.6756 | 0.6798 | -0.0043 | **Yes** |
| Oct | 132 | 0.6873 | 0.6758 | +0.0115 | No |

### DraftKings(2022-2025)

| Month | N | Model LL | Market LL | Gap | Beats Market? |
|---|---:|---:|---:|---:|---|
| Mar/Apr | 128 | 0.6836 | 0.6737 | +0.0099 | No |
| April | 1,460 | 0.6728 | 0.6674 | +0.0054 | No |
| May | 1,581 | 0.6791 | 0.6790 | +0.0001 | No |
| June | 1,543 | 0.6808 | 0.6708 | +0.0100 | No |
| July | 1,433 | 0.6883 | 0.6878 | +0.0005 | No |
| August | 1,417 | 0.6685 | 0.6650 | +0.0035 | No |
| September | 1,159 | 0.6780 | 0.6688 | +0.0092 | No |
| Oct | 86 | 0.7021 | 0.6779 | +0.0242 | No |

---

## Summary & Key Takeaways

1. **Odds source era is the dominant factor.** Model beats Kaggle-era lines (-0.0017) but loses to DraftKings lines (+0.0049). This strongly suggests the Phase 1-3 'edge' was partly an artifact of softer historical odds.
2. **Monthly variation**: Model beats market in 3/8 months (May, August, September).
3. **Line size**: Model beats market in 2/3 buckets (A_heavy_fav(|ML|>200), B_moderate(150-200)).
4. **Day of week**: Weekend gap=+0.0010, Weekday gap=+0.0001. Weekday slightly better.

**Overall**: The model does not reliably beat modern (2022+) MLB closing lines in any game context. Consistent with the Phase 4 finding that clean features produce ~tied results. The historical 'edge' likely reflects softer Kaggle-era odds, not genuine model superiority.
