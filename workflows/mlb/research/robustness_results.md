# Experiment 5.7: Robustness Checks

**Date**: 2026-04-01

**Model**: LogReg(C=0.005) with StandardScaler

**Features** (8): elo_diff, diff_fg_bat_wRC+, diff_fg_bat_wOBA, diff_fg_bat_WAR, diff_fg_pit_ERA, diff_fg_pit_FIP, diff_fg_pit_WHIP, diff_fg_pit_WAR

**Data**: 29523 games with odds, seasons 2012-2025

---

## Test 1: Permutation Test

**Method**: Shuffle `home_win` labels within each season, rerun walk-forward (expanding window, train on seasons < hold, test on hold). 500 iterations.

**Hold-out seasons**: [2021, 2022, 2023, 2024]


| Metric | Value |
|--------|-------|
| Real model LL | 0.677460 |
| Null distribution mean | 0.691510 |
| Null distribution std | 0.000191 |
| Z-score | -73.63 |
| P-value | 0.0000 (0/500) |
| Statistically significant | YES (p < 0.05) |

**Interpretation**: The model's log loss (0.6775) is significantly better than chance. Only 0 of 500 random permutations achieved a LL as good or better, confirming the model has genuine predictive power.

---

## Test 2: Fixed-Window vs Expanding Walk-Forward

**Fixed window**: most recent 5 seasons for training. **Expanding**: all prior seasons.


| Mode | Walk-Forward LL | Market LL | Gap | Acc | Games |
|------|----------------|-----------|-----|-----|-------|
| Expanding | 0.6793 | 0.6789 | +0.0004 | 56.7% | 27,394 |
| Fixed (5-yr) | 0.6795 | 0.6789 | +0.0006 | 56.7% | 27,394 |

### Per-Season Breakdown

| Season | Expanding LL | Fixed LL | Diff (Fixed-Expand) | Better |
|--------|-------------|----------|---------------------|--------|
| 2013 | 0.6818 | 0.6818 | +0.0000 | Fixed |
| 2014 | 0.6884 | 0.6884 | +0.0000 | Fixed |
| 2015 | 0.6868 | 0.6868 | +0.0000 | Fixed |
| 2016 | 0.6852 | 0.6852 | +0.0000 | Fixed |
| 2017 | 0.6826 | 0.6826 | +0.0000 | Fixed |
| 2018 | 0.6735 | 0.6741 | +0.0006 | Expanding |
| 2019 | 0.6677 | 0.6681 | +0.0004 | Expanding |
| 2020 | 0.6727 | 0.6751 | +0.0024 | Expanding |
| 2021 | 0.6744 | 0.6751 | +0.0006 | Expanding |
| 2022 | 0.6736 | 0.6738 | +0.0002 | Expanding |
| 2023 | 0.6806 | 0.6810 | +0.0004 | Expanding |
| 2024 | 0.6809 | 0.6809 | +0.0000 | Expanding |
| 2025 | 0.6779 | 0.6769 | -0.0010 | Fixed |

**Verdict**: Expanding window LL=0.6793 vs Fixed LL=0.6795 (diff=+0.0002). Expanding wins 7/13 seasons. Old data helps — expanding window is better.

---

## Test 3: Vig-Adjusted Edge

**Method**: Flat $100 bets where model probability disagrees with vig-free market probability by > 2%. P&L computed against **raw moneyline odds** (with full vig).


| Season | Bets | Wins | Win% | P&L | ROI |
|--------|------|------|------|-----|-----|
| 2013 | 1,706 | 856 | 50.2% | $+6,522 | +3.8% |
| 2014 | 1,605 | 773 | 48.2% | $+596 | +0.4% |
| 2015 | 1,631 | 798 | 48.9% | $+6,084 | +3.7% |
| 2016 | 1,628 | 760 | 46.7% | $+2,338 | +1.4% |
| 2017 | 1,614 | 810 | 50.2% | $+15,773 | +9.8% |
| 2018 | 1,709 | 889 | 52.0% | $+18,961 | +11.1% |
| 2019 | 1,703 | 903 | 53.0% | $+21,857 | +12.8% |
| 2020 | 580 | 307 | 52.9% | $+7,900 | +13.6% |
| 2021 | 1,664 | 879 | 52.8% | $+22,277 | +13.4% |
| 2022 | 1,715 | 747 | 43.6% | $-7,859 | -4.6% |
| 2023 | 1,698 | 794 | 46.8% | $-624 | -0.4% |
| 2024 | 1,699 | 750 | 44.1% | $-11,401 | -6.7% |
| 2025 | 1,186 | 550 | 46.4% | $+10 | +0.0% |
| **TOTAL** | **20,138** | **9,816** | **48.7%** | **$+82,434** | **+4.1%** |
| **2022-25** | **6,298** | **2,841** | **45.1%** | **$-19,875** | **-3.2%** |

**Verdict**: After full vig, the model is profitable overall (+4.1% ROI on 20,138 bets). Profitable in 10/13 seasons. 
In the recent DraftKings era (2022-2025): -3.2% ROI on 6,298 bets.

---

## Test 4: Model vs Elo — Value of FG Stats

**Question**: How much predictive value do FanGraphs prior-season stats add beyond Elo ratings alone?


| Season | Model LL | Elo LL | Market LL | Model-Elo | Model-Market |
|--------|----------|--------|-----------|-----------|-------------|
| 2013 | 0.6818 | 0.6817 | 0.6809 | +0.0000 | +0.0008 |
| 2014 | 0.6884 | 0.6915 | 0.6856 | -0.0031 | +0.0028 |
| 2015 | 0.6868 | 0.6880 | 0.6837 | -0.0012 | +0.0031 |
| 2016 | 0.6852 | 0.6878 | 0.6805 | -0.0026 | +0.0047 |
| 2017 | 0.6826 | 0.6835 | 0.6864 | -0.0009 | -0.0038 |
| 2018 | 0.6735 | 0.6729 | 0.6809 | +0.0006 | -0.0074 |
| 2019 | 0.6677 | 0.6675 | 0.6739 | +0.0002 | -0.0062 |
| 2020 | 0.6727 | 0.6751 | 0.6767 | -0.0024 | -0.0040 |
| 2021 | 0.6744 | 0.6746 | 0.6816 | -0.0002 | -0.0072 |
| 2022 | 0.6736 | 0.6748 | 0.6679 | -0.0012 | +0.0057 |
| 2023 | 0.6806 | 0.6830 | 0.6773 | -0.0024 | +0.0033 |
| 2024 | 0.6809 | 0.6829 | 0.6723 | -0.0020 | +0.0085 |
| 2025 | 0.6779 | 0.6795 | 0.6770 | -0.0016 | +0.0009 |
| **OVERALL** | **0.6793** | **0.6807** | **0.6789** | **-0.0013** | **+0.0004** |

**FG stats add +0.0013 LL** improvement over Elo alone. The gap between our model and the market is +0.0004 LL. 
FG stats close 76% of the Elo-to-market gap (0.6807 -> 0.6793, target 0.6789).

---

## Summary

1. **Permutation test**: p=0.0000 — model HAS statistically significant predictive power.
2. **Fixed vs expanding window**: Expanding window is better (LL 0.6793 vs 0.6795). Historical data helps.
3. **Vig-adjusted ROI**: +4.1% on 20,138 bets (profitable after full vig). 2022-2025: -3.2%.
4. **FG stats vs Elo**: FG features add +0.0013 LL over Elo alone (0.6807 -> 0.6793).