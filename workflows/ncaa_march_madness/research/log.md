# NCAA March Madness Research Log

Track experiments, findings, and decisions. Most recent entries at top. Read at start of every session.

---

## 2026-03-15 — Experiment 1: XGBoost Baseline + Backtest

### Model
XGBoost on 16 stat differentials, isotonic calibration on validation set.

### Backtest Results (21 seasons, 1,017 tournament games)
| Metric | Value |
|--------|-------|
| **Accuracy** | **79.5%** |
| Log Loss | 1.021 |
| Brier Score | 0.152 |

### Per-Season Highlights
- Best accuracy: 2026 (96.8%), 2022 (93.8%), 2021 (91.2%)
- Worst accuracy: 2006 (66.0%), 2011 (67.7%), 2013 (69.1%)
- Worst log loss: 2025 (2.056), 2005 (1.286), 2011 (1.090)

### Feature Importance (top 5)
1. diff_elo — 43.6%
2. diff_adj_em — 20.8%
3. diff_adj_oe — 5.4%
4. diff_kenpom_rank — 4.8%
5. diff_adj_de — 3.8%

### Key Observations
- **79.5% accuracy exceeds target (>72%)** and matches Odds Gods (77.6%)
- Log loss of 1.021 is above target (<0.55) — isotonic calibration hurts when validation set is small (reconstructed tournament games only have 30-67 samples)
- Elo diff alone accounts for 43.6% of feature importance — it's the dominant signal
- Accuracy is high but calibration needs work — model is overconfident on some predictions
- Seasons with major upsets (2011 VCU/Butler, 2006 George Mason) show worst performance

### Bracket Output (2026)
- **Chalk pick: Michigan** (champion in all 10K simulations)
- Balanced: Michigan 51.6%, Duke 22.3%, Arizona 18.6%, Florida 7.5%
- Contrarian: Michigan 33.0%, Arizona 26.1%, Duke 23.4%, Florida 14.8%
- Final Four (all strategies): Michigan, Duke, Florida, Arizona

### Issues to Address
1. **Calibration**: Isotonic regression needs more validation data. Try Platt scaling or larger val windows
2. **Data gap**: 2018-2025 tournament matchups are reconstructed (R64 only from seeds), not real game results
3. **Feature**: diff_kenpom_rank is all zeros for some seasons (no Massey ordinals) — need real KenPom data
4. **Overfitting to Elo**: Model relies too heavily on Elo diff. Consider capping or regularizing

### Next Steps
- [ ] Fix calibration: use Platt scaling or rolling 3-season val window
- [ ] Test LightGBM as alternative backend
- [ ] Add real game results for 2018-2025 (Kaggle competition access or scraping)
- [ ] Explore ensemble: Logistic Regression + XGBoost + LightGBM
- [ ] Reduce Elo dominance: try training without Elo to see stat-only performance

---

## 2026-03-15 — Project Setup

### What
Initialized youBet repo with core library and NCAA March Madness workflow structure.

### Design Decisions
- 16 stat differentials as features (experience included via KenPom)
- XGBoost baseline with isotonic calibration
- Temporal split by season: train 2003-2022, val 2023-2024, test 2025
- Monte Carlo bracket simulation with 3 strategies (chalk/balanced/contrarian)
- Quarter Kelly for bet sizing

### Data Sources Used
- KenPom dataset (Kaggle): 165 columns, 2002-2026 (primary features)
- Historical Elo ratings (Kaggle): 1985-2026
- College Basketball Dataset (Kaggle): 2013-2025 (win percentages)
- Kaggle March ML Mania box scores: 2003-2017 (game-level results)

### Key References
- Odds Gods: 77.6% tournament accuracy with LightGBM
- XGBalling: 90% regular season with XGBoost
- arXiv: calibration > accuracy → ~70% higher betting returns
