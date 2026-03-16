# NCAA March Madness Research Log

Track experiments, findings, and decisions. Most recent entries at top. Read at start of every session.

---

## 2026-03-15 — Project Setup

### What
Initialized youBet repo with core library and NCAA March Madness workflow structure.

### Design Decisions
- 18 stat differentials as features (see CLAUDE.md for full list)
- XGBoost baseline with isotonic calibration
- Temporal split by season: train 2003-2022, val 2023-2024, test 2025
- Monte Carlo bracket simulation with 3 strategies (chalk/balanced/contrarian)
- Quarter Kelly for bet sizing

### Key References
- Odds Gods: 77.6% tournament accuracy with LightGBM
- XGBalling: 90% regular season with XGBoost
- arXiv: calibration > accuracy → ~70% higher betting returns

### Next Steps
- [ ] Implement `collect_data.py` — start with Kaggle CSV download
- [ ] Implement `build_features.py` — 18 differentials
- [ ] Train baseline model and measure log loss
