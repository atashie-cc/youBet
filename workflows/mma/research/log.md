# MMA Research Log

## Phase 1: Data Collection (2026-04-03)

Starting MMA/UFC fight prediction workflow. Market research indicates:
- UFC closing line LL estimated ~0.65-0.68 (less efficient than NBA 0.59, comparable to MLB 0.68)
- Documented longshot bias in MMA markets
- Poor calibration near pick'em fights
- ~500-550 UFC fights/year, $10.3B handle in 2024

Strategy: Kaggle dataset first for speed, defer scraping to Phase 3.

### Data Summary
- 7,169 UFC fights (2010-2026), 2,241 fighters
- 96% odds coverage (6,892 fights), avg vig 3.54%
- Red corner (fighter_a) wins 57.8%, 48.6% finishes

## Phase 2: Baseline Model + Market Efficiency (2026-04-03)

### Elo Tuning Results (Quick Grid — 36 combos)
- **Best overall Elo LL: 0.6750** (K=80, decay=0.10, finish_bonus=1.2, WC_penalty=0.0)
- **Weight-class Elo LL: 0.6855** — slightly worse due to fewer fights per class
- Time decay saves 0.0046, finish bonus saves 0.0008, all enhancements: 0.0059 improvement
- High K (80) is optimal — confirms sparse data needs aggressive updates

### Model Evolution Through Codex Reviews

| Version | Model LL | Gap to Opening | Fix Applied |
|---------|----------|----------------|-------------|
| Initial XGBoost | 0.6991 | — | Baseline (worse than Elo) |
| + Cal leak fix (Codex R1) | 0.6623 | +0.053 | Calibration year excluded from training |
| + Age PIT + WC Elo (Codex R2) | 0.6676 | +0.035 | Per-fight age, weight-class Elo, paired closing line, scraper validation |
| + Config features + smart cal + reach PIT (Codex R3) | **0.6608** | **+0.033** | Explicit feature schema, 80/20 cal split, PIT reach/height |

### Market Efficiency Test — Final Results

| Source | Log Loss | Accuracy | N |
|--------|----------|----------|---|
| **Market closing line** | **0.6130** | **65.6%** | 4,050 |
| **Market opening line** | **0.6318** | **64.0%** | 4,050 |
| XGBoost model (final) | 0.6608 | 60.1% | 5,461 |
| Elo-only (overall) | 0.6754 | 57.6% | 5,461 |

**Gaps:**
- Model vs closing: +0.050 LL — STOP (kill threshold was 0.02)
- Model vs opening: +0.033 LL — even opening lines beat the model
- Opening → Closing: +0.019 LL — late-breaking info worth 0.019 LL (larger than NBA's ~0.007)

### Key Observations
1. **MMA market far more efficient than literature predicted** — closing LL 0.61, not 0.65-0.68
2. Market beats model in every year, every weight class, every odds range
3. Opening lines (set 5-7 days out) are also efficient — gap +0.033 LL
4. MMA lines move more than NBA (0.019 vs 0.007 LL) due to weigh-ins, injuries, camp news
5. Opening vig is 5.7% vs closing 1.8% — wider spreads on opening lines
6. Weight-class Elo (0.6855) is worse than overall Elo (0.6754) — fewer fights per class means slower convergence
7. XGBoost beats Elo by 0.015 LL but both are far behind the market

### Codex Adversarial Review Findings (3 rounds, 7 fixes)
**Round 1:** Calibration fold leaks into training — early stopping and Platt scaling on training data
**Round 2:** Age from future profile snapshots, Elo tuning protocol, synthetic closing line (best-of-both-sides), scraper name collisions
**Round 3:** Feature selection ignores config (grabs all diff_ columns), cal year exclusion too aggressive (model never trains on freshest data), reach/height from future profile snapshots

All fixes improved methodology correctness. Net model improvement: 0.6991 → 0.6608 (0.038 LL gain).

### Conclusion
**MMA moneyline betting is NOT viable with public statistics.** Both opening and closing lines are efficient.

This is the third sport (after NBA and MLB) to reach the same conclusion. The pattern is now confirmed across 3 independent experiments: moneyline closing lines on any well-known sport incorporate more information than public statistical models can match.
