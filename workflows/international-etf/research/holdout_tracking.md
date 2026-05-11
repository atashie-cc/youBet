# International ETF Workflow - Prospective Holdout Tracking

**Pre-committed in plan v1.3:**
- Holdout-start date: 2026-05-02
- No decision power until: 2028-05-02 (>=24 months) AND >=1 DXY direction-change
- Tracked strategies (4 descriptive baselines; none passed Phase 0-4 strict gate):
  - **A**: 100% VTI buy-and-hold (anchor)
  - **B**: 60/40 VTI/VXUS annual rebalance (Vanguard variance-min point)
  - **C**: 60/40 VTI/HEFA annual rebalance (C1b - near-miss; rejected by P1a placebo; tracked for falsification)
  - **D**: 60/40 VTI/VEA annual rebalance (unhedged hedging-comparison baseline)

**Decision rule (pre-committed):** results inform but do not gate. After 2028-05 + DXY direction-change, run Round 3 codex review on whatever the holdout shows.

## Rolling log

| As-of | Price-end | n_days | DXY | DXY 12m % | A cum | B cum | C cum | D cum | C-A | C-B | C-D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-05 | 2026-05-04 | 1 | 98.47 | -1.53% (negative) | -0.0037 | -0.0057 | -0.0060 | -0.0069 | -0.0023 | -0.0003 | +0.0009 |
