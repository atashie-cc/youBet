# Strategy Dashboard — Multi-Criteria Cross-Workflow Comparison

## Premise

vti-as-challenger (2026-05-12) confirmed that strict statistical gates cannot resolve which investment strategy is best given youBet's data window — 0/27 candidates survived inverted Holm, 0/27 reached TOST equivalence at workflow-native MEIs. The 9 workflows' uniformly null verdicts are **power-limited, not power-definitive**.

This workflow pivots to **multi-criteria comparative scoring**: rank ~65 strategies across 6 reusable parquet workflows + 6 reference baselines on a battery of 16 metrics in 6 dimensional groups (return, risk-adjusted, drawdown, tail/distribution, regime, stability). No formal gates. Apply rankings as judgment aids, not statistical proof.

## Conventions

- **Native window** is the primary scoring window for each strategy.
- **Common window 2010-01-05 to 2026-04-30** is the parallel sensitivity window so rankings can be compared apples-to-apples; factor-timing's pre-2010 history is reported separately.
- VTI buy-and-hold (from `workflows/real-world-test/artifacts/panel.parquet:VTI`) is the universal benchmark for the Information Ratio metric.
- z-scores are MAD-robust (`z = (x - median) / MAD`), not std-z, so extreme outliers like UPRO's MaxDD don't dominate rankings.
- "Lower-is-better" metrics (drawdown duration, rolling Sharpe std) are sign-flipped before z-scoring so higher composite always = better.
- For regime-period returns where a strategy's window doesn't cover that epoch (e.g., stock-selection 2010+ misses 2008 GFC), the per-metric table shows raw NaN. Composite imputes regime-column mean within the universe.

## Files
- `config.yaml` — locked: metric weights, regime epoch dates, common window
- `experiments/_shared.py` — load_strategy, compute_full_metrics, z-score, composite
- `experiments/phase0_assemble_roster.py` — writes `artifacts/roster.json` (frozen)
- `experiments/phase1_compute_metrics.py` — per-strategy 16-metric × 2-window battery
- `experiments/phase2_compose_rankings.py` — composite + per-metric + per-group rankings
- `experiments/phase3_visualize.py` — Pareto scatter PNGs
- `experiments/test_dashboard.py` — unit tests
- `artifacts/roster.json` — ~65-strategy roster (frozen, hashed)
- `artifacts/metrics.parquet` — strategy × metric matrix (long format)
- `artifacts/composite_ranking.json` — full ranked list with per-metric ranks
- `artifacts/{strategy_id}_metrics.json` — per-strategy detail
- `artifacts/pareto/*.png` — scatter plots
- `research/final-dashboard.md` — primary deliverable
- `research/log.md` — experiment log
