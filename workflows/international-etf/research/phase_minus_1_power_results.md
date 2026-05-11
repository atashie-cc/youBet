# Phase -1 — Power Analysis Results

- Run date: 2026-05-01
- N (Holm denominator): 19
- Simulations per effect size: 100
- Sample length: 24.7 years (~6224 trading days)
- Correlation: 0.7
- Bootstrap: stationary, block=22d, n=600, ci=0.9
- Gate: ExSharpe>0.2 OR fallback 0.4; Holm p<0.05; CI_lower>0.0

## Power and Type-I error by target excess Sharpe

| Target ExSh | Power @ 0.20 gate | Power @ 0.40 gate | FWER (any null pass) | Mean point | Point SD |
|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.030 | -0.001 | 0.155 |
| 0.20 | 0.040 | 0.040 | 0.070 | 0.199 | 0.155 |
| 0.30 | 0.130 | 0.130 | 0.150 | 0.299 | 0.154 |
| 0.40 | 0.300 | 0.300 | 0.320 | 0.399 | 0.154 |

## Pre-committed decision rule application

- Power at ExSharpe=0.20, primary gate: **0.040** (threshold: 0.50)
- Power at ExSharpe=0.40, fallback gate: **0.300** (threshold: 0.50)
- FWER at ExSharpe=0.0 (target=null): **0.030** (target: <=0.05)

**Decision: HALT.** Even ExSharpe = 0.40 is underpowered with N=19 + ~25yr daily data. Either reduce N (re-prune plan) or accept the workflow as descriptive-only.

FWER controlled at null (observed 0.030 <= nominal 0.05).