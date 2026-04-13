# 2026 FIFA World Cup Bracket Prediction

## Domain Context
Predict match outcomes and generate bracket picks for the 2026 FIFA World Cup. New 48-team format: 12 groups of 4 → top 2 from each group + 8 best third-placed teams advance to a 32-team knockout bracket → R16 → QF → SF → Final (104 total matches).

**Primary objective**: Bracket pool optimization, not betting edge. Top-flight soccer is documented as market-efficient in `docs/research/multi-market-plan.md`, so the workflow optimizes for calibrated probabilities + bracket accuracy, with a Stage-2 market LL sanity check per CLAUDE.md principle #9.

Structural complications vs `workflows/ncaa_march_madness/`:
1. Three-way outcomes (W/D/L) in group stage — core infra is binary-only
2. Knockout matches have ET + penalty tails — cannot model as simple binary
3. Group correlation + FIFA tiebreakers + best-third-placed ranking — NCAA simulator doesn't handle this
4. 2026 48-team format is structurally novel — prior WCs were 32 teams / 8 groups

## Phase Plan
See `C:\Users\18033\.claude\plans\linear-greeting-turing.md` for the full plan. Phase summary:

- **Phase 0**: Data feasibility spike (Kaggle confirmed; FBref/Transfermarkt/OddsPortal feasibility to be proven before downstream use)
- **Phase A**: Core infra decision — extend `core/experiment.py` for multi-class, or bypass Experiment runner
- **Phase 1**: Elo + 6 minimal features (no xG, no squad features)
- **Phase 2**: 3-way XGBoost baseline + Stage-2 market gate
- **Phase 3**: Multi-stage knockout model (regulation → ET → penalties)
- **Phase 4**: Group simulator with FIFA tiebreakers + knockout tree (mostly new code)
- **Phase 5**: Backtest on 2018/2022 + final report
- **Phase 6 (conditional)**: Feature expansion if baseline underperforms

## Pipeline (planned)
```bash
python scripts/collect_data.py             # Phase 0: Kaggle international results
python scripts/fbref_feasibility.py        # Phase 0 spike: FBref international xG
python scripts/transfermarkt_feasibility.py # Phase 0 spike: squad scraping
python scripts/oddsportal_feasibility.py   # Phase 0 spike: historical WC odds
python scripts/compute_elo.py              # Phase 1
python scripts/build_features.py           # Phase 1
python scripts/train.py                    # Phase 2 (after Phase A decision)
python scripts/backtest.py                 # Phase 5
python scripts/generate_bracket.py         # Phase 4 → post-draw Dec 2025
```

## Data Sources (confirmed status pending Phase 0)
1. **Kaggle "International football results 1872-2024"** (Jürisoo) — ~45K matches, score, tournament type, venue, neutral flag. **Status**: assumed available, to be verified in Phase 0.
2. **FBref international match xG** — FEASIBILITY SPIKE in Phase 0.
3. **Transfermarkt / FIFA squads** — FEASIBILITY SPIKE (deferred to Phase 6 regardless).
4. **OddsPortal historical WC closing odds** (2010/2014/2018/2022) — FEASIBILITY SPIKE.

## Features (Phase 1 v2 baseline — 5 differentials + 1 match-level flag)
Differentials computed as home stat minus away stat:
1. `diff_elo` — International Elo rating (pre-match)
2. `diff_elo_trend_12mo` — Elo change over trailing 365 days
3. `diff_win_rate_l10` — Last 10 international matches W / decided
4. `diff_goal_diff_per_match_l10` — Rolling goal differential
5. `diff_rest_days` — Days since last international (capped at 365)

Match-level flag (NOT a differential):
6. `is_neutral` — Neutral venue flag (binary). Per the Phase 0 audit, most
   WC matches are neutral EXCEPT host-nation home games: in 2026, 9 of 72
   WC group-stage fixtures are non-neutral (Mexico/Canada/USA home games).

**Neutral-match orientation**: for `is_neutral == 1` rows, the home/away
labels are deterministically flipped (SHA-256 hash of date + sorted team
names) so the neutral-slice class distribution is symmetric between home_win
and away_win. Non-neutral rows preserve their raw venue ordering. An audit
column `was_flipped` tags flipped rows for downstream join/debugging.

**Deferred to Phase 6**: xG features, squad continuity, club league strength, confederation flag, host bonus, Dixon-Coles goal model.

## Key Considerations
- **Multi-class target** — core binary infra needs Phase A decision before Phase 2
- **Knockout semantics** — regulation win / draw+ET / draw+penalties is a three-stage model, not binary "decided match"
- **Format shift** — training on 32-team WCs does not directly validate 2026 bracket accuracy
- **Group draw is Dec 2025** — team-level 2026 predictions cannot run until after the draw
- **PIT discipline** — only features with trivial cutoff semantics in v1 baseline; anything with forward-fill or entity-resolution hazards is deferred to Phase 6 with documented audit
- **No host bonus in baseline** — US/CAN/MEX 2026 host bonus is unvalidated historically and deferred to Phase 6 with sensitivity analysis

## Research Log
See `research/log.md` — read at start of every session.
