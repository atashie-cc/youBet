# NCAA Men's Basketball March Madness

## Domain Context
Predict outcomes of NCAA Men's Basketball Tournament games and generate bracket picks. 68 teams, single elimination, seeded 1-16 in 4 regions.

## Data Sources
1. **Kaggle March ML Mania** (primary): Free CSV download. Season results, tournament results, seeds, ordinals. 2003-present.
2. **KenPom** (subscription $24.95/yr): Adjusted efficiency metrics (AdjOE, AdjDE, AdjEM, AdjTempo), SOS. Gold standard for college basketball analytics.
3. **Bart Torvik T-Rank** (free): Similar to KenPom. Good free alternative.
4. **Sports Reference** (free, scrape at 1 req/3s): Box scores, advanced stats.

## Features (18 differentials)
All computed as Team A stat - Team B stat:
1. AdjOE diff — Adjusted offensive efficiency
2. AdjDE diff — Adjusted defensive efficiency
3. NetEff diff — Net efficiency margin
4. AdjTempo diff — Adjusted tempo
5. SOS diff — Strength of schedule
6. Seed diff — Tournament seed
7. Elo diff — Custom Elo rating
8. KenPom rank diff — KenPom ranking
9. Win% diff — Season win percentage
10. Win% last 5 diff — Recent form
11. TO rate diff — Turnover rate
12. FT rate diff — Free throw rate
13. 3PT rate diff — Three-point shooting rate
14. OReb rate diff — Offensive rebound rate
15. Ast rate diff — Assist rate
16. Blk rate diff — Block rate
17. Stl rate diff — Steal rate
18. Experience diff — Team experience/continuity

## Key Considerations
- Tournament games are different from regular season (single elimination, neutral courts, pressure)
- Seed matchups have historical patterns (12-5 upsets are common, ~35%)
- First Four games and play-in teams behave differently
- Conference tournament performance is a recency signal
- Coach tournament experience matters but is hard to quantify

## Research Log
See `research/log.md` — read at start of every session.
