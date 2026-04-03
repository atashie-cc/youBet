"""Retroactive betting simulation for MLB using ensemble disagreement.

Loads ensemble predictions from analyze_disagreement.py output and tests
multiple betting strategies with Kelly sizing, risk limits, and CLV tracking.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add core library to path
sys.path.insert(0, str(Path("C:/github/youBet/src")))
from youbet.core.bankroll import (
    confidence_kelly,
    fractional_kelly,
    remove_vig,
    american_to_decimal,
    RiskLimits,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_CSV = Path("C:/github/youBet/workflows/mlb/betting/output/reports/ensemble_predictions.csv")
OUTPUT_DIR = Path("C:/github/youBet/workflows/mlb/betting/output/reports")

STARTING_BANKROLL = 10_000.0

STRATEGIES: dict[str, dict[str, Any]] = {
    "flat_eighth_open": dict(
        base_kelly=0.125, edge_threshold=0.0,
        quintile_filter=None, quintile_multipliers=None, line_type="open",
    ),
    "flat_eighth_close": dict(
        base_kelly=0.125, edge_threshold=0.0,
        quintile_filter=None, quintile_multipliers=None, line_type="close",
    ),
    "q1q2_only": dict(
        base_kelly=0.125, edge_threshold=0.0,
        quintile_filter=[0, 1], quintile_multipliers=None, line_type="open",
    ),
    "confidence_scaled": dict(
        base_kelly=0.125, edge_threshold=0.0,
        quintile_filter=None,
        quintile_multipliers={0: 1.5, 1: 1.2, 2: 1.0, 3: 0.5, 4: 0.0},
        line_type="open",
    ),
    "conservative": dict(
        base_kelly=0.125, edge_threshold=0.02,
        quintile_filter=[0, 1], quintile_multipliers=None, line_type="open",
    ),
    "selective": dict(
        base_kelly=0.125, edge_threshold=0.03,
        quintile_filter=[0, 1, 2],
        quintile_multipliers={0: 1.5, 1: 1.2, 2: 0.8},
        line_type="open",
    ),
    "aggressive": dict(
        base_kelly=0.25, edge_threshold=0.0,
        quintile_filter=None,
        quintile_multipliers={0: 2.0, 1: 1.5, 2: 1.0, 3: 0.5, 4: 0.0},
        line_type="open",
    ),
}

RISK = RiskLimits(
    max_daily_exposure=0.20,
    max_single_bet=0.05,
    stop_loss_threshold=0.50,
)


def _precompute_bet_candidates(
    games: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Pre-compute all candidate bets with their fractions, odds, and CLV.

    Returns a DataFrame with one row per candidate bet (can be 0, 1, or 2
    per game -- home side, away side, or both). Each row has:
    - game_idx: index into the original games DataFrame
    - game_date, season
    - side: "home" or "away"
    - bet_frac: Kelly fraction before daily cap scaling
    - dec_odds: decimal odds for this side
    - clv: closing line value
    - home_win_col: 1 if home wins (from original data)
    """
    base_kelly = cfg["base_kelly"]
    edge_threshold = cfg["edge_threshold"]
    quintile_filter = cfg["quintile_filter"]
    quintile_multipliers = cfg["quintile_multipliers"]
    line_type = cfg["line_type"]

    records: list[dict] = []

    for idx, game in games.iterrows():
        quintile = int(game["agreement_quintile"])
        logreg_pred = game["logreg_pred"]

        # Choose lines
        if line_type == "open":
            home_ml = game["home_open_ml"]
            away_ml = game["away_open_ml"]
        else:
            home_ml = game["home_close_ml"]
            away_ml = game["away_close_ml"]

        # Skip invalid lines
        try:
            vf_home, vf_away, overround = remove_vig(home_ml, away_ml)
        except (ValueError, ZeroDivisionError):
            continue

        vig_per_side = overround / 2.0

        # Check both sides
        for side in ["home", "away"]:
            if side == "home":
                model_prob = logreg_pred
                implied_prob = vf_home
                ml = home_ml
                open_ml_side = game["home_open_ml"]
                close_ml_side = game["home_close_ml"]
            else:
                model_prob = 1.0 - logreg_pred
                implied_prob = vf_away
                ml = away_ml
                open_ml_side = game["away_open_ml"]
                close_ml_side = game["away_close_ml"]

            edge = model_prob - implied_prob

            # Edge must exceed vig + threshold
            if edge <= vig_per_side + edge_threshold:
                continue

            # Quintile filter
            if quintile_filter is not None and quintile not in quintile_filter:
                continue

            # Compute multiplier
            multiplier = 1.0
            if quintile_multipliers is not None:
                multiplier = quintile_multipliers.get(quintile, 0.0)
                if multiplier <= 0:
                    continue

            # Compute bet fraction
            try:
                dec_odds = american_to_decimal(ml)
            except ValueError:
                continue

            bet_frac = confidence_kelly(
                model_prob, dec_odds, base_kelly, multiplier,
                max_bet_fraction=RISK.max_single_bet,
            )

            if bet_frac <= 0:
                continue

            # CLV calculation (only for bets placed at opening line)
            clv = 0.0
            if line_type == "open":
                try:
                    open_implied = 1.0 / american_to_decimal(open_ml_side)
                    close_implied = 1.0 / american_to_decimal(close_ml_side)
                    clv = open_implied - close_implied
                except (ValueError, ZeroDivisionError):
                    pass

            records.append({
                "game_idx": idx,
                "game_date": game["game_date"],
                "season": int(game["season"]),
                "side": side,
                "bet_frac": bet_frac,
                "dec_odds": dec_odds,
                "clv": clv,
                "home_win": int(game["home_win"]),
            })

    return pd.DataFrame(records)


def _run_bankroll_sim(
    candidates: pd.DataFrame,
    home_win_override: np.ndarray | None = None,
    games_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Execute the bankroll simulation given pre-computed bet candidates.

    If home_win_override is provided (indexed by game_idx), use those labels
    instead of the ones stored in candidates. This is used by the permutation test.
    """
    if len(candidates) == 0:
        return {
            "total_pnl": 0.0, "roi": 0.0, "compound_roi": 0.0,
            "win_rate": 0.0, "total_bets": 0,
            "max_drawdown": 0.0, "sharpe": 0.0, "avg_clv": 0.0,
            "final_bankroll": STARTING_BANKROLL, "total_wagered": 0.0,
            "stopped": False, "season_breakdown": [],
        }

    bankroll = STARTING_BANKROLL
    peak_bankroll = bankroll
    max_drawdown = 0.0
    total_wagered = 0.0
    total_pnl = 0.0
    wins = 0
    bets_placed = 0
    clv_sum = 0.0
    season_results: dict[int, dict] = {}
    stopped = False

    # Pre-sort and group by date
    candidates_sorted = candidates.sort_values(["game_date", "game_idx"]).reset_index(drop=True)

    # Convert to numpy for speed
    game_dates = candidates_sorted["game_date"].values
    seasons = candidates_sorted["season"].values
    bet_fracs = candidates_sorted["bet_frac"].values.copy()
    dec_odds_arr = candidates_sorted["dec_odds"].values
    clv_arr = candidates_sorted["clv"].values
    sides = candidates_sorted["side"].values
    game_idxs = candidates_sorted["game_idx"].values

    if home_win_override is not None:
        hw_arr = home_win_override[game_idxs]
    else:
        hw_arr = candidates_sorted["home_win"].values

    # Determine won for each bet
    won_arr = np.where(sides == "home", hw_arr == 1, hw_arr == 0)

    # Group by date boundaries
    date_changes = np.where(game_dates[:-1] != game_dates[1:])[0] + 1
    day_starts = np.concatenate([[0], date_changes])
    day_ends = np.concatenate([date_changes, [len(candidates_sorted)]])

    for d_start, d_end in zip(day_starts, day_ends):
        if stopped:
            break

        # Check stop-loss
        if bankroll < STARTING_BANKROLL * RISK.stop_loss_threshold:
            stopped = True
            break

        season = int(seasons[d_start])
        if season not in season_results:
            season_results[season] = {
                "start_bankroll": bankroll,
                "n_bets": 0, "wins": 0, "wagered": 0.0, "pnl": 0.0,
            }

        # Apply daily exposure cap
        day_fracs = bet_fracs[d_start:d_end].copy()
        total_exposure = day_fracs.sum()
        if total_exposure > RISK.max_daily_exposure:
            day_fracs *= RISK.max_daily_exposure / total_exposure

        # Execute bets for this day
        for i in range(d_start, d_end):
            frac = day_fracs[i - d_start]
            bet_amount = frac * bankroll
            total_wagered += bet_amount
            bets_placed += 1
            szn = int(seasons[i])
            if szn not in season_results:
                season_results[szn] = {
                    "start_bankroll": bankroll,
                    "n_bets": 0, "wins": 0, "wagered": 0.0, "pnl": 0.0,
                }
            season_results[szn]["n_bets"] += 1
            season_results[szn]["wagered"] += bet_amount

            if won_arr[i]:
                profit = bet_amount * (dec_odds_arr[i] - 1)
                bankroll += profit
                total_pnl += profit
                wins += 1
                season_results[szn]["wins"] += 1
                season_results[szn]["pnl"] += profit
            else:
                bankroll -= bet_amount
                total_pnl -= bet_amount
                season_results[szn]["pnl"] -= bet_amount

            clv_sum += clv_arr[i]

            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            dd = (peak_bankroll - bankroll) / peak_bankroll
            if dd > max_drawdown:
                max_drawdown = dd

    roi = total_pnl / total_wagered if total_wagered > 0 else 0.0
    compound_roi = (bankroll / STARTING_BANKROLL) - 1.0
    win_rate = wins / bets_placed if bets_placed > 0 else 0.0
    avg_clv = clv_sum / bets_placed if bets_placed > 0 else 0.0

    # Sharpe from per-season ROIs
    season_rois = []
    season_breakdown = []
    for szn in sorted(season_results.keys()):
        sr = season_results[szn]
        szn_roi = sr["pnl"] / sr["wagered"] if sr["wagered"] > 0 else 0.0
        szn_wr = sr["wins"] / sr["n_bets"] if sr["n_bets"] > 0 else 0.0
        end_br = sr["start_bankroll"] + sr["pnl"]
        season_rois.append(szn_roi)
        season_breakdown.append({
            "season": szn,
            "n_bets": sr["n_bets"],
            "win_rate": szn_wr,
            "pnl": sr["pnl"],
            "roi": szn_roi,
            "end_bankroll": end_br,
        })

    sharpe = 0.0
    if len(season_rois) > 1:
        s_std = np.std(season_rois, ddof=1)
        if s_std > 0:
            sharpe = np.mean(season_rois) / s_std

    return {
        "total_pnl": total_pnl,
        "roi": roi,
        "compound_roi": compound_roi,
        "win_rate": win_rate,
        "total_bets": bets_placed,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "avg_clv": avg_clv,
        "final_bankroll": bankroll,
        "total_wagered": total_wagered,
        "stopped": stopped,
        "season_breakdown": season_breakdown,
    }


def simulate_strategy(
    games: pd.DataFrame,
    strategy_name: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Run a single betting strategy on the full game set."""
    candidates = _precompute_bet_candidates(games, cfg)
    result = _run_bankroll_sim(candidates)
    result["strategy"] = strategy_name
    return result


def permutation_test(
    games: pd.DataFrame,
    best_strategy_name: str,
    best_cfg: dict[str, Any],
    actual_pnl: float,
    n_permutations: int = 1000,
) -> float:
    """Shuffle home_win labels within each season and re-run the best strategy.

    Optimized: pre-compute bet candidates once (bet selection doesn't change
    when labels change), then only re-run bankroll sim with shuffled outcomes.
    """
    logger.info("Running permutation test with %d iterations...", n_permutations)

    # Pre-compute candidates once
    candidates = _precompute_bet_candidates(games, best_cfg)
    if len(candidates) == 0:
        return 1.0

    # Build an array of home_win indexed by game position
    # We need to map game_idx -> home_win for shuffling
    games_sorted = games.sort_values(["game_date", "home"]).reset_index(drop=True)
    max_idx = int(candidates["game_idx"].max()) + 1
    base_hw = np.zeros(max_idx, dtype=int)
    for idx, row in games_sorted.iterrows():
        if idx < max_idx:
            base_hw[idx] = int(row["home_win"])

    # Build season membership for each game index
    season_arr = np.zeros(max_idx, dtype=int)
    for idx, row in games_sorted.iterrows():
        if idx < max_idx:
            season_arr[idx] = int(row["season"])

    unique_seasons = np.unique(season_arr[season_arr > 0])
    season_masks = {s: np.where(season_arr == s)[0] for s in unique_seasons}

    rng = np.random.RandomState(42)
    count_ge = 0

    for i in range(n_permutations):
        hw_shuffled = base_hw.copy()
        for s, idxs in season_masks.items():
            labels = hw_shuffled[idxs].copy()
            rng.shuffle(labels)
            hw_shuffled[idxs] = labels

        result = _run_bankroll_sim(candidates, home_win_override=hw_shuffled)
        if result["total_pnl"] >= actual_pnl:
            count_ge += 1

        if (i + 1) % 100 == 0:
            logger.info("Permutation %d/%d done (count_ge=%d)", i + 1, n_permutations, count_ge)

    p_value = count_ge / n_permutations
    return p_value


def era_analysis(
    games: pd.DataFrame,
    strategy_name: str,
    cfg: dict[str, Any],
) -> dict[str, dict]:
    """Run the strategy separately on 2016-2021 and 2022-2025 eras."""
    results = {}
    for era_name, seasons in [("2016-2021", range(2016, 2022)), ("2022-2025", range(2022, 2026))]:
        era_games = games[games["season"].isin(seasons)]
        if len(era_games) == 0:
            continue
        res = simulate_strategy(era_games, f"{strategy_name}_{era_name}", cfg)
        results[era_name] = res
    return results


def generate_report(
    all_results: list[dict],
    best_name: str,
    p_value: float,
    era_results: dict[str, dict],
) -> str:
    """Generate markdown report of all simulation results."""
    lines: list[str] = []
    lines.append("# MLB Retroactive Betting Simulation\n")
    lines.append(f"**Generated**: 2026-04-01")
    lines.append(f"**Starting bankroll**: ${STARTING_BANKROLL:,.0f}")
    lines.append(f"**Risk limits**: max daily {RISK.max_daily_exposure:.0%}, "
                 f"max single {RISK.max_single_bet:.0%}, "
                 f"stop-loss {RISK.stop_loss_threshold:.0%}\n")

    # -- Strategy comparison table --
    lines.append("## Strategy Comparison\n")
    lines.append("| Strategy | Bets | Win Rate | P&L | ROI | Compound ROI | Max DD | Sharpe | Avg CLV | Final Bankroll |")
    lines.append("|----------|------|---------|-----|-----|-------------|--------|--------|---------|----------------|")

    for r in all_results:
        stopped_flag = " (stopped)" if r["stopped"] else ""
        lines.append(
            f"| {r['strategy']}{stopped_flag} | {r['total_bets']} | {r['win_rate']:.3f} | "
            f"${r['total_pnl']:+,.0f} | {r['roi']:+.3f} | {r['compound_roi']:+.3f} | "
            f"{r['max_drawdown']:.3f} | {r['sharpe']:.2f} | {r['avg_clv']:+.4f} | "
            f"${r['final_bankroll']:,.0f} |"
        )
    lines.append("")

    # -- Best strategy details --
    best = next(r for r in all_results if r["strategy"] == best_name)
    lines.append(f"## Best Strategy: {best_name}\n")
    lines.append(f"Selected by highest ROI among strategies with >100 bets.\n")
    lines.append(f"- **Total P&L**: ${best['total_pnl']:+,.2f}")
    lines.append(f"- **ROI**: {best['roi']:+.4f}")
    lines.append(f"- **Compound ROI**: {best['compound_roi']:+.4f}")
    lines.append(f"- **Total bets**: {best['total_bets']}")
    lines.append(f"- **Win rate**: {best['win_rate']:.4f}")
    lines.append(f"- **Max drawdown**: {best['max_drawdown']:.4f}")
    lines.append(f"- **Sharpe ratio**: {best['sharpe']:.3f}")
    lines.append(f"- **Avg CLV**: {best['avg_clv']:+.5f}")
    lines.append(f"- **Total wagered**: ${best['total_wagered']:,.0f}")
    lines.append(f"- **Final bankroll**: ${best['final_bankroll']:,.2f}")
    lines.append("")

    # -- Per-season breakdown for best strategy --
    lines.append(f"### Per-Season Breakdown ({best_name})\n")
    lines.append("| Season | Bets | Win Rate | P&L | ROI | End Bankroll |")
    lines.append("|--------|------|---------|-----|-----|-------------|")
    for sb in best["season_breakdown"]:
        lines.append(
            f"| {sb['season']} | {sb['n_bets']} | {sb['win_rate']:.3f} | "
            f"${sb['pnl']:+,.0f} | {sb['roi']:+.3f} | ${sb['end_bankroll']:,.0f} |"
        )
    lines.append("")

    # -- Permutation test --
    lines.append("## Permutation Test\n")
    lines.append(f"Shuffled home_win labels 1000 times within each season, re-ran {best_name}.\n")
    lines.append(f"- **Actual P&L**: ${best['total_pnl']:+,.2f}")
    lines.append(f"- **p-value**: {p_value:.4f} (fraction of permutations with P&L >= actual)")
    if p_value < 0.05:
        lines.append(f"- **Interpretation**: Statistically significant at p < 0.05. Results unlikely due to chance.")
    elif p_value < 0.10:
        lines.append(f"- **Interpretation**: Marginally significant (p < 0.10). Suggestive but not conclusive.")
    else:
        lines.append(f"- **Interpretation**: Not statistically significant. Results could be due to chance.")
    lines.append("")

    # -- Era analysis --
    lines.append("## Era Analysis\n")
    lines.append(f"Separate results for {best_name} on early vs recent data.\n")
    lines.append("| Era | Bets | Win Rate | P&L | ROI | Max DD | Sharpe |")
    lines.append("|-----|------|---------|-----|-----|--------|--------|")
    for era_name, er in era_results.items():
        lines.append(
            f"| {era_name} | {er['total_bets']} | {er['win_rate']:.3f} | "
            f"${er['total_pnl']:+,.0f} | {er['roi']:+.3f} | "
            f"{er['max_drawdown']:.3f} | {er['sharpe']:.2f} |"
        )
    lines.append("")

    # -- Strategy configurations --
    lines.append("## Strategy Configurations\n")
    for name, cfg in STRATEGIES.items():
        lines.append(f"**{name}**: base_kelly={cfg['base_kelly']}, "
                     f"edge_threshold={cfg['edge_threshold']}, "
                     f"quintile_filter={cfg['quintile_filter']}, "
                     f"quintile_multipliers={cfg['quintile_multipliers']}, "
                     f"line_type={cfg['line_type']}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ensemble predictions from %s", INPUT_CSV)
    games = pd.read_csv(INPUT_CSV)
    logger.info("Loaded %d games", len(games))

    # Run all strategies
    all_results: list[dict] = []
    for name, cfg in STRATEGIES.items():
        logger.info("Running strategy: %s", name)
        result = simulate_strategy(games, name, cfg)
        all_results.append(result)
        logger.info(
            "  %s: bets=%d, PnL=$%.0f, ROI=%.4f, WR=%.3f",
            name, result["total_bets"], result["total_pnl"],
            result["roi"], result["win_rate"],
        )

    # Find best strategy (highest ROI among those with >100 bets)
    viable = [r for r in all_results if r["total_bets"] > 100]
    if not viable:
        viable = all_results  # fallback
    best = max(viable, key=lambda r: r["roi"])
    best_name = best["strategy"]
    logger.info("Best strategy: %s (ROI=%.4f)", best_name, best["roi"])

    # Permutation test on best strategy
    p_value = permutation_test(
        games, best_name, STRATEGIES[best_name], best["total_pnl"],
        n_permutations=1000,
    )
    logger.info("Permutation test p-value: %.4f", p_value)

    # Era analysis on best strategy
    era_results = era_analysis(games, best_name, STRATEGIES[best_name])

    # Generate report
    report = generate_report(all_results, best_name, p_value, era_results)
    report_path = OUTPUT_DIR / "retroactive_simulation.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Saved simulation report to %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RETROACTIVE SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Best strategy: {best_name}")
    print(f"  P&L: ${best['total_pnl']:+,.2f}")
    print(f"  ROI: {best['roi']:+.4f}")
    print(f"  Bets: {best['total_bets']}")
    print(f"  Sharpe: {best['sharpe']:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
