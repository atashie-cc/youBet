"""Retroactive P&L simulation: Opening Lines vs Closing Lines comparison.

Uses the nba_archive_10Y.json dataset which has both opening spreads
and closing moneylines for 13,903 games (2011-2021). Converts opening
spreads to approximate opening moneylines using an empirically-derived
conversion table, then re-runs the P&L simulation against both.

Key question: Can our model beat OPENING lines (same lead time) even
though it loses to CLOSING lines (which have same-day information)?

Usage:
    python betting/scripts/retroactive_opening_lines.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from youbet.core.bankroll import american_to_decimal, fractional_kelly, remove_vig

logger = logging.getLogger(__name__)
WORKFLOW_DIR = Path(__file__).resolve().parents[2]

SAMPLE_DECAY = 0.3
ES_ROUNDS = 50

# Team short names (from JSON) → abbreviations (from our features)
TEAM_SHORT_TO_ABBR = {
    "Hawks": "ATL", "Celtics": "BOS", "Nets": "BKN", "Bobcats": "CHA",
    "Hornets": "CHA", "Bulls": "CHI", "Cavaliers": "CLE", "Mavericks": "DAL",
    "Nuggets": "DEN", "Pistons": "DET", "Warriors": "GSW", "Rockets": "HOU",
    "Pacers": "IND", "Clippers": "LAC", "Lakers": "LAL", "Grizzlies": "MEM",
    "Heat": "MIA", "Bucks": "MIL", "Timberwolves": "MIN", "Pelicans": "NOP",
    "Knicks": "NYK", "Thunder": "OKC", "Magic": "ORL", "76ers": "PHI",
    "Suns": "PHX", "Blazers": "POR", "Trail Blazers": "POR", "Kings": "SAC",
    "Spurs": "SAS", "Raptors": "TOR", "Jazz": "UTA", "Wizards": "WAS",
    "SuperSonics": "OKC",
}

# Empirical spread-to-moneyline conversion (derived from closing data)
# Interpolated from the empirical table above
SPREAD_ML_TABLE = np.array([
    # (spread, favorite_ml)
    (0.0,  -105),
    (-0.5, -108),
    (-1.0, -120),
    (-1.5, -125),
    (-2.0, -134),
    (-2.5, -142),
    (-3.0, -155),
    (-3.5, -165),
    (-4.0, -180),
    (-4.5, -195),
    (-5.0, -210),
    (-5.5, -230),
    (-6.0, -250),
    (-6.5, -275),
    (-7.0, -300),
    (-7.5, -330),
    (-8.0, -360),
    (-8.5, -400),
    (-9.0, -450),
    (-9.5, -520),
    (-10.0, -600),
    (-11.0, -780),
    (-12.0, -990),
    (-13.0, -1300),
    (-14.0, -1800),
    (-15.0, -2500),
    (-17.0, -5000),
    (-20.0, -10000),
])


def spread_to_ml(home_spread: float) -> tuple[float, float]:
    """Convert point spread to approximate moneylines.

    Returns (home_ml, away_ml) in American odds format.
    """
    if pd.isna(home_spread) or abs(home_spread) > 25:
        return np.nan, np.nan

    abs_spread = abs(home_spread)
    # Interpolate from table
    fav_ml = np.interp(abs_spread, -SPREAD_ML_TABLE[:, 0], -SPREAD_ML_TABLE[:, 1])
    fav_ml = -fav_ml  # Convert back to negative

    # Underdog ML: approximate from favorite ML
    # At -150, underdog is ~+130; at -300, underdog is ~+250
    # Rough formula: underdog = |fav_ml| - vig_offset
    if fav_ml <= -200:
        dog_ml = abs(fav_ml) - 50
    elif fav_ml <= -150:
        dog_ml = abs(fav_ml) - 20
    else:
        dog_ml = abs(fav_ml) - 10

    dog_ml = max(dog_ml, 100)  # Floor at +100

    if home_spread < 0:
        # Home is favorite
        return fav_ml, dog_ml
    elif home_spread > 0:
        # Away is favorite
        return dog_ml, fav_ml
    else:
        return -105, -105  # Pick'em


def compute_sample_weights(df, decay):
    df = df.reset_index(drop=True)
    weights = np.ones(len(df))
    if decay <= 0:
        return weights
    for _, group in df.groupby("season"):
        idx = group.index
        dates = pd.to_datetime(group["GAME_DATE"])
        days = (dates - dates.min()).dt.days
        mx = days.max() if days.max() > 0 else 1
        weights[idx] = np.exp(-decay * (1 - days / mx))
    return weights


def build_model(arch):
    if arch == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            max_depth=3, learning_rate=0.08, n_estimators=750,
            min_child_weight=3, subsample=0.7, colsample_bytree=1.0,
            objective="binary:logistic", eval_metric="logloss",
            reg_alpha=0.1, reg_lambda=1.0, random_state=42)
    elif arch == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])


ENSEMBLE = [
    ("xgboost", [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "rest_days_home", "rest_days_away", "is_playoff",
    ]),
    ("logistic", [
        "diff_elo", "diff_win_pct", "diff_win_pct_last10", "diff_scoring_margin",
        "diff_fg_pct", "diff_fg3_pct", "diff_three_pt_rate", "diff_ft_rate",
        "diff_oreb_rate", "diff_tov_rate", "diff_ast_rate",
        "is_b2b_home", "is_b2b_away", "is_3in4_home",
        "consecutive_away_away", "tz_change_home", "altitude_diff",
        "rest_days_away", "is_playoff",
    ]),
]


def simulate(games_df, line_type, kelly_frac=0.25, edge_threshold=0.0,
             bankroll_start=10000.0):
    """Simulate betting on a set of games."""
    ml_home_col = f"{line_type}_home_ml"
    ml_away_col = f"{line_type}_away_ml"

    bankroll = bankroll_start
    bets = 0
    won = 0
    wagered = 0.0
    season_results = []
    peak = bankroll
    max_dd = 0.0

    for season in sorted(games_df["season"].unique()):
        sdata = games_df[games_df["season"] == season]
        s_start = bankroll
        s_bets = 0
        s_won = 0

        for _, g in sdata.iterrows():
            h_ml = g[ml_home_col]
            a_ml = g[ml_away_col]
            if pd.isna(h_ml) or pd.isna(a_ml) or abs(h_ml) < 100 or abs(a_ml) < 100:
                continue

            h_dec = american_to_decimal(h_ml)
            a_dec = american_to_decimal(a_ml)
            h_vf, a_vf, overround = remove_vig(h_ml, a_ml)
            vig = overround / 2

            for side in ["home", "away"]:
                prob = g["model_home_prob"] if side == "home" else 1 - g["model_home_prob"]
                vf = h_vf if side == "home" else a_vf
                dec = h_dec if side == "home" else a_dec
                edge = prob - vf

                if edge <= vig + edge_threshold:
                    continue

                bet_frac = min(fractional_kelly(prob, dec, kelly_frac), 0.10)
                if bet_frac <= 0:
                    continue

                bet_amt = bankroll * bet_frac
                actual_won = (g["actual_home_won"] == 1) if side == "home" else (g["actual_home_won"] == 0)

                bets += 1
                s_bets += 1
                wagered += bet_amt
                if actual_won:
                    bankroll += bet_amt * (dec - 1)
                    won += 1
                    s_won += 1
                else:
                    bankroll -= bet_amt

                if bankroll > peak:
                    peak = bankroll
                dd = (peak - bankroll) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

                if bankroll <= 0:
                    bankroll = 0
                    break
            if bankroll <= 0:
                break

        s_pnl = bankroll - s_start
        season_results.append({
            "season": season, "pnl": s_pnl,
            "roi": s_pnl / s_start if s_start > 0 else 0,
            "n_bets": s_bets, "wr": s_won / s_bets if s_bets > 0 else 0,
        })

    return {
        "final": bankroll, "pnl": bankroll - bankroll_start,
        "roi": (bankroll - bankroll_start) / bankroll_start,
        "bets": bets, "won": won,
        "wr": won / bets if bets > 0 else 0,
        "wagered": wagered, "max_dd": max_dd,
        "seasons": season_results,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load features
    feat_df = pd.read_csv(WORKFLOW_DIR / "data" / "processed" / "matchup_features.csv")

    # Load JSON with opening spreads + closing MLs
    with open(WORKFLOW_DIR / "betting" / "data" / "lines" / "nba_archive_10Y.json") as f:
        raw = json.load(f)
    lines = pd.DataFrame(raw)

    # Map team names
    lines["home_abbr"] = lines["home_team"].map(TEAM_SHORT_TO_ABBR)
    lines["away_abbr"] = lines["away_team"].map(TEAM_SHORT_TO_ABBR)

    # Convert date
    lines["date_str"] = pd.to_datetime(lines["date"].astype(int).astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")

    # Convert opening spreads to approximate opening MLs
    open_mls = lines["home_open_spread"].apply(lambda s: spread_to_ml(s))
    lines["open_home_ml"] = [m[0] for m in open_mls]
    lines["open_away_ml"] = [m[1] for m in open_mls]

    # Rename closing MLs
    lines["close_home_ml"] = pd.to_numeric(lines["home_close_ml"], errors="coerce")
    lines["close_away_ml"] = pd.to_numeric(lines["away_close_ml"], errors="coerce")

    # Home won
    lines["home_final"] = pd.to_numeric(lines["home_final"], errors="coerce")
    lines["away_final"] = pd.to_numeric(lines["away_final"], errors="coerce")

    unmapped = lines[lines["home_abbr"].isna()]["home_team"].unique()
    if len(unmapped) > 0:
        logger.warning("Unmapped: %s", unmapped)
    lines = lines.dropna(subset=["home_abbr", "away_abbr", "open_home_ml", "close_home_ml"])

    logger.info("Lines: %d games with opening + closing", len(lines))

    # Match with features
    feat_df["date_str"] = pd.to_datetime(feat_df["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    merged = feat_df.merge(
        lines[["date_str", "home_abbr", "away_abbr",
               "open_home_ml", "open_away_ml", "close_home_ml", "close_away_ml",
               "home_final", "away_final"]],
        left_on=["date_str", "home_team", "away_team"],
        right_on=["date_str", "home_abbr", "away_abbr"],
        how="inner",
    )
    merged["actual_home_won"] = (merged["home_final"] > merged["away_final"]).astype(int)
    logger.info("Matched: %d games", len(merged))

    # Walk-forward ensemble predictions
    all_feats = set()
    for _, feats in ENSEMBLE:
        all_feats.update(feats)

    all_seasons = sorted(merged["season"].unique())
    test_seasons = [s for s in all_seasons if s >= 2012]

    logger.info("Walk-forward: %d test seasons", len(test_seasons))

    game_results = []
    t0 = time.time()

    for ts in test_seasons:
        train = merged[merged["season"] < ts].dropna(subset=list(all_feats)).reset_index(drop=True)
        test = merged[merged["season"] == ts].dropna(subset=list(all_feats)).reset_index(drop=True)
        if len(train) < 1000 or len(test) == 0:
            continue

        weights = compute_sample_weights(train, SAMPLE_DECAY)
        preds_list = []
        for arch, feats in ENSEMBLE:
            model = build_model(arch)
            if arch == "xgboost":
                n_val = max(1, int(len(train) * 0.1))
                tr, va = train.head(len(train) - n_val), train.tail(n_val).reset_index(drop=True)
                model.set_params(early_stopping_rounds=ES_ROUNDS)
                model.fit(tr[feats], tr["home_win"], sample_weight=weights[:len(tr)],
                          eval_set=[(va[feats], va["home_win"])], verbose=False)
            else:
                model.fit(train[feats], train["home_win"])
            preds_list.append(model.predict_proba(test[feats])[:, 1])

        ens_pred = np.mean(preds_list, axis=0)
        ens_std = np.std(preds_list, axis=0)

        for idx in range(len(test)):
            r = test.iloc[idx]
            game_results.append({
                "season": int(r["season"]),
                "date": r["date_str"],
                "model_home_prob": ens_pred[idx],
                "ensemble_std": ens_std[idx],
                "open_home_ml": r["open_home_ml"],
                "open_away_ml": r["open_away_ml"],
                "close_home_ml": r["close_home_ml"],
                "close_away_ml": r["close_away_ml"],
                "actual_home_won": int(r["actual_home_won"]),
            })

        logger.info("  Season %d: %d games", ts, len(test))

    games = pd.DataFrame(game_results)
    elapsed = time.time() - t0
    logger.info("Predictions complete: %d games in %.1f min", len(games), elapsed / 60)

    # Log loss comparison: model vs opening vs closing
    model_ll = log_loss(games["actual_home_won"], games["model_home_prob"].clip(0.01, 0.99))

    # Opening line implied probs
    valid_open = games[games["open_home_ml"].notna() & (games["open_home_ml"].abs() >= 100)]
    open_probs = valid_open.apply(
        lambda r: remove_vig(r["open_home_ml"], r["open_away_ml"])[0], axis=1)
    open_ll = log_loss(valid_open["actual_home_won"], open_probs.clip(0.01, 0.99))

    valid_close = games[games["close_home_ml"].notna() & (games["close_home_ml"].abs() >= 100)]
    close_probs = valid_close.apply(
        lambda r: remove_vig(r["close_home_ml"], r["close_away_ml"])[0], axis=1)
    close_ll = log_loss(valid_close["actual_home_won"], close_probs.clip(0.01, 0.99))

    print("\n" + "=" * 100)
    print("OPENING vs CLOSING LINE COMPARISON")
    print(f"Games: {len(games)} | Seasons: {test_seasons[0]}-{test_seasons[-1]}")
    print("=" * 100)

    print(f"\n  Model LL:   {model_ll:.4f}")
    print(f"  Opening LL: {open_ll:.4f} (approx from spread conversion)")
    print(f"  Closing LL: {close_ll:.4f}")
    print(f"  Model vs Opening: {model_ll - open_ll:+.4f}")
    print(f"  Model vs Closing: {model_ll - close_ll:+.4f}")
    print(f"  Opening vs Closing: {open_ll - close_ll:+.4f} (info gained during day)")

    # Simulate strategies against BOTH line types
    strats = {
        "flat_kelly_025": {"kelly_frac": 0.25, "edge_threshold": 0.0},
        "edge_gt_3pct": {"kelly_frac": 0.25, "edge_threshold": 0.03},
        "edge_gt_5pct": {"kelly_frac": 0.25, "edge_threshold": 0.05},
    }

    print(f"\n--- P&L Simulation: Opening Lines vs Closing Lines ---")
    print(f"{'Strategy':<20} {'Line':>8} {'Final $':>10} {'P&L':>10} {'ROI':>8} "
          f"{'Bets':>6} {'WR':>6} {'MaxDD':>7}")
    print("-" * 85)

    for sname, sparams in strats.items():
        for line_type in ["open", "close"]:
            r = simulate(games, line_type, **sparams)
            print(f"{sname:<20} {line_type:>8} {r['final']:>10,.0f} {r['pnl']:>+10,.0f} "
                  f"{r['roi']:>7.1%} {r['bets']:>6} {r['wr']:>5.1%} {r['max_dd']:>6.1%}")

    # Per-season detail for best strategy on opening lines
    print(f"\n--- Per-Season: flat_kelly_025 on Opening Lines ---")
    r = simulate(games, "open", kelly_frac=0.25, edge_threshold=0.0)
    print(f"{'Season':>8} {'P&L':>10} {'ROI':>8} {'Bets':>6} {'WR':>6}")
    print("-" * 42)
    for s in r["seasons"]:
        print(f"{s['season']:>8} {s['pnl']:>+10,.0f} {s['roi']:>7.1%} {s['n_bets']:>6} {s['wr']:>5.1%}")

    print("=" * 100)

    # Save
    reports_dir = WORKFLOW_DIR / "betting" / "output" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    games.to_csv(reports_dir / "retroactive_opening_lines_games.csv", index=False)
    out = {
        "n_games": len(games),
        "model_ll": model_ll, "opening_ll": open_ll, "closing_ll": close_ll,
        "model_vs_opening": model_ll - open_ll,
        "model_vs_closing": model_ll - close_ll,
    }
    (reports_dir / "retroactive_opening_lines.json").write_text(json.dumps(out, indent=2, default=str))
    logger.info("Saved")


if __name__ == "__main__":
    main()
