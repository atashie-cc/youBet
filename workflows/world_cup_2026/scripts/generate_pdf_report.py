"""Generate a concise PDF experiment summary for the WC 2026 bracket workflow."""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKFLOW_DIR / "output"


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, "2026 FIFA World Cup Bracket Prediction - Experiment Summary", align="C", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.ln(3)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def subsection(self, title: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")

    def body(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None):
        if col_widths is None:
            col_widths = [int(190 / len(headers))] * len(headers)
        self.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 5, h, border=1, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        for row in rows:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 5, str(cell), border=1, align="C")
            self.ln()
        self.ln(2)


def main():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2026 FIFA World Cup", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, "Bracket Prediction Experiments", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "youBet Framework | Phases 0-6 | 10 Codex Adversarial Reviews", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Executive Summary
    pdf.section("Executive Summary")
    pdf.body(
        "This report summarizes the experiments conducted to build a Monte Carlo bracket simulator "
        "for the 2026 FIFA World Cup (48-team format). The workflow progressed through 7 phases "
        "(0-6), with 10 adversarial Codex review rounds that identified and fixed 15 blockers. "
        "The final assessment: the model captures ~60% of the signal between random and market on "
        "World Cup match prediction. Its genuine value is structural bracket-path simulation and "
        "pool-play strategy optimization, not beating bookmaker odds."
    )

    # Key Results
    pdf.section("Key Results")
    pdf.subsection("Champion Probabilities (Phase 6, 10K Monte Carlo sims)")
    pdf.table(
        ["Rank", "Team", "Balanced", "Contrarian", "Market"],
        [
            ["1", "Spain", "22.8%", "13.3%", "18-20%"],
            ["2", "Argentina", "20.9%", "12.1%", "~11%"],
            ["3", "France", "12.1%", "7.5%", "13-15%"],
            ["4", "England", "7.8%", "5.2%", "13-15%"],
            ["5", "Brazil", "7.2%", "5.7%", "10-12%"],
            ["6", "Colombia", "6.0%", "5.1%", "~3%"],
            ["7", "Portugal", "5.8%", "4.6%", "~5%"],
            ["8", "Morocco", "1.4%", "2.7%", "~1.5%"],
        ],
        [12, 40, 28, 28, 30],
    )

    pdf.subsection("Per-Match Backtest (192 WC matches, walk-forward)")
    pdf.table(
        ["WC", "N", "Elo LL", "XGB LL", "Pinnacle"],
        [
            ["2014", "64", "0.957", "0.938", "N/A"],
            ["2018", "64", "1.000", "0.990", "N/A"],
            ["2022", "64", "1.041", "1.068", "~0.992"],
            ["Agg", "192", "0.995", "0.999", "-"],
        ],
        [20, 20, 40, 40, 40],
    )
    pdf.body("Random baseline: 1.099. XGBoost adds +0.004 LL (slightly harmful) on top of Elo.")

    # Market Efficiency
    pdf.section("Market Efficiency Finding")
    pdf.body(
        "Market outright odds (27 bookmakers, WC 2022) beat our Elo by 0.028 LL on 64 matches. "
        "Blending Elo with market signal degrades performance monotonically - pure market is best, "
        "pure Elo is worst. This is consistent with the cross-sport pattern: the market wins in "
        "NBA (+0.030 LL gap), MMA (+0.048), and now WC soccer (~0.05)."
    )
    pdf.table(
        ["Sport", "Model LL", "Market LL", "Gap"],
        [
            ["NBA", "0.622", "0.592", "+0.030"],
            ["MLB", "0.677", "0.678", "-0.001"],
            ["MMA", "0.661", "0.613", "+0.048"],
            ["WC Soccer", "~1.041", "~0.992", "~+0.049"],
        ],
        [40, 40, 40, 40],
    )

    # Experiment Journey
    pdf.section("Experiment Journey (Phases 0-6)")
    pdf.table(
        ["Phase", "Goal", "Key Finding"],
        [
            ["0", "Data feasibility", "49K matches; StatsBomb+Wikipedia confirmed"],
            ["A", "Core infra", "Extended experiment.py for 3-class targets"],
            ["1 v2", "Elo + features", "K=18 tuned; 193-tournament taxonomy"],
            ["2 v2", "3-way XGBoost", "LL 1.008; XGB beats Elo by 0.015"],
            ["3 v2", "Knockout model", "PK tail +0 LL; all models same argmax"],
            ["4 v2", "Bracket sim", "Real FIFA bracket; H2H tiebreakers"],
            ["6A", "Elo recalib", "Morocco #3->#11; Brazil #10->#4"],
            ["6B", "Squad quality", "Partially corrected; Track B unvalidated"],
        ],
        [14, 30, 100],
    )

    # Codex Reviews
    pdf.section("Codex Adversarial Review Impact")
    pdf.body(
        "10 Codex review rounds found 15 blockers across all phases. Key catches: Phase 1 "
        "taxonomy was 18/193 mapped (systematic bias); Phase 2 training window was misclaimed; "
        "Phase 3 acceptance test drifted from plan; Phase 4 R32 bracket was fabricated; Phase 6 "
        "blend experiment had a team-name bug producing inverted results. Each fix materially "
        "improved the output."
    )

    # Phase 6 Ablation
    pdf.section("Phase 6: Elo Recalibration Ablation")
    pdf.body(
        "A 4-way ablation (v1/v2 taxonomy x old/tuned hyperparameters) showed Morocco was already "
        "Elo #2 in the v1 baseline -- the bias was not caused by the taxonomy fix. The root cause "
        "is structural: Elo cannot distinguish opponent quality across confederations. The Phase 6 "
        "recalibration (eloratings-inspired K schedule + no mean reversion) was the most impactful "
        "single change, validated on WC 2010 held-out (LL 0.9812 vs old 1.0049)."
    )
    pdf.table(
        ["Config", "Morocco Rank", "Brazil Rank", "Mor-Bra Gap"],
        [
            ["v1+old (K=12,mr=0.80)", "#2", "#17", "+55.1"],
            ["v2+old (K=12,mr=0.80)", "#1", "#16", "+63.9"],
            ["v1+tuned (K=18,mr=0.90)", "#3", "#9", "+40.4"],
            ["v2+tuned [PROD]", "#3", "#10", "+51.3"],
            ["Phase 6 (K=20x,mr=1.0)", "#11", "#4", "-"],
        ],
        [55, 35, 35, 35],
    )

    # Strategic Recommendation
    pdf.section("Strategic Recommendation")
    pdf.body(
        "Do not invest further in feature engineering for match prediction. The ceiling is defined "
        "by the market (~0.05 LL gap). The model's genuine value is in two areas bookmakers don't "
        "publish: (1) structural bracket-path simulation (groups, tiebreakers, best-thirds, knockout "
        "tree) and (2) pool-play strategy optimization (contrarian differentiation by pool size).\n\n"
        "Recommended approach for 2026: when pre-tournament outright odds are available (May-June "
        "2026), anchor team strengths to market odds, use the bracket simulator for structural path "
        "effects, and optimize the upset-boost parameter for your specific pool size."
    )

    # Output
    out = OUTPUT_DIR / "wc2026_experiment_summary.pdf"
    pdf.output(str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
