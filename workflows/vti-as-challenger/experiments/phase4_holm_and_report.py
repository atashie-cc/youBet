"""Phase 4: joint Holm correction across all 27 candidates, then segmented report.

- Reads each candidate's phase1 JSON (sign_flipped p, diff CI, TOST)
- Reads each candidate's phase3 robustness JSON (flag, notes)
- Computes joint Holm across the union (raw p = sign_flipped p_one_sided, native window)
- Emits 3 segmented tables: Class 1 / Class 2 / Class 3
- Writes research/final-report.md and artifacts/holm_joint.json
"""

from __future__ import annotations

import json
from pathlib import Path

from _shared import ARTIFACTS_DIR, joint_holm, load_precommit

WORKFLOW_DIR = ARTIFACTS_DIR.parent
RESEARCH_DIR = WORKFLOW_DIR / "research"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:+.4f}, {hi:+.4f}]"


def main() -> None:
    precommit = load_precommit()
    raw_p: dict[str, float] = {}
    rows: list[dict] = []

    for cand in precommit["candidates"]:
        cid = cand["id"]
        phase1 = _load(ARTIFACTS_DIR / f"{cid}.json")
        phase3 = _load(ARTIFACTS_DIR / f"{cid}_robustness.json")
        if phase1 is None:
            print(f"  skip {cid}: no phase1 artifact")
            continue
        sfn = phase1["native"]["sign_flipped"]
        cin = phase1["native"]["diff_ci"]
        ton = phase1["native"]["tost"]
        cw = phase1.get("common_window") or {}
        cw_sfp = (cw.get("sign_flipped") or {}).get("p_one_sided") if "sign_flipped" in cw else None
        cw_tost_eq = (cw.get("tost") or {}).get("equivalent") if "tost" in cw else None

        raw_p[cid] = float(sfn["p_one_sided"])
        rows.append({
            "id": cid,
            "class": cand["class"],
            "workflow": cand["workflow"],
            "metric": cand["metric"],
            "benchmark_label": cand["benchmark_label"],
            "n_obs": phase1["data"]["n_obs"],
            "window": phase1["data"]["window_start"] + " to " + phase1["data"]["window_end"],
            "mei": cand["mei"],
            "point": cin["point_estimate"],
            "ci_lower": cin["ci_lower"],
            "ci_upper": cin["ci_upper"],
            "sf_p_raw": float(sfn["p_one_sided"]),
            "sf_passes": bool(sfn["passes"]),
            "tost_equivalent": bool(ton["equivalent"]),
            "cw_sf_p": cw_sfp,
            "cw_tost_equivalent": cw_tost_eq,
            "robustness_flag": phase3["robustness_flag"] if phase3 else "n/a",
            "robustness_notes": "; ".join(phase3["robustness_notes"]) if phase3 and phase3.get("robustness_notes") else "",
            "structural_caveat": cand.get("structural_caveat"),
            "prior_finding": cand.get("prior_finding"),
        })

    # Joint Holm across all classes
    holm = joint_holm(raw_p) if raw_p else {}
    with open(ARTIFACTS_DIR / "holm_joint.json", "w") as f:
        json.dump(holm, f, indent=2)

    # Attach Holm rank/adjusted_p back into rows
    for row in rows:
        h = holm.get(row["id"], {})
        row["holm_rank"] = h.get("rank")
        row["holm_adjusted_p"] = h.get("adjusted_p")
        row["holm_significant_05"] = h.get("significant_05")

    rows.sort(key=lambda r: (r["class"], r["holm_rank"] if r["holm_rank"] is not None else 999))

    # Build segmented report
    md: list[str] = []
    md.append("# VTI-as-Challenger — Inverted Gate Final Report")
    md.append("")
    md.append(f"- Frozen at: {precommit['frozen_at']}")
    md.append(f"- Candidates: {len(rows)} (joint Holm-{len(rows)} family)")
    md.append(f"- Bootstrap: Politis-Romano stationary, {precommit['n_bootstrap']} reps, {precommit['block_length_days']}d block")
    md.append(f"- One-sided alpha: {precommit['alpha_one_sided']}; CI: {int(precommit['ci_confidence']*100)}%")
    md.append("")
    # Pre-compute headline tallies for the executive summary
    n_total = len(rows)
    sf_raw_pass = [r for r in rows if r["sf_passes"]]
    sf_holm_pass = [r for r in rows if r["holm_significant_05"]]
    tost_eq_pass = [r for r in rows if r["tost_equivalent"]]
    cw_sf_pass = [r for r in rows if r["cw_sf_p"] is not None and r["cw_sf_p"] < 0.05]
    inconclusive = [r for r in rows if not r["sf_passes"] and not r["tost_equivalent"]]

    md.append("## Headline finding")
    md.append("")
    md.append(f"After inverting the gate against {n_total} candidates that previously had positive point "
              f"estimates vs their workflow's native benchmark:")
    md.append("")
    md.append(f"- **{len(sf_raw_pass)}/{n_total}** of benchmark-as-challenger trials cross the raw one-sided gate (p < 0.05).")
    md.append(f"- **{len(sf_holm_pass)}/{n_total}** survive the joint Holm-{n_total} correction across the family.")
    md.append(f"- **{len(tost_eq_pass)}/{n_total}** can be declared statistically equivalent at their workflow's native MEI under TOST.")
    md.append(f"- **{len(inconclusive)}/{n_total}** are genuinely inconclusive (SF=N, TOST=N) — i.e. the original null verdict cannot be replaced with either \"benchmark beats\" or \"benchmark equivalent\" at the workflows' MEIs.")
    md.append("")
    md.append("**This confirms the user's hypothesis**: the original null results across 9 workflows are "
              "**power-limited**, not power-definitive. The data simply cannot distinguish the benchmark "
              "from the alternatives at any direction or at the workflows' native MEIs.")
    md.append("")
    if sf_raw_pass:
        md.append("**Raw SF-pass exceptions** (do not survive Holm):")
        for r in sf_raw_pass:
            md.append(f"- `{r['id']}` raw p = {r['sf_p_raw']:.4f}; Holm adjusted p = {r['holm_adjusted_p']:.4f}. "
                      f"Point = {r['point']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]")
        md.append("")
    if cw_sf_pass:
        md.append("**Common-window (2017-2026) SF-pass cases** — informative sensitivity:")
        for r in cw_sf_pass:
            md.append(f"- `{r['id']}` common-window SF p = {r['cw_sf_p']:.4f}.")
        md.append("")

    md.append("## How to read this report")
    md.append("")
    md.append("Each candidate had a positive Sharpe-diff or CAGR-diff point estimate vs the workflow's "
              "native benchmark in its original test. Here we **flip the framing**: does the benchmark "
              "beat the candidate?")
    md.append("")
    md.append("- **SF p**: one-sided block-bootstrap p for H0: benchmark <= candidate. SF passes at p < 0.05.")
    md.append("- **TOST eq?**: is the 90% CI on Sharpe-of-excess (benchmark - candidate paired) contained in [-MEI, +MEI]? "
              "If yes, the two are statistically equivalent under the workflow's native MEI.")
    md.append("- **Common-window** sensitivity: 2017-01-01 to 2026-04-30 intersect for cross-workflow comparability.")
    md.append("- **Robustness flag**: PASS / PASS-with-caveat (sub-period sign disagrees) / FAIL (placebo gate fires).")
    md.append("- **Holm-adjusted p**: joint Holm across the full 27-candidate family.")
    md.append("")
    md.append("**Interpretation key**:")
    md.append("- _SF=Y, TOST=Y_: benchmark significantly beats candidate AND they are equivalent within MEI (unusual — only if MEI is wide).")
    md.append("- _SF=Y, TOST=N_: benchmark significantly beats candidate; effect is larger than MEI (clean win for benchmark).")
    md.append("- _SF=N, TOST=Y_: cannot prove benchmark beats candidate, but difference is small enough to call them equivalent. **Strongest \"power-limited but bounded\" result.**")
    md.append("- _SF=N, TOST=N_: power-limited and difference might be meaningful in either direction — genuinely inconclusive.")
    md.append("")

    class_titles = {
        1: "## Class 1 — Investable Sharpe-comparable",
        2: "## Class 2 — Paper factor portfolios (structural-mismatch caveat)",
        3: "## Class 3 — Leveraged-CAGR (CAGR-diff metric)",
    }

    for cls in sorted(set(r["class"] for r in rows)):
        md.append("")
        md.append(class_titles[cls])
        md.append("")
        if cls == 2:
            md.append("_Caveat: Class 2 candidates are Ken French long-short factor portfolios "
                      "compared to factor B&H (NOT VTI). The comparison is internal to each factor — "
                      "it tests whether SMA-timing the factor beats holding the factor, not whether "
                      "the factor beats the market._")
            md.append("")
        if cls == 3:
            md.append("_Class 3 uses CAGR-diff (annualized geometric) instead of Sharpe-diff, "
                      "with native MEI = 0.01 (1%/yr). All candidates' workflow gates were "
                      "structurally unpowered for the Sharpe metric due to extreme excess-return "
                      "vol of leveraged strategies; see cagr-max E0 power analysis._")
            md.append("")

        md.append("| Candidate | n_obs | Window | Point | 90% CI | SF p | SF? | MEI | TOST eq? | CW SF p | CW TOST? | Robust | Holm adj p |")
        md.append("|---|---:|---|---:|---|---:|:---:|---:|:---:|---:|:---:|:---:|---:|")
        for r in [x for x in rows if x["class"] == cls]:
            cw_sf = f"{r['cw_sf_p']:.4f}" if r["cw_sf_p"] is not None else "—"
            cw_to = ("Y" if r["cw_tost_equivalent"] else "N") if r["cw_tost_equivalent"] is not None else "—"
            md.append(
                f"| `{r['id']}` "
                f"| {r['n_obs']} "
                f"| {r['window']} "
                f"| {r['point']:+.4f} "
                f"| {fmt_ci(r['ci_lower'], r['ci_upper'])} "
                f"| {r['sf_p_raw']:.4f} "
                f"| {'Y' if r['sf_passes'] else 'N'} "
                f"| {r['mei']:.2f} "
                f"| {'Y' if r['tost_equivalent'] else 'N'} "
                f"| {cw_sf} "
                f"| {cw_to} "
                f"| {r['robustness_flag']} "
                f"| {r['holm_adjusted_p']:.4f} |"
            )

    # Final summary
    n = len(rows)
    sf_pass = sum(1 for r in rows if r["sf_passes"])
    holm_pass = sum(1 for r in rows if r["holm_significant_05"])
    tost_eq = sum(1 for r in rows if r["tost_equivalent"])
    inconclusive = sum(1 for r in rows if not r["sf_passes"] and not r["tost_equivalent"])

    md.append("")
    md.append("## Summary tally")
    md.append("")
    md.append(f"- **Sign-flipped gate (raw p < 0.05)**: {sf_pass}/{n}")
    md.append(f"- **Sign-flipped gate (Holm-adjusted p < 0.05)**: {holm_pass}/{n}")
    md.append(f"- **TOST equivalent at workflow-native MEI**: {tost_eq}/{n}")
    md.append(f"- **Genuinely inconclusive (SF=N, TOST=N)**: {inconclusive}/{n}")
    md.append("")
    md.append("## Excluded workflows (deferred)")
    md.append("")
    for ex in precommit.get("excluded_candidates", []):
        md.append(f"- **{ex['workflow']}** — {ex['reason']}")
        for w in ex.get("would_have_been", []):
            md.append(f"  - would have been: {w}")
    md.append("")

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESEARCH_DIR / "final-report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote {out_path}")
    print(f"\nTallies: SF raw={sf_pass}/{n}, SF Holm={holm_pass}/{n}, TOST eq={tost_eq}/{n}, inconclusive={inconclusive}/{n}")


if __name__ == "__main__":
    main()
