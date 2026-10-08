#!/usr/bin/env python3
"""
Export the topic labels as one tidy file, for whoever runs the moderation analysis.

The labels are the moderator variable in "does the real-vs-synthetic empathy gap
vary by topic". This writes them in a form that joins to the analysis dataset on
submission_id, alongside a data dictionary that carries the measurement
properties an analyst needs and cannot recover from the CSV itself.

One deliberate choice about the discipline column
-------------------------------------------------
`discipline` is the MODEL's label for every row, including the 200 posts that
were hand-coded. It is tempting to substitute the human label where we have one,
but that would make 3% of the corpus better-measured than the rest, and
differential measurement error in a moderator is worse than uniform error: it
biases the interaction estimates unevenly across topics.

The human labels ship in `discipline_gold`, flagged by `in_validation_sample`,
so the analyst can use them as a validation subset, estimate the attenuation, or
overrule this choice deliberately rather than by accident.

Usage
-----
  python export_labels.py                      # writes CSV + data dictionary
  python export_labels.py --require-complete   # refuse to export a partial run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import Counter
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")

from agreement import cohens_kappa, min_detectable_moderation
from score_handcoding import read_adjudications, read_coder


def read_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[r["submission_id"]] = r  # last write wins
    return out


def build_gold(hc_dir: Path, adj_path: Path | None) -> tuple[dict, dict, set, dict]:
    """Human gold: coder consensus, then adjudication, then rules. Mirrors score_handcoding."""
    books = sorted(hc_dir.glob("handcode_*_*.xlsx"))
    coders = {b.stem.rsplit("_", 1)[-1]: read_coder(b) for b in books}
    if not coders:
        return {}, {}, set(), {}
    names = sorted(coders)
    adjudged, contested = ({}, set())
    if adj_path and adj_path.exists():
        adjudged, contested = read_adjudications(adj_path)

    gold_disc: dict[str, str] = {}
    gold_cancer: dict[str, bool] = {}
    if len(names) >= 2:
        ca, cb = coders[names[0]], coders[names[1]]
        for i in sorted(set(ca) & set(cb)):
            da, db = ca[i]["discipline"], cb[i]["discipline"]
            if da and db and da == db:
                gold_disc[i] = da
            elif i in adjudged:
                gold_disc[i] = adjudged[i]
            ra, rb = ca[i]["cancer_relevant"], cb[i]["cancer_relevant"]
            if ra and rb and ra == rb:
                gold_cancer[i] = (ra == "yes")
    else:
        for i, r in coders[names[0]].items():
            if r["discipline"]:
                gold_disc[i] = r["discipline"]
            if r["cancer_relevant"]:
                gold_cancer[i] = (r["cancer_relevant"] == "yes")
    for i, v in adjudged.items():
        if i in gold_disc:
            gold_disc[i] = v
    return gold_disc, gold_cancer, contested, coders


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export topic labels and a data dictionary for the analyst.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--labels-dir", default="../Piloting/Round 5/output/corpora/labels")
    ap.add_argument("--handcoding-dir",
                    default="../Piloting/Round 5/output/corpora/handcoding")
    ap.add_argument("--adjudicated",
                    default="../Piloting/Round 5/output/corpora/handcoding/"
                            "adjudication_recommended.xlsx")
    ap.add_argument("--taxonomy", default="taxonomy.json")
    ap.add_argument("--model", default="gemini")
    ap.add_argument("--out-dir", default="../Piloting/Round 5/output/corpora/export")
    ap.add_argument("--n-expected", type=int, default=6600)
    ap.add_argument("--require-complete", action="store_true",
                    help="exit rather than export a partial run")
    args = ap.parse_args()

    lab = Path(args.labels_dir)
    disc = read_jsonl(lab / f"labels_{args.model}_discipline.jsonl")
    canc = read_jsonl(lab / f"labels_{args.model}_cancer.jsonl")
    if not disc:
        sys.exit(f"ERROR: no discipline labels in {lab}. Run classify_topics.py first.")

    tax = json.load(open(args.taxonomy))
    n_exp = args.n_expected
    complete = len(disc) >= n_exp and len(canc) >= n_exp
    print(f"discipline labels: {len(disc):,} of {n_exp:,}")
    print(f"cancer labels:     {len(canc):,} of {n_exp:,}")
    if not complete:
        msg = ("the run is incomplete; the export will cover only the labelled posts "
               "and the analyst will be short of rows")
        if args.require_complete:
            sys.exit(f"ERROR: {msg}")
        print(f"WARNING: {msg}")

    gold_disc, gold_cancer, contested, coders = build_gold(
        Path(args.handcoding_dir), Path(args.adjudicated) if args.adjudicated else None)
    print(f"human gold:        {len(gold_disc):,} posts "
          f"({len(contested)} contested)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "topic_labels.csv"

    ids = sorted(set(disc) | set(canc))
    cols = ["submission_id", "discipline", "cancer_relevant", "cancer_stance",
            "model_confidence", "evidence_quote_verified", "in_validation_sample",
            "discipline_gold", "cancer_relevant_gold", "discipline_contested",
            "label_model"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in ids:
            d, c = disc.get(i), canc.get(i)
            w.writerow({
                "submission_id": i,
                "discipline": (d or {}).get("label"),
                "cancer_relevant": (c or {}).get("cancer_relevant"),
                "cancer_stance": (c or {}).get("label"),
                "model_confidence": (d or {}).get("confidence"),
                "evidence_quote_verified": (d or {}).get("evidence_quote_verified"),
                "in_validation_sample": i in gold_disc,
                "discipline_gold": gold_disc.get(i, ""),
                "cancer_relevant_gold": gold_cancer.get(i, ""),
                "discipline_contested": i in contested,
                "label_model": (d or {}).get("model_id"),
            })
    print(f"\nwrote {csv_path}  ({len(ids):,} rows)")

    # --- accuracy against the gold, for the dictionary ----------------------
    both = [i for i in gold_disc if i in disc]
    acc = (sum(1 for i in both if disc[i]["label"] == gold_disc[i]) / len(both)
           if both else None)
    iaa = None
    if len(coders) >= 2:
        names = sorted(coders)
        ca, cb = coders[names[0]], coders[names[1]]
        pairs = [(ca[i]["discipline"], cb[i]["discipline"])
                 for i in set(ca) & set(cb)
                 if ca[i]["discipline"] and cb[i]["discipline"]]
        iaa = cohens_kappa(pairs)

    counts = Counter(r["label"] for r in disc.values())
    n_tot = len(disc)

    # --- data dictionary ----------------------------------------------------
    md = out_dir / "topic_labels_README.md"
    with open(md, "w") as f:
        W = f.write
        W(f"# Topic labels for the r/AskDocs real-vs-synthetic analysis\n\n")
        W(f"Generated {date.today().isoformat()} from `topic_labels.csv`. "
          f"{len(ids):,} posts.\n\n")
        W("Join to the analysis dataset on `submission_id`. Same corpus and same "
          "seed (42) as the generation run, so the ids line up exactly.\n\n")

        W("## Columns\n\n")
        W("| column | meaning |\n|---|---|\n")
        for c, m in [
            ("submission_id", "Reddit post id. Join key."),
            ("discipline", "**The moderator variable.** The model's label, for every "
                           "row, one of the 17 disciplines."),
            ("cancer_relevant", "Model's flag: does the post involve a malignancy."),
            ("cancer_stance", "current_patient / survivor / not_applicable / unclear. "
                              "Collected but DEFERRED from the analysis -- see below."),
            ("model_confidence", "Model's self-reported confidence. Near-useless: it "
                                 "is 1.0 almost everywhere. Do not filter on it."),
            ("evidence_quote_verified", "Whether the model's supporting quote was found "
                                        "verbatim in the post. A weak per-row quality flag."),
            ("in_validation_sample", "TRUE for the 200 hand-coded posts."),
            ("discipline_gold", "Human label where it exists, else blank."),
            ("cancer_relevant_gold", "Human cancer flag where it exists."),
            ("discipline_contested", "TRUE where the adjudicators recorded the post as "
                                     "genuinely ambiguous."),
            ("label_model", "Model id that produced the label."),
        ]:
            W(f"| `{c}` | {m} |\n")

        W("\n## Use `discipline` for all rows, not a mix\n\n")
        W("`discipline` is the model's label even on the 200 hand-coded posts. "
          "Substituting human labels there would make 3% of the corpus "
          "better-measured than the rest, and differential measurement error in a "
          "moderator distorts interaction estimates unevenly across topics. The human "
          "labels are in `discipline_gold` for validation, attenuation estimates, or a "
          "deliberate override.\n\n")

        W("## Measurement properties you need\n\n")
        if iaa and iaa.get("kappa") is not None:
            lo, hi = iaa["ci"]
            W(f"- **Human ceiling.** Two coders independently labelled 200 posts: raw "
              f"agreement {iaa['observed']:.3f}, Cohen's kappa {iaa['kappa']:.3f} "
              f"(95% CI {lo:.3f}-{hi:.3f}). This bounds what any classifier can be "
              f"shown to achieve.\n")
        if acc is not None:
            W(f"- **Classifier accuracy** against the adjudicated gold: {acc:.3f} "
              f"on {len(both)} posts.\n")
        W("- **So the moderator carries real misclassification.** Roughly one label in "
          "four disagrees with the human gold. Misclassification in a moderator "
          "attenuates interaction estimates toward zero, so a null topic effect is "
          "weaker evidence of no effect than it would be with a clean moderator. Worth "
          "either an attenuation correction or an explicit caveat.\n")
        W("- **`Nonmedical` and `Medical; other` are residual categories**, not topics. "
          "They collect posts that fit nothing else. Treating them as substantive "
          "levels will produce uninterpretable coefficients; consider dropping or "
          "pooling them.\n")
        W("- **`Neoplasms` follows the ICD-10 chapter**, so it covers benign neoplasms "
          "as well as malignant, and a malignancy counts only once a clinician or test "
          "has raised it. It is not the same variable as `cancer_relevant`.\n")
        W("- **`cancer_stance` is deferred.** current_patient vs survivor was dropped "
          "from the analysis as underpowered; the labels are collected so it can be "
          "revisited post hoc.\n\n")

        W("## Realized prevalence and what is testable\n\n")
        W("Minimum detectable moderation at 80% power, from the pilot-2 Explorations "
          "proportions (real 0.191, synthetic 0.073, gap 11.8 pp). A category whose "
          "figure exceeds ~5.9 pp can only detect a moderation more than half the size "
          "of the main effect.\n\n")
        W(f"| discipline | n | share | min. detectable |\n|---|---|---|---|\n")
        for lab_, n in counts.most_common():
            mde = min_detectable_moderation(n, n_tot)
            cell = f"{mde:.1f} pp" if mde else "n/a"
            W(f"| {lab_} | {n:,} | {100*n/n_tot:.1f}% | {cell} |\n")

        W("\n## Codebook rules\n\n")
        W("Applied to both the model prompt and the human adjudication, so the two are "
          "held to one rubric.\n\n")
        for i, r in enumerate(tax.get("category_rules", []), 1):
            W(f"{i}. {r}\n\n")
        W(f"Taxonomy source: {tax['source']['title']}, "
          f"{tax['source']['journal']} {tax['source']['year']};"
          f"{tax['source']['volume']}:{tax['source']['article']}, "
          f"doi:{tax['source']['doi']} ({tax['source']['table']}). "
          "Only the 17 discipline names are taken from that paper; the labelling method "
          "is ours.\n")
    print(f"wrote {md}")
    print("\nHand the analyst both files. The README carries the ceiling, the")
    print("accuracy, the prevalence table and the rules, so the CSV is not")
    print("self-explanatory on its own.")


if __name__ == "__main__":
    main()
