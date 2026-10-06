#!/usr/bin/env python3
"""
Join the two cancer labellers, report agreement, and queue the rest for hand-coding.

Third stage of the plan in Piloting/Round 5/topic_classification_plan.md:
"Agreement gives a confidence value. Disagreements and low-confidence cases go to
manual coding. We can report kappa to quantify agreement."

Inputs (from the two previous stages, all optional except labeller A's cancer file):
  labels_<model>_cancer.jsonl       labeller A, the LLM
  labels_scispacy_cancer.jsonl      labeller B, scispaCy entity linking
  labels_<model>_discipline.jsonl   for the prevalence and power table

Outputs:
  cancer_consensus.jsonl            per-OP joined label + confidence tier
  manual_coding_queue.csv           what a human has to read, highest priority first
  (a report on stdout)

Two kappas, and they are not equally meaningful
-----------------------------------------------
  cancer_relevant   Both labellers decide this independently and by different
                    means (LLM judgement vs UMLS semantic type). This kappa is
                    the real measure of agreement.

  stance            Labeller B has no way to read current-patient vs survivor; it
                    scores surface cues. The stance kappa is reported because the
                    plan asks for it, but it is a floor on agreement, not a
                    validation of the stance labels. Validating stance needs the
                    hand-coding step, which is not implemented.

Usage
-----
  python agreement.py
  python agreement.py --labels-dir "../Piloting/Round 5/output/corpora/labels"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

STANCES = ("current_patient", "survivor", "not_applicable", "unclear")

# --- Power model, calibrated against the plan's own table ------------------
#
# The plan gives the smallest detectable moderation at 80% power for five
# prevalences, from the pilot-2 Explorations proportions (real 0.191, synthetic
# 0.073). Treating a topic contrast as a difference of the real-vs-synthetic gap
# between a subgroup of n threads and its complement,
#
#     MDE = (z_0.975 + z_0.80) * SD * sqrt(1/n_sub + 1/n_complement)
#
# SD = 0.3857 reproduces all five published rows to within 0.03 pp, so this is
# the plan's arithmetic rather than a new approximation:
#
#     prevalence   n      published   this model
#     2%           132    9.5 pp      9.50 pp
#     5%           330    6.1 pp      6.10 pp
#     10%          660    4.4 pp      4.43 pp
#     20%          1320   3.3 pp      3.32 pp
#     30%          1980   2.9 pp      2.90 pp
Z_SUM = 1.959963985 + 0.8416212336  # two-sided alpha 0.05, power 0.80
GAP_SD = 0.3857


def min_detectable_moderation(n_sub: int, n_total: int) -> float | None:
    """Smallest detectable topic moderation, in percentage points."""
    n_comp = n_total - n_sub
    if n_sub <= 0 or n_comp <= 0:
        return None
    return 100 * Z_SUM * GAP_SD * math.sqrt(1 / n_sub + 1 / n_comp)


# ---------------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------------

def cohens_kappa(pairs: list[tuple[str, str]]) -> dict:
    """
    Cohen's kappa with its asymptotic standard error and a 95% CI.

    Returns observed and expected agreement too, because kappa alone is
    misleading when one category dominates: with 97% not_applicable, chance
    agreement is already high and a mediocre kappa can sit under an excellent
    raw agreement. Both numbers belong in the paper.
    """
    n = len(pairs)
    if n == 0:
        return {"n": 0}

    cats = sorted({c for p in pairs for c in p})
    obs = sum(1 for a, b in pairs if a == b) / n
    ra = Counter(a for a, _ in pairs)
    rb = Counter(b for _, b in pairs)
    exp = sum((ra[c] / n) * (rb[c] / n) for c in cats)

    if exp >= 1.0:
        # Every rating in one category: kappa is undefined, agreement is 1.
        return {"n": n, "observed": obs, "expected": exp, "kappa": None,
                "se": None, "ci": None, "categories": cats}

    kappa = (obs - exp) / (1 - exp)
    se = math.sqrt(obs * (1 - obs) / (n * (1 - exp) ** 2))
    lo, hi = kappa - 1.96 * se, kappa + 1.96 * se
    return {
        "n": n,
        "observed": obs,
        "expected": exp,
        "kappa": kappa,
        "se": se,
        "ci": (max(lo, -1.0), min(hi, 1.0)),
        "categories": cats,
    }


def interpret(kappa: float | None) -> str:
    """Landis & Koch bands, for orientation only."""
    if kappa is None:
        return "undefined"
    if kappa < 0:
        return "worse than chance"
    for bound, name in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                        (0.80, "substantial")):
        if kappa <= bound:
            return name
    return "almost perfect"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a truncated final line
    return out


def latest_per_op(records: list[dict]) -> dict[str, dict]:
    """Last write wins, so a re-run that appends corrections supersedes."""
    by_id: dict[str, dict] = {}
    for r in records:
        sid = r.get("submission_id")
        if sid:
            by_id[sid] = r
    return by_id


def discover(labels_dir: Path, task: str) -> dict[str, Path]:
    """Map model name -> label file for a task, from labels_<model>_<task>.jsonl."""
    found = {}
    for p in sorted(labels_dir.glob(f"labels_*_{task}.jsonl")):
        stem = p.stem[len("labels_"):-len(f"_{task}")]
        if stem != "scispacy":
            found[stem] = p
    return found


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_quote_verification(a_records: list[dict]) -> None:
    """
    Quote verification, counted only where a quote was actually required.

    The cancer prompt explicitly allows an empty quote when the post is not
    cancer-related, so scoring those as "unverifiable" would put the rate at
    whatever the non-cancer share happens to be -- around 97% -- and say nothing
    about the labeller. The number that means something is the rate among posts
    the labeller did flag, where it asserted a cancer history and should be able
    to point at the words it read that from.
    """
    total = len(a_records)
    if not total:
        return
    required = [r for r in a_records if r.get("cancer_relevant")]
    bad = sum(1 for r in required if not r.get("evidence_quote_verified"))
    empty_ok = total - len(required)
    off = sum(1 for r in a_records if r.get("label_on_list") is False)

    print("\n--- Labeller A sanity checks ---")
    print(f"  records:                              {total:,}")
    print(f"  no quote required (not cancer):       {empty_ok:,}")
    if required:
        print(f"  quote required (cancer-flagged):      {len(required):,}")
        print(f"    unverifiable against the post text: {bad:,} "
              f"({100*bad/len(required):.1f}%)")
        print("    (the labeller asserted a cancer history it could not point to;")
        print("     these are worth reading by hand whatever the confidence said)")
    print(f"  off-list labels coerced:              {off:,}")


def report_agreement(a: dict[str, dict], b: dict[str, dict]) -> set[str]:
    """Print both kappas. Returns the set of ids both labellers covered."""
    shared = sorted(set(a) & set(b))
    print("\n--- Agreement: labeller A (LLM) vs labeller B (scispaCy) ---")
    print(f"  OPs labelled by A: {len(a):,}   by B: {len(b):,}   by both: {len(shared):,}")
    if not shared:
        print("  No overlap, so no kappa. Run both labellers over the same OP set.")
        return set()

    rel_pairs = [
        ("cancer" if a[i].get("cancer_relevant") else "not_cancer",
         "cancer" if b[i].get("cancer_relevant") else "not_cancer")
        for i in shared
    ]
    rel = cohens_kappa(rel_pairs)
    print("\n  cancer_relevant  (STRONG: independent instruments)")
    print(f"    n={rel['n']:,}  observed agreement {rel['observed']:.3f}  "
          f"expected {rel['expected']:.3f}")
    if rel["kappa"] is None:
        print("    kappa undefined (one category only)")
    else:
        lo, hi = rel["ci"]
        print(f"    kappa {rel['kappa']:.3f}  (95% CI {lo:.3f} to {hi:.3f})  "
              f"-- {interpret(rel['kappa'])}")

    cross = Counter(rel_pairs)
    print(f"    both cancer: {cross[('cancer','cancer')]:,}   "
          f"A only: {cross[('cancer','not_cancer')]:,}   "
          f"B only: {cross[('not_cancer','cancer')]:,}   "
          f"neither: {cross[('not_cancer','not_cancer')]:,}")

    # Stance, on the posts both called cancer-relevant.
    both_rel = [i for i in shared
                if a[i].get("cancer_relevant") and b[i].get("cancer_relevant")]
    print("\n  stance  (WEAK: labeller B is surface cues, not a second reader)")
    if not both_rel:
        print("    no posts both labellers called cancer-relevant")
    else:
        st = cohens_kappa([(a[i].get("label", "unclear"), b[i].get("label", "unclear"))
                           for i in both_rel])
        print(f"    n={st['n']:,}  observed agreement {st['observed']:.3f}  "
              f"expected {st['expected']:.3f}")
        if st["kappa"] is None:
            print("    kappa undefined (one category only)")
        else:
            lo, hi = st["ci"]
            print(f"    kappa {st['kappa']:.3f}  (95% CI {lo:.3f} to {hi:.3f})  "
                  f"-- {interpret(st['kappa'])}")
        print("    Do not read this as validation of the stance labels. It is a")
        print("    floor: hand-coding is what settles stance (plan, 'Validation').")
    return set(shared)


def build_consensus(
    a: dict[str, dict], b: dict[str, dict], conf_floor: float, have_b: bool
) -> list[dict]:
    """
    Per-OP consensus plus a confidence tier.

    have_b says whether labeller B was run at all. It is optional (scispaCy is a
    heavy install), and when it is absent every OP would otherwise fall into
    'single' and the queue would be the whole corpus. So without B we tier on
    labeller A alone, and 'single' keeps its real meaning: B ran, but did not
    cover this OP.

      low       the labellers disagree on relevance, A could not read the
                stance, or the evidence quote is not in the post. Reading the
                post settles it, so these go first.
      medium    cancer-flagged and agreed, but A was below the confidence floor
      high      cancer-flagged, agreed, and A was confident
      negative  no cancer here, and nothing disagrees
      single    labeller B covered this OP but A did not, so there is nothing
                to tier on

    Only low, medium and single are queued. 'negative' is deliberately not: a
    post that the LLM reads as non-cancer and in which UMLS linking finds no
    malignancy concept at all does not need a human, and queueing those on a low
    confidence score alone would put most of the corpus in front of a coder
    (~50% at a 0.70 floor) and bury the few hundred cases that matter.

    An OP that B has not covered is tiered on A alone and flagged b_missing,
    rather than being queued for that reason. A partly finished labeller-B run
    is a reason to finish the run, not to put thousands of plainly non-cancer
    posts in front of a human; the coverage gap is reported separately.
    """
    out = []
    for sid in sorted(set(a) | set(b)):
        ra, rb = a.get(sid), b.get(sid)
        conf = (ra or {}).get("confidence")
        a_rel = (ra or {}).get("cancer_relevant")
        b_rel = (rb or {}).get("cancer_relevant")
        stance = (ra or {}).get("label")

        dual = rb is not None  # this OP has both labellers, so they can be compared

        if ra is None:
            tier, why = "single", "only labeller B covered this OP"
        elif dual and a_rel != b_rel:
            tier, why = "low", f"relevance disagreement (A={a_rel}, B={b_rel})"
        elif not a_rel:
            # Both labellers agree there is no cancer here. Settled.
            tier, why = "negative", ""
        elif stance == "unclear":
            tier, why = "low", "labeller A could not read the stance"
        elif not ra.get("evidence_quote_verified"):
            # A cancer history asserted without a quote that occurs in the post.
            # The report says these get read by hand; this is what makes that true.
            tier, why = "low", "evidence quote not found in the post text"
        elif conf is None or conf < conf_floor:
            tier, why = "medium", f"labeller A confidence {conf}"
        else:
            tier, why = "high", ""

        out.append(
            {
                "submission_id": sid,
                "cancer_relevant_a": a_rel,
                "cancer_relevant_b": b_rel,
                "stance_a": stance,
                "stance_b_cue_based": (rb or {}).get("label"),
                "confidence_a": conf,
                "evidence_quote": (ra or {}).get("evidence_quote"),
                "evidence_quote_verified": (ra or {}).get("evidence_quote_verified"),
                "concepts_b": [c.get("name") for c in (rb or {}).get("concepts", [])],
                "tier": tier,
                "queue_reason": why,
                "b_missing": have_b and ra is not None and not dual,
            }
        )
    return out


def report_prevalence(consensus: list[dict], disciplines: dict[str, dict],
                      n_total: int) -> None:
    print("\n--- Prevalence and power ---")
    print(f"  Denominator for the power model: {n_total:,} threads\n")

    if disciplines:
        counts = Counter(r.get("label", "?") for r in disciplines.values())
        print(f"  {'discipline':<42} {'n':>6} {'share':>7} {'min. detectable':>16}")
        print("  " + "-" * 74)
        for label, n in counts.most_common():
            mde = min_detectable_moderation(n, n_total)
            cell = f"{mde:.1f} pp" if mde else "n/a"
            flag = "" if n >= 0.05 * n_total else "   (exploratory)"
            print(f"  {label[:42]:<42} {n:>6,} {100*n/n_total:>6.1f}% {cell:>16}{flag}")
        print("\n  Below ~5% prevalence the detectable moderation is more than half the")
        print("  main effect (11.8 pp), so those rows are exploratory, per the plan.")
    else:
        print("  (no discipline labels found; run --tasks discipline for this table)")

    # The cancer cells: this is the contrast the binary exists to support.
    cancer = [r for r in consensus if r.get("cancer_relevant_a")]
    stance_counts = Counter(r.get("stance_a") for r in cancer)
    print(f"\n  Cancer-flagged (labeller A): {len(cancer):,} "
          f"({100*len(cancer)/max(n_total,1):.1f}% of {n_total:,})")
    print(f"  {'stance':<20} {'n':>6} {'min. detectable':>16}")
    print("  " + "-" * 44)
    for s in STANCES:
        n = stance_counts.get(s, 0)
        mde = min_detectable_moderation(n, n_total)
        cell = f"{mde:.1f} pp" if mde else "n/a"
        print(f"  {s:<20} {n:>6,} {cell:>16}")

    n_pat = stance_counts.get("current_patient", 0)
    n_surv = stance_counts.get("survivor", 0)
    print(f"\n  The patient-vs-survivor contrast rests on {n_pat:,} vs {n_surv:,} threads.")
    if min(n_pat, n_surv) == 0:
        print("  One cell is empty: no contrast is estimable.")
    else:
        # Here the comparison is between the two cells, not cell vs complement.
        mde = 100 * Z_SUM * GAP_SD * math.sqrt(1 / n_pat + 1 / n_surv)
        print(f"  Smallest difference in the real-vs-synthetic gap detectable between")
        print(f"  those two cells at 80% power: {mde:.1f} pp.")
        if mde > 11.8:
            print("  That exceeds the main effect itself (11.8 pp), so treat the")
            print("  patient-vs-survivor split as descriptive, not as a powered test.")
        elif mde > 5.9:
            print("  That is more than half the main effect (11.8 pp): exploratory.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Agreement, consensus and manual-coding queue for the cancer flag.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--labels-dir", default="../Piloting/Round 5/output/corpora/labels")
    ap.add_argument("--model", default=None,
                    help="labeller-A model to use; default is the only one present, "
                         "or 'gemini' if several")
    ap.add_argument("--confidence-floor", type=float, default=0.70,
                    help="labeller A confidence below this is queued for a human")
    ap.add_argument("--n-total", type=int, default=0,
                    help="denominator for the power table; 0 means use the number of "
                         "OPs actually labelled")
    ap.add_argument("--out-prefix", default=None,
                    help="where to write cancer_consensus.jsonl and "
                         "manual_coding_queue.csv (default: --labels-dir)")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        sys.exit(
            f"ERROR: labels directory not found: {labels_dir}\n"
            "       Run classify_topics.py first."
        )

    cancer_files = discover(labels_dir, "cancer")
    disc_files = discover(labels_dir, "discipline")
    if not cancer_files:
        sys.exit(
            f"ERROR: no labeller-A cancer labels in {labels_dir}\n"
            "       Expected labels_<model>_cancer.jsonl from classify_topics.py."
        )

    model = args.model or ("gemini" if "gemini" in cancer_files
                           else sorted(cancer_files)[0])
    if model not in cancer_files:
        sys.exit(f"ERROR: no cancer labels for model '{model}'. "
                 f"Found: {sorted(cancer_files)}")
    print(f"Labeller A: {model}  ({cancer_files[model].name})")

    a_records = read_jsonl(cancer_files[model])
    a = latest_per_op(a_records)

    b_path = labels_dir / "labels_scispacy_cancer.jsonl"
    if b_path.exists():
        b = latest_per_op(read_jsonl(b_path))
        print(f"Labeller B: scispacy  ({b_path.name})")
    else:
        b = {}
        print("Labeller B: not found -- skipping kappa.\n"
              "            Run classify_cancer_scispacy.py to enable it.")

    disciplines = (
        latest_per_op(read_jsonl(disc_files[model])) if model in disc_files else {}
    )
    if disciplines:
        print(f"Disciplines: {disc_files[model].name}")

    report_quote_verification(a_records)
    if b:
        report_agreement(a, b)

    consensus = build_consensus(a, b, args.confidence_floor, have_b=bool(b))
    if not b:
        print(
            "\nNOTE: single-labeller mode. Tiers below come from labeller A alone, so\n"
            "      there is no agreement-based confidence and no kappa -- the plan's\n"
            "      dual-labelling is not satisfied by this run."
        )
    n_total = args.n_total or max(len(a), len(disciplines))
    report_prevalence(consensus, disciplines, n_total)

    n_missing_b = sum(1 for r in consensus if r.get("b_missing"))
    if n_missing_b:
        print(
            f"\nWARNING: labeller B has no label for {n_missing_b:,} of "
            f"{len(consensus):,} OPs\n"
            "         ({:.1f}%). Those are tiered on labeller A alone and are not\n"
            "         dual-labelled, so they are outside the kappa above. Re-run\n"
            "         classify_cancer_scispacy.py to finish the coverage; it resumes."
            .format(100 * n_missing_b / max(len(consensus), 1))
        )

    tiers = Counter(r["tier"] for r in consensus)
    print("\n--- Confidence tiers ---")
    for t in ("high", "medium", "low", "negative", "single"):
        n = tiers.get(t, 0)
        note = "   (not queued: both labellers say no cancer)" if t == "negative" else ""
        print(f"  {t:<9} {n:>6,} ({100*n/max(len(consensus),1):.1f}%){note}")

    out_dir = Path(args.out_prefix) if args.out_prefix else labels_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cons_path = out_dir / "cancer_consensus.jsonl"
    with open(cons_path, "w") as f:
        for r in consensus:
            f.write(json.dumps(r) + "\n")

    # Queue: disagreements first, then single-labeller OPs, then low confidence.
    # A disagreement is the case where reading the post actually settles
    # something, so it is worth a coder's attention before a merely unconfident
    # agreement. 'high' and 'negative' are not queued.
    priority = {"low": 0, "single": 1, "medium": 2}
    queue = sorted(
        (r for r in consensus if r["tier"] in priority),
        key=lambda r: (priority[r["tier"]], r["confidence_a"] or 0.0),
    )
    queue_path = out_dir / "manual_coding_queue.csv"
    cols = [
        "submission_id", "tier", "queue_reason", "cancer_relevant_a",
        "cancer_relevant_b", "stance_a", "stance_b_cue_based", "confidence_a",
        "evidence_quote_verified", "evidence_quote", "concepts_b",
    ]
    with open(queue_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*cols, "human_stance", "human_notes"])
        w.writeheader()
        for r in queue:
            row = {c: r.get(c) for c in cols}
            row["concepts_b"] = "; ".join(r.get("concepts_b") or [])
            row["human_stance"] = ""   # for the coder to fill
            row["human_notes"] = ""
            w.writerow(row)

    print(f"\nWrote {cons_path}")
    print(f"Wrote {queue_path}  ({len(queue):,} OPs for a human, "
          f"{100*len(queue)/max(len(consensus),1):.1f}% of the labelled set)")
    print(
        "\nStill outstanding from the plan: the blind hand-coding of 200 random OPs\n"
        "before anyone looks at automated output, two coders, and a reportable\n"
        "inter-annotator kappa. That is the ceiling on everything above, and it is\n"
        "not built here."
    )


if __name__ == "__main__":
    main()
