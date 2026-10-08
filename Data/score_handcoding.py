#!/usr/bin/env python3
"""
Score the hand-coded workbooks: agreement, gold standard, and classifier accuracy.

The second half of the plan's validation step: "report precision, recall, and
inter-annotator agreement as the ceiling on automated performance."

It reports, in this order, because each step bounds the next:

  1. Inter-annotator agreement between the coders. This is the ceiling. If two
     trained humans agree only 70% of the time on a category, no classifier can
     be meaningfully said to exceed that on it.
  2. The gold standard: the OPs both coders agreed on. Disagreements are written
     to an adjudication queue rather than silently resolved by picking one coder.
  3. The classifier against that gold: accuracy, macro-F1, and per-class
     precision/recall/F1 for discipline; precision/recall/F1 for cancer_relevant.

Adjudication
------------
Where the coders disagree, a human decides. Record that decision in the
adjudicated column of adjudication_queue.csv and pass it back with
--adjudicated, and those OPs rejoin the gold standard. Until then they are
excluded, and the exclusion is reported, since a gold standard built only from
the easy cases flatters the classifier.

Usage
-----
  python score_handcoding.py
  python score_handcoding.py --adjudicated ../Piloting/Round 5/output/corpora/handcoding/adjudication_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from openpyxl import load_workbook
except ImportError:
    sys.exit("ERROR: openpyxl is required.  pip install openpyxl")

from agreement import cohens_kappa, interpret, latest_per_op, read_jsonl

TASK_COLS = ("discipline", "cancer_relevant", "cancer_stance")

# The discipline that codebook Rule A routes malignancies to.
CANCER_NODE = "Neoplasms"


# ---------------------------------------------------------------------------
# Reading the workbooks
# ---------------------------------------------------------------------------

def read_coder(path: Path) -> dict[str, dict]:
    """Pull {submission_id: {discipline, cancer_relevant, cancer_stance, notes}}."""
    wb = load_workbook(path, data_only=True)
    if "Coding" not in wb.sheetnames:
        sys.exit(f"ERROR: {path.name} has no 'Coding' sheet. Is it one of ours?")
    ws = wb["Coding"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h else "" for h in next(rows)]
    try:
        idx = {c: header.index(c) for c in (*TASK_COLS, "submission_id", "notes")}
    except ValueError as e:
        sys.exit(f"ERROR: {path.name} is missing a column: {e}")

    out: dict[str, dict] = {}
    for row in rows:
        sid = row[idx["submission_id"]]
        if not sid:
            continue
        rec = {}
        for c in (*TASK_COLS, "notes"):
            v = row[idx[c]]
            rec[c] = str(v).strip() if v is not None and str(v).strip() else None
        if rec["cancer_relevant"]:
            rec["cancer_relevant"] = rec["cancer_relevant"].lower()
        out[str(sid)] = rec
    return out


def coverage(coded: dict[str, dict], field: str) -> int:
    return sum(1 for r in coded.values() if r.get(field))


# ---------------------------------------------------------------------------
# Precision / recall
# ---------------------------------------------------------------------------

def prf(tp: int, fp: int, fn: int):
    """
    Precision, recall and F1, or None where the quantity is undefined.

    Undefined is not zero, and conflating them misreads badly: a classifier that
    flagged nothing because there was nothing to flag has undefined precision,
    not precision 0.0, which would read as total failure.
    """
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = (2 * p * r / (p + r)) if p and r else (0.0 if p is not None or r is not None else None)
    return p, r, f


def _fmt(x, width: int = 6) -> str:
    return f"{x:>{width}.2f}" if x is not None else f"{'n/a':>{width}}"


def report_multiclass(gold: dict[str, str], pred: dict[str, str], title: str) -> None:
    """Per-class precision/recall/F1 plus accuracy and macro-F1."""
    shared = sorted(set(gold) & set(pred))
    print(f"\n--- {title} ---")
    if not shared:
        print("  No OPs have both a gold label and a classifier label.")
        return
    correct = sum(1 for i in shared if gold[i] == pred[i])
    print(f"  n={len(shared):,}   accuracy {correct/len(shared):.3f} "
          f"({correct}/{len(shared)})")

    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for i in shared:
        g, p = gold[i], pred[i]
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    classes = sorted(set(gold[i] for i in shared) | set(pred[i] for i in shared))
    print(f"\n  {'class':<46} {'n':>4} {'prec':>6} {'rec':>6} {'F1':>6}")
    print("  " + "-" * 72)
    f1s = []
    for c in classes:
        support = sum(1 for i in shared if gold[i] == c)
        p, r, f = prf(tp[c], fp[c], fn[c])
        if support:
            f1s.append(f if f is not None else 0.0)
        flag = "  (no gold support)" if not support else ""
        print(f"  {c[:46]:<46} {support:>4} {_fmt(p)} {_fmt(r)} {_fmt(f)}{flag}")
    if f1s:
        print(f"\n  macro-F1 over the {len(f1s)} classes with gold support: "
              f"{sum(f1s)/len(f1s):.3f}")

    # The confusions worth reading, not the whole 17x17 matrix.
    conf = Counter((gold[i], pred[i]) for i in shared if gold[i] != pred[i])
    if conf:
        print("\n  most common confusions (gold -> classifier):")
        for (g, p), n in conf.most_common(8):
            print(f"    {n:>3}  {g[:32]:<32} -> {p[:32]}")


def report_binary(gold: dict[str, bool], pred: dict[str, bool], title: str) -> None:
    shared = sorted(set(gold) & set(pred))
    print(f"\n--- {title} ---")
    if not shared:
        print("  No overlap.")
        return
    tp = sum(1 for i in shared if gold[i] and pred[i])
    fp = sum(1 for i in shared if not gold[i] and pred[i])
    fn = sum(1 for i in shared if gold[i] and not pred[i])
    tn = sum(1 for i in shared if not gold[i] and not pred[i])
    p, r, f = prf(tp, fp, fn)
    print(f"  n={len(shared):,}   accuracy {(tp+tn)/len(shared):.3f}")
    print(f"  positives in gold: {tp+fn}   flagged by the classifier: {tp+fp}")
    print(f"  precision {_fmt(p,5)}   recall {_fmt(r,5)}   F1 {_fmt(f,5)}")
    print(f"  tp {tp}  fp {fp}  fn {fn}  tn {tn}")
    if p is None or r is None:
        print("  (n/a = undefined, not zero: there were no positives to score on this "
              "side)")
    if fn:
        print(f"  {fn} missed positive(s): worth reading, since recall is what a "
              "prefilter would have capped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score hand-coded workbooks against each other and the classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--handcoding-dir",
                    default="../Piloting/Round 5/output/corpora/handcoding")
    ap.add_argument("--labels-dir", default="../Piloting/Round 5/output/corpora/labels")
    ap.add_argument("--model", default="gemini", help="classifier whose labels to score")
    ap.add_argument("--adjudicated", default=None, metavar="CSV",
                    help="completed adjudication_queue.csv; its 'adjudicated' column "
                         "resolves coder disagreements back into the gold standard")
    args = ap.parse_args()

    hc_dir = Path(args.handcoding_dir)
    books = sorted(hc_dir.glob("handcode_*_*.xlsx"))
    if not books:
        sys.exit(f"ERROR: no handcode_*.xlsx in {hc_dir}\n"
                 "       Run build_handcoding_sheets.py first.")

    coders = {}
    for b in books:
        name = b.stem.rsplit("_", 1)[-1]
        coders[name] = read_coder(b)
        got = coverage(coders[name], "discipline")
        print(f"Coder {name}: {len(coders[name]):,} rows, {got:,} discipline codes filled"
              f" ({b.name})")
    if not any(coverage(c, "discipline") for c in coders.values()):
        sys.exit("\nERROR: no discipline codes filled in yet. Nothing to score.")

    names = sorted(coders)
    disagreements: list[dict] = []

    # --- 1. inter-annotator agreement: the ceiling --------------------------
    if len(names) >= 2:
        a, b = names[0], names[1]
        ca, cb = coders[a], coders[b]
        both = [i for i in sorted(set(ca) & set(cb))
                if ca[i]["discipline"] and cb[i]["discipline"]]
        print(f"\n=== 1. Inter-annotator agreement ({a} vs {b}) ===")
        print(f"  {len(both):,} OPs coded by both")
        if both:
            k = cohens_kappa([(ca[i]["discipline"], cb[i]["discipline"]) for i in both])
            print(f"\n  discipline: observed agreement {k['observed']:.3f}, "
                  f"expected {k['expected']:.3f}")
            if k["kappa"] is not None:
                lo, hi = k["ci"]
                print(f"    kappa {k['kappa']:.3f} (95% CI {lo:.3f} to {hi:.3f}) "
                      f"-- {interpret(k['kappa'])}")
            print("    This is the ceiling: a classifier cannot be shown to beat the")
            print("    agreement of the humans defining the target.")

        bothc = [i for i in sorted(set(ca) & set(cb))
                 if ca[i]["cancer_relevant"] and cb[i]["cancer_relevant"]]
        if bothc:
            kc = cohens_kappa([(ca[i]["cancer_relevant"], cb[i]["cancer_relevant"])
                               for i in bothc])
            print(f"\n  cancer_relevant: n={kc['n']:,}, observed "
                  f"{kc['observed']:.3f}", end="")
            if kc["kappa"] is not None:
                print(f", kappa {kc['kappa']:.3f} -- {interpret(kc['kappa'])}")
            else:
                print(" (kappa undefined: one category only)")

        for i in both:
            if ca[i]["discipline"] != cb[i]["discipline"]:
                disagreements.append({
                    "submission_id": i,
                    "field": "discipline",
                    f"coder_{a}": ca[i]["discipline"],
                    f"coder_{b}": cb[i]["discipline"],
                    "notes_a": ca[i].get("notes") or "",
                    "notes_b": cb[i].get("notes") or "",
                    "adjudicated": "",
                })
    else:
        print("\n=== 1. Inter-annotator agreement ===")
        print("  Only one coder, so there is none. The plan asks for two; without a")
        print("  second the numbers below have no ceiling to be read against.")

    # --- 2. gold standard ---------------------------------------------------
    adjudged: dict[str, str] = {}
    if args.adjudicated:
        p = Path(args.adjudicated)
        if not p.exists():
            sys.exit(f"ERROR: adjudication file not found: {p}")
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                v = (row.get("adjudicated") or "").strip()
                if v:
                    adjudged[row["submission_id"]] = v
        print(f"\nAdjudication: {len(adjudged):,} resolved decisions read from {p.name}")

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
        only = coders[names[0]]
        for i, r in only.items():
            if r["discipline"]:
                gold_disc[i] = r["discipline"]
            if r["cancer_relevant"]:
                gold_cancer[i] = (r["cancer_relevant"] == "yes")

    # A rule decision recorded for a post the coders agreed on overrides that consensus.
    # The gold standard then matches the rubric the classifier was given, which is the
    # point of putting the rules in taxonomy.json; the count is reported, not buried.
    overridden = []
    for i, v in adjudged.items():
        if i in gold_disc and gold_disc[i] != v:
            overridden.append((i, gold_disc[i], v))
            gold_disc[i] = v

    print(f"\n=== 2. Gold standard ===")
    print(f"  discipline:      {len(gold_disc):,} OPs")
    print(f"  cancer_relevant: {len(gold_cancer):,} OPs")
    if overridden:
        print(f"  {len(overridden):,} post(s) recoded by rule, overriding coder consensus:")
        for i, was, now in overridden:
            print(f"    {i}  {was[:34]} -> {now}")
    # Rule candidates: posts where a codebook rule may override the coders. Both coders
    # called the post cancer-relevant but the gold label is not the cancer node, so Rule A
    # may apply -- whether it does needs a human to confirm the malignancy was clinically
    # raised rather than merely worried about, which no script can decide. These join the
    # same queue the disagreements go to, so there is one file to work through.
    cancer_both: set[str] = set()
    if len(names) >= 2:
        ca, cb = coders[names[0]], coders[names[1]]
        cancer_both = {i for i in set(ca) & set(cb)
                       if ca[i]["cancer_relevant"] == "yes" == cb[i]["cancer_relevant"]}
    else:
        cancer_both = {i for i, r in coders[names[0]].items()
                       if r["cancer_relevant"] == "yes"}
    rule_cands = [
        {
            "submission_id": i,
            "field": "discipline (rule candidate)",
            **{f"coder_{n}": coders[n].get(i, {}).get("discipline") for n in names[:2]},
            "notes_a": "both coders agreed; both flagged cancer_relevant",
            "notes_b": f"Rule A -> {CANCER_NODE} if the malignancy was clinically raised",
            "adjudicated": "",
        }
        for i in sorted(cancer_both)
        if i in gold_disc and gold_disc[i] != CANCER_NODE and i not in adjudged
    ]

    unresolved = [d for d in disagreements if d["submission_id"] not in adjudged]
    queue = unresolved + rule_cands
    if queue:
        out = hc_dir / "adjudication_queue.csv"
        # Carry over decisions already recorded, so re-running never wipes partial work.
        prior: dict[str, str] = {}
        if out.exists():
            with open(out, newline="") as f:
                for row in csv.DictReader(f):
                    v = (row.get("adjudicated") or "").strip()
                    if v:
                        prior[row["submission_id"]] = v
        kept = 0
        for d in queue:
            if d["submission_id"] in prior:
                d["adjudicated"] = prior[d["submission_id"]]
                kept += 1
        cols = list(queue[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(queue)
        print(f"  {len(unresolved):,} disagreement(s) excluded, written to {out.name}")
        if rule_cands:
            print(f"  {len(rule_cands):,} rule candidate(s) added to the same file: posts")
            print(f"     both coders agreed on, where Rule A may override them")
        if kept:
            print(f"  {kept:,} decision(s) already in that file were preserved")
        print("  Fill its 'adjudicated' column and re-run with --adjudicated to fold")
        print("  them back in. Until then the gold holds only the cases both coders")
        print("  found easy, which flatters the classifier below.")

    # --- 3. the classifier against gold -------------------------------------
    labels_dir = Path(args.labels_dir)
    dpath = labels_dir / f"labels_{args.model}_discipline.jsonl"
    cpath = labels_dir / f"labels_{args.model}_cancer.jsonl"
    print(f"\n=== 3. Classifier ({args.model}) against the gold standard ===")
    if not dpath.exists() and not cpath.exists():
        print(f"  No labels in {labels_dir}. Run classify_topics.py, then re-run this.")
        return

    if dpath.exists():
        pred = {k: v["label"] for k, v in latest_per_op(read_jsonl(dpath)).items()}
        missing = len(set(gold_disc) - set(pred))
        if missing:
            print(f"  NOTE: {missing:,} gold OPs have no classifier label yet "
                  "(partial run); they are skipped.")
        report_multiclass(gold_disc, pred, "discipline")

    if cpath.exists() and gold_cancer:
        praw = latest_per_op(read_jsonl(cpath))
        pred_c = {k: bool(v.get("cancer_relevant")) for k, v in praw.items()}
        report_binary(gold_cancer, pred_c, "cancer_relevant")

    print("\nRead the section 3 numbers against the section 1 ceiling, not against 1.0.")


if __name__ == "__main__":
    main()
