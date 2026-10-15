#!/usr/bin/env python3
"""
Map each thread's first responder to a standardised role and specialty.

Answers: does the person who answers first tend to match the subject of the
post -- a psychiatrist on a mental health question, a dermatologist on a rash?

r/AskDocs flair is semi-structured and mod-verified, e.g.

    Physician
    Physician - Dermatologist | Top Contributor
    Registered Nurse
    Layperson/not verified as healthcare professional

so the first segment gives the role and a later segment, when present, gives
the specialty. "Top Contributor" and "Moderator" are decoration and dropped.

Read the match rate as lift, not as a count
-------------------------------------------
Only about a fifth of first responders declare a specialty, and that fifth is
dominated by a handful of prolific contributors -- a third of all declared
specialties are one dermatologist tag. Counting raw matches would therefore
mostly measure who posts most. So for each specialty this reports the observed
match rate against the base rate of that discipline in the corpus, and the
number of distinct authors behind it. A lift of 1.0 is what you would see if
responders ignored the subject entirely.

Output
------
  output/corpora/export/first_responder_roles.csv
      one row per thread, joinable to topic_labels.csv on submission_id

Usage
-----
  python map_responder_roles.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Flair fragments that are status, not role or specialty.
NOISE = re.compile(r"top contributor|moderator|^mod$|not verified", re.I)
SPLIT = re.compile(r"\s*[\|\-–/]\s*(?![a-z])|\s*\|\s*|\s+-\s+")

# --- role normalisation ----------------------------------------------------
ROLE_PATTERNS = [
    (r"^physician assistant|^\s*pa-c", "Physician assistant"),
    (r"physician|^md$|^do$|surgeon|/md", "Physician"),
    (r"nurse practitioner|^np$", "Nurse practitioner"),
    (r"registered nurse|^rn\b|^nurse\b", "Nurse"),
    (r"medical student|^ms[1-4]\b|^m[1-4]\b", "Medical student"),
    (r"pharmacist|pharm\.?d", "Pharmacist"),
    (r"psychologist", "Psychologist"),
    (r"dentist|dental", "Dentist"),
    (r"paramedic|emt|respiratory therapist|technologist|sonograph|lab scien"
     r"|dietician|dietitian|physical therap|speech|optometr|midwife|audiolog",
     "Allied health"),
    (r"layperson|^lay\b", "Layperson"),
]

# --- specialty -> one of the 17 disciplines --------------------------------
# Only where the correspondence is defensible. Specialties that span every body
# system (radiology, pathology) or that are generalist by definition (emergency,
# family, internal medicine) map to no discipline and are reported separately:
# counting them as mismatches would be wrong, since matching is not their job.
SPECIALTY_TO_DISCIPLINE = {
    "dermatolog": "Dermatology",
    "gastroenterolog": "Gastroenterology",
    "hepatolog": "Gastroenterology",
    "psychiatr": "Psychiatry and mental health",
    "addiction medicine": "Psychiatry and mental health",
    "cardiolog": "Cardiovascular",
    "electrophysiolog": "Cardiovascular",
    "pulmonary": "Pulmonology",
    "pulm": "Pulmonology",
    "neurolog": "Neurology",
    "neurosurg": "Neurology",
    "endocrinolog": "Endocrinology, nutrition, and metabolism",
    "diabetes": "Endocrinology, nutrition, and metabolism",
    "nutrition": "Endocrinology, nutrition, and metabolism",
    "ob/gyn": "Genitourinary and reproductive health",
    "obstetric": "Genitourinary and reproductive health",
    "gynaecolog": "Genitourinary and reproductive health",
    "gynecolog": "Genitourinary and reproductive health",
    "urolog": "Genitourinary and reproductive health",
    "nephrolog": "Genitourinary and reproductive health",
    "otolaryngolog": "Otolaryngology",
    # Word-bounded: a bare substring match would fire inside "Medical Student",
    # which silently mapped 342 students to ENT on the first run.
    r"\bent\b": "Otolaryngology",
    "rheumatolog": "Musculoskeletal and connective tissue conditions",
    "orthopedic": "Musculoskeletal and connective tissue conditions",
    "orthopaedic": "Musculoskeletal and connective tissue conditions",
    "ophthalmolog": "Ophthalmology",
    "optometr": "Ophthalmology",
    "infectious disease": "Infectious diseases",
    "allerg": "Hematology and immunology",
    "immunolog": "Hematology and immunology",
    "hematolog": "Hematology and immunology",
    # Heme/Onc spans two disciplines; see the note printed in the report.
    "heme/onc": "Hematology and immunology",
    "oncolog": "Neoplasms",
}
GENERALIST = re.compile(
    r"emergency|family medicine|^fm\b|internal medicine|^im\b|primary care"
    r"|pediatric|paediatric|hospitalist|general practice|^gp\b|phpm", re.I)
CROSS_CUTTING = re.compile(
    r"radiolog|patholog|anesthes|anaesthes|critical care|surgery|surgeon"
    r"|neonatal|palliative|geriatric|sports medicine", re.I)


def segments(flair: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*\|\s*|\s+-\s+|\s+–\s+", flair or "")]
    return [p for p in parts if p and not NOISE.search(p)]


def normalise_role(first_segment: str) -> str:
    s = (first_segment or "").lower()
    if not s:
        return "Unflaired"
    for pat, name in ROLE_PATTERNS:
        if re.search(pat, s):
            return name
    return "Other"


def map_specialty(segs: list[str]) -> tuple[str | None, str | None, str]:
    """Return (raw specialty, mapped discipline, kind)."""
    for seg in segs[1:] + segs[:1]:  # later segments first; role may itself name one
        low = seg.lower()
        for key, disc in SPECIALTY_TO_DISCIPLINE.items():
            hit = (re.search(key, low) if key.startswith(r"\b") else key in low)
            if hit:
                return seg, disc, "body-system"
        if GENERALIST.search(low):
            return seg, None, "generalist"
        if CROSS_CUTTING.search(low):
            return seg, None, "cross-cutting"
    return None, None, "none"


def load(labels_csv: Path, subs: Path, comments: Path):
    labels = {r["submission_id"]: r["discipline"]
              for r in csv.DictReader(open(labels_csv)) if r["discipline"]}
    if not labels:
        sys.exit(f"ERROR: no labels in {labels_csv}")
    op_author = {}
    with open(subs) as f:
        for line in f:
            s = json.loads(line)
            if s["id"] in labels:
                op_author[s["id"]] = s.get("author")

    # Earliest non-OP, non-removed top-level comment -- the same filters the
    # generation run used, so the thread set matches.
    first: dict[str, tuple[int, str, str]] = {}
    with open(comments) as f:
        for line in f:
            c = json.loads(line)
            pid = c.get("parent_id", "")
            if not pid.startswith("t3_"):
                continue
            sid = pid[3:]
            if sid not in labels:
                continue
            if c.get("body") in ("[removed]", "[deleted]"):
                continue
            if c.get("author") and c["author"] == op_author.get(sid):
                continue
            t = c.get("created_utc") or 0
            if sid not in first or t < first[sid][0]:
                first[sid] = (t, c.get("author_flair_text") or "", c.get("author") or "")
    return labels, first


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Map first responders to roles and specialties.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--labels", default="output/corpora/export/topic_labels.csv")
    ap.add_argument("--corpora-dir", default="../Piloting/Round 2")
    ap.add_argument("--out", default="output/corpora/export/first_responder_roles.csv")
    args = ap.parse_args()

    corp = Path(args.corpora_dir)
    labels, first = load(Path(args.labels), corp / "submissions_corpus.jsonl",
                         corp / "comments_corpus.jsonl")
    print(f"threads labelled: {len(labels):,}   with a first responder: {len(first):,}")

    rows = []
    for sid, (_, flair, author) in sorted(first.items()):
        segs = segments(flair)
        role = normalise_role(segs[0] if segs else "")
        spec, disc, kind = map_specialty(segs)
        rows.append({
            "submission_id": sid,
            "topic": labels[sid],
            "responder_role": role,
            "responder_specialty": spec or "",
            "specialty_discipline": disc or "",
            "specialty_kind": kind,
            "specialty_matches_topic": (disc == labels[sid]) if disc else "",
            "responder_flair": flair,
            "responder_author": author,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows):,} rows)\n")

    # ---- report ----------------------------------------------------------
    print("=== who answers first ===")
    for r, n in Counter(x["responder_role"] for x in rows).most_common():
        print(f"  {r:<22} {n:>5}  {100*n/len(rows):>5.1f}%")

    print("\n=== is a specialty declared? ===")
    for k, n in Counter(x["specialty_kind"] for x in rows).most_common():
        lab = {"none": "no specialty given", "body-system": "maps to a discipline",
               "generalist": "generalist (EM, FM, IM, paeds)",
               "cross-cutting": "cross-cutting (radiology, pathology, surgery)"}[k]
        print(f"  {lab:<46} {n:>5}  {100*n/len(rows):>5.1f}%")

    testable = [x for x in rows if x["specialty_kind"] == "body-system"]
    if not testable:
        print("\nNo body-system specialties; nothing to test.")
        return

    base = Counter(x["topic"] for x in rows)
    n_all = len(rows)
    print(f"\n=== does the specialty match the topic?  (n={len(testable):,}) ===")
    matched = sum(1 for x in testable if x["specialty_matches_topic"] is True)
    exp = sum(base[x["specialty_discipline"]] / n_all for x in testable)
    print(f"  matched {matched} of {len(testable)} = {100*matched/len(testable):.1f}%")
    print(f"  expected if responders ignored the subject: {100*exp/len(testable):.1f}%")
    print(f"  lift: {matched/exp:.1f}x\n")

    print(f"  {'specialty':<28} {'n':>4} {'authors':>8} {'match':>7} {'base':>7} {'lift':>6}")
    print("  " + "-" * 68)
    by_spec = defaultdict(list)
    for x in testable:
        by_spec[x["specialty_discipline"]].append(x)
    for disc, xs in sorted(by_spec.items(), key=lambda kv: -len(kv[1])):
        m = sum(1 for x in xs if x["specialty_matches_topic"] is True)
        b = base[disc] / n_all
        authors = len({x["responder_author"] for x in xs})
        lift = (m / len(xs)) / b if b else float("nan")
        print(f"  {disc[:28]:<28} {len(xs):>4} {authors:>8} "
              f"{100*m/len(xs):>6.1f}% {100*b:>6.1f}% {lift:>5.1f}x")

    print("\n  'authors' is the number of distinct accounts behind each row count.")
    print("  Where it is small the figure describes those individuals, not the")
    print("  specialty. Heme/Onc is mapped to Hematology and immunology, so posts")
    print("  it answers that are labelled Neoplasms count as misses.")


if __name__ == "__main__":
    main()
