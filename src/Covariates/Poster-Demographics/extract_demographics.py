#!/usr/bin/env python3
"""
Extract self-reported demographics from each thread's opening post.

r/AskDocs asks posters to state age and sex, and many volunteer height and
weight too, so the information is there in semi-standard forms:

    20 F Can someone please tell me if I should be worried?
    Hi M20 here. I've been on 15mg of Mirtazapine ...
    Age: 30 Sex: Female Height: 5'3 Weight: 114 pounds Race: Caucasian
    F24 150lbs 4'11 I'm having upper stomach pain
    My 12-week-old daughter has had persistent vomiting

Rules, not a model
------------------
Extraction is regex-based rather than LLM-based, deliberately. The formats are
formulaic because the subreddit mandates them, so rules get most of it; and
unlike a model, a rule is free, deterministic, auditable, and needs no second
validation exercise to be trusted. The cost is recall on unusual phrasings,
which is measured below as coverage rather than hidden.

Whose demographics?
-------------------
Plenty of posts are written on someone else's behalf -- a parent about a child,
a spouse about a spouse. The age and sex then belong to the PATIENT, not the
poster. Rather than guess which the analyst wants, every row carries `subject`
(first_person / third_party / unknown) so either can be filtered. Treat the
demographics as the subject's.

Precision over recall
---------------------
Several traps are specifically avoided:
  * medication doses are not weights -- a unit (lb/kg) is always required
  * "99 F" after the word fever is a temperature, not a 99-year-old female
  * bare colour words are never read as race, because "black stool" and "white
    patches" are symptoms; only an explicit "Race:" field is taken
  * ages are only accepted from an explicit marker (a joined sex letter, "yo",
    "years old", an "Age:" field), never from a loose number
  * implausible values are dropped, not stored: see the PLAUSIBLE bounds

Output
------
  poster_demographics.csv        one row per thread, joins on submission_id
  results_coverage.csv           how often each field is found, overall and by topic
  results_distributions.csv      the distribution of each field

Usage
-----
  python extract_demographics.py
  python extract_demographics.py --audit 40    # print a sample to hand-check
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Demographic self-ID clusters at the start; a window keeps stray numbers out.
HEAD_CHARS = 350

PLAUSIBLE = {"age": (0.0, 110.0), "height_cm": (40.0, 230.0),
             "weight_kg": (1.5, 350.0), "bmi": (8.0, 90.0)}

LB_TO_KG = 0.453592
IN_TO_CM = 2.54

# --- age ------------------------------------------------------------------
AGE_UNIT = {"day": 1 / 365.25, "week": 7 / 365.25, "month": 1 / 12, "year": 1.0}
RE_AGE_SUBYEAR = re.compile(
    r"\b(\d{1,3})[-\s]*(day|week|month|year)s?[-\s]*(?:old\b|of\s+age\b)", re.I)
RE_AGE_YEARS = re.compile(
    r"\b(\d{1,3})\s*(?:y\.?\s?o\.?\b|y/o\b|yrs?\b|years?\s*[-]?\s*old\b)", re.I)
# The colon is optional: "Age 23 female" is as common as "Age: 23".
RE_AGE_FIELD = re.compile(r"\bage\s*[:=]?\s*(\d{1,3})\b", re.I)
# "40y Female" -- a bare "y" only counts when a sex follows, since "y" alone is
# too weak a marker to trust on its own.
RE_AGE_Y = re.compile(r"\b(\d{1,3})\s*y\b(?=\s*[,;]?\s*(?:male|female|m\b|f\b))", re.I)
# "20F" / "F24" -- the dominant form. The sex letter must end a word so that
# "30mg" and "5 ft" cannot match.
RE_AGE_SEX = re.compile(r"\b(\d{1,2})\s*([mf])\b", re.I)
# Letter first, as in "M25". The lookbehind is essential: without it the "m" in
# "I'm 38" matches and every such poster is recorded as male.
RE_SEX_AGE = re.compile(r"(?<![a-z'’])([mf])\s*(\d{1,2})\b", re.I)
# "38, female" and "Male, 40" -- weaker, so tried only after the forms above.
# The lookarounds stop "5'2 female" and "woman 5'2" yielding ages of 2 and 5.
_SEXW = r"male|female|man|woman|boy|girl"
RE_AGE_SEXWORD = re.compile(
    rf"(?<![\'’\d])\b(\d{{1,2}})\s*[,;/\s-]\s*({_SEXW})\b", re.I)
RE_SEXWORD_AGE = re.compile(
    rf"\b({_SEXW})\s*[,;/\s-]\s*(\d{{1,2}})\b(?!\s*(?:['’\"]|ft|feet|cm|lbs?|kg|pounds))", re.I)
RE_TEMP_NEAR = re.compile(r"(?:fever|temp\w*|degrees?)\D{0,12}$", re.I)

# --- sex ------------------------------------------------------------------
RE_SEX_FIELD = re.compile(r"\bsex\s*[:=]\s*(male|female|m|f|other)\b", re.I)
RE_SEX_WORD = re.compile(r"\b(male|female|man|woman|boy|girl)\b", re.I)
RE_TRANS = re.compile(r"\b(ftm|mtf|trans(?:gender)?|non[- ]?binary|nb)\b", re.I)
# Relational terms carry sex reliably, but only tell you about the SUBJECT, so
# they are used only when the post is third-party: "my daughter" in a
# first-person post says nothing about the poster.
REL_SEX = {
    "female": r"daughter|sister|mother|mom|mum|wife|girlfriend|aunt|niece|"
              r"grandmother|grandma",
    "male": r"son|brother|father|dad|husband|boyfriend|uncle|nephew|"
            r"grandfather|grandpa",
}
RE_REL_SEX = {k: re.compile(rf"\bmy\s+(?:[\w-]+\s+){{0,3}}?({v})\b", re.I)
              for k, v in REL_SEX.items()}

# --- height / weight ------------------------------------------------------
RE_FT_IN = re.compile(
    r"\b([3-7])\s*(?:'|’|ft\b\.?|feet\b)\s*(\d{1,2})?\s*(?:\"|''|”|in\b\.?|inches\b)?",
    re.I)
RE_CM = re.compile(r"\b(\d{2,3})(?:\.\d)?\s*cm\b", re.I)
RE_LB = re.compile(r"\b(\d{2,3})(?:\.\d)?\s*(?:lbs?\b\.?|pounds?\b|#)", re.I)
RE_KG = re.compile(r"\b(\d{1,3})(?:\.\d)?\s*(?:kg\b|kilos?\b|kilograms?\b)", re.I)

# --- subject --------------------------------------------------------------
RELATIVE = (r"son|daughter|child|kid|baby|infant|toddler|husband|wife|spouse|"
            r"partner|boyfriend|girlfriend|mother|father|mom|mum|dad|parent|"
            r"brother|sister|sibling|grandmother|grandfather|grandma|grandpa|"
            r"aunt|uncle|cousin|friend|roommate|patient|niece|nephew")
RE_THIRD = re.compile(rf"\bmy\s+(?:[\w-]+\s+){{0,3}}?({RELATIVE})\b", re.I)
RE_FIRST = re.compile(
    r"\bi\s*(?:'m|’m|\s+am|\s+was|\s+have|\s+had)\b|\bi\s*\(|\bmy\s+(?:own\s+)?"
    r"(?:age|height|weight|bmi|body)\b|\bme\b", re.I)

# --- behaviour ------------------------------------------------------------
RE_NONSMOKER = re.compile(r"\bnon[-\s]?smok\w*|\b(?:don'?t|do not|never)\s+smok\w*"
                          r"|\bno\s+smok\w*|\bnever\s+smoked\b", re.I)
RE_SMOKER = re.compile(r"\bi\s+smoke\b|\bsmoker\b|\bvap(?:e|ing)\b|\bcigarett",
                       re.I)
RE_NONDRINK = re.compile(r"\bnon[-\s]?drink\w*|\b(?:don'?t|do not|never)\s+drink\w*"
                         r"|\bno\s+alcohol\b|\bteetotal", re.I)
RE_DRINK = re.compile(r"\bi\s+drink\b|\bdrinker\b|\balcoholic\b", re.I)
RE_RACE_FIELD = re.compile(
    r"\brace\s*[:=]\s*([A-Za-z][A-Za-z /-]{2,24})", re.I)


def _ok(field: str, v: float | None) -> float | None:
    if v is None:
        return None
    lo, hi = PLAUSIBLE[field]
    return v if lo <= v <= hi else None


def _age_candidates(scope: str):
    """Every age the text states, as (position, years, how)."""
    out = []
    for m in RE_AGE_SUBYEAR.finditer(scope):
        n, unit = int(m.group(1)), m.group(2).lower()
        out.append((m.start(), n * AGE_UNIT[unit], f"{unit}s-old"))
    for rx, how in ((RE_AGE_FIELD, "age-field"), (RE_AGE_YEARS, "years-old"),
                    (RE_AGE_Y, "age-y")):
        for m in rx.finditer(scope):
            out.append((m.start(), float(m.group(1)), how))
    for rx, how, gi in ((RE_AGE_SEX, "age+sex", 1), (RE_SEX_AGE, "sex+age", 2),
                        (RE_AGE_SEXWORD, "age+sexword", 1),
                        (RE_SEXWORD_AGE, "sexword+age", 2)):
        for m in rx.finditer(scope):
            # guard: "fever of 99 F" is a temperature, not a 99-year-old
            if RE_TEMP_NEAR.search(scope[max(0, m.start() - 14):m.start()]):
                continue
            out.append((m.start(), float(m.group(gi)), how))
    return out


def extract_age(head: str, full: str) -> tuple[float | None, str | None]:
    """
    Return (age in years, how it was found) -- the EARLIEST age stated.

    Position beats pattern strength, which took three attempts to get right. A
    post titled "[23M NJ]" that later says "some random 19-year-old" was
    recorded as 19 twice over: first because the explicit patterns searched the
    whole post while the joined age+sex searched only the head, and then
    because "19-year-old" is a stronger pattern than "23M" even though it comes
    300 characters later. The subject is introduced first; other people are
    mentioned in passing afterwards. So take the first age the post states.
    """
    for scope in (head, full):
        cands = _age_candidates(scope)
        if cands:
            pos, years, how = min(cands, key=lambda c: c[0])
            v = _ok("age", years)
            if v is not None:
                return v, how
    return None, None


def relational_sex(head: str) -> tuple[str | None, int]:
    """Sex implied by a relative term, with the position of the match."""
    best = (None, 10**9)
    for sex, rx in RE_REL_SEX.items():
        m = rx.search(head)
        if m and m.start() < best[1]:
            best = (sex, m.start())
    return best


def extract_sex(head: str, full: str) -> tuple[str | None, str | None, bool]:
    trans = bool(RE_TRANS.search(full))
    m = RE_SEX_FIELD.search(full)
    if m:
        v = m.group(1).lower()
        return ("male" if v in ("m", "male") else
                "female" if v in ("f", "female") else "other"), "sex-field", trans
    for rx, gi, how in ((RE_AGE_SEX, 2, "age+sex"), (RE_SEX_AGE, 1, "sex+age")):
        for mm in rx.finditer(head):
            if RE_TEMP_NEAR.search(head[max(0, mm.start() - 14):mm.start()]):
                continue
            return ("male" if mm.group(gi).lower() == "m" else "female"), how, trans
    m = RE_SEX_WORD.search(head)
    if m:
        w = m.group(1).lower()
        return ("male" if w in ("male", "man", "boy") else "female"), "sex-word", trans
    return None, None, trans


def extract_height(full: str) -> tuple[float | None, str | None]:
    m = RE_CM.search(full)
    if m:
        return _ok("height_cm", float(m.group(1))), "cm"
    m = RE_FT_IN.search(full)
    if m:
        ft = int(m.group(1))
        inch = int(m.group(2)) if m.group(2) and int(m.group(2)) < 12 else 0
        return _ok("height_cm", (ft * 12 + inch) * IN_TO_CM), "ft-in"
    return None, None


def extract_weight(full: str) -> tuple[float | None, str | None]:
    m = RE_KG.search(full)
    if m:
        return _ok("weight_kg", float(m.group(1))), "kg"
    m = RE_LB.search(full)
    if m:
        return _ok("weight_kg", float(m.group(1)) * LB_TO_KG), "lb"
    return None, None


def extract_subject(head: str) -> str:
    third = RE_THIRD.search(head)
    first = RE_FIRST.search(head)
    if third and not first:
        return "third_party"
    if first and not third:
        return "first_person"
    if third and first:
        # whichever is introduced first usually owns the demographics
        return "first_person" if first.start() < third.start() else "third_party"
    return "unknown"


def tri(pos, neg) -> str | None:
    if neg:
        return "no"
    if pos:
        return "yes"
    return None


def extract(title: str, body: str) -> dict:
    full = f"{title} || {body}"
    head = f"{title} || {body[:HEAD_CHARS]}"
    age, age_src = extract_age(head, full)
    sex, sex_src, trans = extract_sex(head, full)
    subject = extract_subject(head)
    if sex is None and subject == "third_party":
        sex, _ = relational_sex(head)
        sex_src = "relation" if sex else None
    h, h_src = extract_height(full)
    w, w_src = extract_weight(full)
    bmi = _ok("bmi", w / (h / 100) ** 2) if (h and w) else None
    race = RE_RACE_FIELD.search(full)
    return {
        "subject": subject,
        "age_years": round(age, 2) if age is not None else "",
        "age_source": age_src or "",
        "sex": sex or "",
        "sex_source": sex_src or "",
        "gender_minority_mention": "yes" if trans else "",
        "height_cm": round(h, 1) if h else "",
        "height_source": h_src or "",
        "weight_kg": round(w, 1) if w else "",
        "weight_source": w_src or "",
        "bmi": round(bmi, 1) if bmi else "",
        "smoker": tri(RE_SMOKER.search(full), RE_NONSMOKER.search(full)) or "",
        "drinker": tri(RE_DRINK.search(full), RE_NONDRINK.search(full)) or "",
        "race_stated": (race.group(1).strip().rstrip(".,").title() if race else ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract self-reported demographics from opening posts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--labels", default="../../output/corpora/export/topic_labels.csv")
    ap.add_argument("--corpora-dir", default="../../../Piloting/Round 2")
    ap.add_argument("--out", default="poster_demographics.csv")
    ap.add_argument("--results-dir", default=".")
    ap.add_argument("--audit", type=int, default=0,
                    help="print this many random posts with what was extracted, "
                         "to hand-check precision")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    labels = {r["submission_id"]: r["discipline"]
              for r in csv.DictReader(open(args.labels))}
    if not labels:
        sys.exit(f"ERROR: no labels in {args.labels}")
    posts = {}
    with open(Path(args.corpora_dir) / "submissions_corpus.jsonl") as f:
        for line in f:
            s = json.loads(line)
            if s["id"] in labels:
                posts[s["id"]] = (s.get("title") or "", s.get("selftext") or "")
    print(f"threads: {len(posts):,}")

    rows = []
    for sid in sorted(posts):
        t, b = posts[sid]
        rows.append({"submission_id": sid, "topic": labels[sid], **extract(t, b)})

    if args.audit:
        rng = random.Random(args.seed)
        print(f"\n=== audit sample of {args.audit} ===")
        for r in rng.sample(rows, args.audit):
            t, b = posts[r["submission_id"]]
            txt = re.sub(r"\s+", " ", f"{t} || {b}")[:150]
            got = " ".join(f"{k}={r[k]}" for k in
                           ("subject", "age_years", "sex", "height_cm", "weight_kg")
                           if r[k] != "")
            print(f"\n  {txt}\n    -> {got}")
        return

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows):,} rows)\n")

    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    FIELDS = ["age_years", "sex", "height_cm", "weight_kg", "bmi",
              "smoker", "drinker", "race_stated"]

    print("=== coverage: how often is each field stated? ===")
    cov = {}
    for k in FIELDS:
        n = sum(1 for r in rows if r[k] != "")
        cov[k] = n
        print(f"  {k:<16} {n:>5}  {100*n/len(rows):>5.1f}%")

    print("\n=== whose demographics are they? ===")
    for s, n in Counter(r["subject"] for r in rows).most_common():
        print(f"  {s:<16} {n:>5}  {100*n/len(rows):>5.1f}%")

    # Coverage by topic: if missingness tracks the subject, the covariate is
    # missing-not-at-random and that has to be reported, not discovered later.
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    with open(res / "results_coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["topic", "threads"] + [f"{k}_coverage" for k in FIELDS])
        w.writerow(["ALL", len(rows)] + [round(cov[k] / len(rows), 4) for k in FIELDS])
        for t, xs in sorted(by_topic.items(), key=lambda kv: -len(kv[1])):
            w.writerow([t, len(xs)] +
                       [round(sum(1 for r in xs if r[k] != "") / len(xs), 4)
                        for k in FIELDS])

    print("\n=== age and sex coverage by topic (the missingness worth knowing) ===")
    print(f"  {'topic':<42} {'n':>5} {'age':>7} {'sex':>7} {'wt':>7}")
    for t, xs in sorted(by_topic.items(), key=lambda kv: -len(kv[1])):
        f_ = lambda k: 100 * sum(1 for r in xs if r[k] != "") / len(xs)
        print(f"  {t[:42]:<42} {len(xs):>5} {f_('age_years'):>6.1f}% "
              f"{f_('sex'):>6.1f}% {f_('weight_kg'):>6.1f}%")

    print("\n=== distributions ===")
    dist_rows = []
    for k in ("age_years", "height_cm", "weight_kg", "bmi"):
        v = sorted(float(r[k]) for r in rows if r[k] != "")
        if not v:
            continue
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        print(f"  {k:<12} n={len(v):>5}  median {statistics.median(v):>6.1f}  "
              f"p10 {q(.10):>6.1f}  p90 {q(.90):>6.1f}  min {v[0]:>5.1f}  max {v[-1]:>6.1f}")
        dist_rows.append({"field": k, "n": len(v),
                          "median": round(statistics.median(v), 1),
                          "p10": round(q(.10), 1), "p90": round(q(.90), 1),
                          "min": round(v[0], 1), "max": round(v[-1], 1)})
    for k in ("sex", "smoker", "drinker", "subject"):
        c = Counter(r[k] for r in rows if r[k] != "")
        tot = sum(c.values())
        print(f"  {k:<12} " + "  ".join(f"{a} {100*b/tot:.0f}%" for a, b in c.most_common(4)))
        for a, b in c.most_common():
            dist_rows.append({"field": k, "n": b, "median": a, "p10": "",
                              "p90": round(b / tot, 4), "min": "", "max": ""})
    with open(res / "results_distributions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["field", "n", "median", "p10", "p90", "min", "max"])
        w.writeheader()
        w.writerows(dist_rows)
    print(f"\nwrote {res/'results_coverage.csv'} and {res/'results_distributions.csv'}")


if __name__ == "__main__":
    main()
