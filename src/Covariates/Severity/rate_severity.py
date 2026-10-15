#!/usr/bin/env python3
"""
Rate the clinical severity of the problem described in each opening post.

A moderator for the empathy analysis: an abscessed tooth plausibly draws more
empathy than a cavity, and that distinction is clinical, not tonal.

Why not sentiment
-----------------
Sentiment measures how upset the writing sounds, which in this corpus is nearly
orthogonal to how serious the problem is. Of the posts that read as frightened,
only 4.7% contain a clinical red flag; of the posts that do contain one, only
14% sound frightened. The failure is systematic in both directions -- people
who are genuinely very sick, medically literate, or writing about an infant
tend to write calmly. So the prompt below says, in as many words, to rate the
condition and ignore the tone.

Three signals, deliberately independent
---------------------------------------
severity_level   An LLM rating on a four-point triage scale (plus
                 not_applicable). The primary measure, because the distinction
                 that matters sits mid-range: a dental abscess and a cavity are
                 both invisible to red-flag rules, and only 3% of the corpus has
                 a red flag at all, so rules alone would put 97% of threads in
                 one bucket.

red_flag,        Rule-based, high precision, no model. A cross-check on the top
care_escalated   of the LLM scale rather than a measure in itself: red_flag
                 catches named emergencies, care_escalated catches posts where a
                 clinician has already acted (an ER visit, an ordered scan, a
                 referral), which is the closest thing here to an external
                 judgement of seriousness.

hours_to_first_  Behaviour rather than text: how long the subreddit took to
response,        answer, and who answered. Confounded by time of day and topic
first_responder  popularity, so it is a triangulation signal, never a primary
                 one -- but it is the only measure here not derived from the
                 post's own words.

Usage
-----
  python rate_severity.py --dry-run          # cost estimate, no API calls
  python rate_severity.py                    # rate, resuming if interrupted
  python rate_severity.py --rules-only       # skip the API entirely
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]
# Reuse the API plumbing from the topic labeller rather than copying it: the
# retry, backoff and empty-response handling there were debugged against a real
# 13,000-call run, and a second hand-rolled copy would not be.
sys.path.insert(0, str(SRC / "Covariates" / "Topics"))
sys.path.insert(0, str(SRC / "Generation"))
from classify_topics import Checkpoint, Usage, call_json, verify_quote, _confidence
from generate_synthetic import PROVIDERS

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

LEVELS = ("self_care", "routine", "urgent", "emergency", "not_applicable")
LEVEL_NUM = {"self_care": 1, "routine": 2, "urgent": 3, "emergency": 4,
             "not_applicable": ""}

# --- rule-based signals ----------------------------------------------------
RED_FLAGS = {
    "gi_bleed": r"coffee[- ]ground|hematemesis|haematemesis|vomiting blood|melena|black tarry",
    "cardiac": r"crushing chest pain|chest pain radiat|chest pain.{0,30}(?:arm|jaw)",
    "stroke": r"face droop|slurred speech|sudden.{0,20}weakness on one side|stroke symptom",
    "airway": r"anaphyla|can'?t breathe|cannot breathe|throat clos|unable to breathe",
    "sepsis": r"\bsepsis\b|\bseptic\b",
    "neuro": r"worst headache of my life|sudden vision loss|unresponsive",
    "self_harm": r"suicidal|kill myself|end my life",
    "haemoptysis": r"coughing up blood|hemoptysis",
}
RE_REDFLAG = {k: re.compile(v, re.I) for k, v in RED_FLAGS.items()}

CARE_ESCALATED = {
    "emergency_dept": r"\b(?:in|at|went to|going to|visited|taken to) (?:the )?(?:er|ed|a&e|emergency room|emergency department)\b|\bambulance\b|\b911\b|\b999\b",
    "admitted": r"\b(?:was |been )?(?:hospitali[sz]ed|admitted to (?:the )?hospital|in hospital)\b|\bicu\b",
    "imaging_ordered": r"\b(?:ordered|scheduled|had|getting) (?:an? )?(?:mri|ct scan|ct\b|ultrasound|x-?ray|biopsy|endoscopy|colonoscopy)\b",
    "referral": r"\breferred (?:me |him |her |to )|\breferral to\b|\bspecialist appointment\b",
    "urgent_advised": r"\btold (?:me )?to go to (?:the )?(?:er|ed|a&e|hospital)\b|\bsaid (?:it'?s |this is )?(?:an )?emergenc",
}
RE_CARE = {k: re.compile(v, re.I) for k, v in CARE_ESCALATED.items()}

RE_ACUTE = re.compile(r"\bsudden(?:ly)?\b|\bout of nowhere\b|\bwoke up (?:to|with)\b"
                      r"|\b(?:this|last) (?:morning|night|evening)\b|\bfew hours ago\b"
                      r"|\btoday\b|\byesterday\b", re.I)
RE_CHRONIC = re.compile(r"\bfor (?:the (?:past|last) )?(?:\d+|a few|several|many)\s*"
                        r"(?:weeks?|months?|years?)\b|\bchronic\b|\bongoing for\b"
                        r"|\bsince (?:\d{4}|childhood)\b", re.I)


def rule_signals(text: str) -> dict:
    flags = [k for k, rx in RE_REDFLAG.items() if rx.search(text)]
    care = [k for k, rx in RE_CARE.items() if rx.search(text)]
    acute, chronic = bool(RE_ACUTE.search(text)), bool(RE_CHRONIC.search(text))
    return {
        "red_flag": "yes" if flags else "no",
        "red_flag_kinds": ";".join(flags),
        "care_escalated": "yes" if care else "no",
        "care_escalated_kinds": ";".join(care),
        "onset": "acute" if acute and not chronic else
                 "chronic" if chronic and not acute else
                 "both" if acute and chronic else "",
    }


# --- the prompt ------------------------------------------------------------

SCALE = """1. "self_care" -- no clinician needed. Reassurance, an over-the-counter
   remedy, or waiting it out is the right answer. A mild cold, a small bruise, a
   question about whether a supplement is safe.
2. "routine" -- should be seen, but in the coming days or weeks. A persistent
   rash, a dental cavity, a long-standing mild symptom, a medication question
   that needs a prescriber.
3. "urgent" -- should be assessed today or tomorrow. A dental abscess, a
   suspected fracture, an infection with fever, a new lump that needs working up,
   a medication reaction that is worsening.
4. "emergency" -- needs care immediately, now. Chest pain radiating to the arm,
   vomiting blood, anaphylaxis, stroke signs, active suicidal intent.
5. "not_applicable" -- the post describes no medical problem: a question about
   the subreddit, about how to talk to a doctor, or about medical careers."""


def build_prompt(title: str, selftext: str) -> str:
    return (
        "You are triaging a post from r/AskDocs by how serious the medical "
        "problem is.\n\n"
        f"TITLE: {title}\n\nBODY: {selftext}\n\n"
        "Choose the one level that best describes what the person described "
        "actually needs:\n\n"
        f"{SCALE}\n\n"
        "Rules:\n"
        "- **Rate the condition, not the tone.** How worried the writing sounds "
        "is irrelevant. A calmly written account of vomiting blood is an "
        "emergency; a panicked question about having touched something dirty is "
        "self_care. People who are genuinely very unwell, or who are medically "
        "trained, or who are writing about their baby, often write calmly.\n"
        "- Rate what the post describes, not the worst thing it could turn out "
        "to be. Almost any symptom could be something terrible; judge what is "
        "actually reported.\n"
        "- If a clinician has already acted -- an ER visit, an ordered scan, a "
        "referral -- that is evidence the problem is at least urgent, unless the "
        "post says the result was reassuring.\n"
        "- Judge the title and body together.\n\n"
        "Return one JSON object and nothing else:\n"
        "{\n"
        '  "label": "<one of: self_care, routine, urgent, emergency, not_applicable>",\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "evidence_quote": "<ONE short span copied character-for-character from '
        'the title or body that decided it; a single continuous run of text, no '
        'ellipsis, no TITLE:/BODY: labels>"\n'
        "}\n"
    )


# --- loading ---------------------------------------------------------------

def load(labels_csv: Path, corpora: Path):
    labels = {r["submission_id"]: r["discipline"]
              for r in csv.DictReader(open(labels_csv))}
    posts, created = {}, {}
    with open(corpora / "submissions_corpus.jsonl") as f:
        for line in f:
            s = json.loads(line)
            if s["id"] in labels:
                posts[s["id"]] = (s.get("title") or "", s.get("selftext") or "")
                created[s["id"]] = s.get("created_utc") or 0
    # time to the first non-OP top-level comment: the behavioural signal
    op_author = {}
    with open(corpora / "submissions_corpus.jsonl") as f:
        for line in f:
            s = json.loads(line)
            if s["id"] in labels:
                op_author[s["id"]] = s.get("author")
    first = {}
    with open(corpora / "comments_corpus.jsonl") as f:
        for line in f:
            c = json.loads(line)
            pid = c.get("parent_id", "")
            if not pid.startswith("t3_"):
                continue
            sid = pid[3:]
            if sid not in labels or c.get("body") in ("[removed]", "[deleted]"):
                continue
            if c.get("author") and c["author"] == op_author.get(sid):
                continue
            t = c.get("created_utc") or 0
            if sid not in first or t < first[sid]:
                first[sid] = t
    latency = {}
    for sid, t in first.items():
        if created.get(sid) and t > created[sid]:
            latency[sid] = round((t - created[sid]) / 3600, 2)
    return labels, posts, latency


# --- the API loop ----------------------------------------------------------

async def rate_one(client, cfg, sid, title, body, sem, usage, ckpt, pbar):
    payload = await call_json(client, cfg, build_prompt(title, body), sem, usage)
    if payload is None:
        usage.skipped += 1
        if pbar:
            pbar.update(1)
        return
    raw = str(payload.get("label", "")).strip().lower().replace(" ", "_").replace("-", "_")
    level = raw if raw in LEVELS else "not_applicable"
    if raw not in LEVELS:
        usage.off_list += 1
    quote = str(payload.get("evidence_quote", "") or "")
    await ckpt.write({
        "submission_id": sid, "model": cfg.name, "model_id": cfg.model_id,
        "severity_level": level, "label_on_list": raw in LEVELS,
        "confidence": _confidence(payload), "evidence_quote": quote,
        "evidence_quote_verified": verify_quote(quote, title, body),
    })
    if pbar:
        pbar.update(1)


async def run_model(cfg, todo, posts, out_dir, concurrency):
    from openai import AsyncOpenAI
    key = os.environ.get(cfg.api_key_env)
    if not key:
        sys.exit(f"ERROR: ${cfg.api_key_env} is not set")
    kwargs = {"api_key": key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    client = AsyncOpenAI(**kwargs, timeout=120.0, max_retries=0)
    ckpt = Checkpoint(out_dir / f"severity_{cfg.name}.jsonl")
    todo = [s for s in todo if s not in ckpt.done]
    print(f"  {len(todo):,} to rate ({len(ckpt.done):,} already done)")
    if not todo:
        await client.close()
        return Usage(), ckpt
    usage = Usage()
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(todo), unit="op", desc="severity") if tqdm else None
    try:
        await asyncio.gather(*[
            rate_one(client, cfg, s, posts[s][0], posts[s][1], sem, usage, ckpt, pbar)
            for s in todo])
    finally:
        if pbar:
            pbar.close()
        await client.close()
    return usage, ckpt


def main_sync(args, labels, posts, latency, ratings):
    rows = []
    for sid in sorted(posts):
        t, b = posts[sid]
        r = ratings.get(sid, {})
        lvl = r.get("severity_level", "")
        rows.append({
            "submission_id": sid, "topic": labels[sid],
            "severity_level": lvl,
            "severity_score": LEVEL_NUM.get(lvl, ""),
            "severity_confidence": r.get("confidence", "") or "",
            "evidence_quote_verified": r.get("evidence_quote_verified", ""),
            **rule_signals(f"{t} || {b}"),
            "hours_to_first_response": latency.get(sid, ""),
        })
    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows):,} rows)")

    res = Path(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)
    rated = [r for r in rows if r["severity_level"]]

    print("\n=== rule-based signals (no model) ===")
    for k in ("red_flag", "care_escalated"):
        n = sum(1 for r in rows if r[k] == "yes")
        print(f"  {k:<16} {n:>5}  {100*n/len(rows):>5.1f}%")
    for k, n in Counter(r["onset"] for r in rows if r["onset"]).most_common():
        print(f"  onset {k:<10} {n:>5}  {100*n/len(rows):>5.1f}%")

    if not rated:
        print("\nNo LLM ratings yet; rule signals and latency are written.")
        return

    print(f"\n=== severity distribution (n={len(rated):,}) ===")
    dist = Counter(r["severity_level"] for r in rated)
    with open(res / "results_severity_distribution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "score", "threads", "share"])
        for lvl in LEVELS:
            n = dist.get(lvl, 0)
            print(f"  {lvl:<16} {n:>5}  {100*n/len(rated):>5.1f}%")
            w.writerow([lvl, LEVEL_NUM[lvl], n, round(n / len(rated), 4)])

    # Does the rating agree with the two independent signals?
    print("\n=== cross-check: do the independent signals agree? ===")
    xrows = []
    for lvl in LEVELS:
        xs = [r for r in rated if r["severity_level"] == lvl]
        if not xs:
            continue
        rf = 100 * sum(1 for r in xs if r["red_flag"] == "yes") / len(xs)
        ce = 100 * sum(1 for r in xs if r["care_escalated"] == "yes") / len(xs)
        lat = [float(r["hours_to_first_response"]) for r in xs
               if r["hours_to_first_response"] != ""]
        med = statistics.median(lat) if lat else float("nan")
        print(f"  {lvl:<16} red_flag {rf:>5.1f}%   care_escalated {ce:>5.1f}%   "
              f"median hours to reply {med:>6.2f}")
        xrows.append({"level": lvl, "threads": len(xs),
                      "red_flag_pct": round(rf, 1), "care_escalated_pct": round(ce, 1),
                      "median_hours_to_first_response": round(med, 2) if lat else ""})
    with open(res / "results_cross_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(xrows[0]))
        w.writeheader()
        w.writerows(xrows)

    # Severity by topic, and crucially the spread WITHIN a topic, since that is
    # what separates a severity effect from a relabelled topic effect.
    print("\n=== mean severity by topic, and the spread within it ===")
    by_t = defaultdict(list)
    for r in rated:
        if r["severity_score"] != "":
            by_t[r["topic"]].append(int(r["severity_score"]))
    trows = []
    for t, v in sorted(by_t.items(), key=lambda kv: -statistics.mean(kv[1])):
        sd = statistics.pstdev(v) if len(v) > 1 else 0.0
        print(f"  {t[:42]:<42} n={len(v):>4}  mean {statistics.mean(v):>4.2f}  sd {sd:>4.2f}")
        trows.append({"topic": t, "threads": len(v),
                      "mean_severity": round(statistics.mean(v), 2),
                      "sd_within_topic": round(sd, 2)})
    with open(res / "results_by_topic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0]))
        w.writeheader()
        w.writerows(trows)
    print(f"\nwrote three results tables to {res}")


async def main() -> None:
    ap = argparse.ArgumentParser(
        description="Rate clinical severity of each opening post.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--labels", default="../../output/corpora/export/topic_labels.csv")
    ap.add_argument("--corpora-dir", default="../../../Piloting/Round 2")
    ap.add_argument("--out", default="severity.csv")
    ap.add_argument("--results-dir", default=".")
    ap.add_argument("--labels-out-dir", default="../../output/corpora/labels",
                    help="where the resumable rating checkpoint is kept")
    ap.add_argument("--model", default="gemini")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="10 is the tested ceiling for this project")
    ap.add_argument("--rules-only", action="store_true",
                    help="skip the API; write rule signals and latency only")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    labels, posts, latency = load(Path(args.labels), Path(args.corpora_dir))
    print(f"threads: {len(posts):,}   with a response time: {len(latency):,}")

    if args.model not in PROVIDERS:
        sys.exit(f"ERROR: unknown model '{args.model}'")
    cfg = PROVIDERS[args.model]

    if args.dry_run:
        instr = len(build_prompt("", "")) / 4
        t_in = sum(len(t + b) / 4 + instr for t, b in posts.values())
        t_out = 60 * len(posts)
        cost = t_in / 1e6 * cfg.price_in + t_out / 1e6 * cfg.price_out
        print(f"\n--- estimate ---\n  {len(posts):,} calls, instruction block ~{instr:,.0f} tok")
        print(f"  {t_in/1e6:.1f}M in / {t_out/1e6:.1f}M out")
        print(f"  {cfg.model_id}: ~${cost:,.2f}\n\nDry run: no API calls made.")
        return

    ratings = {}
    out_dir = Path(args.labels_out_dir)
    if not args.rules_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        usage, ckpt = await run_model(cfg, sorted(posts), posts, out_dir, args.concurrency)
        if usage.calls:
            cost = usage.tokens_in / 1e6 * cfg.price_in + usage.tokens_out / 1e6 * cfg.price_out
            print(f"  {usage.calls:,} calls, {usage.failures} failed, {usage.empty} empty, "
                  f"{usage.skipped} skipped, {usage.off_list} off-list  ~${cost:,.2f}")
        with open(out_dir / f"severity_{cfg.name}.jsonl") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    ratings[r["submission_id"]] = r
    else:
        p = out_dir / f"severity_{cfg.name}.jsonl"
        if p.exists():
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        ratings[r["submission_id"]] = r

    main_sync(args, labels, posts, latency, ratings)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Ratings are checkpointed; re-run to resume.")
