#!/usr/bin/env python3
"""
Cancer flag, labeller B: scispaCy entity linking.

The second of the two labellers in the "targeted flags, dual-labelled" design in
Piloting/Round 5/topic_classification_plan.md. Labeller A is the LLM in
classify_topics.py; this one is deliberately a different kind of instrument, so
that agreement between them is informative rather than two views of the same
model's priors.

What it decides, and how well
-----------------------------
cancer_relevant  STRONG. Every entity in the post is linked to UMLS, and the post
                 is flagged if any linked concept carries semantic type T191
                 (Neoplastic Process). This is an ontology lookup, not a guess,
                 and it is what the kappa in agreement.py is really measuring.

stance           WEAK, and labelled as such everywhere it is reported. Current
                 patient vs survivor is a temporal and discourse distinction;
                 entity linking says nothing about it. All we can do without a
                 second language model is score surface cues ("in remission",
                 "5 years cancer free" vs "starting chemo next week"), which is
                 what _stance_from_cues does. Treat the stance kappa as a floor
                 on agreement, not as a validation of the stance labels. The
                 real check on stance is hand-coding, which the plan specifies
                 and which is not implemented here.

Install
-------
scispaCy is an optional dependency: nothing else in the pipeline needs it, and
the UMLS linker is a ~1 GB download gated behind a licence click-through.

    pip install scispacy spacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

The first run downloads and caches the linker index, which takes a while and a
few GB of RAM. UMLS requires a free UTS account; see
https://uts.nlm.nih.gov/uts/signup-login for the licence terms.

Usage
-----
  python classify_cancer_scispacy.py \
      --from-analysis-dataset "../Piloting/Round 5/output/corpora/analysis_dataset_6600.jsonl"

  # Check the install and the cancer-concept filter on a handful of posts
  python classify_cancer_scispacy.py --self-test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    import spacy
except ImportError:
    spacy = None

# UMLS semantic type for malignancy. T191 "Neoplastic Process" covers malignant
# neoplasms; benign lesions generally link to T190 (Anatomical Abnormality) or
# T033 (Finding), which is the behaviour we want, since the prompt given to
# labeller A also excludes benign tumours and cysts.
CANCER_TUIS = {"T191"}

# Concepts that carry T191 but are not a cancer diagnosis in the poster, and which
# would otherwise fire on routine posts. Checked by CUI so wording does not matter.
CANCER_CUI_BLOCKLIST = {
    "C0027651",  # Neoplasms (the generic parent concept)
    "C0598798",  # Neoplastic cell
}


# ---------------------------------------------------------------------------
# Stance cues  (weak, surface-level, and reported as such)
# ---------------------------------------------------------------------------

SURVIVOR_CUES = re.compile(
    r"\b("
    r"in remission|remission|cancer[- ]free|no evidence of disease|"
    r"survivor|survived|beat (?:the )?cancer|"
    r"(?:finished|completed|done with|after) (?:my )?(?:chemo\w*|radiation|radiotherapy|treatment)|"
    r"(?:\d+|one|two|three|four|five|ten) years? (?:post|after|out from|cancer[- ]free|in remission)|"
    r"surveillance|follow[- ]?up scans?|all clear"
    r")\b",
    re.IGNORECASE,
)

CURRENT_CUES = re.compile(
    r"\b("
    r"(?:just|recently|newly) diagnosed|"
    r"(?:currently|now) (?:on|undergoing|receiving|getting) (?:chemo\w*|radiation|immunotherapy|treatment)|"
    r"(?:starting|start|begin|beginning) (?:chemo\w*|radiation|immunotherapy|treatment)"
    r"|(?:next|this) (?:chemo|infusion|cycle|round)|"
    r"stage (?:III|IV|3|4)\b|metastatic|metastasi[sz]ed|spread to|"
    r"on (?:chemo\w*|immunotherapy)|mid[- ]treatment|active (?:disease|cancer)|"
    r"recurren(?:ce|t)|relapsed?"
    r")\b",
    re.IGNORECASE,
)

CANCER_TERMS = (
    r"cancer|carcinoma|sarcoma|lymphoma|leuka?emia|melanoma|myeloma|tumou?r|"
    r"chemo\w*|radiotherapy|immunotherapy|remission|mastectomy|lumpectomy"
)

# Does the POSTER claim the cancer history as their own? This is what decides
# whether a third-party cue ("my mum was diagnosed") wins. Two ways to qualify:
# a first-person pronoun with a verb, followed closely by a cancer term; or a
# possessive attached directly to a cancer noun. "my mum" does not qualify,
# which is the whole point. Requiring a verb after the pronoun keeps "stage I
# breast cancer" from reading as first person.
FIRST_PERSON_OWNERSHIP = re.compile(
    r"\bi\s*(?:'m|'ve|am|was|were|have|has|had|got|finished|completed|started|"
    rf"beat|underwent|survived)\b[^.?!]{{0,60}}?(?:{CANCER_TERMS})"
    rf"|\bmy\s+(?:{CANCER_TERMS}|oncologist|diagnosis|biopsy results)\b",
    re.IGNORECASE,
)

# Someone else's cancer, not the poster's.
THIRD_PARTY_CUES = re.compile(
    r"\b(my (?:mum|mom|mother|dad|father|wife|husband|partner|son|daughter|sister|"
    r"brother|friend|grandmother|grandfather|aunt|uncle|cousin|patient)|"
    r"(?:he|she|they) (?:was|were|has|have|is|are) (?:just )?diagnosed)\b",
    re.IGNORECASE,
)

# No diagnosis yet: worried-well, screening, awaiting a first result.
UNDIAGNOSED_CUES = re.compile(
    r"\b(could (?:this|it) be cancer|is (?:this|it) cancer|do i have cancer|"
    r"worried (?:about|it(?:'s| is)) cancer|scared it(?:'s| is) cancer|"
    r"waiting (?:on|for) (?:the )?(?:biopsy|results?|pathology)|"
    r"biopsy (?:is )?(?:scheduled|next|pending)|screening|"
    r"family history of cancer|risk of cancer)\b",
    re.IGNORECASE,
)


def _stance_from_cues(text: str) -> tuple[str, list[str]]:
    """
    Crude surface-cue stance. Returns (stance, cues_matched).

    Order matters. A third-party cue wins unless the poster also claims a cancer
    history of their own: "my mum was just diagnosed" carries a current-treatment
    cue ("just diagnosed") that would otherwise make it look like the poster's
    own active disease. A cancer nobody has diagnosed yet is likewise
    not_applicable. Only then do the stance cues apply, and an explicit
    current-treatment cue outranks a survivorship cue, because posts often
    recite a treatment history before describing a current problem.
    """
    cues: list[str] = []

    owns = FIRST_PERSON_OWNERSHIP.search(text)
    third = [m.group(0) for m in THIRD_PARTY_CUES.finditer(text)]
    undiag = [m.group(0) for m in UNDIAGNOSED_CUES.finditer(text)]
    current = [m.group(0) for m in CURRENT_CUES.finditer(text)]
    survivor = [m.group(0) for m in SURVIVOR_CUES.finditer(text)]

    if third and not owns:
        return "not_applicable", third[:5]
    if undiag and not (current or survivor):
        return "not_applicable", undiag[:5]
    if current:
        cues = current[:5]
        return "current_patient", cues
    if survivor:
        cues = survivor[:5]
        return "survivor", cues
    return "unclear", []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_nlp(model_name: str, linker: str):
    if spacy is None:
        sys.exit(
            "ERROR: spaCy/scispaCy are not installed. This labeller is optional;\n"
            "       the rest of the pipeline runs without it.\n\n"
            "  pip install scispacy spacy\n"
            "  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            "releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz\n"
        )
    try:
        import scispacy  # noqa: F401  (registers the pipe factories)
        from scispacy.linking import EntityLinker  # noqa: F401
    except ImportError:
        sys.exit(
            "ERROR: the 'scispacy' package is missing (spaCy alone is not enough).\n"
            "       pip install scispacy\n"
        )

    try:
        nlp = spacy.load(model_name)
    except OSError:
        sys.exit(
            f"ERROR: spaCy model '{model_name}' is not installed.\n"
            "       pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            f"releases/v0.5.4/{model_name}-0.5.4.tar.gz\n"
        )

    print(f"Loading the {linker} linker (first run downloads ~1 GB and is slow)...")
    nlp.add_pipe(
        "scispacy_linker",
        config={"resolve_abbreviations": True, "linker_name": linker, "max_entities_per_mention": 3},
    )
    return nlp


def cancer_concepts(doc, kb, threshold: float) -> list[dict]:
    """Linked concepts in this doc that are malignancies, above the score threshold."""
    hits: dict[str, dict] = {}
    for ent in doc.ents:
        for cui, score in getattr(ent._, "kb_ents", []):
            if score < threshold or cui in CANCER_CUI_BLOCKLIST:
                continue
            entry = kb.cui_to_entity.get(cui)
            if entry is None or not (set(entry.types) & CANCER_TUIS):
                continue
            prev = hits.get(cui)
            if prev is None or score > prev["score"]:
                hits[cui] = {
                    "cui": cui,
                    "name": entry.canonical_name,
                    "mention": ent.text,
                    "score": round(float(score), 3),
                }
    return sorted(hits.values(), key=lambda h: -h["score"])


def load_ops(args) -> list[dict]:
    if args.from_analysis_dataset:
        path = Path(args.from_analysis_dataset)
        if not path.exists():
            sys.exit(f"ERROR: analysis dataset not found: {path}")
        ops = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    ops.append(
                        {
                            "id": r["submission_id"],
                            "title": r.get("title") or "",
                            "selftext": r.get("selftext") or "",
                        }
                    )
        print(f"Loaded {len(ops):,} OPs from {path.name}")
        return ops

    # Fall back to the corpus, reusing the generation run's filters and sample.
    from generate_synthetic import load_corpus, select_sample

    eligible, _ = load_corpus(Path(args.corpora_dir))
    sample = select_sample(eligible, args.n_ops, args.seed)
    print(f"Selected {len(sample):,} OPs (seed={args.seed}).")
    return [
        {"id": s["id"], "title": s.get("title") or "", "selftext": s.get("selftext") or ""}
        for s in sample
    ]


SELF_TEST_POSTS = [
    ("Finished chemo last year", "I had breast cancer, finished chemo in 2024 and I'm in remission now. Is this fatigue normal?"),
    ("Just diagnosed", "I was just diagnosed with stage IV colon cancer and start chemo next week."),
    ("Worried about a mole", "This mole changed shape. Could this be melanoma? I have a biopsy scheduled."),
    ("My mum's lymphoma", "My mum was just diagnosed with lymphoma, what questions should she ask?"),
    ("Sore throat", "I've had a sore throat for three days and a mild fever."),
]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cancer flag labeller B: scispaCy UMLS entity linking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--from-analysis-dataset", default=None, metavar="PATH")
    ap.add_argument("--corpora-dir", default="../Piloting/Round 5/output/corpora")
    ap.add_argument("--out-dir", default="../Piloting/Round 5/output/corpora/labels")
    ap.add_argument("--spacy-model", default="en_core_sci_sm",
                    help="en_core_sci_sm is enough for entity spans; en_core_sci_md "
                         "detects slightly more at ~5x the size")
    ap.add_argument("--linker", default="umls", choices=("umls", "mesh"),
                    help="UMLS gives semantic types directly; mesh is smaller")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="minimum linker score for a concept to count")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-ops", type=int, default=6600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true",
                    help="run on five built-in posts and print the labels, to check "
                         "the install and the concept filter")
    args = ap.parse_args()

    nlp = build_nlp(args.spacy_model, args.linker)
    linker = nlp.get_pipe("scispacy_linker")
    kb = linker.kb

    if args.self_test:
        print("\n--- self test ---")
        for title, body in SELF_TEST_POSTS:
            text = f"{title}. {body}"
            doc = nlp(text)
            hits = cancer_concepts(doc, kb, args.threshold)
            stance, cues = _stance_from_cues(text)
            print(f"\n{title!r}")
            print(f"  cancer_relevant: {bool(hits)}")
            print(f"  concepts: {[h['name'] for h in hits][:3]}")
            print(f"  stance (weak): {stance}  cues={cues}")
        return

    ops = load_ops(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "labels_scispacy_cancer.jsonl"

    # Resume support, to match the rest of the pipeline.
    done: set[str] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["submission_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = [o for o in ops if o["id"] not in done]
    print(f"{len(ops):,} OPs in scope, {len(done):,} already done, {len(todo):,} to go")
    if not todo:
        return

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    texts = [f"{o['title']}. {o['selftext']}" for o in todo]
    started = time.time()
    n_relevant = 0
    stance_counts: dict[str, int] = {}

    with open(out_path, "a") as out:
        stream = nlp.pipe(texts, batch_size=args.batch_size)
        if tqdm:
            stream = tqdm(stream, total=len(texts), unit="op", desc="scispacy")
        for op, text, doc in zip(todo, texts, stream):
            hits = cancer_concepts(doc, kb, args.threshold)
            relevant = bool(hits)
            if relevant:
                n_relevant += 1
                stance, cues = _stance_from_cues(text)
            else:
                stance, cues = "not_applicable", []
            stance_counts[stance] = stance_counts.get(stance, 0) + 1
            out.write(
                json.dumps(
                    {
                        "submission_id": op["id"],
                        "task": "cancer",
                        "labeller": "scispacy",
                        "spacy_model": args.spacy_model,
                        "linker": args.linker,
                        "cancer_relevant": relevant,
                        "label": stance,
                        "stance_is_cue_based": True,
                        "concepts": hits[:5],
                        "stance_cues": cues,
                    }
                )
                + "\n"
            )

    elapsed = time.time() - started
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"cancer_relevant: {n_relevant:,} of {len(todo):,} ({100*n_relevant/len(todo):.1f}%)")
    print("stance (cue-based, weak):")
    for k, v in sorted(stance_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<16} {v:>6,}")
    print(f"\nOutput: {out_path}")
    print("Next:   python agreement.py")


if __name__ == "__main__":
    main()
