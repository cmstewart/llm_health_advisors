#!/usr/bin/env python3
"""
Topic classification of r/AskDocs opening posts -- labeller A (LLM).

Implements the supervised half of Piloting/Round 5/topic_classification_plan.md:

  task "discipline"  Fixed taxonomy, LLM-classified. The model picks exactly one
                     of the 17 JMIR disciplines (Data/taxonomy.json), which carry
                     their own residual categories ("Medical; other", "Nonmedical").
                     It never generates a category freely.

  task "cancer"      Targeted flag, labeller A of two. Returns cancer relevance
                     and, for first-person cancer threads, the binary stance
                     current_patient vs survivor.

Both tasks return {label, confidence, evidence_quote} per the plan. The quote is
checked against the post text, so a quote the model invented is recorded as
unverified rather than taken on trust.

Labeller B (scispaCy entity linking) lives in classify_cancer_scispacy.py;
agreement.py joins the two, reports kappa, and writes the manual-coding queue.

The cancer stance definitions
-----------------------------
NCI and the NCCS define a "survivor" as anyone from diagnosis onward, which would
put every current patient in the survivor class and collapse the contrast we want.
We therefore use the restricted clinical sense, and say so in the prompt:

  current_patient  active malignancy, or in/awaiting primary treatment
  survivor         primary treatment complete, no evidence of active disease
  not_applicable   the cancer history is not the OP's own (caregiver, family),
                   or there is no diagnosis (worried-well, screening, risk question)
  unclear          a cancer history is present but the stance cannot be read

This is an adjudication decision, and the plan commits to publishing those: it is
restated verbatim in the prompt below so the prompt and the paper cannot drift.

OP set alignment
----------------
The OPs classified must be exactly the OPs in the analysis dataset, or the labels
will not join. Two ways to guarantee that:

  --from-analysis-dataset PATH   read the submission ids straight from the
                                 analysis dataset (preferred; exact by construction)
  --corpora-dir PATH             re-derive them from the corpus, reusing
                                 generate_synthetic.load_corpus/select_sample so the
                                 filters and the seeded sample are identical

Usage
-----
  # Dry run: no API calls, prints workload + cost estimate
  python classify_topics.py --dry-run --n-ops 6600

  # Both tasks over the analysis dataset
  python classify_topics.py \
      --from-analysis-dataset "../Piloting/Round 5/output/corpora/analysis_dataset_6600.jsonl"

  # Discipline only, and with a second model as an LLM-vs-LLM check
  python classify_topics.py --tasks discipline --models gemini,openai

Environment variables:
  GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # only needed for live runs, not --dry-run

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# Reuse the generation run's corpus filters and seeded sampling verbatim, so the
# classified OP set cannot drift from the generated one.
from generate_synthetic import PROVIDERS, load_corpus, select_sample

TASKS = ("discipline", "cancer")

CANCER_STANCES = ("current_patient", "survivor", "not_applicable", "unclear")

PLACEHOLDER_RE = re.compile(r"^<TODO\b.*>$")


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

@dataclass
class Taxonomy:
    disciplines: list[str]
    other_label: str | None
    fallback_label: str
    residual_guidance: str
    cancer_node: str | None
    verified: bool
    placeholders: list[str] = field(default_factory=list)

    @property
    def choices(self) -> list[str]:
        """
        The closed list shown to the model.

        The JMIR taxonomy already ends in its own two residual categories,
        "Medical; other" and "Nonmedical", so other_label is null and nothing is
        appended: a third catch-all would just split the residue across three
        overlapping options. The append is still supported for a taxonomy that
        has no escape hatch of its own.
        """
        if self.other_label:
            return [*self.disciplines, self.other_label]
        return list(self.disciplines)


def load_taxonomy(path: Path) -> Taxonomy:
    if not path.exists():
        sys.exit(f"ERROR: taxonomy file not found: {path}")
    with open(path) as f:
        raw = json.load(f)

    disciplines = raw.get("disciplines") or []
    if not isinstance(disciplines, list) or not disciplines:
        sys.exit(f"ERROR: {path} has no 'disciplines' list")

    placeholders = [d for d in disciplines if PLACEHOLDER_RE.match(d.strip())]
    cancer_node = raw.get("cancer_node")
    if cancer_node is False:
        cancer_node = None

    other_label = raw.get("other_label") or None
    tax = Taxonomy(
        disciplines=disciplines,
        other_label=other_label,
        # Where an off-list label lands. Falls back to the appended catch-all for
        # a taxonomy that has one.
        fallback_label=raw.get("fallback_label") or other_label or "",
        residual_guidance=raw.get("residual_guidance", ""),
        cancer_node=cancer_node,
        verified=bool(raw.get("verified", False)),
        placeholders=placeholders,
    )
    if not tax.fallback_label:
        sys.exit(
            f"ERROR: {path} sets neither 'fallback_label' nor 'other_label'.\n"
            "       One is required: it is where a label outside the closed list goes."
        )
    if tax.fallback_label not in tax.choices:
        sys.exit(
            f"ERROR: {path} has fallback_label {tax.fallback_label!r}, which is not "
            "one of the categories offered to the model."
        )
    return tax


# ---------------------------------------------------------------------------
# Prompts
#
# Both prompts demand one JSON object, a label from a closed list, a 0-1
# confidence, and a verbatim quote from the post. Asking for the quote is not
# decoration: it is checked against the source text in verify_quote(), which
# turns "the model asserted X" into "the model pointed at the span it read X from".
# ---------------------------------------------------------------------------

def _post_block(title: str, selftext: str) -> str:
    return f"TITLE: {title}\n\nBODY: {selftext}"


def build_discipline_prompt(title: str, selftext: str, tax: Taxonomy) -> str:
    numbered = "\n".join(f"  {i}. {d}" for i, d in enumerate(tax.choices, 1))
    return (
        "You are classifying a post from r/AskDocs, a forum where people ask medical "
        "questions, by medical discipline.\n\n"
        f"{_post_block(title, selftext)}\n\n"
        "Choose the ONE discipline that best fits the post's primary medical subject "
        "matter, from this closed list:\n"
        f"{numbered}\n\n"
        "Rules:\n"
        "- You must copy one label from the list exactly. Do not invent a category.\n"
        + (f"- {tax.residual_guidance}\n" if tax.residual_guidance else "")
        + "- Classify by the post's own primary concern, not by every condition it mentions "
        "in passing.\n"
        "- Judge the title and body together. The subject is often only in the body.\n\n"
        "Return one JSON object and nothing else:\n"
        "{\n"
        '  "label": "<one label, copied exactly from the list>",\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "evidence_quote": "<a short verbatim span, copied character-for-character '
        'from the title or body, that decided it>"\n'
        "}\n"
    )


def build_cancer_prompt(title: str, selftext: str) -> str:
    return (
        "You are labelling a post from r/AskDocs for the poster's own cancer status.\n\n"
        f"{_post_block(title, selftext)}\n\n"
        "First decide whether the post involves cancer (a malignancy: carcinoma, sarcoma, "
        "lymphoma, leukaemia, melanoma, myeloma, or a named cancer of any organ). Benign "
        "tumours, cysts and non-neoplastic lumps are NOT cancer.\n\n"
        "Then assign exactly one stance. Note the definitions carefully: the usual broad "
        "definition of \"survivor\" (anyone from diagnosis onward) is NOT what we want "
        "here, because it would include people currently under treatment. Use these:\n\n"
        "- \"current_patient\": the POSTER has an active malignancy, or is undergoing or "
        "awaiting primary treatment for one (surgery, chemotherapy, radiotherapy, "
        "immunotherapy), or has known residual, recurrent or metastatic disease.\n"
        "- \"survivor\": the POSTER had a malignancy, primary treatment is complete, and "
        "there is no evidence of active disease (in remission, 'cancer free', 'no evidence "
        "of disease', on surveillance only). Long-term maintenance or endocrine therapy "
        "after primary treatment still counts as survivor.\n"
        "- \"not_applicable\": the cancer history is not the poster's own (they are asking "
        "about a parent, partner, child or patient), OR there is no diagnosis at all (they "
        "are worried they might have cancer, asking about screening, risk, or symptoms, or "
        "awaiting a first diagnostic result), OR the post does not involve cancer.\n"
        "- \"unclear\": the poster does have a cancer history, but the post does not say "
        "enough to tell whether disease or primary treatment is current.\n\n"
        "A first diagnostic biopsy that has not yet returned is not_applicable, not "
        "current_patient. Do not guess a stance from the fact that someone sounds worried.\n\n"
        "Return one JSON object and nothing else:\n"
        "{\n"
        '  "cancer_relevant": <true or false>,\n'
        '  "label": "<one of: current_patient, survivor, not_applicable, unclear>",\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "evidence_quote": "<a short verbatim span, copied character-for-character '
        'from the title or body, that decided it; empty string if not cancer-related>"\n'
        "}\n"
    )


# ---------------------------------------------------------------------------
# Evidence-quote verification
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().casefold()


def verify_quote(quote: str, title: str, selftext: str) -> bool:
    """
    True if the quote really occurs in the post, comparing on collapsed whitespace
    and case. An empty quote is not a match: callers record it as unverified.
    """
    q = _normalize(quote)
    if not q:
        return False
    return q in _normalize(f"{title} {selftext}")


# ---------------------------------------------------------------------------
# OP selection
# ---------------------------------------------------------------------------

def load_from_analysis_dataset(path: Path) -> list[dict]:
    """Read OPs straight from the analysis dataset, so the id set matches exactly."""
    if not path.exists():
        sys.exit(
            f"ERROR: analysis dataset not found: {path}\n"
            "       It is not in the repository (26.7 MB, shared via Google Drive).\n"
            "       Either place it there or use --corpora-dir to re-derive the sample."
        )
    ops: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
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


def load_from_corpus(corpora_dir: Path, n_ops: int, seed: int) -> list[dict]:
    eligible, _ = load_corpus(corpora_dir)
    sample = select_sample(eligible, n_ops, seed)
    print(f"Selected {len(sample):,} OPs (seed={seed}).")
    return [
        {
            "id": s["id"],
            "title": s.get("title") or "",
            "selftext": s.get("selftext") or "",
        }
        for s in sample
    ]


# ---------------------------------------------------------------------------
# Checkpointing  (append-only JSONL, one file per model x task)
# ---------------------------------------------------------------------------

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: set[str] = set()
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    try:
                        self.done.add(json.loads(line)["submission_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a truncated final line
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self.done.add(record["submission_id"])


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    calls: int = 0
    failures: int = 0
    skipped: int = 0        # OPs left unwritten because the call failed
    empty: int = 0          # responses with no content at all
    off_list: int = 0       # labels not in the closed list, coerced
    unverified_quote: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    finish_reasons: dict = field(default_factory=dict)


async def call_json(
    client,
    cfg,
    prompt: str,
    sem: asyncio.Semaphore,
    usage: Usage,
    retries: int = 5,
) -> dict | None:
    """
    Send one prompt and parse a single JSON object. Returns None on failure or on
    an empty response; the caller decides whether to checkpoint.

    Mirrors generate_synthetic.call_model: an empty response is not retried,
    because every attempt would send a byte-identical prompt and each retry is
    still billed. The finish_reason is tallied instead.
    """
    kwargs = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        # Classification, not generation: take the mode, not a sample.
        "temperature": 0.0,
    }
    if cfg.reasoning_effort:
        kwargs["reasoning_effort"] = cfg.reasoning_effort

    for attempt in range(retries):
        try:
            async with sem:
                resp = await client.chat.completions.create(**kwargs)
            usage.calls += 1
            if getattr(resp, "usage", None):
                usage.tokens_in += resp.usage.prompt_tokens or 0
                usage.tokens_out += resp.usage.completion_tokens or 0

            choice = resp.choices[0]
            content = getattr(choice.message, "content", None)
            if content is None or not content.strip():
                reason = getattr(choice, "finish_reason", None) or "unknown"
                usage.empty += 1
                usage.finish_reasons[reason] = usage.finish_reasons.get(reason, 0) + 1
                return None

            payload = json.loads(content)
            if not isinstance(payload, dict):
                raise ValueError(f"expected an object, got {type(payload).__name__}")
            return payload

        except Exception as e:  # noqa: BLE001 - retry on anything transient
            if attempt == retries - 1:
                usage.failures += 1
                print(f"  [{cfg.name}] giving up after {retries} attempts: {e}")
                return None
            await asyncio.sleep(min(2**attempt, 30) + random.random())
    return None


def _confidence(payload: dict) -> float | None:
    try:
        c = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return None
    return min(max(c, 0.0), 1.0)


def coerce_discipline(payload: dict, tax: Taxonomy, usage: Usage) -> tuple[str, bool]:
    """
    Map a returned label onto the closed list. Case- and whitespace-insensitive,
    since that is the whole of the drift we see in practice. Anything else becomes
    'Other' and is tallied as off_list rather than silently accepted.
    """
    raw = str(payload.get("label", "")).strip()
    lookup = {_normalize(c): c for c in tax.choices}
    hit = lookup.get(_normalize(raw))
    if hit is not None:
        return hit, True
    usage.off_list += 1
    return tax.fallback_label, False


def coerce_stance(payload: dict, usage: Usage) -> tuple[str, bool]:
    raw = _normalize(payload.get("label", "")).replace(" ", "_").replace("-", "_")
    if raw in CANCER_STANCES:
        return raw, True
    usage.off_list += 1
    return "unclear", False


# ---------------------------------------------------------------------------
# Per-OP work
# ---------------------------------------------------------------------------

async def process_op(
    client, cfg, task: str, op: dict, tax: Taxonomy, sem, usage, ckpt, pbar
) -> None:
    title, selftext = op["title"], op["selftext"]

    if task == "discipline":
        prompt = build_discipline_prompt(title, selftext, tax)
    else:
        prompt = build_cancer_prompt(title, selftext)

    payload = await call_json(client, cfg, prompt, sem, usage)

    if payload is None:
        # Not checkpointed: a failed OP stays on the to-do list for the next run
        # rather than being written as a bogus label. Same rule as the generation
        # script, for the same reason.
        usage.skipped += 1
        if pbar is not None:
            pbar.update(1)
        return

    quote = str(payload.get("evidence_quote", "") or "")
    quote_ok = verify_quote(quote, title, selftext)

    record = {
        "submission_id": op["id"],
        "task": task,
        "model": cfg.name,
        "model_id": cfg.model_id,
        "confidence": _confidence(payload),
        "evidence_quote": quote,
        "evidence_quote_verified": quote_ok,
    }

    if task == "discipline":
        label, on_list = coerce_discipline(payload, tax, usage)
        record["label"] = label
        record["label_on_list"] = on_list
    else:
        label, on_list = coerce_stance(payload, usage)
        relevant = payload.get("cancer_relevant")
        if not isinstance(relevant, bool):
            # Infer it from the stance rather than dropping the record.
            relevant = label in ("current_patient", "survivor", "unclear")
        # A stance of current_patient/survivor asserts a cancer history, so an
        # explicit cancer_relevant=false alongside one is self-contradictory.
        if label in ("current_patient", "survivor") and not relevant:
            relevant = True
            record["coerced_relevance"] = True
        record["label"] = label
        record["label_on_list"] = on_list
        record["cancer_relevant"] = relevant

    if not quote_ok and (task == "discipline" or record.get("cancer_relevant")):
        # An unverifiable quote on a substantive label is the signal worth counting:
        # for a not_applicable cancer label the prompt allows an empty quote.
        usage.unverified_quote += 1

    await ckpt.write(record)
    if pbar is not None:
        pbar.update(1)


async def run_model_task(
    cfg, task: str, ops: list[dict], tax: Taxonomy, out_dir: Path, concurrency: int
) -> Usage:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        print(f"SKIP {cfg.name}/{task}: ${cfg.api_key_env} not set")
        return Usage()
    if AsyncOpenAI is None:
        sys.exit("ERROR: the 'openai' package is required. pip install openai")

    ckpt = Checkpoint(out_dir / f"labels_{cfg.name}_{task}.jsonl")
    todo = [o for o in ops if o["id"] not in ckpt.done]

    print(
        f"\n=== {cfg.name} / {task} ===\n"
        f"  model: {cfg.model_id}\n"
        f"  {len(ops):,} OPs in scope, {len(ckpt.done):,} already done, "
        f"{len(todo):,} to go"
    )
    if not todo:
        return Usage()

    kwargs = {"api_key": api_key}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    client = AsyncOpenAI(**kwargs, timeout=120.0, max_retries=0)

    usage = Usage()
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(todo), desc=f"{cfg.name}/{task}", unit="op") if tqdm else None
    try:
        await asyncio.gather(
            *[
                process_op(client, cfg, task, o, tax, sem, usage, ckpt, pbar)
                for o in todo
            ]
        )
    finally:
        if pbar is not None:
            pbar.close()
        await client.close()

    if usage.empty:
        detail = ", ".join(f"{k}={v:,}" for k, v in sorted(usage.finish_reasons.items()))
        print(f"  empty responses: {usage.empty:,}  (finish_reason: {detail})")
    if usage.off_list:
        print(f"  off-list labels coerced: {usage.off_list:,}")
    if usage.unverified_quote:
        print(
            f"  unverifiable evidence quotes: {usage.unverified_quote:,} "
            f"({100*usage.unverified_quote/max(usage.calls,1):.1f}% of calls)"
        )

    return usage


# ---------------------------------------------------------------------------
# Dry run / estimation
# ---------------------------------------------------------------------------

def estimate(ops: list[dict], models: list[str], tasks: list[str]) -> None:
    n = len(ops)
    print("\n--- Workload estimate ---")
    print(f"OPs: {n:,}   tasks: {', '.join(tasks)}")
    print(f"  {n:,} calls per model per task")
    print(f"  {n*len(tasks):,} calls per model  x{len(models)} models "
          f"= {n*len(tasks)*len(models):,} total")

    # ~4 chars/token. The prompt carries the post plus its instruction block; the
    # taxonomy list makes the discipline prompt the longer of the two. Output is a
    # single small JSON object.
    INSTR = {"discipline": 420, "cancer": 520}
    OUT = 60
    t_in = t_out = 0.0
    for o in ops:
        post = len(o["title"] + o["selftext"]) / 4
        for t in tasks:
            t_in += post + INSTR[t]
            t_out += OUT

    print(f"\nEst. tokens/model: {t_in/1e6:.1f}M in / {t_out/1e6:.1f}M out")
    print("\nEst. cost (sync pricing):")
    total = 0.0
    for m in models:
        cfg = PROVIDERS[m]
        c = t_in / 1e6 * cfg.price_in + t_out / 1e6 * cfg.price_out
        total += c
        print(f"  {m:<8} ${c:>8,.2f}   ({cfg.model_id})")
    print(f"  {'TOTAL':<8} ${total:>8,.2f}")
    print("  (verify current per-token prices before a large run)")
    print("  NOTE: reasoning models bill hidden reasoning tokens as output, so a")
    print("        reasoning-heavy model can cost several times this. Classification")
    print("        does not need it: leave --reasoning-effort unset, or set it low.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _apply_overrides(args) -> None:
    for override in args.model_id or []:
        if "=" not in override:
            sys.exit(f"ERROR: --model-id expects name=value, got '{override}'")
        name, value = override.split("=", 1)
        if name not in PROVIDERS:
            sys.exit(f"ERROR: unknown model '{name}' in --model-id")
        PROVIDERS[name].model_id = value

    for override in args.reasoning_effort or []:
        if "=" not in override:
            sys.exit(f"ERROR: --reasoning-effort expects name=level, got '{override}'")
        name, value = override.split("=", 1)
        if name not in PROVIDERS:
            sys.exit(f"ERROR: unknown model '{name}' in --reasoning-effort")
        if value not in ("minimal", "low", "medium", "high"):
            sys.exit(f"ERROR: reasoning effort must be minimal/low/medium/high, got '{value}'")
        PROVIDERS[name].reasoning_effort = value


def _cancer_scope(
    ops: list[dict], scope: str, tax: Taxonomy, out_dir: Path, models: list[str]
) -> list[dict]:
    """
    Which OPs the cancer task runs over.

    'all' is the default and the methodologically safest: a prefilter can only
    lower recall, and the plan already rules keyword matching out as a labeller.
    At one cheap call per OP the saving is not worth a recall ceiling we would
    then have to argue about. 'oncology' and 'screen' exist for budget runs.
    """
    if scope == "all":
        return ops

    if scope == "oncology":
        if not tax.cancer_node:
            sys.exit(
                "ERROR: --cancer-scope oncology needs 'cancer_node' set in the taxonomy\n"
                "       file, and the discipline task to have been run first."
            )
        keep: set[str] = set()
        found = False
        for m in models:
            p = out_dir / f"labels_{m}_discipline.jsonl"
            if not p.exists():
                continue
            found = True
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("label") == tax.cancer_node:
                        keep.add(r["submission_id"])
        if not found:
            sys.exit(
                "ERROR: --cancer-scope oncology found no discipline labels in\n"
                f"       {out_dir}. Run --tasks discipline first."
            )
        sel = [o for o in ops if o["id"] in keep]
        print(f"Cancer scope 'oncology': {len(sel):,} of {len(ops):,} OPs "
              f"labelled {tax.cancer_node!r}.")
        return sel

    # scope == "screen": recall-oriented prefilter, never a label in its own right
    pattern = re.compile(
        r"\b(cancer|carcinoma|sarcoma|lymphoma|leuka?emia|melanoma|myeloma|"
        r"metasta(?:sis|tic|ses)|malignan(?:t|cy)|oncolog(?:y|ist)|chemo(?:therapy)?|"
        r"radiotherapy|immunotherapy|tumour|tumor|neoplasm|biopsy|remission|"
        r"in situ|mastectomy|lumpectomy)\b",
        re.IGNORECASE,
    )
    sel = [o for o in ops if pattern.search(f"{o['title']} {o['selftext']}")]
    print(
        f"Cancer scope 'screen': {len(sel):,} of {len(ops):,} OPs matched the term "
        "screen.\n"
        "  WARNING: this caps recall. Threads that describe a cancer history without\n"
        "  any of these terms are never seen by the labeller, and that loss is not\n"
        "  measurable from the output. Prefer --cancer-scope all for the real run."
    )
    return sel


async def main_async(args) -> None:
    tax = load_taxonomy(Path(args.taxonomy))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for m in models:
        if m not in PROVIDERS:
            sys.exit(f"ERROR: unknown model '{m}'. Choose from {list(PROVIDERS)}")
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"ERROR: unknown task '{t}'. Choose from {list(TASKS)}")
    _apply_overrides(args)

    if args.from_analysis_dataset:
        ops = load_from_analysis_dataset(Path(args.from_analysis_dataset))
    else:
        ops = load_from_corpus(Path(args.corpora_dir), args.n_ops, args.seed)
    if not ops:
        sys.exit("ERROR: no OPs selected.")

    # The discipline task cannot run against a taxonomy that is still placeholders:
    # the model would be offered "<TODO 05 of 17: ...>" as a category.
    if "discipline" in tasks and tax.placeholders:
        msg = (
            f"{len(tax.placeholders)} of {len(tax.disciplines)} disciplines in "
            f"{args.taxonomy} are still placeholders.\n"
            "       Fill them in from Table 2 of the source paper and set "
            "'verified': true."
        )
        if not args.dry_run:
            sys.exit(f"ERROR: {msg}")
        print(f"\nWARNING: {msg}")
    elif "discipline" in tasks and not tax.verified:
        print(
            f"\nWARNING: {args.taxonomy} has no placeholders left but 'verified' is "
            "still false.\n"
            "         Confirm the names are verbatim from Table 2, then set it true."
        )

    estimate(ops, models, tasks)

    if args.dry_run:
        print("\nDry run: no API calls made.")
        return

    out_dir = Path(args.out_dir)
    started = time.time()
    totals: dict[str, Usage] = {}

    for m in models:
        cfg = PROVIDERS[m]
        agg = Usage()
        for t in tasks:
            scope = ops if t == "discipline" else _cancer_scope(
                ops, args.cancer_scope, tax, out_dir, models
            )
            u = await run_model_task(cfg, t, scope, tax, out_dir, args.concurrency)
            agg.calls += u.calls
            agg.failures += u.failures
            agg.skipped += u.skipped
            agg.empty += u.empty
            agg.off_list += u.off_list
            agg.unverified_quote += u.unverified_quote
            agg.tokens_in += u.tokens_in
            agg.tokens_out += u.tokens_out
            for k, v in u.finish_reasons.items():
                agg.finish_reasons[k] = agg.finish_reasons.get(k, 0) + v
        totals[m] = agg

    elapsed = time.time() - started
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    grand = 0.0
    for m, u in totals.items():
        cfg = PROVIDERS[m]
        cost = u.tokens_in / 1e6 * cfg.price_in + u.tokens_out / 1e6 * cfg.price_out
        grand += cost
        print(
            f"  {m:<8} {u.calls:>7,} calls  {u.failures:>5,} failed  "
            f"{u.empty:>5,} empty  {u.skipped:>5,} skipped  "
            f"{u.off_list:>5,} off-list  ~${cost:,.2f}"
        )
    print(f"  {'TOTAL':<8} ~${grand:,.2f} (actual, from reported token usage)")

    if any(u.skipped for u in totals.values()):
        print(
            "\nNOTE: skipped OPs were NOT checkpointed. Fix the cause (usually\n"
            "      depleted credits or a bad key), then re-run to pick them up."
        )
    print(f"\nOutput: {out_dir}/labels_<model>_<task>.jsonl")
    print("Next:   python classify_cancer_scispacy.py   (labeller B)")
    print("        python agreement.py                  (kappa + manual queue)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Classify r/AskDocs OPs by discipline and cancer stance (labeller A).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--taxonomy", default="taxonomy.json",
                   help="fixed taxonomy JSON (the 17 disciplines)")
    p.add_argument("--from-analysis-dataset", default=None, metavar="PATH",
                   help="read the OP set from the analysis dataset (exact id match; "
                        "preferred over re-deriving it)")
    p.add_argument("--corpora-dir", default="../Piloting/Round 5/output/corpora",
                   help="used when --from-analysis-dataset is not given")
    p.add_argument("--out-dir", default="../Piloting/Round 5/output/corpora/labels",
                   help="where label checkpoints are written")
    p.add_argument("--models", default="gemini",
                   help="comma-separated labeller-A models: gemini,openai,grok. More "
                        "than one gives an LLM-vs-LLM agreement check.")
    p.add_argument("--tasks", default="discipline,cancer",
                   help="comma-separated: discipline,cancer")
    p.add_argument("--cancer-scope", default="all", choices=("all", "oncology", "screen"),
                   help="which OPs the cancer task sees. 'all' has no recall ceiling; "
                        "'oncology' gates on the discipline label; 'screen' uses a term "
                        "prefilter and caps recall")
    p.add_argument("--n-ops", type=int, default=6600,
                   help="number of OPs when sampling from the corpus; 0 means all")
    p.add_argument("--seed", type=int, default=42,
                   help="sampling seed; must match the generation run")
    p.add_argument("--concurrency", type=int, default=25,
                   help="max in-flight API requests")
    p.add_argument("--model-id", action="append", metavar="NAME=ID",
                   help="override a model id, e.g. --model-id gemini=gemini-2.5-pro")
    p.add_argument("--reasoning-effort", action="append", metavar="NAME=LEVEL",
                   help="set reasoning budget per model (minimal/low/medium/high). "
                        "Unset is right for classification.")
    p.add_argument("--dry-run", action="store_true",
                   help="print workload and cost estimate without calling any API")
    return p.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is checkpointed; re-run to resume.")
