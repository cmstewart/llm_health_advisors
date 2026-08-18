#!/usr/bin/env python3
"""
Multi-model synthetic comment generation for r/AskDocs.

Generates synthetic clinician comments for opening posts (OPs) using three
prompting strategies (DG, SAG, SAGII) across multiple LLM providers.

All three providers are accessed through the OpenAI wire format:
  - OpenAI      -> api.openai.com
  - Gemini      -> generativelanguage.googleapis.com/v1beta/openai/
  - Grok        -> openrouter.ai/api/v1

Design notes:
  * Async with a global semaphore bounding total in-flight requests.
  * DG   : 1 call per OP                      -> parallel across OPs
  * SAG  : n independent calls per OP         -> parallel within AND across OPs
  * SAGII: n sequential calls per OP          -> sequential within, parallel across OPs
  * Checkpointed per (model, strategy) as JSONL. Re-running skips completed OPs.

Usage
-----
  # Dry run: no API calls, prints workload + cost estimate
  python generate_synthetic.py --dry-run --n-ops 3000

  # Phase 1: 3,000 OPs, all models, all strategies
  python generate_synthetic.py --n-ops 3000 --models gemini,openai,grok

  # Phase 2: full corpus (resumes; already-done OPs are skipped)
  python generate_synthetic.py --n-ops 0 --models gemini,openai,grok

Environment variables (or use a .env file):
  GEMINI_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Provider configuration
#
# NOTE: model IDs change over time. Verify these against each provider's current
# model list before a large run, and override with --model-id if needed, e.g.
#   --model-id gemini=gemini-2.5-pro --model-id openai=gpt-5
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    name: str
    model_id: str
    base_url: str | None
    api_key_env: str
    # USD per 1M tokens, for the end-of-run cost report
    price_in: float
    price_out: float
    # Optional reasoning budget for reasoning models ("minimal"/"low"/"medium"/"high").
    # Reasoning tokens bill as output, so this is the main cost lever on GPT/Grok.
    # Left as None the parameter is not sent at all.
    reasoning_effort: str | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "gemini": ProviderConfig(
        name="gemini",
        model_id="gemini-2.5-pro",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        price_in=1.25,
        price_out=10.00,
    ),
    "openai": ProviderConfig(
        name="openai",
        model_id="gpt-5",
        base_url=None,  # SDK default
        api_key_env="OPENAI_API_KEY",
        price_in=1.25,
        price_out=15.00,
    ),
    "grok": ProviderConfig(
        name="grok",
        model_id="x-ai/grok-4.6",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        # Calibrated against a real 16,128-call run: 14.4M in / 3.3M out billed
        # $38.55 by OpenRouter. Treat as approximate; check your provider dashboard
        # for authoritative spend.
        price_in=1.50,
        price_out=5.00,
    ),
}

STRATEGIES = ("dg", "sag", "sagii")


# ---------------------------------------------------------------------------
# Prompts  (kept identical to the Round 4 notebook)
# ---------------------------------------------------------------------------

COMMENT_GUIDELINES = (
    "Important guidelines for the comments:\n"
    "- Keep comments brief and natural. Match the typical length of real Reddit "
    "comments (usually 1-4 sentences).\n"
    "- Layperson comments do NOT need to be medically accurate or provide correct "
    "medical advice.\n"
    "- Each comment can focus on just one particular aspect of the OP rather than "
    "responding to the entire post.\n"
    "- Clinicians should NOT introduce themselves by stating their role or credentials "
    "(e.g. do not start with 'I'm a doctor' or 'As a nurse').\n"
    "- Clinicians should NOT include disclaimers like 'It's important to consult a "
    "physician' or 'You will want to ask a doctor about this'.\n"
    "- Laypersons should NOT disclaim their lack of medical training "
    "(e.g. do not say 'I'm not a doctor but' or 'I have no medical background').\n"
)


def build_dg_prompt(title: str, selftext: str, n: int) -> str:
    return (
        "The following is the opening post from a thread in the subreddit r/AskDocs. "
        f"Generate exactly {n} comments (no more, no fewer) in which each comment "
        "responds to the opening post and any subsequent comments prior to it, i.e. none "
        "of the comments should be threaded. Each comment should imitate the response "
        "of a clinician.\n\n"
        f"{COMMENT_GUIDELINES}\n"
        f'Return your response as a JSON object with a single key "comments" whose value '
        f"is an array of exactly {n} objects, where each object has these fields:\n"
        '- "body": the text of the comment\n'
        '- "author_type": "clinician"\n\n'
        f"Opening Post:\nTitle: {title}\n\n{selftext}"
    )


def build_sag_prompt(title: str, selftext: str) -> str:
    return (
        "The following is the opening post from a thread in the subreddit r/AskDocs. "
        "Generate exactly 1 comment that responds to the opening post. "
        "The comment should imitate the response of a clinician.\n\n"
        f"{COMMENT_GUIDELINES}\n"
        'Return your response as a JSON object with a single key "comments" whose value '
        "is an array containing exactly 1 object with these fields:\n"
        '- "body": the text of the comment\n'
        '- "author_type": "clinician"\n\n'
        f"Opening Post:\nTitle: {title}\n\n{selftext}"
    )


def build_sagii_prompt(title: str, selftext: str, prior: list[dict]) -> str:
    if prior:
        prior_text = (
            "\n\nThe following comments have already been posted in response to this OP:\n\n"
        )
        for i, pc in enumerate(prior, 1):
            prior_text += f"Comment {i}: {pc['body']}\n\n"
        context = (
            "Your comment should respond to the opening post. You may also acknowledge "
            "or build on the existing comments, but do not simply repeat what has already "
            "been said."
        )
    else:
        prior_text = ""
        context = "Your comment should respond to the opening post."

    return (
        "The following is the opening post from a thread in the subreddit r/AskDocs. "
        f"Generate exactly 1 comment. {context} "
        "The comment should imitate the response of a clinician.\n\n"
        f"{COMMENT_GUIDELINES}\n"
        'Return your response as a JSON object with a single key "comments" whose value '
        "is an array containing exactly 1 object with these fields:\n"
        '- "body": the text of the comment\n'
        '- "author_type": "clinician"\n\n'
        f"Opening Post:\nTitle: {title}\n\n{selftext}{prior_text}"
    )


# ---------------------------------------------------------------------------
# Corpus loading  (mirrors the Round 4 notebook exactly)
# ---------------------------------------------------------------------------

IMG_URL_PATTERN = re.compile(
    r"(i\.redd\.it|preview\.redd\.it|imgur\.com|\.jpg|\.jpeg|\.png|\.gif|\.webp|\.heic)",
    re.IGNORECASE,
)


def has_image(submission: dict) -> bool:
    """True if the submission embeds or links an image."""
    if submission.get("preview"):
        return True
    return bool(IMG_URL_PATTERN.search(submission.get("selftext", "") or ""))


def load_corpus(corpora_dir: Path) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Load submissions and top-level comments, applying the Round 4 filters:
      * drop self-posts (comment author == OP author)
      * drop OPs containing images
      * keep OPs with >= 1 non-removed comment
      * NO keyword/topic filter
    """
    submissions_path = corpora_dir / "submissions_corpus.jsonl"
    comments_path = corpora_dir / "comments_corpus.jsonl"

    for p in (submissions_path, comments_path):
        if not p.exists():
            sys.exit(f"ERROR: corpus file not found: {p}")

    submissions: list[dict] = []
    with open(submissions_path) as f:
        for line in f:
            submissions.append(json.loads(line))
    by_id = {s["id"]: s for s in submissions}

    comments_by_sub: dict[str, list[dict]] = defaultdict(list)
    n_self = 0
    with open(comments_path) as f:
        for line in f:
            c = json.loads(line)
            if not c.get("parent_id", "").startswith("t3_"):
                continue
            sid = c["parent_id"][3:]
            # Drop self-posts: the OP replying under their own thread.
            op_author = by_id.get(sid, {}).get("author")
            c_author = c.get("author")
            if c_author and op_author and c_author == op_author:
                n_self += 1
                continue
            comments_by_sub[sid].append(c)

    def has_real(sid: str) -> bool:
        return any(
            c.get("body", "") not in ("[removed]", "[deleted]")
            for c in comments_by_sub.get(sid, [])
        )

    eligible = [
        s for s in submissions if not has_image(s) and has_real(s["id"])
    ]
    for s in eligible:
        s["_n_comments"] = len(comments_by_sub[s["id"]])

    print(f"Loaded {len(submissions):,} submissions; dropped {n_self:,} self-posts.")
    print(f"Eligible OPs (image-free, >=1 real comment): {len(eligible):,}")
    return eligible, comments_by_sub


def select_sample(eligible: list[dict], n_ops: int, seed: int) -> list[dict]:
    """
    Deterministically select n_ops OPs. n_ops <= 0 means 'use everything'.

    Sampling is seeded and sorted by id first, so the 3,000-OP phase-1 sample is a
    stable subset: re-running later with a larger n_ops keeps the earlier OPs and
    their checkpoints valid.
    """
    ordered = sorted(eligible, key=lambda s: s["id"])
    if n_ops <= 0 or n_ops >= len(ordered):
        return ordered
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ordered)), n_ops))
    return [ordered[i] for i in idx]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

class Checkpoint:
    """Append-only JSONL checkpoint, one file per (model, strategy)."""

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
    tokens_in: int = 0
    tokens_out: int = 0


async def call_model(
    client,
    cfg: ProviderConfig,
    prompt: str,
    n_expected: int,
    sem: asyncio.Semaphore,
    usage: Usage,
    retries: int = 5,
) -> list[dict]:
    """
    Send one prompt, parse a JSON object of the form {"comments": [...]}.
    Returns a list of comment dicts, truncated to n_expected. On total failure
    returns [] and the caller records placeholders.
    """
    kwargs = {
        "model": cfg.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 1.0,
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

            payload = json.loads(resp.choices[0].message.content)
            # Accept {"comments": [...]} or a bare [...] just in case.
            items = payload.get("comments") if isinstance(payload, dict) else payload
            if not isinstance(items, list):
                raise ValueError(f"expected a list, got {type(items).__name__}")

            out = [
                {
                    "body": str(it.get("body", "")),
                    "author_type": str(it.get("author_type", "clinician")),
                }
                for it in items
                if isinstance(it, dict)
            ]
            return out[:n_expected]

        except Exception as e:  # noqa: BLE001 - retry on anything transient
            if attempt == retries - 1:
                usage.failures += 1
                print(f"  [{cfg.name}] giving up after {retries} attempts: {e}")
                return []
            # exponential backoff with jitter; rate limits are the common case
            await asyncio.sleep(min(2**attempt, 30) + random.random())
    return []


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

async def run_dg(client, cfg, sub, sem, usage) -> list[dict]:
    n = sub["_n_comments"]
    prompt = build_dg_prompt(sub.get("title", ""), sub.get("selftext", ""), n)
    return await call_model(client, cfg, prompt, n, sem, usage)


async def run_sag(client, cfg, sub, sem, usage) -> list[dict]:
    """n independent calls -> fire them all concurrently."""
    n = sub["_n_comments"]
    prompt = build_sag_prompt(sub.get("title", ""), sub.get("selftext", ""))
    results = await asyncio.gather(
        *[call_model(client, cfg, prompt, 1, sem, usage) for _ in range(n)]
    )
    return [
        r[0] if r else {"body": "", "author_type": "clinician"} for r in results
    ]


async def run_sagii(client, cfg, sub, sem, usage) -> list[dict]:
    """n sequential calls -- each sees everything generated so far in this thread."""
    n = sub["_n_comments"]
    title, selftext = sub.get("title", ""), sub.get("selftext", "")
    acc: list[dict] = []
    for _ in range(n):
        prompt = build_sagii_prompt(title, selftext, acc)
        r = await call_model(client, cfg, prompt, 1, sem, usage)
        acc.append(r[0] if r else {"body": "", "author_type": "clinician"})
    return acc


RUNNERS = {"dg": run_dg, "sag": run_sag, "sagii": run_sagii}


async def process_op(client, cfg, strategy, sub, sem, usage, ckpt, pbar) -> None:
    try:
        comments = await RUNNERS[strategy](client, cfg, sub, sem, usage)
        await ckpt.write(
            {
                "submission_id": sub["id"],
                "model": cfg.name,
                "model_id": cfg.model_id,
                "strategy": strategy,
                "n_comments": sub["_n_comments"],
                "comments": comments,
            }
        )
    except Exception as e:  # noqa: BLE001 - never let one OP kill the run
        print(f"  [{cfg.name}/{strategy}] error on {sub['id']}: {e}")
    finally:
        if pbar is not None:
            pbar.update(1)


async def run_model_strategy(
    cfg: ProviderConfig,
    strategy: str,
    sample: list[dict],
    out_dir: Path,
    concurrency: int,
) -> Usage:
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        print(f"SKIP {cfg.name}/{strategy}: ${cfg.api_key_env} not set")
        return Usage()
    if AsyncOpenAI is None:
        sys.exit("ERROR: the 'openai' package is required. pip install openai")

    ckpt = Checkpoint(out_dir / f"{cfg.name}_{strategy}.jsonl")
    todo = [s for s in sample if s["id"] not in ckpt.done]

    print(
        f"\n=== {cfg.name} / {strategy} ===\n"
        f"  model: {cfg.model_id}\n"
        f"  {len(sample):,} OPs in sample, {len(ckpt.done):,} already done, "
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
    pbar = (
        tqdm(total=len(todo), desc=f"{cfg.name}/{strategy}", unit="op")
        if tqdm
        else None
    )
    try:
        await asyncio.gather(
            *[
                process_op(client, cfg, strategy, s, sem, usage, ckpt, pbar)
                for s in todo
            ]
        )
    finally:
        if pbar is not None:
            pbar.close()
        await client.close()

    return usage


# ---------------------------------------------------------------------------
# Dry run / estimation
# ---------------------------------------------------------------------------

def estimate(sample: list[dict], models: list[str], strategies: list[str]) -> None:
    n_ops = len(sample)
    n_comments = sum(s["_n_comments"] for s in sample)
    sizes = [s["_n_comments"] for s in sample]

    calls = {"dg": n_ops, "sag": n_comments, "sagii": n_comments}
    per_model = sum(calls[s] for s in strategies)

    print("\n--- Workload estimate ---")
    print(f"OPs: {n_ops:,}   comments/strategy: {n_comments:,}")
    print(
        f"thread size: mean {statistics.mean(sizes):.2f}, "
        f"median {statistics.median(sizes):.0f}, max {max(sizes)}"
    )
    for s in strategies:
        print(f"  {s:<6} {calls[s]:>8,} calls/model")
    print(f"  {'TOTAL':<6} {per_model:>8,} calls/model"
          f"  x{len(models)} models = {per_model*len(models):,}")

    # Token estimate: ~4 chars/token, ~200 tok of guidelines, ~80 tok per comment out.
    GUIDE, OUT = 200, 80
    t_in = t_out = 0.0
    for s in sample:
        base = len((s.get("title") or "") + (s.get("selftext") or "")) / 4 + GUIDE
        n = s["_n_comments"]
        if "dg" in strategies:
            t_in += base
            t_out += n * OUT
        if "sag" in strategies:
            t_in += base * n
            t_out += n * OUT
        if "sagii" in strategies:
            t_in += sum(base + i * OUT for i in range(n))
            t_out += n * OUT

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args) -> None:
    corpora = Path(args.corpora_dir)
    eligible, _ = load_corpus(corpora)
    sample = select_sample(eligible, args.n_ops, args.seed)
    print(f"Selected {len(sample):,} OPs (seed={args.seed}).")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for m in models:
        if m not in PROVIDERS:
            sys.exit(f"ERROR: unknown model '{m}'. Choose from {list(PROVIDERS)}")
    for s in strategies:
        if s not in STRATEGIES:
            sys.exit(f"ERROR: unknown strategy '{s}'. Choose from {list(STRATEGIES)}")

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

    estimate(sample, models, strategies)

    if args.dry_run:
        print("\nDry run: no API calls made.")
        return

    out_dir = Path(args.out_dir)
    started = time.time()
    totals: dict[str, Usage] = {}

    for m in models:
        cfg = PROVIDERS[m]
        agg = Usage()
        for s in strategies:
            u = await run_model_strategy(cfg, s, sample, out_dir, args.concurrency)
            agg.calls += u.calls
            agg.failures += u.failures
            agg.tokens_in += u.tokens_in
            agg.tokens_out += u.tokens_out
        totals[m] = agg

    elapsed = time.time() - started
    print(f"\n=== Done in {elapsed/3600:.2f} h ===")
    grand = 0.0
    for m, u in totals.items():
        cfg = PROVIDERS[m]
        cost = u.tokens_in / 1e6 * cfg.price_in + u.tokens_out / 1e6 * cfg.price_out
        grand += cost
        print(
            f"  {m:<8} {u.calls:>7,} calls  {u.failures:>5,} failed  "
            f"{u.tokens_in/1e6:>6.1f}M in  {u.tokens_out/1e6:>5.1f}M out  ~${cost:,.2f}"
        )
    print(f"  {'TOTAL':<8} ~${grand:,.2f} (actual, from reported token usage)")
    print(f"\nOutput: {out_dir}/<model>_<strategy>.jsonl")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate synthetic r/AskDocs clinician comments across models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpora-dir", default="../output/corpora",
                   help="directory holding submissions_corpus.jsonl / comments_corpus.jsonl")
    p.add_argument("--out-dir", default="../output/corpora/generated",
                   help="where per-(model,strategy) JSONL checkpoints are written")
    p.add_argument("--models", default="gemini,openai,grok",
                   help="comma-separated: gemini,openai,grok")
    p.add_argument("--strategies", default="dg,sag,sagii",
                   help="comma-separated: dg,sag,sagii")
    p.add_argument("--n-ops", type=int, default=3000,
                   help="number of OPs; 0 or negative means the full eligible set")
    p.add_argument("--seed", type=int, default=42, help="sampling seed")
    p.add_argument("--concurrency", type=int, default=25,
                   help="max in-flight API requests (lower this if you hit rate limits)")
    p.add_argument("--model-id", action="append", metavar="NAME=ID",
                   help="override a model id, e.g. --model-id openai=gpt-5")
    p.add_argument("--reasoning-effort", action="append", metavar="NAME=LEVEL",
                   help="set reasoning budget per model, e.g. --reasoning-effort openai=low "
                        "(minimal/low/medium/high). Reasoning tokens bill as output, so this "
                        "is the main cost lever. Omit for a provider that doesn't support it.")
    p.add_argument("--dry-run", action="store_true",
                   help="print workload and cost estimate without calling any API")
    return p.parse_args(argv)


if __name__ == "__main__":
    try:
        asyncio.run(main_async(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is checkpointed; re-run to resume.")
