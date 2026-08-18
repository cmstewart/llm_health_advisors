# Multi-Model Synthetic Comment Generation

`generate_synthetic.py` scales the Round 4 notebook up to the full corpus and adds
multiple model providers. It replaces the notebook for large runs; the notebook is
still the right tool for inspecting a handful of threads by hand.

## What it does

For each opening post (OP), it generates synthetic **clinician** comments using three
prompting strategies, for each model you enable:

| Strategy | Calls per OP | Parallelism |
|---|---|---|
| **DG** — whole thread in one prompt | 1 | across OPs |
| **SAG** — each comment independent, sees only the OP | *n* | within *and* across OPs |
| **SAGII** — each comment sees all prior comments | *n* | across OPs (sequential within) |

where *n* = the number of real, non-self top-level comments in that thread.

Corpus filtering is identical to the Round 4 notebook: self-posts removed, image OPs
excluded, no keyword filter, OPs must have at least one real comment. This yields
**12,464 eligible OPs / 27,275 comments per strategy**.

## Setup

```bash
pip install openai tqdm
```

Set whichever keys you need. Models without a key are skipped with a warning, so you
can run one provider at a time.

```bash
export GEMINI_API_KEY=...       # Gemini, direct
export OPENAI_API_KEY=...       # GPT, direct
export OPENROUTER_API_KEY=...   # Grok, via OpenRouter
```

All three providers are called through the OpenAI wire format — Gemini via its
[OpenAI-compatibility endpoint](https://ai.google.dev/gemini-api/docs/openai) and Grok
via OpenRouter — so there is one code path rather than three SDKs.

## Running

Always dry-run first. It prints the exact workload and cost estimate and makes no API
calls:

```bash
python generate_synthetic.py --dry-run --n-ops 3000
```

Phase 1 — the 3,000-OP sample:

```bash
python generate_synthetic.py --n-ops 3000 --models gemini,openai,grok
```

Phase 2 — the full corpus. The 3,000-OP sample is a stable subset, and checkpoints are
keyed by submission ID, so this resumes rather than regenerating:

```bash
python generate_synthetic.py --n-ops 0 --models gemini,openai,grok
```

For a long run, use `tmux` or `nohup` so it survives a disconnect:

```bash
nohup python generate_synthetic.py --n-ops 0 > gen.log 2>&1 &
tail -f gen.log
```

## Scale and cost

Estimates from the dry run, at sync (non-batch) pricing:

| | Calls/model | ×3 models | Est. cost |
|---|---|---|---|
| 3,000 OPs | 15,952 | 47,856 | **~$122** |
| Full 12,464 OPs | 67,014 | 201,042 | **~$508** |

Runtime depends almost entirely on `--concurrency`. The notebook's sequential
1-second-delay approach would take ~154 h per model on the full corpus; at the default
concurrency of 25 that drops to roughly 6–8 h per model.

Prices move — the script prints an estimate from the table in `PROVIDERS`, then reports
**actual** cost at the end from the token counts the APIs return. Verify current
per-token pricing before committing to a full run.

## Resuming

Progress is checkpointed after every OP to `<out-dir>/<model>_<strategy>.jsonl`. If the
run crashes, is interrupted, or hits a rate-limit wall, just re-run the same command —
completed OPs are skipped. A truncated final line from a hard kill is tolerated.

## Output

One JSONL file per (model, strategy), e.g. `gemini_dg.jsonl`, one record per OP:

```json
{
  "submission_id": "1abc234",
  "model": "gemini",
  "model_id": "gemini-2.5-pro",
  "strategy": "dg",
  "n_comments": 3,
  "comments": [{"body": "...", "author_type": "clinician"}]
}
```

Deliberately not Excel — 200k comments is not a spreadsheet. Use the notebook's export
cell to build review spreadsheets from a sample.

## Options

| Flag | Default | Notes |
|---|---|---|
| `--n-ops` | `3000` | `0` or negative = full eligible set |
| `--models` | `gemini,openai,grok` | any comma-separated subset |
| `--strategies` | `dg,sag,sagii` | any comma-separated subset |
| `--concurrency` | `25` | lower it if you hit rate limits |
| `--seed` | `42` | sampling seed |
| `--model-id` | — | override a model id, e.g. `--model-id openai=gpt-5` |
| `--dry-run` | off | estimate only, no API calls |
| `--corpora-dir` | `../output/corpora` | where the corpus JSONL files live |
| `--out-dir` | `../output/corpora/generated` | where checkpoints are written |

## Things to watch

**Model IDs.** The defaults in `PROVIDERS` (`gemini-2.5-pro`, `gpt-5`, `x-ai/grok-4`)
should be checked against each provider's current model list before a big run. Override
with `--model-id` rather than editing the file.

**Rate limits.** Concurrency 25 is a reasonable starting point. If you see repeated
`429` retries in the log, drop to 10. The script retries with exponential backoff and
only gives up after 5 attempts, so transient limits are handled, but sustained
throttling just wastes wall-clock time.

**No thread cap.** Per the current design decision, threads are generated at their true
length. The largest thread has 126 comments, which means 126 *sequential* SAGII calls
for that one OP. A handful of such threads will be the long tail of each run. If they
become a bottleneck, adding a cap is a small change to `select_sample`.

**Cost control.** Start with one model and one strategy (`--models gemini --strategies dg`)
to confirm output quality before spending the full budget.
