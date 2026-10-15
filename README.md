# LLM Health Advisors

Code and analysis for a research project evaluating how AI-generated medical advice
compares to real clinician and layperson responses in online health forums.

## Overview

This project studies how LLM-generated comments in medical advice threads (r/AskDocs)
compare to real clinician and layperson responses, and how the choice of prompting
strategy affects the character of synthetic discourse. Synthetic comment threads are
generated under controlled experimental conditions, then compared against the real data
along dimensions such as linguistic style, empathy, and diversity.

The underlying corpus derives from the
[Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) Reddit datasets and
contains 19,984 submissions and 40,537 top-level comments from r/AskDocs.

## Where things are

All current work lives in [`src/`](src/) — code, documentation and output together.
`Piloting/` holds the pilot rounds that led up to it, as history, and is not maintained.

| Looking for | It's in |
|---|---|
| **Topic labels for the 6,600 threads** (the current deliverable) | `src/output/corpora/export/` |
| The topic classification pipeline and what it did | [`src/Covariates/Topics/`](src/Covariates/Topics/) — start with its [README](src/Covariates/Topics/README.md) |
| Whether the first responder's specialty matches the topic | [`src/Covariates/Topic-Clinician-Match/`](src/Covariates/Topic-Clinician-Match/) |
| The generation script | `src/Generation/generate_synthetic.py` |
| The hand-annotated 200-post sample and its adjudication | `src/output/corpora/handcoding/` |
| Findings from the 3,000-thread run | [`Piloting/Round 5/README.md`](Piloting/Round%205/README.md) |
| The corpus itself (gitignored, shared separately) | `Piloting/Round 2/*.jsonl` |
| `requirements.txt` | `Piloting/Round 1/` |

Two things worth knowing before you go looking:

- **`src/` holds the output too**, under `src/output/corpora/`, rather than at the
  repository root. Every script's defaults are relative to its own directory, which keeps
  them short. The source corpus is the exception and lives in `Piloting/Round 2/`.
- **Large files are gitignored**, so the corpus, the generated comments, and the merged
  analysis dataset are not on GitHub and are shared out of band. Those directories will
  look empty here. The topic labels and the annotated 200-post sample are the exceptions:
  small enough to track, and checked in.

```
src/                                  # all current work, by what it produces
  README.md                           #   map of the four folders
  Generation/                         # the experiment: synthetic comment generation
    generate_synthetic.py
    SCALEUP_6600.md                   #   runbook for the 6,600-thread run
    power_analysis.md                 #   why 6,600 rather than 3,000
  Covariates/                         # variables for the downstream model
    Topics/                           #   17-discipline label per thread
      README.md                       #     what it did and why -- read this first
      topic_classification_plan.md    #     the full decision log
      taxonomy.json                   #     the 17 disciplines and the codebook rules
      classify_topics.py              #     LLM labeller
      build_handcoding_sheets.py      #     blind annotation workbooks
      score_handcoding.py             #     agreement and model accuracy
      agreement.py export_labels.py sweep.sh
    Topic-Clinician-Match/            #   does the first responder's specialty fit?
      map_responder_roles.py
  Analysis/
    build_analysis_dataset.py         #   merges corpus + generated output
    build_review_spreadsheet.py
    5_exploratory_analysis_6600.ipynb
  output/corpora/                     # shared output
    export/                           #   deliverables -- checked in
    handcoding/                       #   the 200-post sample -- checked in
    labels/ generated/                #   raw model output -- gitignored

Piloting/                             # history; not maintained
  Round 1/   corpus construction from Arctic Shift .zst archives; requirements.txt
  Round 2/   the built corpus (gitignored JSONL) and an early generation notebook
  Round 3/   DG / SAG / SAGII prompting strategies introduced
  Round 4/   clinician-only design, 100 threads, one model; empathy instruction removed
  Round 5/   the 3,000-thread run: its findings README and analysis notebook
```

## Prompting strategies

For each opening post, synthetic comments are generated using three strategies:

- **DG (Discourse Generation)** — a single prompt generates the entire thread of *n*
  replies.
- **SAG (Single Advice Generation)** — *n* independent API calls, each generating one
  response. Each call sees only the opening post.
- **SAGII (SAG with Incremental Information)** — *n* sequential calls, each also seeing
  all previously generated responses.

Round 5 ran all three across Gemini 2.5 Pro, GPT-5 and Grok 4.6, giving nine synthetic
sets per thread plus the real comments.

## Corpus details

The corpus was built by streaming Reddit submissions and comments from compressed `.zst`
files. Submissions were retained if `selftext` exists and is at least 30 characters.
Comments were retained if top-level, not by AutoModerator, and not removed or deleted.
Each submission carries pre-computed `total_top_level_comments`,
`layperson_top_level_comments` and `layperson_comment_fraction`; layperson status comes
from the commenter's flair containing "layperson" (case-insensitive). Schemas follow the
Arctic Shift format
([comments](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RC/2026.ts),
[submissions](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RS/2026.ts)).

## Setup

```bash
pip install -r "Piloting/Round 1/requirements.txt"
```

The generation and classification scripts additionally need `openai`, `tqdm` and
`openpyxl`, plus API keys in the environment — `GEMINI_API_KEY`, `OPENAI_API_KEY`,
`OPENROUTER_API_KEY`. `src/Covariates/Topics/sweep.sh` reads them from `src/.env`, which
is gitignored.

## Collaborators

Forked from
[lucienbaumgartner/llm_health_advisors](https://github.com/lucienbaumgartner/llm_health_advisors).
