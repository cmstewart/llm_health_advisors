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

The work is organised by **round**, each a successive experiment. `Piloting/Round 5/` is
the current one; rounds 1–4 are kept as history and are not maintained.

| Looking for | It's in |
|---|---|
| **Topic labels for the 6,600 threads** (the current deliverable) | `Piloting/Round 5/output/corpora/export/` |
| The topic classification pipeline and what it did | [`Data/`](Data/) — start with its [README](Data/README.md) |
| The current generation script | `Data/generate_synthetic.py` |
| Round 5 findings | [`Piloting/Round 5/README.md`](Piloting/Round%205/README.md) |
| The corpus itself (gitignored, shared separately) | `Piloting/Round 2/*.jsonl` |
| `requirements.txt` | `Piloting/Round 1/` |

Two naming quirks worth knowing before you go looking:

- **`Data/` holds code, not data.** It began as a home for the 6,600-thread scale-up and
  grew into the topic classification pipeline. The corpus lives in `Piloting/Round 2/`.
- **Large files are gitignored**, so the corpus, the generated comments, and the merged
  analysis dataset are not on GitHub and are shared out of band. The directories will look
  empty here. The topic labels are the exception: small enough to track, and checked in.

```
Data/                         # scale-up generation + topic classification (code)
  README.md                   #   what the classification did and why -- read this first
  taxonomy.json               #   the 17 disciplines and the codebook rules
  classify_topics.py          #   LLM labeller
  build_handcoding_sheets.py  #   blind annotation workbooks (200-post sample)
  score_handcoding.py         #   agreement, gold standard, classifier accuracy
  agreement.py                #   cancer consensus, prevalence, power
  export_labels.py            #   produces the deliverable
  generate_synthetic.py       #   Round 5 generation script
  SCALEUP_6600.md             #   runbook for the 6,600-thread generation
  power_analysis.md           #   why 6,600 rather than 3,000

Piloting/
  Round 1/   corpus construction from Arctic Shift .zst archives; requirements.txt
  Round 2/   the built corpus (gitignored JSONL) and an early generation notebook
  Round 3/   DG / SAG / SAGII prompting strategies introduced
  Round 4/   clinician-only design, 100 threads, one model; empathy instruction removed
  Round 5/   scale-up to 3,000 then 6,600 threads across three models; current work
    README.md                      findings
    topic_classification_plan.md   the topic labelling decision log
    5_exploratory_analysis.ipynb   analysis of the 3,000-thread run
    build_analysis_dataset.py      merges corpus + generated output
    output/corpora/                all Round 5 output (mostly gitignored)
      export/                      the topic labels -- checked in
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
`OPENROUTER_API_KEY`. `Data/sweep.sh` reads them from `Data/.env`, which is gitignored.

## Collaborators

Forked from
[lucienbaumgartner/llm_health_advisors](https://github.com/lucienbaumgartner/llm_health_advisors).
