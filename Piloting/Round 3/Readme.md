# LLM Health Advisors

Evaluating the impact of prompting strategies on the quality of AI-generated medical discourse in online health forums.

## Overview

This project studies how LLM-generated comments in medical advice threads (r/AskDocs) compare to real clinician and layperson responses, and how the choice of prompting strategy affects the character of synthetic discourse. We use Google's Gemini 2.5 Pro to generate synthetic comment threads under controlled experimental conditions, then compare them against the real data along dimensions such as linguistic style, empathy, and diversity.

The underlying corpus is derived from the [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) Reddit datasets and contains 19,984 submissions and 40,537 top-level comments from r/AskDocs.

## Repository Structure

```
src/
  1_preprocessing/          # Scripts for filtering raw Reddit data from .zst archives
  2_synthetic_comment_generation.ipynb   # Round 1: Fixed 75/25, 25/75, no-ratio conditions
  3_synthetic_comment_generation.ipynb   # Round 3: DG, SAG, SAGII prompting strategies
output/
  corpora/                  # Corpus JSONL files, metadata CSV, generated spreadsheets
requirements.txt
```

## Experimental Rounds

### Round 1

Three fixed authorship conditions were applied to every OP: 75% clinician / 25% layperson, 25% clinician / 75% layperson, and no specified ratio. The number of synthetic comments per OP matched the number of real top-level comments. This round used minimal prompt constraints and served as a baseline.

### Round 2

The prompts were revised to produce more realistic output: comments were kept brief (1-4 sentences), clinicians were instructed not to announce credentials or add disclaimers, layperson comments were not required to be medically accurate, and clinicians were instructed to show empathy. OPs were filtered to those whose titles contained health-related terms. Five OPs were sampled and output was exported to a spreadsheet for manual review.

### Round 3

Three distinct prompting strategies were introduced, each generating a thread of *n* comments matching the real thread's size and clinician/layperson distribution:

- **DG (Discourse Generation)**: A single prompt generates the entire thread, specifying the exact number of clinician and layperson comments.
- **SAG (Single Advice Generation)**: *n* independent API calls, each generating one response with a specified role (clinician or layperson). Each call sees only the OP.
- **SAGII (SAG with Incremental Information)**: *n* sequential API calls. Each call generates one response with a specified role, but also sees all previously generated responses in the thread.

Key changes from earlier rounds:

- OPs are filtered by an expanded term list (`healthy`, `ill`, `sick`, `unhealthy`, `unwell`, `well`, `disease`, `dysfunction`, `illness`, `health`, `disorder`) searched in both the title and body, yielding 5,935 matching OPs.
- Clinician/layperson counts are computed directly from comment flairs rather than from a pre-computed submission field that had a bug (always reported 0 layperson comments). About 24% of matching OPs have at least one layperson comment.
- 100 OPs were sampled for generation (seed 42), producing approximately 566 API calls across all three strategies.
- Checkpointing allows the generation to resume after interruption.

## Corpus Details

The corpus was built by streaming Reddit submissions and comments from compressed `.zst` files. Submissions were retained if `selftext` exists and is at least 30 characters. Comments were retained if they are top-level, not by AutoModerator, and not removed/deleted.

On r/AskDocs, each commenter has a "flair" -- a label indicating whether they are a verified medical professional (e.g. "Physician", "Registered Nurse") or an unverified user ("Layperson/not verified as healthcare professional"). The notebook computes clinician/layperson counts directly from these flair strings by checking whether the flair contains "layperson" (case-insensitive); all other commenters are counted as clinicians.

Full data schemas follow the Arctic Shift format ([comments](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RC/2026.ts), [submissions](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RS/2026.ts)).

## Setup

```bash
pip install -r requirements.txt
```

The generation notebooks additionally require `openpyxl` for spreadsheet export and a Google Gemini API key.

## Collaborators

Forked from [lucienbaumgartner/llm_health_advisors](https://github.com/lucienbaumgartner/llm_health_advisors).
