# LLM Health Advisors

This respository contains code and analysis for a research project evaluating the impact of prompting strategies on the quality of AI-generated medical discourse in online health forums.

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

## Prompting Strategies (Round 3)

OPs are filtered to those whose titles contain health-related terms and have at least one non-removed comment. For each OP, synthetic comments are generated using three strategies:

- **DG (Discourse Generation)**: A single prompt generates the entire thread of *n* replies, matching the actual clinician/layperson distribution from the real thread.
- **SAG (Single Advice Generation)**: *n* independent API calls, each generating one response with a specified role. Each call sees only the OP.
- **SAGII (SAG with Incremental Information)**: *n* sequential API calls, each generating one response with a specified role, but also seeing all previously generated responses.

## Corpus Details

The corpus was built by streaming Reddit submissions and comments from compressed `.zst` files. Submissions were retained if `selftext` exists and is at least 30 characters. Comments were retained if they are top-level, not by AutoModerator, and not removed/deleted. Each submission includes pre-computed metadata fields: `total_top_level_comments`, `layperson_top_level_comments`, and `layperson_comment_fraction`. Layperson status is determined by checking whether the commenter's flair contains the string "layperson" (case-insensitive). Full data schemas follow the Arctic Shift format ([comments](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RC/2026.ts), [submissions](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RS/2026.ts)).

## Setup

```bash
pip install -r requirements.txt
```

The generation notebooks additionally require `openpyxl` for spreadsheet export and a Google Gemini API key.

## Collaborators

Forked from [lucienbaumgartner/llm_health_advisors](https://github.com/lucienbaumgartner/llm_health_advisors).
