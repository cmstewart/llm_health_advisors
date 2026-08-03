# Round 4: Clinician-Only Synthetic Comment Generation

This round generates synthetic **clinician** comments for r/AskDocs opening posts (OPs) using three prompting strategies (DG, SAG, SAGII) and lines them up against the real clinician responses for the same threads. It refines Round 3 in response to our meeting discussions about it and a data-quality finding about layperson comments, i.e. there are basically none!

## Why this round differs from Round 3

The central change is a shift to an **AI vs. verified-clinician** comparison. While preparing this round we discovered that genuine layperson responses barely exist in the corpus. Of the ~5,900 comments carrying the "Layperson/not verified" flair, roughly 5,160 are the OP replying to their own thread and ~440 are bot/moderator messages. Only about 300 genuine human layperson comments remain across the entire 40,537-comment corpus. This is a consequence of how r/AskDocs works: it is built around verified professionals answering, and moderators remove layperson answers, so the "layperson" flair mostly tags the people *asking* questions rather than answering them. We therefore dropped the layperson arm and generate all synthetic comments in a clinician voice.

Specific changes from Round 3:

- **Clinician-only generation.** All layperson logic was removed from the prompt builders and generation functions. Each OP receives *n* synthetic clinician comments, where *n* matches the number of real (non-self) top-level comments in the thread.
- **Empathy instruction removed.** The prompt no longer tells clinicians to show empathy, i.e. the model decides the level of empathy itself.
- **Self-posts filtered.** Top-level comments written by the OP (matched by username) are removed, since these are the poster following up on their own thread rather than genuine responses.
- **Image OPs excluded.** Submissions that embed or link an image (detected via Reddit's `preview` object or an image URL/extension in the body) are dropped, leaving text-only threads.
- **Keyword/topic filter removed.** Round 3 required health-related terms in the title or body. That filter is gone. The pool is now every image-free OP with at least one real comment (**12,464 OPs**).

## Prompting strategies

- **DG (Discourse Generation):** a single prompt generates the entire thread of *n* clinician comments at once.
- **SAG (Single Advice Generation):** *n* independent API calls, each producing one clinician comment that sees only the OP.
- **SAGII (SAG with Incremental Information):** *n* sequential API calls, each producing one clinician comment that also sees all previously generated comments in the thread.

## Run summary

- Model: Gemini 2.5 Pro
- Sample: 100 OPs (random, seed 42), matching the real thread size for each
- API calls: ~402 (100 DG + 151 SAG + 151 SAGII); runtime ~1h14m
- Output: `round3_clinician_synthetic_comments.xlsx` with one tab per OP with the real comments and the DG/SAG/SAGII outputs side by side for review

## Files

- `3_synthetic_comment_generation.ipynb` — the notebook, with cell outputs from the run above can be found [here](https://docs.google.com/spreadsheets/d/1I7wR4KRQ-8NPk6tL05DnNUK23dVoT3c9/edit?usp=drive_web&ouid=101955053890019658525&rtpof=true)

## Reproducing

Open the notebook from a directory where `../output/corpora/` contains `submissions_corpus.jsonl` and `comments_corpus.jsonl`, then run all cells top to bottom. A Google Gemini API key is required (entered at the prompt), along with `google-genai`, `tqdm`, and `openpyxl`.
