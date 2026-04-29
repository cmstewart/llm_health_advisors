# Reddit Discussion Corpus

This repository contains a filtered corpus of Reddit submissions and comments derived from the Arctic Shift Reddit datasets.

## Corpus Files

The final corpora are located at:

- `output/corpora/comments_corpus.jsonl`
- `output/corpora/submissions_corpus.jsonl`

Each file is in **JSON Lines (JSONL)** format, where each line is a complete JSON object.

---

## Data Schemas

The original data follow the Arctic Shift Reddit schemas:

- **Comments schema:** [RC/2026.ts](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RC/2026.ts)  
- **Submissions schema:** [RS/2026.ts](https://github.com/ArthurHeitmann/arctic_shift/blob/master/schemas/RS/2026.ts)  

All objects in the corpus retain the **full original schema fields**, except that the submission objects include additional metadata keys described below.

---

## Filtering Procedure

The corpus was created through the following filtering steps.

### Submission Filtering

All Reddit submissions were streamed from compressed `.zst` files.

Submissions were retained if:

- `selftext` exists
- the length of `selftext` is **≥ 30 characters**

These retained submissions were used as the **candidate submission set**.

---

### Comment Filtering

Comments were streamed from `.zst` files and retained only if all of the following conditions were met:

1. The comment belongs to a **candidate submission**.
2. The comment is a **top-level comment**, meaning: parent_id == link_id
3. The comment is **not written by AutoModerator**.
4. The comment is **not deleted or removed**, based on the `_meta.removal_type` field.

All retained comments are written to: `output/corpora/comments_corpus.jsonl`


The full comment objects are preserved according to the Arctic Shift schema.

---

### Submission Metadata

Only submissions that received **at least one retained top-level comment** are included in: `output/corpora/submissions_corpus.jsonl`


Each submission object preserves the original schema and includes the following **additional metadata fields**:

| Key | Description |
|----|----|
| `total_top_level_comments` | Number of retained top-level comments for the submission |
| `layperson_top_level_comments` | Number of those comments authored by users whose flair contains the string `"layperson"` (case-insensitive) |
| `layperson_comment_fraction` | Fraction of retained comments written by laypeople |

Layperson status is determined by checking whether:

```python
author_flair_text.lower() contains "layperson"
```

---

### Metadata File

A summary table is written to: `output/corpora/submission_metadata.csv`


Each row corresponds to a submission and contains:

| Column | Description |
|--------|-------------|
| `submission_id` | Reddit submission ID |
| `total_comments` | Number of retained top-level comments |
| `layperson_comments` | Number of retained comments authored by laypeople |
| `layperson_fraction` | Proportion of layperson comments |

This file provides a convenient overview of discussion sizes and layperson participation.

---

## Corpus Statistics

The corpus was produced from the following dataset pass:

**Submission processing**

- Submissions scanned: 67,233  
- Candidate submissions (≥30 characters): 44,561  

**Comment processing**

- Comments scanned: 255,228  
- Top-level comments retained: 40,537  

**Final corpus**

- Submissions with at least one retained comment: 19,984  

---

## Final Dataset Size

| Object | Count |
|--------|-------|
| Submissions | 19,984 |
| Top-level comments | 40,537 |

Only discussions with at least one retained top-level comment are included in the final corpus.

