# Topic Classification of Opening Posts

The goal is to label the 6,600 threads by subject matter, to describe the corpus and to 
test whether the real-vs-synthetic empathy gap varies by topic. The code for approaches 1
and 2 is in `Data/` (see [Implementation](#implementation)); nothing has been run yet.

## Approach

Topic models are typically unsupervised. For example, BERTopic finds whatever structure 
exists, and cannot be asked for a category that we have in mind. Wherever we have a 
hypothesis in advance, that is a classification problem suitable for a supervised approach.

1. **Fixed taxonomy, LLM-classified** This will be our primary approach. We can use the 17
   disciplines from the [JMIR Reddit health taxonomy](https://formative.jmir.org/2025/1/e55309).
   Its 82 finer topics would be too sparse at this `n`, so we let an LLM pick from the fixed list
   plus "other", never generating freely.
3. **Targeted flags, dual-labelled.** We can use this for any pre-specified category, e.g. cancer
   vs. other disease. Labeller A is an LLM with a published definition, returning
   `{label, confidence, evidence_quote}`. Labeller B is a
   [scispaCy](https://github.com/allenai/scispacy) entity linking to the relevant MeSH tree or
   UMLS semantic type. Agreement gives a confidence value. Disagreements and low-confidence
   cases go to manual coding. We can report kappa to quantify agreement.
5. **Use BERTopic once** as a coverage check on whether the fixed taxonomy misses real
   structure. This would be diagnostic in nature.

Keyword matching alone is not viable as spot checks show the signal often sits in the post
body rather than the title and there is a lot of potential for false positives.

## Feasibility by prevalence

Using pilot 2 Explorations proportions (real 0.191, synthetic 0.073, gap 11.8 pp), the 
smallest moderation detectable at 80% power:

| Prevalence | Threads | Min. detectable moderation |
|---|---|---|
| 2% | 132 | 9.5 pp |
| 5% | 330 | 6.1 pp |
| 10% | 660 | 4.4 pp |
| 20% | 1,320 | 3.3 pp |
| 30% | 1,980 | 2.9 pp |

Below ~5% we can only detect a moderation half the size of the main effect. Thinner
categories should be pooled or called exploratory. A topic × model × strategy interaction
across 17 disciplines is underpowered regardless.

## Validation and cost

We can hand-code 200 random OPs before seeing any automated output, ideally two coders. 
We should report precision, recall, and inter-annotator agreement as the ceiling on automated
performance. Also, we will publish prompts and adjudication decisions.

Compute is ~$7 and an afternoon. The hand-coding is the real cost, and what makes it
publishable.

## Implementation

In `Data/`, at the repository root. Run in this order:

| File | What it does |
|---|---|
| `taxonomy.json` | The fixed category list. No discipline name is hardcoded in the Python. |
| `classify_topics.py` | Labeller A. Task `discipline` picks one of the 17 + "Other"; task `cancer` returns relevance and the patient/survivor stance. Both return `{label, confidence, evidence_quote}`. |
| `classify_cancer_scispacy.py` | Labeller B. UMLS entity linking for cancer relevance; cue-based stance. |
| `agreement.py` | Joins the two, reports both kappas, writes `cancer_consensus.jsonl` and `manual_coding_queue.csv`. |

```bash
cd Data
python classify_topics.py --dry-run --n-ops 6600     # cost first, always
python classify_topics.py --from-analysis-dataset "../Piloting/Round 5/output/corpora/analysis_dataset_6600.jsonl"
python classify_cancer_scispacy.py --from-analysis-dataset "...same..."
python agreement.py
```

Decisions worth recording, since the plan commits to publishing them:

- **The cancer binary is patient vs survivor on first-person threads only.** Cancer
  threads where the history is someone else's (caregiver), or where there is no diagnosis
  (worried-well, screening, awaiting a first biopsy), get `not_applicable`; a cancer
  history whose currency cannot be read gets `unclear` and goes to the queue. Forcing
  those into one of two classes would have put caregiver and worried-well posts in the
  patient or survivor cells.
- **"Survivor" is used in the restricted sense**, not the NCI/NCCS sense. NCI counts
  anyone from diagnosis onward as a survivor, which would place every current patient in
  the survivor class and collapse the contrast. Ours is: primary treatment complete, no
  evidence of active disease. The prompt states this verbatim.
- **Only `cancer_relevant` is genuinely dual-labelled.** Entity linking cannot read
  current-vs-past, so labeller B scores surface cues for stance. That kappa is reported
  as a floor on agreement, not as validation of the stance labels.
- **The cancer task runs on all threads by default**, not on the oncology node or a
  keyword prefilter. A prefilter can only lower recall, and the loss would not be
  measurable from the output.
- **Evidence quotes are checked against the post text.** A quote the model invented is
  recorded as unverified, and an unverifiable quote on a cancer-flagged post is queued
  for a human whatever the confidence said.
- The power table above is reproduced in `agreement.py` to within 0.03 pp and applied to
  realized prevalences, so every topic row carries its own minimum detectable moderation.

- **No "Other" is appended to the taxonomy.** The published list already ends in its own
  two residual categories, `Medical; other` and `Nonmedical`, so adding a third catch-all
  would split the residue across overlapping options. Off-list labels are coerced to
  `Medical; other` and counted.

Not built, from the sections above: the BERTopic coverage check (approach 3), and the
blind hand-coding of 200 OPs with two coders. The second is the ceiling on everything
here, and `agreement.py` says so when it finishes.

### The taxonomy

All 17 names are in `taxonomy.json`, transcribed verbatim from **Table 2** of Chan et al.
(the disciplines follow the ICD-10 chapter codes). In the paper's own
descending-frequency order, with its relative frequencies:

| | | | | |
|---|---|---|---|---|
| Psychiatry and mental health | .266 | | Nonmedical | .0387 |
| Genitourinary and reproductive health | .136 | | Neurology | .0373 |
| Infectious diseases | .0895 | | Cardiovascular | .0353 |
| Endocrinology, nutrition, and metabolism | .0891 | | Pulmonology | .0183 |
| Hematology and immunology | .0588 | | Medical; other | .0160 |
| Ophthalmology | .0572 | | **Neoplasms** | **.0157** |
| Musculoskeletal and connective tissue conditions | .0476 | | Dentistry | .0095 |
| Gastroenterology | .0426 | | Otolaryngology | .0022 |
| Dermatology | .0410 | | | |

Those frequencies are Chan et al.'s, measured on their 78,890-post general-Reddit corpus.
They are a prior, not our expected prevalences: r/AskDocs is a single advice subreddit and
will not share that distribution.

**The cancer node is `Neoplasms`, and in the source corpus it is the third rarest
discipline at 1.57%.** If r/AskDocs is similar, that is ~104 of our 6,600 threads, below
the smallest row in the feasibility table above, and the patient-vs-survivor split would
divide it further. Two things soften this and neither is a fix: the cancer flag runs on
all threads rather than on the node, so it also catches malignancies filed under
Dermatology (melanoma) or Hematology (leukaemia); and the ICD-10 `Neoplasms` chapter
covers benign tumours too, so the node is broader than cancer in the other direction. The
honest reading is that the patient-vs-survivor contrast is likely descriptive rather than
powered, and `agreement.py` prints the realized arithmetic once labels exist. Worth
settling before the API spend: pool, oversample cancer threads beyond the seeded sample,
or pre-register it as exploratory.

## Open questions

1. Which categories do we have hypotheses about, committed **before** seeing results?
2. OP-level labels only, or also code comment-level engagement?
3. Who hand-codes, and do we want two coders for a reportable kappa?
