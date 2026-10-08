# Topic Classification of Opening Posts

The goal is to label the 6,600 threads by subject matter, to describe the corpus and to 
test whether the real-vs-synthetic empathy gap varies by topic. The code for approaches 1
and 2 is in `Data/` (see [Implementation](#implementation)). A 100-OP pilot has run; the
full run has not.

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
| `build_handcoding_sheets.py` | Draws 200 OPs and writes one blind workbook per coder: no label, confidence or quote reaches them. Run this FIRST. |
| `score_handcoding.py` | Inter-annotator agreement, the gold standard, the adjudication queue, and the classifier's precision/recall against it. |

```bash
cd Data
python build_handcoding_sheets.py                    # first: code 200 OPs blind
python score_handcoding.py                           # agreement + gold standard

python classify_topics.py --dry-run --n-ops 6600     # cost first, always
python classify_topics.py --from-analysis-dataset "../Piloting/Round 5/output/corpora/analysis_dataset_6600.jsonl"
python classify_cancer_scispacy.py --from-analysis-dataset "...same..."
python agreement.py
```

Decisions worth recording, since the plan commits to publishing them:

- **The live cancer contrast is relevant vs not. Current patient vs survivor is
  deferred** (see below), though the stance labels are still collected and stored, so
  reviving it later means reading `cancer_consensus.jsonl` rather than re-running the
  cancer task. When it does apply it is first-person threads only: cancer threads where
  the history is someone else's (caregiver), or where there is no diagnosis
  (worried-well, screening, awaiting a first biopsy), get `not_applicable`, and a cancer
  history whose currency cannot be read gets `unclear`. Forcing those into one of two
  classes would have put caregiver and worried-well posts in the patient or survivor
  cells.
- **Neoplasms outranks the organ system** (adopted 2026-10-08, after the hand-coding).
  A confirmed or suspected malignancy is coded `Neoplasms` whichever organ is involved;
  the organ system is coded only when the post's primary concern is a non-malignant
  problem alongside a cancer history. This follows the ICD-10 chapter convention the
  taxonomy is built on. It lives in `taxonomy.json` as `category_rules` and is rendered
  into the prompt, so the classifier and the human coders are held to one rubric —
  scoring the model against a rule it was never given would understate it.
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

Not built, from the sections above: the BERTopic coverage check (approach 3).

The hand-coding harness is built. `build_handcoding_sheets.py` draws 200 OPs from the same
6,600-OP seeded sample, so every coded OP joins the analysis, and writes one workbook per
coder with dropdowns bound to the codebook. Blindness is verified, not asserted: the
workbooks were audited for any label artefact and carry none, and the script reports how
many sampled OPs already have an automated label (2 of 200, from the pilot) so the overlap
can be disclosed rather than discovered. The codebook shown to coders restates the model's
own rubric verbatim, since judging humans and model against different definitions would
make the precision and recall meaningless.

`score_handcoding.py` then reports, in the order each bounds the next: inter-annotator
agreement as the ceiling; the gold standard, with coder disagreements written to an
adjudication queue rather than resolved by picking a coder; and the classifier's accuracy,
macro-F1 and per-class precision/recall against that gold. Undefined metrics print as
`n/a`, never as 0.00.

### Pilot findings (100 OPs, gemini-2.5-pro, $0.27)

A 100-OP pilot ran twice over the same posts, once before and once after a prompt fix.
The second run is the one checkpointed; the full run resumes past it.

**It works.** 200/200 calls returned, none empty, none refused, and not one label fell
outside the closed list. All 17 disciplines were used, and the two residual categories
were used sparingly and close to the source paper's rates (`Nonmedical` 4%,
`Medical; other` 1%), so the model is not dumping hard cases into the escape hatches.

**r/AskDocs does not look like Chan et al.'s corpus**, in ways that make sense:
dermatology 12% against their 4.1%, psychiatry 13% against their 26.6%. People bring
rashes and moles to an advice forum and take mental health elsewhere. Their frequencies
are a prior about general Reddit, not a prediction about this subreddit.

**Running the cancer flag on all threads was the right call.** 7 of 100 posts were
cancer-relevant, while only 1 was labelled `Neoplasms`. Gating the cancer task on the
oncology node would have missed most of them.

**Self-reported confidence is close to useless.** Every discipline label came back at
0.70 or above, most at 1.0, and every cancer label at exactly 1.0. The confidence floor
in `agreement.py` will therefore never fire: triage rests on labeller disagreement and
quote verification, not on the model's own certainty.

**Test-retest reliability is about 95%.** The two runs used identical inputs at
temperature 0, and 5 of 100 discipline labels still changed between them (3 of the 5
moved to psychiatry); `cancer_relevant` changed on 1 of 100. That is an upper bound on
reliability that is independent of accuracy, and it is worth reporting alongside the
hand-coded validation rather than in place of it.

**Quoting the evidence mostly works.** 9 of 100 quotes failed a strict substring test,
but 8 of those were real text either split across an ellipsis or carrying the prompt's
own `TITLE:` label. The true paraphrase rate is 1%. The prompt now asks for a single
contiguous span and the checker handles both shapes, which took the measured rate from
9% to 1% and multi-span quotes from 7 to 3.

### Validation result (6,600 posts labelled; 200 hand-coded)

**The classifier performs at the human ceiling.** Against the adjudicated gold standard
it is right on 154 of 200 posts, accuracy 0.770 (95% CI 0.712-0.828), macro-F1
0.726. The two coders agreed with each other on 0.795 of the same posts (kappa 0.779,
95% CI 0.718-0.839). The 2.5-point gap between 0.770 and 0.795 sits inside the interval,
so the model's disagreement with the gold is about as large as two trained humans'
disagreement with each other, and no better performance could be demonstrated on this
scheme without first improving the humans.

`cancer_relevant` is stronger: accuracy 0.977 with recall 1.00 and precision 0.80 on the
labelled subset. It missed no cancer-relevant post and over-flagged one.

Where it fails, and why it matters for the moderation analysis:

- **Psychiatry is over-assigned**: recall 1.00 but precision 0.59, so it finds every
  psychiatric post and two in five of its psychiatric calls are wrong. Neurology and
  Nonmedical posts leak into it. The same failure mode the discarded first coding pass
  showed, which suggests posts written anxiously read as psychiatric to a careless reader
  of any kind.
- **The residual categories are weakest**: `Medical; other` recall 0.55 and `Nonmedical`
  recall 0.40. The model resists them, pushing posts into substantive categories instead.
  Neither is a topic, so this matters less than it looks, but it is a reason not to model
  them as levels.
- **`Neoplasms` is 0.57 both ways** on n=7. It is the most rule-dependent category, so
  it is also the one where the codebook rules are doing the most work and the sample is
  thinnest.
- **Two systematic confusions worth a rule each if this is ever rerun**: `Medical; other`
  -> Neurology three times, which is the sleep and circadian posts the adjudicators sent
  to the residual category and the model reads as neurological; and `Infectious diseases`
  -> Gastroenterology three times, the food-poisoning posts, where the gold follows
  ICD-10's placement of intestinal infectious disease in Chapter I and the model follows
  the symptom.

The practical consequence for whoever models the topic moderation: roughly one label in
four disagrees with the gold, which attenuates interaction estimates toward zero. A null
topic effect is therefore weaker evidence of no effect than a clean moderator would give.

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
powered.

**Decision: that contrast is deferred.** It is out of the current analysis rather than
pre-registered as exploratory, and can be revisited post hoc. The stance labels are
collected and stored regardless, because they come from the same API call and cost
nothing extra: reviving the contrast later is a matter of reading
`cancer_consensus.jsonl`, not of re-running the cancer task. `agreement.py` reports the
stance counts and their realized power under a `[deferred]` heading, so the decision can
be revisited against actual numbers rather than against the source paper's prior. The
live cancer contrast is relevance, cancer vs rest.

## Open questions

1. Which categories do we have hypotheses about, committed **before** seeing results?
2. OP-level labels only, or also code comment-level engagement?
3. Who hand-codes, and do we want two coders for a reportable kappa?
