# Topic classification of r/AskDocs opening posts

This folder holds the scale-up generation script and the topic classification pipeline
built on top of it. The detailed decision log is in
[`../Piloting/Round 5/topic_classification_plan.md`](../Piloting/Round%205/topic_classification_plan.md).
This short README explains what was done and why.

## The goal

Label all 6,600 r/AskDocs threads by medical subject, so the real-vs-synthetic empathy gap
can be tested for variation by topic.

The lables are in `output/corpora/export/topic_labels.csv`, which joins to the analysis
dataset on `submission_id` and is ready for use as an independent variable.

## What I did

I sampled 200 of the threads, use the ([Chan et al. 2025](https://formative.jmir.org/2025/1/e55309) taxonomy
to label them and iron out issues, generated topic labels with 2 LLMs (Claude Sonnet 5.5 and GPT-6),
isolated the 41 threads where the models disagreed, manually adjudicated between the two labels. This
provided a level of ground truth. I then labeled all of the posts with a separate LLM (Gemini 2.5 Pro) with
the same prompt.

## Why I did it

**A fixed taxonomy rather than topic modelling.** Unsupervised methods like BERTopic find
whatever structure exists and cannot be asked for a category of interest.

**The 17 disciplines come from a published taxonomy**
([Chan et al. 2025](https://formative.jmir.org/2025/1/e55309), Table 2), which groups them
on ICD-10 chapters. Only the category names were borrowed. The labelling method is ours,
and their 82 finer topics were too sparse at this sample size. Their reported frequencies
are a prior about general Reddit, not a prediction about this subreddit. The data
is a reflection of that.

**An LLM does the labelling.** Gemini 2.5 Pro was shown each post and picked one of the 17 
topic labels from a closed list, returning a confidence and a supporting quote. 6,600 posts 
is far too many to read by hand, and keyword matching was ruled out early on.

**Two LLMs establish the ceiling.** Two pseudo-raters (Claude Sonnet 5.5 and GPT 6)
labelled the same 200 posts blind. The 41 posts that they assigned different labels to were 
adjudicated by a human.

**Codebook rules are given to the model and the humans alike.** Where category boundaries
were genuinely ambiguous, rules were written and resolved by ICD-10 chapter. For example, 
malignancies and benign neoplasms to `Neoplasms`, poisoning and exposures to `Medical; other`, and
care-process questions to the complaint the post describes. Those rules can be found in `taxonomy.json`
as data and are rendered into the prompt, because scoring a classifier against a rule it
was never given would understate it.

## What came out

| | |
|---|---|
| Classifier accuracy | **0.770** (154/200), 95% CI 0.712–0.828 |
| Human/LLM ceiling | 0.795 agreement, Cohen's κ 0.779 |
| macro-F1 | 0.726 |
| `cancer_relevant` | accuracy 0.980, recall 0.90 |
| Labels | 6,600 discipline, 6,598 cancer |
| Cost | ~$21 in API calls |

**The classifier performs at the human/separate LLM ceiling.** The 2.5-point gap between 0.770 and
0.795 sits inside the confidence interval, meaning that the model disagrees with the adjudicated gold
about as much as the separate LLMs disagree with each other.

Two things the full run settled:

- **r/AskDocs does not resemble the source corpus.** Dermatology is the most common topic
  label at 10.8% where the paper had psychiatry at 26.6%. People bring visible physical
  complaints to an advice subreddit and take mental health elsewhere.
- **Cancer vs rest is adequately powered, but current-patient vs survivor is not.** The cancer
  flag fires on 7.8% of posts, five times what the oncology node alone implied. But most
  of those are a relative's cancer or no diagnosis at all, leaving 18 current patients
  against 38 survivors. The contrast that needs a cancer-enriched sample, not more
  labelling, but the labels are collected so it can be revisited.

## The caveat that matters downstream

Roughly **one label in four disagrees with the human/LLM golden labels**. That is not a defect,
it is a ceiling. Misclassification in a moderator **attenuates interaction estimates toward zero**. 
A null topic effect is therefore weaker evidence of no effect than a clean moderator would give. 


## Files

| File | |
|---|---|
| `taxonomy.json` | The 17 disciplines and the codebook rules. No category name is hardcoded in Python. |
| `classify_topics.py` | Labeller A. Discipline and cancer tasks over the corpus. |
| `build_handcoding_sheets.py` | Blind workbooks for the human coders. |
| `score_handcoding.py` | Inter-annotator agreement, gold standard, adjudication queue, classifier accuracy. |
| `agreement.py` | Cancer consensus, prevalence and power. |
| `export_labels.py` | The deliverable: CSV plus a data dictionary. |
| `classify_cancer_scispacy.py` | Labeller B, UMLS entity linking. Built, never installed — see below. |
| `sweep.sh` | Re-runs the classifier until the corpus is labelled. |
| `generate_synthetic.py` | The Round 5 generation script, which the above reuse for corpus loading and sampling. |

Run order: hand-coding first, then classification, then scoring.

```bash
cd Data
python build_handcoding_sheets.py       # 1. two blind workbooks, 200 posts
python score_handcoding.py              # 2. agreement, gold standard, adjudication queue
python classify_topics.py --dry-run     # 3. cost estimate, always first
./sweep.sh                              # 4. label the corpus, resuming until complete
python score_handcoding.py --adjudicated <adjudication file>   # 5. validation
python export_labels.py                 # 6. the deliverable
```

Corpus input is `../Piloting/Round 2/`; output goes under
`../Piloting/Round 5/output/corpora/`. Needs `GEMINI_API_KEY`, which `sweep.sh` reads from
`Data/.env`.

## What was not done, and why

- **Labeller B (scispaCy).** The plan specified the cancer flag as dual-labelled, with
  UMLS entity linking as a non-LLM second opinion. The code is here but was never
  installed, because the flag ended up validated against two LLMs and a human, a strictly
  better reference than entity linking, which cannot read current-versus-past anyway.
- **BERTopic coverage check.** Meant to test whether the fixed taxonomy misses real
  structure. The residual categories answer it for free: a taxonomy that failed to cover
  its domain would show a bloated `Medical; other`, and the two residuals together take
  only 6.1% of posts, with all 17 categories in use.
- **The empathy measure itself.** Still the question-mark proxy rather than the three
  EPITOME classifiers. This is the real outstanding item, and it belongs to whoever runs
  the modelling: the proxy under-detects the mechanism in synthetic comments specifically,
  and its blind spots may vary by topic — the same dimension the analysis tests. Note also
  that the published EPITOME code ships no trained checkpoints.
