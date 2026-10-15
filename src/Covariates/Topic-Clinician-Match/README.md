# Does the first responder's specialty match the topic?

We want to know who answers each r/AskDocs thread first, and whether or not their declared
specialty fit what the post is about, e.g. a psychiatrist on a mental health question, 
a dermatologist on a rash. This involves making one covariate per thread for the 
real-vs-synthetic empathy analysis. 

## High-level finding

**Responders strongly match the subject:** Of the 1,130 threads whose first responder
declared a specialty that maps to a body system, **57% were answered on-topic against 9%
expected by chance, a 6.4× lift.** A lift of 1.0 is what you would see if responders
ignored the subject entirely.

| Specialty | threads | authors | match | topic base rate | lift |
|---|---|---|---|---|---|
| Dermatology | 528 | **2** | 72.5% | 10.8% | 6.7× |
| Gastroenterology | 109 | **1** | 70.6% | 10.8% | 6.6× |
| Hematology and immunology | 108 | 2 | 42.6% | 4.9% | 8.6× |
| Pulmonology | 92 | 2 | 30.4% | 2.6% | **11.5×** |
| Psychiatry and mental health | 84 | 8 | 38.1% | 9.0% | 4.2× |
| Neurology | 83 | 9 | 32.5% | 8.0% | 4.1× |
| Cardiovascular | 79 | 4 | 46.8% | 7.5% | 6.2× |
| Endocrinology, nutrition, metabolism | 22 | 3 | 13.6% | 6.7% | 2.1× |
| Genitourinary and reproductive health | 16 | 5 | 68.8% | 10.5% | 6.5× |
| Musculoskeletal and connective tissue | 6 | 3 | 33.3% | 5.5% | 6.1× |
| Otolaryngology | 3 | 3 | 33.3% | 4.3% | 7.8× |

**Read lift, not the match rate.** Pulmonology's 30% looks weak beside dermatology's 72%,
but pulmonology posts are only 2.6% of the corpus, so 30% is the strongest effect in the
table. The match rate measures how common the topic is as much as it measures the
responder.

**Physicians answer first almost everywhere** — 79% of threads, with nurses at 6.6% and
medical students at 5.2%. Not one thread was answered first by someone flaired as a
layperson, which says something about how the subreddit is moderated rather than about
who reads it.

## Three caveats, and the first is the one that matters

**A few accounts dominate.** Look at the `authors` column: 528 dermatology threads come
from **two** accounts, 109 gastroenterology threads from **one**. Those rows describe
individual prolific contributors, not "what dermatologists do". Psychiatry and neurology,
at 8 and 9 authors, are the more trustworthy rows. Across the whole table, 1,130 threads
come from just 42 accounts. This is a caveat that we should mention in our eventual 
publications.

**Most responders declare no specialty.** Only 17% of first responders give one that maps
to a body system. 76% say just "Physician" or similar. Generalists (emergency, family,
internal medicine, paediatrics — 3.5%) and cross-cutting specialties (radiology,
pathology, surgery — 3.8%) are reported separately rather than counted as mismatches,
since matching a body system is not their job. So this covers about a sixth of the corpus
and the rest is invisible.

**The topic labels are themselves noisy.** They come from an LLM and disagree with the
reference labels on about one post in four, so some apparent mismatches are mislabelled
posts rather than off-topic responders. That makes 6.4× a conservative estimate of the
true selection effect.

## Files

| | |
|---|---|
| `map_responder_roles.py` | Builds everything below. |
| `results_specialty_match.csv` | The table above, with an `ALL` row. Regenerated on every run. |
| `results_roles.csv` | Who answers first, by role. |
| `../../output/corpora/export/first_responder_roles.csv` | **One row per thread** — role, specialty, mapped discipline, whether it matches. Joinable to `topic_labels.csv` on `submission_id`. Lives with the other deliverables so the analyst collects them from one place. |

```bash
cd src/Covariates/Topic-Clinician-Match
python map_responder_roles.py
```

## How it works

r/AskDocs flair is mod-verified and semi-structured — `Physician - Dermatologist | Top
Contributor`, so the first segment gives the role and a later segment sometimes gives the
specialty. "Top Contributor" and "Moderator" are decoration and dropped. The first
responder is the earliest top-level comment that is not the OP's own and not removed,
which is the same filter the generation run used, so the thread set matches exactly.

Specialties map onto the 17 disciplines only where the correspondence is defensible. Two
known imperfections: Heme/Onc is mapped to Hematology and immunology, so threads it
answers that are labelled `Neoplasms` count as misses; and a specialty named only in the
role segment (`Physician/Neurosurgeon`) is picked up, but an undeclared subspecialty
cannot be.
