# Self-reported demographics of the person the post is about

r/AskDocs asks posters to state age and sex, and many volunteer height and weight too.
This analysis pulls that out of all 6,600 opening posts, one row per thread, for use as covariates.

## What it found

| Field | Coverage | |
|---|---|---|
| Age | **88.2%** | median 25, p10 16, p90 41; 14% are under 18 |
| Sex | **83.0%** | 62% female, 38% male |
| Weight | 34.1% | median 66.7 kg |
| Height | 28.7% | median 168 cm |
| BMI (derived) | 23.1% | median 23.8; 21% at 30 or above |
| Smoking status | 11.2% | — |
| Drinking status | 4.6% | — |
| Race | **0.7%** | only from an explicit `Race:` field — see below |

**Age and sex are usable as covariates.** At 88% and 83%, age and sex are close to complete. 
Weight and height sit around a third, and BMI needs both, so it exists for under a quarter of threads,
i.e. enough to describe the corpus, but not enough to condition on without dropping three quarters of the data.

**A tenth of posts are about someone else.** 9.9% are third-party, e.g. a parent about a
child, a spouse about a spouse, and 16.3% are ambiguous. The demographics belong to the
**subject of the post**, not necessarily the poster, so every row carries a `subject`
column. Filter to `first_person` (73.8%) if you want the poster specifically.

## The caveat that matters for using these

**Missingness is not random, and it tracks the topic.** Weight is stated by 47% of
endocrinology posters and 26% of oncology posters. This is because people volunteer a 
weight when they think it is relevant. So conditioning on weight silently re-weights the corpus toward
metabolic topics. Age and sex are near-universal because the subreddit rules demand them,
which is exactly why they are the two safe ones.

`results_coverage.csv` gives per-field coverage for all 17 topics so this can be checked
rather than assumed.

**Race is effectively absent.** Only 0.7% state one. Bare color words are risky as they
are exposed to a risk of false positives in terms like "black stool" and "white patches".
Only an explicit `Race:` field is read.

## Files

| | |
|---|---|
| `extract_demographics.py` | Does the extraction. `--audit N` prints a sample with what was pulled from it, for hand-checking. |
| `poster_demographics.csv` | **One row per thread** — joins to `topic_labels.csv` on `submission_id`. Carries each field plus a `*_source` column saying which pattern found it, so any value can be traced. |
| `results_coverage.csv` | Coverage per field, overall and by topic. |
| `results_distributions.csv` | Distribution of each field. |

```bash
cd src/Covariates/Poster-Demographics
python extract_demographics.py
python extract_demographics.py --audit 30   # hand-check precision
```

## How it works, and what it gets wrong

This data is gathered via rules, not a model. The formats are formulaic because the subreddit 
mandates them, so regex-es get most of it. Unlike a model a rule is free, deterministic, auditable, 
and needs no separate validation exercise. The cost is recall on unusual phrasings, which
shows up as coverage rather than hiding in the numbers.

Values are normalised to years, cm and kg, and implausible ones are dropped rather than
stored. Ages are only taken from an explicit marker, something like a joined sex letter (`23F`), `yo`,
`years old`, an `Age:` field. They are never from a loose number as, again, this could result in
false positives.

Four traps the audit caught, each now handled, for some fun with NLP:

- **`I'm 38 female` was recorded as male.** The letter-first pattern matched the `m` in
  `I'm` followed by `38`. Every poster who wrote "I'm NN" was at risk; 49 sexes were wrong.
- **The earliest stated age wins, not the strongest pattern.** A post titled `[23M NJ]`
  that later mentioned "some random 19-year-old" was recorded as 19 — twice over, before
  the rule became positional. The subject is introduced first; other people come up in
  passing later.
- **`99 F` after the word fever is a temperature**, not a 99-year-old female.
- **Medication doses are not weights.** A unit is always required, so `30mg` is never read
  as a weight.

Known limits: `Early 20s M` yields a sex but no age, because it is not a number. Height
and weight are taken from the first stated value, so a post listing pre- and
post-pregnancy weight gives the earlier one. And in a long first-person post that also
mentions a relative's age, the earliest-age rule can still pick the wrong person. The
`*_source` columns allow for finding and checking such cases.
