# How serious is the problem in the post?

A severity rating per thread, for testing whether the real-vs-synthetic empathy gap depends
on how serious the problem is — an abscessed tooth plausibly drawing more empathy than a
cavity.

## What it found

Every thread gets one of four triage levels. 6,599 of 6,600 are rated (one post returned
no content from the model and is left blank).

| Level | | Share |
|---|---|---|
| `emergency` | needs care now | 12.7% |
| `urgent` | today or tomorrow | 32.1% |
| `routine` | days to weeks | 33.4% |
| `self_care` | no clinician needed | 14.4% |
| `not_applicable` | no triageable problem | 7.3% |

**The scale separates things, which is the point.** A red-flag rule would have found only
2.9% of the corpus and left 97% in one bucket. This spreads across the range, so it can
actually moderate anything.

**Severity is not topic under another name.** This is the result that matters for the
design. The spread *within* a topic is **4.4× the spread between topics** — within-topic
SD averages 0.88 against 0.20 for the topic means. So a severity effect is identifiable
separately from a topic effect, which it would not be if severity merely tracked subject
matter.

Your dentistry example comes out as expected: 48 abscess posts average **2.75** against
37 cavity posts at **2.52**, with abscesses landing mostly `urgent` (30 of 48) and
cavities split between `urgent` and `routine`.

Topic means run from Neoplasms at 2.83 down to Endocrinology at 2.22 — a range of 0.6,
smaller than the within-topic SD, as above.

## Two independent signals agree. One doesn't.

The point of building three measures from different sources was to check the rating
against something that is not itself a language model.

**The rule-based signals validate it.** Neither was visible to the model, and both rise
with the scale:

| Level | `red_flag` | `care_escalated` |
|---|---|---|
| emergency | **9.9%** | **24.7%** |
| urgent | 2.3% | 15.8% |
| routine | 1.5% | 8.5% |
| self_care | 1.5% | 3.6% |

`care_escalated` is the more useful of the two — a clinician has already ordered a scan,
admitted the person, or sent them to an ER — which is the closest thing available to an
external judgement of seriousness, and it is nearly seven times more common at the top of
the scale than the bottom.

**Response latency does not work, and should not be used.** I proposed it as a behavioural
cross-check and it failed: median hours to first reply is 0.53 for `self_care`, 0.91
`routine`, 0.82 `urgent`, 0.56 `emergency`. That is U-shaped, not monotonic. The plausible
reading is that it measures how *answerable* a post is rather than how serious — the
extremes are easy to reply to and the middle needs thought. The column is in the CSV
because it may be useful for something else, but it is not a severity measure.

## Limits worth knowing before you model with this

**`not_applicable` is not a zero.** It mixes two different things: genuinely non-medical
posts, and medically substantive questions the triage scale cannot score — *"my Dad died
and we got an autopsy"*, *"why would a hospital send me an oncology survey?"*, *"MRI
outcome"* asking what a result means. 16.4% of these posts have `care_escalated` set,
which is how the problem surfaced; they are questions about care already delivered, where
"what does this person need now?" has no answer. Treat the category as missing data of two
kinds, not as the low end of the scale. It is 7.3% of the corpus.

**It is an LLM rating**, like the topic labels, so it carries the same provenance caveat:
no human rated these. Two things make it less exposed than the topic labels were. The
scale is anchored to a real clinical instrument rather than invented, so the rubric is
externally defined; and the rule-based signals above give a partly external validation
that a second model could not. A human pass over a stratified sample would still be the
right next step if this becomes load-bearing.

**Severity is a moderator here, not a confounder.** The empathy gap is measured within
thread — real and synthetic responses to the same post — so severity cannot confound that
contrast. Both arms see an identical OP. It can only moderate it, which is the question
you wanted to ask.

## Why not sentiment

Sentiment measures how upset the writing sounds, which in this corpus is close to
orthogonal to how serious the problem is. Of posts that read as frightened, only 4.7%
contain a clinical red flag; of posts containing one, only 14% sound frightened. The
failure is systematic: people who are genuinely very unwell, medically literate, or
writing about an infant tend to write calmly, while a panicked question about a trivial
exposure reads as maximally negative. The prompt therefore says in as many words to rate
the condition and ignore the tone.

## Files

| | |
|---|---|
| `rate_severity.py` | Does everything. `--dry-run` costs it, `--rules-only` skips the API. |
| `severity.csv` | **One row per thread** — joins to `topic_labels.csv` on `submission_id`. |
| `results_severity_distribution.csv` | The distribution above. |
| `results_cross_check.csv` | The agreement table above. |
| `results_by_topic.csv` | Mean and within-topic SD per topic. |

```bash
cd src/Covariates/Severity
python rate_severity.py --dry-run     # ~$10
python rate_severity.py               # resumable; checkpointed per thread
```

Columns: `severity_level`, `severity_score` (1–4), `severity_confidence`,
`evidence_quote_verified` (98.9% of quotes were found verbatim in the post), `red_flag`
and `red_flag_kinds`, `care_escalated` and `care_escalated_kinds`, `onset`
(acute/chronic/both), `hours_to_first_response`.

Cost was $9.89 for 6,600 ratings, no failures.
