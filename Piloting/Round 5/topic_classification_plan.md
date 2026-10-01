# Topic Classification of Opening Posts

The goal is to label the 6,600 threads by subject matter, to describe the corpus and to 
test whether the real-vs-synthetic empathy gap varies by topic. Nothing here has been run.

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

We san hand-code 200 random OPs before seeing any automated output, ideally two coders. 
We should report precision, recall, and inter-annotator agreement as the ceiling on automated
performance. Also, we will publish prompts and adjudication decisions.

Compute is ~$7 and an afternoon. The hand-coding is the real cost, and what makes it
publishable.

## Open questions

1. Which categories do we have hypotheses about, committed **before** seeing results?
2. OP-level labels only, or also code comment-level engagement?
3. Who hand-codes, and do we want two coders for a reportable kappa?
