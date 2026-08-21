# Sample size: do we need more than 3,000 OPs?

**Short answer: yes, but only for Interpretations, and only up to roughly 6,600 OPs rather
than the full 12,464.** Explorations and Emotional Reactions are already overpowered by an
order of magnitude, as are all three interaction tests.

## Method

Two-proportion power calculations on P(label = 2), the quantity the binomial mixed models
estimate for Interpretations and Explorations. Emotional Reactions is collapsed to
P(label >= 1). Real and synthetic groups are unequal (3,882 vs 6,475 per cell, ratio 1.67),
which is handled explicitly. Our target is the standard 80% power.

Because the report runs 27 real-vs-synthetic contrasts, two thresholds are reported: the
nominal α = 0.05, and a Bonferroni-conservative α = 0.05/27 = 0.0019 standing in for the
Benjamini-Hochberg (BH) correction actually used. BH is less strict than Bonferroni, so the
truth sits between the two columns and closer to the nominal one.

## Result 1: Required sample per mechanism

Binding (hardest) contrast in each mechanism, at the conservative threshold:

| Mechanism | Binding contrast | Real comments needed | OPs needed | Have |
|---|---|---|---|---|
| Explorations | gemini_dg | 218 | **169** | 3,000 |
| Emotional Reactions | gemini_sag / sagii | 442 | **342** | 3,000 |
| Interpretations | grok_dg | 8,527 | **6,590** | 3,000 |

Explorations and Emotional Reactions were adequately powered at a *twentieth* of the current
sample. This is consistent with Pilot 1, where Explorations was the one mechanism that
survived correction on 87 comments per group.

## Result 2: Interaction tests are fine

Scaling the observed non-centrality (λ ≈ χ² − df) linearly with n:

| Mechanism | Observed χ² (df 4) | OPs needed for 80% | Overpowered |
|---|---|---|---|
| Emotional Reactions | 37.7 | 1,063 | 2.8x |
| Interpretations | 41.8 | 948 | 3.2x |
| Explorations | 52.4 | 740 | 4.1x |

The model × method interaction, usually the most data-hungry term, is comfortably powered.
More data is not needed to support claims about prompting strategy differing by model.

## Result 3: Where the current sample is thin

Achieved power for Interpretations, the only mechanism with a problem:

| Contrast | P(2) synth | Difference | Power now (α=.05) | Power now (corrected) | Power at 12,464 |
|---|---|---|---|---|---|
| gemini_dg | 0.036 | −0.067 | 1.00 | 1.00 | 1.00 |
| gemini_sag | 0.037 | −0.066 | 1.00 | 1.00 | 1.00 |
| gemini_sagii | 0.047 | −0.056 | 1.00 | 1.00 | 1.00 |
| **grok_dg** | 0.087 | −0.016 | 0.76 | **0.33** | 0.99 |
| **grok_sag** | 0.123 | +0.020 | 0.88 | **0.51** | 1.00 |
| **openai_dg** | 0.083 | −0.020 | 0.92 | **0.60** | 1.00 |
| grok_sagii | 0.135 | +0.032 | 1.00 | 0.97 | 1.00 |
| openai_sag | 0.150 | +0.047 | 1.00 | 1.00 | 1.00 |
| openai_sagii | 0.131 | +0.028 | 0.99 | 0.89 | 1.00 |

Three contrasts sit below 80% once multiplicity is accounted for. **`grok_dg` was reported
significant at p = 0.0089 with roughly 33% power.** A significant result from an
underpowered test is the classic profile of a finding that fails to replicate, and this one
sits inside the Interpretations reversal, which is the most theoretically interesting result
in the report.

## Recommendation

We should scale up the sample, but only to **~6,600 OPs, not 12,464**. That is what brings the
binding contrast (`grok_dg`) to 80% power under conservative correction. Beyond that, every
additional OP buys precision on effects that are already resolved.

Approximate incremental cost, at the observed $0.0032 per call across three models:

| Option | Additional OPs | Additional calls | Cost | What it buys |
|---|---|---|---|---|
| Stop at 3,000 | 0 | 0 | $0 | Three Interpretations contrasts stay underpowered |
| **Go to ~6,600** | 3,600 | ~57,000 | **~$185** | All 27 contrasts at 80%+ under correction |
| Full corpus | 9,464 | ~153,000 | ~$490 | Precision on already-settled effects |

The middle option costs about 40% of the full run and resolves the only genuine gap.

## Caveats

- **These are unadjusted two-proportion tests and the actual analysis is a mixed model:** With a
  log(length) covariate and a by-post random intercept. The covariate generally increases
  precision, clustering decreases it. With 1.3 real and 2.2 synthetic comments per post the
  design effect is small (roughly 1.1 at ICC = 0.1), so the ballpark holds, but these are
  approximations rather than exact power for the fitted models.
- **The Grok Interpretations reversal is length-adjusted:** In raw proportions `grok_dg` is
  *lower* than real comments (0.087 vs 0.103). It only exceeds real posts once length is
  controlled. The power calculation above uses raw proportions, so it addresses the precision
  of the contrast rather than the stability of the sign flip. That flip is worth probing directly
  regardless of sample size.
- **Significance was the stated criterion here:** No smallest-effect-of-interest threshold
  was applied. Several contrasts that are comfortably powered describe differences of two
  or three percentage points, and a reviewer may reasonably ask whether those are
  substantively meaningful even though they are statistically unambiguous.
