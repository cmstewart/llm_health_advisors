# src

Four folders, by what the code produces rather than by when it was written.

| | |
|---|---|
| [`Generation/`](Generation/) | Generates the synthetic clinician comments — the experiment itself. `generate_synthetic.py`, its scale-up runbook and power analysis. |
| [`Covariates/Topics/`](Covariates/Topics/) | Labels every thread with one of 17 medical disciplines. The largest piece; **[start here](Covariates/Topics/README.md)**. |
| [`Covariates/Topic-Clinician-Match/`](Covariates/Topic-Clinician-Match/) | Maps each thread's first responder to a role and specialty, and asks whether the specialty matches the topic. |
| [`Analysis/`](Analysis/) | Merges the corpus and generated output into one analysis file, and the exploratory notebook over it. |
| `output/corpora/` | Output from all of the above, shared. `export/` holds the deliverables and is checked in; `labels/` and `generated/` are gitignored. |

Two things that follow from the layout:

- **Scripts are run from their own directory**, since their defaults are relative to it:
  `cd src/Covariates/Topics && python classify_topics.py`.
- **`output/` is shared and sits here, above the four folders**, because the covariate
  analyses feed each other — `Topic-Clinician-Match` reads the topic labels that `Topics`
  produces.

`Covariates/Topics/classify_topics.py` imports `Generation/generate_synthetic.py` to reuse
its corpus filters and seeded sampling, so the labelled thread set cannot drift from the
generated one. That import puts `src/Generation` on the path explicitly; the alternative
was to file the generation script under `Covariates/`, which would misdescribe it.

API keys live in `src/.env` (gitignored) and `Covariates/Topics/sweep.sh` reads them.
