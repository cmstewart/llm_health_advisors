# Round 5: Multi-Model Scale-Up and Exploratory Analysis

Round 4 established the clinician-only design on 100 threads with a single model. Round 5
scales that to **3,000 threads across three models**, and takes a first look at the
results.

## What was generated

For each of 3,000 r/AskDocs threads, synthetic clinician comments were produced by three
models under three prompting strategies:

| | |
|---|---|
| Models | Gemini 2.5 Pro, GPT-5, Grok 4.6 |
| Strategies | DG, SAG, SAGII |
| Threads | 3,000 (seed 42, sampled from 12,464 eligible) |
| Synthetic comments | 58,283 |
| Real comments | 3,882 |
| API calls | ~47,900 |
| Cost | ~$153 |

The number of synthetic comments per thread matches its real comment count. Strategies are
as defined in Round 4: **DG** generates the whole thread in one prompt, **SAG** generates each
comment independently seeing only the OP, and **SAGII** generates them sequentially with each
comment seeing all prior ones.

## Findings

**Real clinicians ask questions; models explain!** 23.3% of real comments contain a question,
vs. 0.8% to 7.4% for every model and every strategy, a gap of 3x-27x. Question rate is
a crude proxy for the EPITOME *Explorations* mechanism, which was the one measure that
separated real from synthetic in the Round 4 pilot. It replicates here at roughly 75x the
sample size.

**The gap is not a length artifact** Within matched length, the distinction actually widens.
Real comments ask *more* questions as they get longer (18.5% under 25 words, rising to 34.9% 
above 100), while synthetic comments ask *fewer* (Gemini falls from 14.4% to 1.1%). At 50 
to 100 words the comparison is 27.1% real against 0.3% to 2.6% synthetic. While human 
clinicians spend extra words inquiring, LLMs appear to spend them explaining.

**Models differ in verbosity but none reproduce real variance** GPT-5 writes about 86 words
per comment to Gemini's and Grok's 32. Real comments have a median of 34 but a standard
deviation of 64.7, against 8 to 28 for the models.

Next step is running the three EPITOME RoBERTa classifiers over these ten groups to replace
the question-mark proxy with the actual measure.

## Files

- `generate_synthetic.py` — multi-model generation across all three strategies
- `build_analysis_dataset.py` — merges corpus and generated output into one analysis file
- `4_exploratory_analysis.ipynb` — the analysis above, with outputs
- `GENERATION.md` — full usage documentation for the generation script

## Data

The notebook reads `analysis_dataset.jsonl`, one record per thread containing the OP, its
real comments, and all nine synthetic sets. It is 26.7 MB (7.7 MB gzipped) and is **not in
this repository**; it is shared separately via Google Drive. Place it in
`output/corpora/` and the notebook will find it.

To rebuild it from source:

```bash
python build_analysis_dataset.py --gzip
```

## Reproducing the generation

```bash
pip install openai tqdm
export GEMINI_API_KEY=...  OPENAI_API_KEY=...  OPENROUTER_API_KEY=...

# Always estimate first: prints workload and cost, makes no API calls
python generate_synthetic.py --dry-run --n-ops 3000

python generate_synthetic.py \
  --n-ops 3000 --models gemini,openai,grok \
  --reasoning-effort openai=low --reasoning-effort grok=low \
  --concurrency 10
```

All three providers are called through the OpenAI wire format (Gemini via its
compatibility endpoint, Grok via OpenRouter), so there is one code path rather than three
SDKs.

Notes:

- **Reasoning effort matters for cost!!!** GPT-5 and Grok bill hidden reasoning tokens as
  output. Setting `low` cut the projected spend from roughly $450 to $153.
- **Concurrency above ~10 was unstable** in testing, producing heap corruption in the async
  HTTP stack. 10 is the tested ceiling.
- **Runs resume** Progress is checkpointed per model and strategy after every thread, so
  re-running the same command skips completed work.
- **Model IDs drift** Verify them against each provider's current list and override with
  `--model-id` rather than editing the file.

Set `--n-ops 0` to run the full 12,464-thread corpus.
