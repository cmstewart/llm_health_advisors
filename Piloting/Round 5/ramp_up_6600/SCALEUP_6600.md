# Scaling from 3,000 to 6,600 OPs

Target set by the power analysis: 6,600 OPs brings the binding contrast
(Interpretations, `grok_dg`) to 80% power under multiplicity correction. Beyond that,
additional OPs buy precision on effects that are already resolved.

| | |
|---|---|
| Already generated | 3,000 OPs / 15,952 calls per model |
| **New** | **3,600 OPs / 19,252 calls per model (57,756 across three models)** |
| Total at 6,600 | 6,600 OPs / 35,204 calls per model |
| Estimated cost | **~$185** (~$37 Gemini, ~$100 OpenAI, ~$47 Grok) |

No code change is needed for the scale-up itself. The 3,000-OP sample is a **verified strict
subset** of the 6,600-OP sample at seed 42, so existing checkpoints stay valid and completed
OPs are skipped.

## Before you start

1. **Top up Gemini credits.** This is what failed last time, 675 calls deep into SAGII.
   Budget ~$40 at [ai.studio](https://ai.studio/projects).
2. **Confirm OpenRouter and OpenAI balances.** Roughly $50 and $100 respectively.
3. **Locate your existing checkpoints.** `output/corpora/generated/*.jsonl`, nine files of
   3,000 records each. These must go onto the VM or the run will regenerate work you have
   already paid for.

## Fix included in this version

`process_op` no longer checkpoints an OP where *every* call failed. Previously a credit
outage wrote records full of empty comments, which marked those OPs done and made the gaps
permanent; recovering meant manually stripping and re-running. Now such OPs are simply left
unwritten and counted as `skipped`, and the next run picks them up.

Partial failures (some comments returned, one blank) are still written, since those are rare
per-call refusals that would otherwise retry forever. Filter empty bodies at analysis time.

## Steps

### 1. Create the VM (on your Mac)

```bash
gcloud compute instances create askdocs-gen \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB

gcloud compute ssh askdocs-gen --zone=us-central1-a --command="mkdir -p ~/corpora ~/generated"
```

`e2-medium` (4 GB) rather than `e2-small`, so you can run all three models in parallel.

### 2. Upload script, corpus, and existing checkpoints

From your local `src/`:

```bash
gcloud compute scp --zone=us-central1-a generate_synthetic.py askdocs-gen:~/

gcloud compute scp --zone=us-central1-a \
  ../output/corpora/submissions_corpus.jsonl \
  ../output/corpora/comments_corpus.jsonl \
  askdocs-gen:~/corpora/

# The checkpoints. Skipping this step costs you ~$150 in regenerated work.
gcloud compute scp --zone=us-central1-a \
  ../output/corpora/generated/*.jsonl \
  askdocs-gen:~/generated/
```

### 3. Environment (on the VM)

```bash
gcloud compute ssh askdocs-gen --zone=us-central1-a

sudo apt-get update && sudo apt-get install -y python3-venv tmux
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install openai tqdm

nano ~/.env      # three export lines, as before
chmod 600 ~/.env && source ~/.env
```

### 4. Pre-flight check

```bash
python generate_synthetic.py --dry-run --n-ops 6600 --corpora-dir ~/corpora
```

Expect **6,600 OPs, 35,204 calls/model**. Then confirm the checkpoints registered:

```bash
wc -l ~/generated/*.jsonl      # nine files, 3,000 lines each
```

Run one model/strategy briefly and check the header says `3,000 already done, 3,600 to go`.
**If it says `0 already done`, stop** — the checkpoints did not upload and you are about to
pay twice.

### 5. Launch

```bash
tmux new -d -s gemini "source ~/venv/bin/activate && source ~/.env && python generate_synthetic.py --n-ops 6600 --models gemini --concurrency 10 --corpora-dir ~/corpora --out-dir ~/generated 2>&1 | tee ~/gen_gemini.log"

tmux new -d -s openai "source ~/venv/bin/activate && source ~/.env && python generate_synthetic.py --n-ops 6600 --models openai --reasoning-effort openai=low --concurrency 10 --corpora-dir ~/corpora --out-dir ~/generated 2>&1 | tee ~/gen_openai.log"

tmux new -d -s grok "source ~/venv/bin/activate && source ~/.env && python generate_synthetic.py --n-ops 6600 --models grok --reasoning-effort grok=low --concurrency 10 --corpora-dir ~/corpora --out-dir ~/generated 2>&1 | tee ~/gen_grok.log"

tmux ls && free -h
```

Concurrency 10 is the tested ceiling; above it the async HTTP stack produced heap corruption.
Expect roughly 3 to 6 hours.

### 6. Verify, retrieve, tear down

```bash
tail -n 1 ~/gen_*.log                     # progress
tail -n 8 ~/gen_*.log                     # final summary; check the skipped count
```

**If any model reports skipped OPs**, that model hit a systemic failure. Fix the cause,
usually credits, and re-run the identical command; nothing was poisoned this time.

```bash
python3 -c "
import json, glob
t=0
for f in sorted(glob.glob('/home/USER/generated/*.jsonl')):
    recs=[json.loads(l) for l in open(f)]
    bad=sum(1 for r in recs if len(r['comments'])!=r['n_comments']
            or not all(c['body'].strip() for c in r['comments']))
    t+=bad; print(f\"{f.split('/')[-1]:22} {len(recs):5} records  {bad} defective\")
print('TOTAL defective:', t)
"
```

Nine files at 6,600 records each. Then, from your Mac:

```bash
gcloud compute scp --recurse --zone=us-central1-a askdocs-gen:~/generated ../output/corpora/
gcloud compute instances delete askdocs-gen --zone=us-central1-a
```

### 7. Rebuild the analysis inputs

```bash
python build_analysis_dataset.py --gzip
python build_review_spreadsheet.py --n-ops 10
```

Then re-run `5_exploratory_analysis.ipynb`, and re-run the empathy classifiers over the
larger set.

## What this does and does not settle

It brings all 27 real-vs-synthetic contrasts to 80%+ power under correction, including the
three Interpretations contrasts currently at 0.33, 0.51, and 0.60.

It does **not** resolve whether the Grok Interpretations reversal is real. That finding only
appears once `log(length)` is controlled; in raw proportions `grok_dg` is *lower* than real
(0.087 vs 0.103). More data tightens the estimate around whichever sign is correct, but the
sensitivity to the length covariate is a modelling question, not a sample size one, and is
worth probing directly.
