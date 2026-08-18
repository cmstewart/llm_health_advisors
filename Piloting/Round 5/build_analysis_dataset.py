#!/usr/bin/env python3
"""
Merge the corpus and all generated output into one analysis-ready JSONL.

Each output record is a single submission with its opening post, the real
(non-self) clinician comments, and every model x strategy synthetic set:

    {
      "submission_id": "1qh5la8",
      "title": ...,
      "selftext": ...,
      "n_comments": 1,
      "real_comments":  [{"body": ..., "author_flair_text": ..., "score": ...}],
      "synthetic": {
        "gemini": {"dg": [...], "sag": [...], "sagii": [...]},
        "openai": {...},
        "grok":   {...}
      }
    }

Filters match the generation run exactly: self-posts removed, image OPs excluded,
no keyword filter.

Usage:
    python build_analysis_dataset.py
    python build_analysis_dataset.py --gzip        # also write a .jsonl.gz for sharing
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

IMG_URL_PATTERN = re.compile(
    r"(i\.redd\.it|preview\.redd\.it|imgur\.com|\.jpg|\.jpeg|\.png|\.gif|\.webp|\.heic)",
    re.IGNORECASE,
)

MODELS = ("gemini", "openai", "grok")
STRATEGIES = ("dg", "sag", "sagii")


def has_image(sub: dict) -> bool:
    if sub.get("preview"):
        return True
    return bool(IMG_URL_PATTERN.search(sub.get("selftext", "") or ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora-dir", default="../output/corpora")
    ap.add_argument("--generated-dir", default="../output/corpora/generated")
    ap.add_argument("--out", default="../output/corpora/analysis_dataset.jsonl")
    ap.add_argument("--gzip", action="store_true", help="also write a gzipped copy")
    args = ap.parse_args()

    corpora = Path(args.corpora_dir)
    gen_dir = Path(args.generated_dir)
    out_path = Path(args.out)

    # ---- submissions ----
    subs_by_id: dict[str, dict] = {}
    with open(corpora / "submissions_corpus.jsonl") as f:
        for line in f:
            s = json.loads(line)
            subs_by_id[s["id"]] = s
    print(f"Loaded {len(subs_by_id):,} submissions")

    # ---- real comments, self-posts removed ----
    comments_by_sub: dict[str, list[dict]] = defaultdict(list)
    n_self = 0
    with open(corpora / "comments_corpus.jsonl") as f:
        for line in f:
            c = json.loads(line)
            if not c.get("parent_id", "").startswith("t3_"):
                continue
            sid = c["parent_id"][3:]
            op_author = subs_by_id.get(sid, {}).get("author")
            if c.get("author") and op_author and c["author"] == op_author:
                n_self += 1
                continue
            comments_by_sub[sid].append(c)
    print(f"Dropped {n_self:,} self-posts")

    # ---- generated output ----
    # generated[submission_id][model][strategy] = [comment, ...]
    generated: dict[str, dict[str, dict[str, list]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for model in MODELS:
        for strat in STRATEGIES:
            p = gen_dir / f"{model}_{strat}.jsonl"
            if not p.exists():
                print(f"  WARNING: missing {p.name}, skipping")
                continue
            n = 0
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    generated[r["submission_id"]][model][strat] = r["comments"]
                    n += 1
            print(f"  {p.name:22} {n:,} records")

    # ---- merge ----
    n_written = 0
    n_incomplete = 0
    with open(out_path, "w") as out:
        for sid, per_model in generated.items():
            sub = subs_by_id.get(sid)
            if sub is None:
                continue

            real = [
                c
                for c in sorted(
                    comments_by_sub.get(sid, []), key=lambda x: x.get("created_utc", 0)
                )
                if c.get("body", "") not in ("[removed]", "[deleted]")
            ]

            synthetic = {
                m: {s: per_model.get(m, {}).get(s, []) for s in STRATEGIES}
                for m in MODELS
            }
            if any(not v for d in synthetic.values() for v in d.values()):
                n_incomplete += 1

            record = {
                "submission_id": sid,
                "title": sub.get("title", ""),
                "selftext": sub.get("selftext", ""),
                "n_comments": len(comments_by_sub.get(sid, [])),
                "permalink": sub.get("permalink"),
                "created_utc": sub.get("created_utc"),
                "real_comments": [
                    {
                        "body": c.get("body", ""),
                        "author_flair_text": c.get("author_flair_text"),
                        "score": c.get("score"),
                    }
                    for c in real
                ],
                "synthetic": synthetic,
            }
            out.write(json.dumps(record) + "\n")
            n_written += 1

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {n_written:,} records to {out_path}  ({size_mb:.1f} MB)")
    if n_incomplete:
        print(f"  {n_incomplete:,} records have at least one empty model/strategy set")

    if args.gzip:
        gz_path = out_path.with_suffix(out_path.suffix + ".gz")
        with open(out_path, "rb") as f_in, gzip.open(gz_path, "wb", compresslevel=9) as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_mb = gz_path.stat().st_size / 1e6
        print(f"Wrote {gz_path}  ({gz_mb:.1f} MB, {100*gz_mb/size_mb:.0f}% of original)")


if __name__ == "__main__":
    main()
