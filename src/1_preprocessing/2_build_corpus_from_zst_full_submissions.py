# build_reddit_corpus_with_comment_metadata.py

import os
import json
import csv
import zstandard as zstd
from tqdm import tqdm

# ------------------------------
# CONFIG
# ------------------------------

SUBMISSIONS_DIR = "../../output/corpora/submissions"
COMMENTS_DIR = "../../output/corpora/comments"

OUTPUT_SUBMISSIONS = "../../output/corpora/submissions_corpus.jsonl"
OUTPUT_COMMENTS = "../../output/corpora/comments_corpus.jsonl"
OUTPUT_METADATA = "../../output/corpora/submission_metadata.csv"

MIN_CHARS = 30
LAYPERSON_TOKEN = "layperson"

# ------------------------------
# ZST STREAMER
# ------------------------------

def read_zst_file(file_path):

    with open(file_path, "rb") as f:

        dctx = zstd.ZstdDecompressor(max_window_size=2**31)

        with dctx.stream_reader(f) as reader:

            buffer = b""

            for chunk in iter(lambda: reader.read(16384), b""):

                buffer += chunk

                while b"\n" in buffer:

                    line, buffer = buffer.split(b"\n", 1)

                    if line.strip():

                        try:
                            yield json.loads(line)
                        except Exception:
                            continue


# ------------------------------
# STEP 1: COLLECT CANDIDATE SUBMISSIONS
# ------------------------------

def collect_submissions(input_dir):

    zst_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".zst")
    )

    submissions = {}
    metadata = {}

    total = 0
    kept = 0

    for file_path in tqdm(zst_files, desc="Scanning submissions"):

        for obj in read_zst_file(file_path):

            total += 1

            text = obj.get("selftext", "")

            if not text:
                continue

            if len(text.strip()) < MIN_CHARS:
                continue

            submission_id = obj.get("id")

            if not submission_id:
                continue

            submissions[submission_id] = obj

            metadata[submission_id] = {
                "submission_id": submission_id,
                "total_comments": 0,
                "layperson_comments": 0
            }

            kept += 1

    print(f"Submissions scanned: {total}")
    print(f"Candidate submissions: {kept}")

    return submissions, metadata


# ------------------------------
# STEP 2: PROCESS COMMENTS
# ------------------------------

def process_comments(input_dir, submissions, metadata):

    valid_submission_ids = set(submissions.keys())

    zst_files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".zst")
    )

    total_comments = 0
    kept_comments = 0

    with open(OUTPUT_COMMENTS, "w", encoding="utf-8") as out:

        for file_path in tqdm(zst_files, desc="Processing comments"):

            for obj in read_zst_file(file_path):

                total_comments += 1

                author = obj.get("author")
                if author == "AutoModerator":
                    continue

                meta = obj.get("_meta", {})
                removal_type = meta.get("removal_type")
                if removal_type in ("deleted", "removed", "removed by reddit"):
                    continue

                link_id = obj.get("link_id")
                parent_id = obj.get("parent_id")

                if not link_id or not parent_id:
                    continue

                submission_id = link_id.split("_")[1]

                if submission_id not in valid_submission_ids:
                    continue

                # retain only top-level comments
                if parent_id != link_id:
                    continue

                metadata_entry = metadata[submission_id]
                metadata_entry["total_comments"] += 1

                flair = obj.get("author_flair_text")

                if flair and LAYPERSON_TOKEN in flair.lower():
                    metadata_entry["layperson_comments"] += 1

                out.write(json.dumps(obj) + "\n")

                kept_comments += 1

    print(f"Comments scanned: {total_comments}")
    print(f"Top-level comments retained: {kept_comments}")


# ------------------------------
# STEP 3: WRITE SUBMISSIONS + METADATA
# ------------------------------

def write_outputs(submissions, metadata):

    kept_submissions = 0

    with open(OUTPUT_SUBMISSIONS, "w", encoding="utf-8") as sub_out, \
         open(OUTPUT_METADATA, "w", newline="", encoding="utf-8") as meta_out:

        writer = csv.writer(meta_out)

        writer.writerow([
            "submission_id",
            "total_comments",
            "layperson_comments",
            "layperson_fraction"
        ])

        for submission_id, submission in submissions.items():

            meta = metadata[submission_id]

            total_comments = meta["total_comments"]
            lay_comments = meta["layperson_comments"]

            if total_comments == 0:
                continue

            fraction = lay_comments / total_comments

            # add metadata to submission object
            submission["total_top_level_comments"] = total_comments
            submission["layperson_comment_fraction"] = fraction

            sub_out.write(json.dumps(submission) + "\n")

            writer.writerow([
                submission_id,
                total_comments,
                lay_comments,
                fraction
            ])

            kept_submissions += 1

    print(f"Submissions written (with comments): {kept_submissions}")


# ------------------------------
# MAIN
# ------------------------------

def main():

    os.makedirs(os.path.dirname(OUTPUT_SUBMISSIONS), exist_ok=True)

    print("\nSTEP 1: Collect submissions")
    submissions, metadata = collect_submissions(SUBMISSIONS_DIR)

    print("\nSTEP 2: Process comments")
    process_comments(COMMENTS_DIR, submissions, metadata)

    print("\nSTEP 3: Write outputs")
    write_outputs(submissions, metadata)

    print("\nDone.")
    print("Outputs:")
    print(OUTPUT_SUBMISSIONS)
    print(OUTPUT_COMMENTS)
    print(OUTPUT_METADATA)


if __name__ == "__main__":
    main()