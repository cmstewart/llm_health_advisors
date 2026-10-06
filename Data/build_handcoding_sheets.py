#!/usr/bin/env python3
"""
Build blind hand-coding workbooks: the validation step the plan puts first.

From Piloting/Round 5/topic_classification_plan.md: "We can hand-code 200 random
OPs before seeing any automated output, ideally two coders. We should report
precision, recall, and inter-annotator agreement as the ceiling on automated
performance."

This writes one workbook per coder. They are blind by construction: no automated
label, confidence or evidence quote is written into them, and this script never
opens the label files except to count the overlap it reports (see
--check-blindness, which is on by default). Score them with score_handcoding.py
once they come back.

Sampling
--------
The 200 are drawn from the same 6,600-OP seeded sample the classification run
uses, so every coded OP joins the analysis set. The draw has its own seed
(--sample-seed) so it is reproducible and independent of the generation seed.

Both coders get the SAME 200 OPs in the SAME order. That is what makes
inter-annotator agreement computable; shuffling per coder would make the two
sheets impossible to align row by row.

Usage
-----
  # Default: 200 OPs, two coders, from the corpus
  python build_handcoding_sheets.py

  # Three coders, 50 OPs, for a dry run of the protocol itself
  python build_handcoding_sheets.py --n 50 --coders A,B,C

Output
------
  <out-dir>/handcode_200_<coder>.xlsx   one per coder, identical content
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from classify_topics import (
    CANCER_STANCES,
    load_from_analysis_dataset,
    load_from_corpus,
    load_taxonomy,
)

# Excel refuses a cell over 32,767 characters.
CELL_LIMIT = 32000

HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")  # the columns a coder fills
NOTE_FILL = PatternFill("solid", fgColor="FCE4D6")


# ---------------------------------------------------------------------------
# The codebook shown to coders
#
# Deliberately the same rubric the model is given in classify_topics.py. If the
# humans are judged against one definition and the model against another, the
# precision and recall numbers mean nothing.
# ---------------------------------------------------------------------------

def instructions(tax, n: int) -> list[tuple[str, str]]:
    return [
        ("What this is",
         f"{n} opening posts from r/AskDocs, to be coded by hand. Your codes are the "
         "gold standard that an automated classifier is measured against, so they are "
         "the ceiling on what we can claim for it."),
        ("Work alone",
         "Do not discuss posts with the other coder while coding, and do not look at "
         "any automated output. Disagreements between coders are informative: they are "
         "measured, then adjudicated afterwards. Resolving them early destroys that."),
        ("", ""),
        ("Column: discipline",
         "Pick exactly ONE from the dropdown. Judge the post's own primary medical "
         "subject, not every condition it mentions in passing. Read the title and the "
         "body together; the subject is often only in the body."),
        ("  residual categories", tax.residual_guidance),
        ("", ""),
        ("Column: cancer_relevant",
         "yes if the post involves a malignancy (carcinoma, sarcoma, lymphoma, "
         "leukaemia, melanoma, myeloma, or a named cancer of any organ), whether or not "
         "it is the poster's own. Benign tumours, cysts and non-neoplastic lumps are "
         "NOT cancer. Otherwise no."),
        ("", ""),
        ("Column: cancer_stance",
         "OPTIONAL, and only when cancer_relevant is yes. This contrast is currently "
         "deferred from the analysis; code it if you can, leave it blank if unsure."),
        ("  current_patient",
         "The POSTER has an active malignancy, or is undergoing or awaiting primary "
         "treatment for one, or has known residual, recurrent or metastatic disease."),
        ("  survivor",
         "The POSTER had a malignancy, primary treatment is complete, and there is no "
         "evidence of active disease. Long-term maintenance or endocrine therapy after "
         "primary treatment still counts as survivor. NOTE: this is narrower than the "
         "NCI definition, which counts anyone from diagnosis onward."),
        ("  not_applicable",
         "The cancer history is not the poster's own (they are asking about a relative "
         "or patient), OR there is no diagnosis at all (worried about cancer, asking "
         "about screening or risk, awaiting a first diagnostic result)."),
        ("  unclear",
         "The poster does have a cancer history, but the post does not say enough to "
         "tell whether disease or primary treatment is current."),
        ("", ""),
        ("Column: notes",
         "Anything that made the post hard to code. These are read during adjudication, "
         "so a sentence on a borderline call is worth more than a perfect code."),
    ]


def write_instructions_sheet(wb: Workbook, tax, n: int, coder: str) -> None:
    ws = wb.create_sheet("Instructions", 0)
    ws["A1"] = f"Hand-coding instructions -- coder {coder}"
    ws["A1"].font = Font(bold=True, size=14)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 100
    row = 3
    for head, body in instructions(tax, n):
        if head:
            ws.cell(row=row, column=1, value=head).font = Font(bold=not head.startswith("  "))
            c = ws.cell(row=row, column=2, value=body)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[row].height = max(15, 14 * (len(body) // 95 + 1))
        row += 1


def write_codebook_sheet(wb: Workbook, tax) -> None:
    """
    The dropdown source. A literal list in a data-validation formula is capped at
    255 characters, and these 17 names are far longer than that, so the options
    live on a sheet and the validation points at the range.
    """
    ws = wb.create_sheet("Codebook")
    ws["A1"] = "discipline"
    ws["A1"].font = Font(bold=True)
    for i, d in enumerate(tax.choices, start=2):
        ws.cell(row=i, column=1, value=d)
    ws["B1"] = "cancer_relevant"
    ws["B1"].font = Font(bold=True)
    for i, v in enumerate(("yes", "no"), start=2):
        ws.cell(row=i, column=2, value=v)
    ws["C1"] = "cancer_stance"
    ws["C1"].font = Font(bold=True)
    for i, v in enumerate(CANCER_STANCES, start=2):
        ws.cell(row=i, column=3, value=v)
    ws.column_dimensions["A"].width = 46
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 20
    return len(tax.choices)


def write_coding_sheet(wb: Workbook, ops: list[dict], n_disc: int) -> None:
    ws = wb.create_sheet("Coding")
    headers = ["#", "submission_id", "title", "body",
               "discipline", "cancer_relevant", "cancer_stance", "notes"]
    widths = [5, 14, 40, 90, 46, 16, 18, 40]
    for col, (h, w) in enumerate(zip(headers, widths), start=1):
        c = ws.cell(row=1, column=col, value=h)
        c.font = Font(bold=True)
        c.fill = HEADER_FILL if col <= 4 else INPUT_FILL
        c.alignment = Alignment(vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = w

    for i, op in enumerate(ops, start=1):
        r = i + 1
        body = op["selftext"] or ""
        if len(body) > CELL_LIMIT:
            body = body[:CELL_LIMIT] + "\n[... truncated for the spreadsheet ...]"
        ws.cell(row=r, column=1, value=i)
        ws.cell(row=r, column=2, value=op["id"])
        ws.cell(row=r, column=3, value=op["title"] or "")
        ws.cell(row=r, column=4, value=body)
        for col in (3, 4):
            ws.cell(row=r, column=col).alignment = Alignment(wrap_text=True, vertical="top")
        for col in (5, 6, 7, 8):
            ws.cell(row=r, column=col).fill = INPUT_FILL
        ws.row_dimensions[r].height = 90

    last = len(ops) + 1
    rules = [
        ("E", f"=Codebook!$A$2:$A${n_disc + 1}", "discipline",
         "Pick one discipline from the list."),
        ("F", "=Codebook!$B$2:$B$3", "cancer_relevant", "yes or no."),
        ("G", f"=Codebook!$C$2:$C${len(CANCER_STANCES) + 1}", "cancer_stance",
         "Optional; only when cancer_relevant is yes."),
    ]
    for col, formula, title, prompt in rules:
        dv = DataValidation(type="list", formula1=formula, allow_blank=True,
                            showDropDown=False)
        dv.promptTitle, dv.prompt = title, prompt
        dv.errorTitle = "Not in the codebook"
        dv.error = "Choose a value from the dropdown."
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")

    ws.freeze_panes = "E2"


def build_workbook(ops: list[dict], tax, coder: str, out_path: Path) -> None:
    wb = Workbook()
    wb.remove(wb.active)  # drop the default sheet
    write_instructions_sheet(wb, tax, len(ops), coder)
    n_disc = write_codebook_sheet(wb, tax)
    write_coding_sheet(wb, ops, n_disc)
    wb._sheets = [wb["Instructions"], wb["Coding"], wb["Codebook"]]
    wb.save(out_path)


# ---------------------------------------------------------------------------
# Blindness check
# ---------------------------------------------------------------------------

def report_blindness(ops: list[dict], labels_dir: Path) -> None:
    """
    Say how many of the sampled OPs already have an automated label.

    This does not weaken the protocol -- blindness is a property of what the
    coders see, and they see none of it -- but the plan promises coding "before
    seeing any automated output", so the overlap is worth stating rather than
    discovering later. A pilot run will usually create a small overlap.
    """
    ids = {o["id"] for o in ops}
    hits: dict[str, int] = {}
    for p in sorted(labels_dir.glob("labels_*.jsonl")) if labels_dir.exists() else []:
        n = 0
        with open(p) as f:
            for line in f:
                try:
                    if json.loads(line).get("submission_id") in ids:
                        n += 1
                except json.JSONDecodeError:
                    continue
        if n:
            hits[p.name] = n
    if not hits:
        print("Blindness: none of the sampled OPs has been labelled yet. Clean.")
        return
    print("Blindness: some sampled OPs already carry an automated label.")
    for name, n in hits.items():
        print(f"  {n:>4} of {len(ids)} in {name}")
    print("  The workbooks contain none of it, so the coders remain blind. Report the\n"
          "  overlap rather than describing the whole set as coded before any output.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build blind hand-coding workbooks for validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=200, help="OPs to code")
    ap.add_argument("--coders", default="A,B",
                    help="comma-separated coder names; one workbook each. Two is what "
                         "makes inter-annotator agreement reportable")
    ap.add_argument("--sample-seed", type=int, default=7,
                    help="seed for drawing the coding sample; distinct from the "
                         "generation seed so the two cannot be confused")
    ap.add_argument("--taxonomy", default="taxonomy.json")
    ap.add_argument("--from-analysis-dataset", default=None, metavar="PATH")
    ap.add_argument("--corpora-dir", default="../Piloting/Round 2")
    ap.add_argument("--out-dir", default="../Piloting/Round 5/output/corpora/handcoding")
    ap.add_argument("--labels-dir", default="../Piloting/Round 5/output/corpora/labels",
                    help="only read to report overlap; never written into the workbooks")
    ap.add_argument("--n-ops", type=int, default=6600,
                    help="the classification sample the 200 are drawn from")
    ap.add_argument("--seed", type=int, default=42,
                    help="generation seed; must match the classification run")
    args = ap.parse_args()

    tax = load_taxonomy(Path(args.taxonomy))
    if args.from_analysis_dataset:
        pool = load_from_analysis_dataset(Path(args.from_analysis_dataset))
    else:
        pool = load_from_corpus(Path(args.corpora_dir), args.n_ops, args.seed)
    if len(pool) < args.n:
        sys.exit(f"ERROR: asked for {args.n} OPs but the sample holds {len(pool)}")

    # Drawn from the classification sample, so every coded OP joins the analysis.
    ordered = sorted(pool, key=lambda o: o["id"])
    rng = random.Random(args.sample_seed)
    ops = [ordered[i] for i in sorted(rng.sample(range(len(ordered)), args.n))]
    print(f"Drew {len(ops):,} OPs from the {len(pool):,}-OP sample "
          f"(sample-seed={args.sample_seed}).")

    report_blindness(ops, Path(args.labels_dir))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coders = [c.strip() for c in args.coders.split(",") if c.strip()]
    if len(coders) < 2:
        print("NOTE: one coder only. Inter-annotator agreement needs two, and the plan\n"
              "      asks for it as the ceiling on automated performance.")
    for coder in coders:
        path = out_dir / f"handcode_{args.n}_{coder}.xlsx"
        build_workbook(ops, tax, coder, path)
        print(f"  wrote {path}")

    print(f"\nBoth workbooks hold the same {len(ops)} OPs in the same order, which is\n"
          "what makes the two coders comparable row by row.")
    print("Score them with:  python score_handcoding.py")


if __name__ == "__main__":
    main()
