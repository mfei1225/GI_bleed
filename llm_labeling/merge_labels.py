#!/usr/bin/env python3
"""Majority-vote several injury CSVs and list disagreements for review.

    python merge_labels.py --inputs llama.csv qwen.csv --out merged.csv --review review.csv
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from label_reports import _cell, _norm_flag

KEYS = ("Bowel", "Kidney", "Liver", "Extravasation")


def _vote(values: list[str]) -> str:
    vals = [v for v in values if v in {"Yes", "No", "Maybe"}]
    if not vals:
        return "Undefined"
    c = Counter(vals)
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "Maybe"
    return top[0][0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--review", default="")
    args = p.parse_args()

    tables = []
    for path in args.inputs:
        df = pd.read_csv(path)
        df["Accession Number"] = df["Accession Number"].map(_cell)
        df = df[df["Accession Number"] != ""]
        df = df.drop_duplicates("Accession Number", keep="last")
        tables.append(df.set_index("Accession Number"))

    accs = sorted(set().union(*[set(t.index) for t in tables]))
    rows = []
    disagree = []
    for acc in accs:
        row = {"Accession Number": acc}
        texts = []
        mrns = []
        for t in tables:
            if acc in t.index:
                texts.append(_cell(t.loc[acc].get("text", "")))
                mrns.append(_cell(t.loc[acc].get("Patient MRN", "")))
        row["text"] = next((x for x in texts if x and x != "nan"), "")
        row["Patient MRN"] = next((x for x in mrns if x and x != "nan"), "")
        any_split = False
        for k in KEYS:
            vals = [_norm_flag(t.loc[acc][k]) for t in tables if acc in t.index and k in t.columns]
            row[k] = _vote(vals)
            uniq = {v for v in vals if v in {"Yes", "No", "Maybe"}}
            if len(uniq) > 1:
                any_split = True
                row[f"{k}_votes"] = ",".join(vals)
        rows.append(row)
        if any_split:
            disagree.append(row)

    out = Path(args.out)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} n={len(rows)} disagreements={len(disagree)}")
    if args.review:
        pd.DataFrame(disagree).to_csv(args.review, index=False)
        print(f"wrote {args.review}")


if __name__ == "__main__":
    main()
