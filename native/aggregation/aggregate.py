"""Turn GI slice scores into one score per study.

    python aggregate.py --slices scores_slices.csv --gi gi_occupancy.json --out scores.csv

Only slices listed in gi_occupancy.json are used. A slice is GI when colon,
small bowel, duodenum, stomach, or esophagus has at least 50 voxels. A
subject missing from that file keeps every slice.

MaxViT is the noisy-OR of the top 8 GI slice probabilities. The detector
score is the mean of GI slice scores above 0.1. The blend is 0.6 times the
MaxViT rank plus 0.4 times the detector rank, divided by the number of
studies. Rank 1 is the lowest score in this file.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _noisy_or(scores: list[float]) -> float:
    s = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    if s.size == 0:
        return 0.0
    s = np.sort(s)[-min(8, s.size) :]
    return float(1.0 - np.prod(1.0 - s))


def _mean_above(scores: list[float]) -> float:
    s = np.asarray(scores, dtype=np.float64)
    keep = s[s > 0.1]
    return float(keep.mean()) if keep.size else 0.0


def _ranks(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return out


def _gi_z(path: Path) -> dict[str, set[int]]:
    raw = json.loads(path.read_text())
    out = {}
    for sid, rec in raw.items():
        if rec and rec.get("gi_z"):
            out[str(sid)] = {int(z) for z in rec["gi_z"]}
    if not out:
        raise SystemExit(f"No gi_z lists in {path}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slices", required=True, help="Slice CSV from inference")
    p.add_argument("--gi", required=True, help="gi_occupancy.json from preprocessing")
    p.add_argument("--out", required=True, help="Patient score CSV")
    args = p.parse_args()

    gi = _gi_z(Path(args.gi))
    studies = defaultdict(lambda: {"nz": "", "rows": []})
    with Path(args.slices).open(newline="") as f:
        reader = csv.DictReader(f)
        need = {"Subject", "Accession", "z", "maxvit_prob", "detector_score"}
        missing = need - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing columns: {sorted(missing)}")
        for row in reader:
            sid = row["Subject"].strip()
            z = int(row["z"])
            if sid in gi and z not in gi[sid]:
                continue
            rec = studies[(sid, row["Accession"].strip())]
            if row.get("nz"):
                rec["nz"] = row["nz"]
            rec["rows"].append((z, float(row["maxvit_prob"]), float(row["detector_score"])))
    studies = {key: rec for key, rec in studies.items() if rec["rows"]}
    if not studies:
        raise SystemExit(f"No GI slices in {args.slices}")

    keys = sorted(studies)
    maxvit_scores = []
    detector_scores = []
    for key in keys:
        rows = studies[key]["rows"]
        maxvit_scores.append(_noisy_or([m for _, m, _ in rows]))
        detector_scores.append(_mean_above([d for _, _, d in rows]))
    blend = 0.6 * _ranks(maxvit_scores) + 0.4 * _ranks(detector_scores)
    n = float(len(keys))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Subject", "Accession", "nz", "n_scored", "gi_z",
        "maxvit_max_slice", "maxvit_score", "detector_score",
        "blend_rank", "blend_score",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (sid, acc) in enumerate(keys):
            rec = studies[(sid, acc)]
            rows = sorted(rec["rows"])
            maxvit = [m for _, m, _ in rows]
            writer.writerow({
                "Subject": sid,
                "Accession": acc,
                "nz": rec["nz"],
                "n_scored": len(rows),
                "gi_z": " ".join(str(z) for z, _, _ in rows),
                "maxvit_max_slice": f"{max(maxvit):.6f}",
                "maxvit_score": f"{maxvit_scores[i]:.6f}",
                "detector_score": f"{detector_scores[i]:.6f}",
                "blend_rank": f"{blend[i]:.4f}",
                "blend_score": f"{(blend[i] / n):.6f}",
            })
    print(f"[agg] {len(keys)} studies -> {out}", flush=True)


if __name__ == "__main__":
    main()
