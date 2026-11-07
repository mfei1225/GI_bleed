"""Which axial slices contain GI tract, from the venous TotalSegmentator masks.

A slice is GI when any of colon, small_bowel, duodenum, stomach, esophagus
has at least 50 voxels. That list is what patient scoring keeps. Slices
outside it are dropped at aggregation. A subject with no masks is omitted;
scoring then keeps every slice for that subject.

Writes <out>/gi_occupancy.json:

    {
      "<Subject>": {
        "acc": "<Accession>",
        "nz": 320,
        "z0": 0,
        "gi_z": [40, 41, ...],
        "n_gi": 180
      }
    }

z in gi_z matches the slice filename index (0000.dcm is z=0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import SimpleITK as sitk

GI_ORGANS = ("colon", "small_bowel", "duodenum", "stomach", "esophagus")
GI_VOXEL_MIN = 50


def _gi_z(seg_dir: Path, nz: int) -> list[int] | None:
    import numpy as np

    gi = np.zeros(int(nz), dtype=bool)
    n_ok = 0
    for organ in GI_ORGANS:
        path = seg_dir / f"{organ}.nii.gz"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        n_ok += 1
        use = min(int(nz), arr.shape[0])
        counts = arr[:use].reshape(use, -1).sum(axis=1)
        gi[:use] |= counts >= GI_VOXEL_MIN
    if n_ok == 0:
        return None
    return [int(i) for i, v in enumerate(gi) if v]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nifti_root", required=True)
    p.add_argument("--seg_root", required=True)
    p.add_argument("--out", required=True, help="Path to gi_occupancy.json")
    p.add_argument("--phase", default="Venous")
    args = p.parse_args()

    nifti_root = Path(args.nifti_root)
    seg_root = Path(args.seg_root)
    out = {}
    n_miss = 0
    for subj in sorted(d for d in nifti_root.iterdir() if d.is_dir() and not d.name.startswith(".")):
        for acc in sorted(d for d in subj.iterdir() if d.is_dir()):
            nii = acc / f"{args.phase}.nii.gz"
            seg_dir = seg_root / subj.name / acc.name / args.phase
            if not nii.is_file():
                continue
            if not seg_dir.is_dir():
                n_miss += 1
                continue
            nz = int(sitk.ReadImage(str(nii)).GetSize()[2])
            zs = _gi_z(seg_dir, nz)
            if zs is None:
                n_miss += 1
                continue
            out[subj.name] = {
                "acc": acc.name,
                "nz": nz,
                "z0": 0,
                "gi_z": zs,
                "n_gi": len(zs),
            }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    print(f"[gi] {len(out)} subjects, {n_miss} without usable masks -> {path}", flush=True)


if __name__ == "__main__":
    main()
