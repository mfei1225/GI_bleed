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

z in gi_z matches the slice filename index (0000.dcm is z=0). A mask whose
grid is coarser than Venous.nii.gz is resampled onto that volume first, so
the slice index is the venous slice and not the segmentation's own index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import SimpleITK as sitk

GI_ORGANS = ("colon", "small_bowel", "duodenum", "stomach", "esophagus")
GI_VOXEL_MIN = 50


def _reference(path: Path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    img = sitk.Image(reader.GetSize(), sitk.sitkUInt8)
    img.SetOrigin(reader.GetOrigin())
    img.SetSpacing(reader.GetSpacing())
    img.SetDirection(reader.GetDirection())
    return img


def _on_reference(mask: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Resample a coarser TotalSegmentator mask onto the venous slice grid."""
    same = (
        mask.GetSize() == ref.GetSize()
        and all(abs(a - b) <= 1e-3 for a, b in zip(mask.GetSpacing(), ref.GetSpacing()))
        and all(abs(a - b) <= 1e-3 for a, b in zip(mask.GetOrigin(), ref.GetOrigin()))
        and all(abs(a - b) <= 1e-5 for a, b in zip(mask.GetDirection(), ref.GetDirection()))
    )
    if same:
        return mask
    filt = sitk.ResampleImageFilter()
    filt.SetReferenceImage(ref)
    filt.SetInterpolator(sitk.sitkNearestNeighbor)
    filt.SetDefaultPixelValue(0)
    return filt.Execute(mask)


def _gi_z(seg_dir: Path, ref: sitk.Image) -> list[int] | None:
    import numpy as np

    nz = int(ref.GetSize()[2])
    gi = np.zeros(nz, dtype=bool)
    n_ok = 0
    for organ in GI_ORGANS:
        path = seg_dir / f"{organ}.nii.gz"
        if not path.is_file() or path.stat().st_size == 0:
            continue
        arr = sitk.GetArrayFromImage(_on_reference(sitk.ReadImage(str(path)), ref))
        n_ok += 1
        use = min(nz, arr.shape[0])
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
            ref = _reference(nii)
            nz = int(ref.GetSize()[2])
            zs = _gi_z(seg_dir, ref)
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
