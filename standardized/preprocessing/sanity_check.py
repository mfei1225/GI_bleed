"""Check a preprocessing folder for the failures that matter before inference.

    python sanity_check.py --out /data/site/preprocessed

Checks, per subject:

- three registered NIfTIs exist and share the same grid
- pixel values are in the Hounsfield range
- slice DICOM count matches the NIfTI, with slope 1 and intercept 0
- ipp_map.json has one position per slice
- the uint8 cache, if present, is (nz, size, size)

Segmentation is optional. When seg/ exists, gi_occupancy.json must list only
slice indexes that exist. Exit code is 1 when any check fails.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

PHASES = ("Venous", "NonCon", "Arterial")


def _grid(img):
    return (
        tuple(int(x) for x in img.GetSize()),
        tuple(round(float(x), 4) for x in img.GetSpacing()),
        tuple(round(float(x), 3) for x in img.GetOrigin()),
    )


def check_out(out: Path) -> list[str]:
    errors = []
    nifti = out / "nifti"
    slices = out / "slices"
    if not nifti.is_dir():
        return [f"missing {nifti}"]

    subjects = [p for p in nifti.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not subjects:
        errors.append(f"no subjects under {nifti}")

    for subj in sorted(subjects):
        idx_path = subj / "index.json"
        if not idx_path.is_file():
            errors.append(f"{subj.name}: missing index.json")
            continue
        idx = json.loads(idx_path.read_text())
        acc = str(idx.get("accession") or "")
        acc_nii = subj / acc
        scans = idx.get("slot_scans") or {}
        chosen = [str(scans.get(ph) or "") for ph in PHASES]
        if any(not s for s in chosen):
            errors.append(f"{subj.name}: a phase has no source scan in index.json")
        if len(set(chosen)) < 3:
            errors.append(f"{subj.name}: phases do not come from three different series {chosen}")

        vols = {}
        for ph in PHASES:
            path = acc_nii / f"{ph}.nii.gz"
            if not path.is_file():
                errors.append(f"{subj.name}: missing {path.name}")
                continue
            img = sitk.ReadImage(str(path))
            vols[ph] = img
            arr = sitk.GetArrayFromImage(img)
            # Air is negative after rescale. A volume with no negatives was stored
            # without the intercept. Metal and contrast can sit well above 4000.
            if arr.size == 0 or int(arr.min()) > -200 or int(arr.max()) > 40000:
                errors.append(
                    f"{subj.name} {ph}: values do not look like HU min={arr.min()} max={arr.max()}"
                )
        if len(vols) == 3:
            grids = {ph: _grid(img) for ph, img in vols.items()}
            if len(set(grids.values())) != 1:
                errors.append(f"{subj.name}: phases are not on the same grid {grids}")
            nz = int(vols["Venous"].GetSize()[2])
        else:
            continue

        acc_sl = slices / subj.name / acc
        ipp_path = acc_sl / "ipp_map.json"
        if not ipp_path.is_file():
            errors.append(f"{subj.name}: missing ipp_map.json")
        else:
            ipp = json.loads(ipp_path.read_text())
            if int(ipp.get("nz", -1)) != nz or len(ipp.get("axial_positions_mm") or []) != nz:
                errors.append(f"{subj.name}: ipp_map nz does not match the NIfTI ({nz})")
        for ph in PHASES:
            dcm_dir = acc_sl / ph
            names = sorted(dcm_dir.glob("*.dcm")) if dcm_dir.is_dir() else []
            if len(names) != nz:
                errors.append(f"{subj.name} {ph}: {len(names)} slice DICOMs, NIfTI nz={nz}")
                continue
            ds = pydicom.dcmread(str(names[0]), stop_before_pixels=True)
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            if abs(slope - 1.0) > 1e-6 or abs(intercept) > 1e-6:
                errors.append(f"{subj.name} {ph}: RescaleSlope={slope} RescaleIntercept={intercept}")

        cache_root = out / "volcache"
        if cache_root.is_dir():
            for npy in (cache_root).glob(f"s*/{subj.name}/{acc}/*.npy"):
                vol = np.load(npy, mmap_mode="r")
                if vol.ndim != 3 or vol.shape[0] != nz or vol.dtype != np.uint8:
                    errors.append(f"{subj.name}: cache {npy.name} shape={vol.shape} dtype={vol.dtype}")
                if vol.shape[1] != vol.shape[2]:
                    errors.append(f"{subj.name}: cache {npy.name} is not square")

    seg = out / "seg"
    occ_path = out / "gi_occupancy.json"
    if seg.is_dir():
        if not occ_path.is_file():
            errors.append("seg/ exists but gi_occupancy.json is missing")
        else:
            occ = json.loads(occ_path.read_text())
            for sid, rec in occ.items():
                nz = int(rec.get("nz") or 0)
                bad = [z for z in rec.get("gi_z") or [] if int(z) < 0 or int(z) >= nz]
                if bad:
                    errors.append(f"{sid}: GI slice indexes outside 0..{nz - 1}")
                for organ in ("colon", "small_bowel", "duodenum", "stomach", "esophagus"):
                    p = seg / sid / str(rec.get("acc")) / "Venous" / f"{organ}.nii.gz"
                    if not p.is_file():
                        errors.append(f"{sid}: missing bowel mask {organ}.nii.gz")
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Folder that contains nifti/ and slices/")
    args = p.parse_args()
    errors = check_out(Path(args.out))
    if errors:
        print(f"[sanity] {len(errors)} problem(s):")
        for err in errors:
            print(f"  {err}")
        raise SystemExit(1)
    print(f"[sanity] ok {args.out}")


if __name__ == "__main__":
    main()
