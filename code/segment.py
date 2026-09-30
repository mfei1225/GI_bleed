"""TotalSegmentator on the registered venous NIfTI.

Input:
    <nifti_root>/<Subject>/<Accession>/Venous.nii.gz

Output:
    <out_root>/<Subject>/<Accession>/Venous/<organ>.nii.gz
    <out_root>/<Subject>/<Accession>/totalseg_meta.json

Default organ list is the GI-filter subset used for this project (bowel, solid
organs, large vessels, heart, lungs). Inference only reads the bowel masks;
the rest are kept so the folder matches a full local run.

Requires a separate install: pip install -r requirements-seg.txt
GPU is the practical device. CPU works with --device cpu.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

# Same subset as scripts/totalseg/run_totalseg_subject.py.
GI_FILT_ROIS = [
    "liver", "spleen", "kidney_left", "kidney_right",
    "kidney_cyst_left", "kidney_cyst_right",
    "aorta", "inferior_vena_cava", "superior_vena_cava",
    "portal_vein_and_splenic_vein", "pulmonary_vein",
    "iliac_artery_left", "iliac_artery_right",
    "iliac_vena_left", "iliac_vena_right",
    "common_carotid_artery_left", "common_carotid_artery_right",
    "subclavian_artery_left", "subclavian_artery_right",
    "brachiocephalic_trunk", "brachiocephalic_vein_left", "brachiocephalic_vein_right",
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
    "heart", "trachea", "esophagus",
    "colon", "small_bowel", "duodenum", "stomach",
]


def _studies(nifti_root: Path, phase: str):
    found = []
    for subj in sorted(p for p in nifti_root.iterdir() if p.is_dir()):
        if subj.name.startswith("."):
            continue
        for acc in sorted(p for p in subj.iterdir() if p.is_dir()):
            if (acc / f"{phase}.nii.gz").is_file():
                found.append((subj.name, acc))
    return found


def _done(out_dir: Path, rois) -> bool:
    return all((out_dir / f"{r}.nii.gz").is_file() for r in rois)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nifti_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--phase", default="Venous", help="Registered NIfTI to segment. Inference uses Venous.")
    p.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    p.add_argument("--task", default="total")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--full_total", action="store_true", help="Save every TotalSegmentator class.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--subjects_only", nargs="*", default=None)
    args = p.parse_args()

    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as e:
        raise SystemExit(
            "TotalSegmentator is not installed. From this folder: pip install -r requirements-seg.txt"
        ) from e

    nifti_root = Path(args.nifti_root)
    out_root = Path(args.out_root)
    rois = None if args.full_total else list(GI_FILT_ROIS)
    want = set(args.subjects_only) if args.subjects_only else None
    studies = [
        (sid, acc) for sid, acc in _studies(nifti_root, args.phase)
        if want is None or sid in want
    ]
    print(f"[seg] {len(studies)} studies, phase={args.phase}, device={args.device}", flush=True)

    for sid, acc in studies:
        nifti = acc / f"{args.phase}.nii.gz"
        out_dir = out_root / sid / acc.name / args.phase
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "subject_id": sid,
            "accession": acc.name,
            "phase": args.phase,
            "task": args.task,
            "device": args.device,
            "roi_subset": rois,
            "input": str(nifti),
        }
        if rois and _done(out_dir, rois) and not args.force:
            print(f"  SKIP {sid} {acc.name}", flush=True)
            meta["status"] = "skipped_exists"
        else:
            t0 = time.time()
            tmp = Path(tempfile.mkdtemp(prefix=f"totalseg_{sid}_"))
            print(f"  {sid} {acc.name} {nifti}", flush=True)
            kw = dict(fast=args.fast, task=args.task, device=args.device, quiet=False)
            if rois:
                kw["roi_subset"] = rois
            totalsegmentator(str(nifti), str(tmp), **kw)
            n_copy = 0
            for src in tmp.glob("*.nii.gz"):
                shutil.copy2(src, out_dir / src.name)
                n_copy += 1
            shutil.rmtree(tmp, ignore_errors=True)
            meta["status"] = "ok"
            meta["seconds"] = time.time() - t0
            meta["n_nifti"] = n_copy
            print(f"    {n_copy} masks in {(time.time() - t0) / 60:.1f} min", flush=True)
        meta_path = out_root / sid / acc.name / "totalseg_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
