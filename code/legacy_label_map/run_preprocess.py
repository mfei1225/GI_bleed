"""Run registration, slice export, and the windowed volume cache on a new cohort.

Example:

    python run_preprocess.py \\
        --image_root /data/site/images \\
        --series_csv /data/site/series.csv \\
        --cohort_csv /data/site/cohort.csv \\
        --out /data/site/preprocessed \\
        --size 512 \\
        --workers 4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(HERE))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_root", required=True)
    p.add_argument("--series_csv", required=True)
    p.add_argument("--cohort_csv", default="")
    p.add_argument("--label_phase_map", default="", help="Optional JSON forcing the reference series per subject.")
    p.add_argument("--out", required=True, help="Parent folder for nifti/, slices/, and volcache/.")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_cache", action="store_true")
    p.add_argument("--skip_seg", action="store_true", help="Skip TotalSegmentator and the GI slice mask.")
    p.add_argument("--seg_device", default="gpu", choices=("gpu", "cpu"))
    args = p.parse_args()

    out = Path(args.out)
    nifti = out / "nifti"
    slices = out / "slices"
    cache = out / "volcache"
    py = sys.executable
    reg = [
        py, str(HERE / "register_and_cache.py"),
        "--image_root", args.image_root,
        "--series_csv", args.series_csv,
        "--out_root", str(nifti),
        "--workers", str(args.workers),
    ]
    if args.cohort_csv:
        reg += ["--cohort_csv", args.cohort_csv]
    if args.label_phase_map:
        reg += ["--extra_label_phase_map", args.label_phase_map]
    if args.overwrite:
        reg.append("--overwrite")
    _run(reg)

    if not args.skip_seg:
        seg = [
            py, str(HERE / "segment.py"),
            "--nifti_root", str(nifti),
            "--out_root", str(out / "seg"),
            "--device", args.seg_device,
        ]
        if args.overwrite:
            seg.append("--force")
        _run(seg)
        _run([
            py, str(HERE / "gi_occupancy.py"),
            "--nifti_root", str(nifti),
            "--seg_root", str(out / "seg"),
            "--out", str(out / "gi_occupancy.json"),
        ])

    ext = [
        py, str(HERE / "extract_registered_slices.py"),
        "--nifti_root", str(nifti),
        "--slice_root", str(slices),
        "--image_root", args.image_root,
        "--series_csv", args.series_csv,
        "--workers", str(args.workers),
        "--min_age_s", "0",
    ]
    if args.overwrite:
        ext.append("--overwrite")
    _run(ext)

    if args.skip_cache:
        return
    cache_cmd = [
        py, str(HERE / "cache_volumes.py"),
        "--slice_root", str(slices),
        "--cache_root", str(cache),
        "--size", str(args.size),
        "--workers", str(args.workers),
    ]
    if args.overwrite:
        cache_cmd.append("--overwrite")
    _run(cache_cmd)


if __name__ == "__main__":
    main()
