"""Register a few subjects and write a contact sheet of the aligned slices.

The sheet is one row per sampled z. Columns are venous, non-contrast, arterial,
and a color overlay (red arterial, green venous, blue non-contrast). Alignment
looks right when the overlay is mostly gray and organ edges sit on top of each
other.

    python sanity_slices.py \
        --image_root /data/site/images \
        --series_csv /data/site/series.csv \
        --subjects SID1 SID2 \
        --out /tmp/sanity_slices

Segmentation is not run. The contact sheet is <out>/registered_slices.png.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
PHASES = ("Venous", "NonCon", "Arterial")
WC, WW = 40.0, 400.0


def _run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def _window(hu: np.ndarray) -> np.ndarray:
    lo, hi = WC - WW / 2.0, WC + WW / 2.0
    x = np.clip((hu.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def _fit(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = side / max(h, w)
    out = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((side, side) if img.ndim == 2 else (side, side, 3), dtype=np.uint8)
    y0 = (side - out.shape[0]) // 2
    x0 = (side - out.shape[1]) // 2
    canvas[y0 : y0 + out.shape[0], x0 : x0 + out.shape[1]] = out
    return canvas


def _label(img: np.ndarray, text: str) -> np.ndarray:
    view = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return view


def _sheet(nifti_root: Path, subjects: list[str], side: int) -> np.ndarray:
    rows = []
    for sid in subjects:
        idx_path = nifti_root / sid / "index.json"
        if not idx_path.is_file():
            raise SystemExit(f"missing {idx_path}")
        idx = json.loads(idx_path.read_text())
        acc = str(idx["accession"])
        vols = {}
        for ph in PHASES:
            img = sitk.ReadImage(str(nifti_root / sid / acc / f"{ph}.nii.gz"))
            vols[ph] = sitk.GetArrayFromImage(img)
        nz = min(v.shape[0] for v in vols.values())
        # Skip the very top and bottom, where the body often leaves the field.
        zs = [int(nz * f) for f in (0.35, 0.5, 0.65)]
        for z in zs:
            v = _window(vols["Venous"][z])
            n = _window(vols["NonCon"][z])
            a = _window(vols["Arterial"][z])
            overlay = np.stack([a, v, n], axis=-1)
            tiles = [
                _label(_fit(v, side), f"{sid} z{z} Venous"),
                _label(_fit(n, side), "NonCon"),
                _label(_fit(a, side), "Arterial"),
                _label(_fit(overlay, side), "R art  G ven  B nc"),
            ]
            rows.append(np.concatenate(tiles, axis=1))
    return np.concatenate(rows, axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_root", required=True)
    p.add_argument("--series_csv", required=True)
    p.add_argument("--subjects", nargs="+", required=True, help="Two or three subject ids is enough.")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--side", type=int, default=320, help="Display size of each slice panel.")
    args = p.parse_args()

    out = Path(args.out)
    nifti = out / "nifti"
    slices = out / "slices"
    py = sys.executable
    _run([
        py, str(HERE / "register_and_cache.py"),
        "--image_root", args.image_root,
        "--series_csv", args.series_csv,
        "--out_root", str(nifti),
        "--workers", str(args.workers),
        "--subjects_only", *args.subjects,
    ])
    _run([
        py, str(HERE / "extract_registered_slices.py"),
        "--nifti_root", str(nifti),
        "--slice_root", str(slices),
        "--image_root", args.image_root,
        "--series_csv", args.series_csv,
        "--workers", str(args.workers),
        "--min_age_s", "0",
        "--subjects_only", *args.subjects,
    ])
    _run([py, str(HERE / "sanity_check.py"), "--out", str(out)])

    present = []
    for sid in args.subjects:
        key = str(int(sid)) if str(sid).isdigit() else str(sid)
        if (nifti / key / "index.json").is_file():
            present.append(key)
        elif (nifti / str(sid) / "index.json").is_file():
            present.append(str(sid))
        else:
            raise SystemExit(f"no index.json for subject {sid} under {nifti}")
    sheet = _sheet(nifti, present, args.side)
    png = out / "registered_slices.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(png), sheet):
        raise SystemExit(f"failed to write {png}")
    print(f"[sanity] wrote {png}", flush=True)


if __name__ == "__main__":
    main()
