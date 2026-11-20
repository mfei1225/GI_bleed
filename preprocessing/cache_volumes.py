"""Window registered slice DICOMs into one uint8 volume per phase.

Output matches the training cache layout:

    <cache_root>/s<size>/<subject>/<accession>/<phase>.npy

Each file is a NumPy array of shape (nz, size, size), dtype uint8.
Value 0 is the low end of the soft-tissue window and 255 is the high end.
Training divides by 255 to recover [0, 1].

Window: center 40 HU, width 400 HU (display range -160 to 240).
Each slice is padded to a square with zeros, then resized with linear interpolation.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pydicom

PHASES = ("Venous", "NonCon", "Arterial")
WC, WW = 40.0, 400.0


def _window_u8(hu: np.ndarray) -> np.ndarray:
    lo, hi = WC - WW / 2.0, WC + WW / 2.0
    x = np.clip((hu.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def _pad_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    out = np.zeros((side, side), dtype=img.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    out[top : top + h, left : left + w] = img
    return out


def _one_volume(phase_dir: Path, size: int):
    files = sorted(phase_dir.glob("*.dcm"))
    if not files:
        return None
    slices = []
    for path in files:
        ds = pydicom.dcmread(str(path))
        hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1)
        hu = hu + float(getattr(ds, "RescaleIntercept", 0) or 0)
        img = _pad_square(_window_u8(hu))
        if img.shape[0] != size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        slices.append(img.astype(np.uint8))
    return np.stack(slices, axis=0)


def _job(task):
    subject, accession, phase, phase_dir, out_path, size, overwrite = task
    if out_path.exists() and not overwrite:
        return subject, phase, "skip"
    vol = _one_volume(Path(phase_dir), size)
    if vol is None or not vol.any():
        return subject, phase, "empty"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".tmp{os.getpid()}.npy")
    with open(tmp, "wb") as f:
        np.save(f, vol)
    os.replace(tmp, out_path)
    return subject, phase, f"done {tuple(vol.shape)}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slice_root", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    tasks = []
    root = Path(args.slice_root)
    for subj in sorted(d for d in root.iterdir() if d.is_dir()):
        for acc in sorted(d for d in subj.iterdir() if d.is_dir()):
            for phase in PHASES:
                phase_dir = acc / phase
                if not phase_dir.is_dir():
                    continue
                out = Path(args.cache_root) / f"s{args.size}" / subj.name / acc.name / f"{phase}.npy"
                tasks.append((subj.name, acc.name, phase, str(phase_dir), out, args.size, args.overwrite))
    print(f"[cache] {len(tasks)} volumes, size={args.size}", flush=True)
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, t) for t in tasks]
        for fut in as_completed(futs):
            subject, phase, status = fut.result()
            key = status.split()[0]
            counts[key] = counts.get(key, 0) + 1
            print(f"  {subject} {phase}: {status}", flush=True)
    print(f"[cache] {counts} -> {args.cache_root}/s{args.size}", flush=True)


if __name__ == "__main__":
    main()
