"""Score slices with the MaxViT and the detector. Writes one row per slice.

    python infer.py --data /data/site/preprocessed --out /data/site/scores_slices.csv

MaxViT reads volcache/s512. The detector reads slices/. Both are produced by
preprocessing. Which slices count toward the patient score is decided
later, in aggregation.

On a GPU, raise --batch and --detector-batch until memory is full. --stride 2
scores every other slice. --n-folds 1 uses a single checkpoint. --models maxvit
skips the detector.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from detector import detect_batch, load_detector, load_phase_slices, slice_tensor
from model import load_bleed_model

PHASES = ("Venous", "NonCon", "Arterial")
ROOT = Path(__file__).resolve().parent


def _studies(data: Path) -> list[tuple[str, str]]:
    found = []
    slices = data / "slices"
    if not slices.is_dir():
        raise SystemExit(f"No slices/ under {data}")
    for sid in sorted(p for p in slices.iterdir() if p.is_dir()):
        for acc in sorted(p for p in sid.iterdir() if p.is_dir()):
            cache = data / "volcache" / "s512" / sid.name / acc.name / "Venous.npy"
            if (acc / "Venous").is_dir() and cache.is_file():
                found.append((sid.name, acc.name))
    if not found:
        raise SystemExit(f"No studies with both slices/ and volcache/s512 under {data}")
    return found


def _checkpoints(folder: Path, n_folds: int) -> list[Path]:
    pts = sorted(p for p in folder.glob("*.pth") if p.is_file())
    if not pts:
        raise SystemExit(f"No checkpoints in {folder}")
    if n_folds:
        pts = pts[:n_folds]
    return pts


def _forward(model, x, device, amp: bool):
    with torch.no_grad():
        if amp and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                return model(x).float().cpu().numpy()
        return model(x).float().cpu().numpy()


def _ncdiff(vols, z: int) -> np.ndarray:
    v = np.asarray(vols["Venous"][z], dtype=np.float32) / 255.0
    nc = vols["NonCon"]
    art = vols["Arterial"]
    if nc is None:
        vnc = np.zeros_like(v)
        anc = np.zeros_like(v)
    else:
        nc_z = np.asarray(nc[z], dtype=np.float32) / 255.0
        vnc = np.clip((v - nc_z) * 0.5 + 0.5, 0.0, 1.0)
        if art is None:
            anc = np.zeros_like(v)
        else:
            anc = np.clip((np.asarray(art[z], dtype=np.float32) / 255.0 - nc_z) * 0.5 + 0.5, 0.0, 1.0)
    return np.stack([vnc, v, anc], axis=0)


def _load_cache(data: Path, sid: str, acc: str):
    root = data / "volcache" / "s512" / sid / acc
    vols = {}
    for phase in PHASES:
        path = root / f"{phase}.npy"
        vols[phase] = np.load(path, mmap_mode="r") if path.is_file() else None
    return vols


def _run_maxvit(data, studies, ckpts, device, stride, batch, amp):
    totals: dict[tuple[str, str, int], float] = {}
    nz_of = {}
    for ck in ckpts:
        model = load_bleed_model(ck, device)
        print(f"[maxvit] {ck.name}", flush=True)
        for sid, acc in studies:
            vols = _load_cache(data, sid, acc)
            nz = int(vols["Venous"].shape[0])
            nz_of[(sid, acc)] = nz
            zs = list(range(0, nz, stride))
            for start in range(0, len(zs), batch):
                batch_z = zs[start : start + batch]
                x = torch.from_numpy(np.stack([_ncdiff(vols, z) for z in batch_z])).to(device)
                prob = _forward(model, x, device, amp)
                for z, pr in zip(batch_z, prob):
                    key = (sid, acc, int(z))
                    totals[key] = totals.get(key, 0.0) + float(pr)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    nfold = float(len(ckpts))
    by_study = {}
    for (sid, acc, z), total in totals.items():
        by_study.setdefault((sid, acc), []).append((z, total / nfold))
    return by_study, nz_of


def _run_detector(data, studies, ckpts, device, stride, batch, amp):
    totals: dict[tuple[str, str, int], float] = {}
    boxes = []
    nz_of = {}
    for fold, ck in enumerate(ckpts):
        model = load_detector(ck, device)
        print(f"[detector] {ck.name}", flush=True)
        for sid, acc in studies:
            root = data / "slices" / sid / acc
            vols = {phase: load_phase_slices(root / phase) for phase in PHASES}
            nz = len(vols["Venous"])
            nz_of[(sid, acc)] = nz
            zs = list(range(0, nz, stride))
            for start in range(0, len(zs), batch):
                chunk = zs[start : start + batch]
                prepared = [slice_tensor(vols, z) for z in chunk]
                founds = detect_batch(
                    model, [t for t, _ in prepared], [g for _, g in prepared], device, amp
                )
                for z, found in zip(chunk, founds):
                    key = (sid, acc, int(z))
                    totals[key] = totals.get(key, 0.0) + (max(d["score"] for d in found) if found else 0.0)
                    for det in found:
                        boxes.append({"Subject": sid, "Accession": acc, "z": int(z), "fold": fold, **det})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    nfold = float(len(ckpts))
    by_study = {}
    for (sid, acc, z), total in totals.items():
        by_study.setdefault((sid, acc), []).append((z, total / nfold))
    return by_study, boxes, nz_of


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="preprocessing output folder")
    p.add_argument("--out", required=True, help="Slice score CSV")
    p.add_argument("--weights", default=str(ROOT / "weights"), help="Folder with maxvit/ and detector/")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--batch", type=int, default=8, help="MaxViT slices per forward")
    p.add_argument("--detector-batch", type=int, default=1, help="Detector slices per forward")
    p.add_argument("--stride", type=int, default=1, help="Score every Nth slice")
    p.add_argument("--n-folds", type=int, default=0, help="First N checkpoints. 0 uses all")
    p.add_argument("--models", choices=("both", "maxvit", "detector"), default="both")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="fp16 on GPU")
    args = p.parse_args()
    if args.batch < 1 or args.detector_batch < 1 or args.stride < 1 or args.n_folds < 0:
        raise SystemExit("--batch, --detector-batch, and --stride must be >= 1")

    data = Path(args.data)
    weights = Path(args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    studies = _studies(data)
    if args.subjects:
        want = set(args.subjects)
        studies = [s for s in studies if s[0] in want]
    if not studies:
        raise SystemExit("No studies matched")

    run_maxvit = args.models in ("both", "maxvit")
    run_detector = args.models in ("both", "detector")
    maxvit_ckpts = _checkpoints(weights / "maxvit", args.n_folds) if run_maxvit else []
    detector_ckpts = _checkpoints(weights / "detector", args.n_folds) if run_detector else []
    print(
        f"[infer] {len(studies)} studies, device={device}, amp={args.amp and device.type == 'cuda'}, "
        f"stride={args.stride}, maxvit_batch={args.batch}, detector_batch={args.detector_batch}, "
        f"folds={len(maxvit_ckpts) or len(detector_ckpts)}",
        flush=True,
    )

    nz_of = {}
    if run_maxvit:
        maxvit, nz_of = _run_maxvit(data, studies, maxvit_ckpts, device, args.stride, args.batch, args.amp)
    else:
        maxvit = {}
    if run_detector:
        detector, boxes, nz_det = _run_detector(
            data, studies, detector_ckpts, device, args.stride, args.detector_batch, args.amp
        )
        nz_of.update(nz_det)
    else:
        detector, boxes = {}, []

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    box_path = out_path.with_name(out_path.stem + "_boxes.csv")
    n_rows = 0
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Subject", "Accession", "nz", "z", "maxvit_prob", "detector_score"]
        )
        writer.writeheader()
        for sid, acc in studies:
            det = {z: score for z, score in detector.get((sid, acc), [])}
            probs = {z: prob for z, prob in maxvit.get((sid, acc), [])}
            for z in sorted(set(det) | set(probs)):
                writer.writerow({
                    "Subject": sid,
                    "Accession": acc,
                    "nz": nz_of.get((sid, acc), 0),
                    "z": z,
                    "maxvit_prob": f"{probs[z]:.6f}" if z in probs else "",
                    "detector_score": f"{det[z]:.6f}" if z in det else "",
                })
                n_rows += 1
    with box_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Subject", "Accession", "z", "fold", "label", "score",
                "x_min", "y_min", "x_max", "y_max",
            ],
        )
        writer.writeheader()
        for box in boxes:
            writer.writerow({
                "Subject": box["Subject"],
                "Accession": box["Accession"],
                "z": box["z"],
                "fold": box["fold"],
                "label": box["label"],
                "score": f"{box['score']:.6f}",
                "x_min": f"{box['x_min']:.2f}",
                "y_min": f"{box['y_min']:.2f}",
                "x_max": f"{box['x_max']:.2f}",
                "y_max": f"{box['y_max']:.2f}",
            })
    print(f"[infer] {n_rows} slices -> {out_path}", flush=True)
    print(f"[infer] {len(boxes)} boxes -> {box_path}", flush=True)


if __name__ == "__main__":
    main()
