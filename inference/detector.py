"""Build a 9-channel slice and return every detection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from detector_model import build_detector

SIZE = 512


def load_detector(path: Path, device: torch.device):
    model = build_detector()
    raw = torch.load(path, map_location="cpu", weights_only=False)
    state = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    cleaned = {}
    for key, value in state.items():
        name = key
        for prefix in ("module.", "_orig_mod."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    bad = [k for k in missing if not k.endswith("num_batches_tracked")]
    if bad or unexpected:
        raise RuntimeError(f"{path.name}: missing {bad[:8]} unexpected {unexpected[:8]}")
    model.to(device).eval()
    return model


def load_phase_slices(folder: Path) -> list[np.ndarray] | None:
    import pydicom

    if not folder.is_dir():
        return None
    files = sorted(folder.glob("*.dcm"))
    if not files:
        return None
    out = []
    lo, hi = -160.0, 240.0
    for path in files:
        ds = pydicom.dcmread(str(path))
        hu = ds.pixel_array.astype(np.float32)
        hu = hu * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
        out.append(np.clip((hu - lo) / (hi - lo), 0.0, 1.0))
    return out


def _at(vol: list[np.ndarray] | None, z: int, shape: tuple[int, int]) -> np.ndarray:
    if not vol:
        return np.zeros(shape, np.float32)
    z = min(max(int(z), 0), len(vol) - 1)
    return np.asarray(vol[z], dtype=np.float32)


def slice_tensor(vols: dict, z: int):
    """Return the 9-channel model input and how to map boxes back to the slice.

    Boxes come back in the 512×512 image. ``geom`` undoes the body crop, the
    square pad, and the resize, into pixels of the original venous slice.
    """
    import cv2

    venous = vols["Venous"]
    shape = venous[0].shape[:2]
    channels = []
    for phase in ("Venous", "NonCon", "Arterial"):
        for dz in (0, -1, 1):
            channels.append(_at(vols[phase], z + dz, shape))
    y0, x0 = 0, 0
    mask = channels[0] > 0
    if mask.any():
        ys, xs = np.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        channels = [c[y0:y1, x0:x1] for c in channels]
    h, w = channels[0].shape[:2]
    side = max(h, w)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    out = []
    for channel in channels:
        if h != w:
            canvas = np.zeros((side, side), np.float32)
            canvas[pad_y : pad_y + h, pad_x : pad_x + w] = channel
            channel = canvas
        if channel.shape[0] != SIZE:
            channel = cv2.resize(channel, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        out.append(channel.astype(np.float32))
    geom = {"x0": x0, "y0": y0, "pad_x": pad_x, "pad_y": pad_y, "side": side}
    return np.stack(out, axis=0), geom


def _to_slice(box: np.ndarray, geom: dict) -> tuple[float, float, float, float]:
    scale = geom["side"] / float(SIZE)
    x1, y1, x2, y2 = [float(v) * scale for v in box]
    x1 = x1 - geom["pad_x"] + geom["x0"]
    x2 = x2 - geom["pad_x"] + geom["x0"]
    y1 = y1 - geom["pad_y"] + geom["y0"]
    y2 = y2 - geom["pad_y"] + geom["y0"]
    return x1, y1, x2, y2


def detect_batch(model, tensors: list[np.ndarray], geoms: list[dict], device: torch.device, amp: bool) -> list[list[dict]]:
    """Every box the model keeps, for a batch of slices. Coordinates are venous-slice pixels."""
    images = [torch.from_numpy(tensor).to(device) for tensor in tensors]
    with torch.no_grad():
        if amp and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                preds = model(images)
        else:
            preds = model(images)
    found_all = []
    for pred, geom in zip(preds, geoms):
        boxes = pred["boxes"].detach().float().cpu().numpy()
        scores = pred["scores"].detach().float().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()
        found = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = _to_slice(box, geom)
            found.append({
                "score": float(score),
                "label": int(label),
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2,
            })
        found_all.append(found)
    return found_all
