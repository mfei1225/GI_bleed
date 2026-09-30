"""Series catalog and path helpers for a new-site preprocessing run.

Replaces the MGB-specific imports in register_and_cache.py / extract_registered_slices.py
(boundingboxDataset.PHASES_CSV, image root, subject-id normalization).
"""
from __future__ import annotations

import os

import pandas as pd

REQUIRED_SERIES_COLUMNS = ("Subject", "Accession", "Scan", "phase")
PHASES = ("Venous", "Delayed", "NonCon", "Arterial")


def normalize_subject_id(s) -> str:
    text = str(s).strip()
    if text.isdigit():
        return str(int(text))
    return text


def load_series_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in REQUIRED_SERIES_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}. Need {list(REQUIRED_SERIES_COLUMNS)}.")
    df["Subject"] = df["Subject"].map(normalize_subject_id)
    df["Accession"] = df["Accession"].astype(str).str.strip()
    df["Scan"] = df["Scan"].astype(str).str.strip()
    df["phase"] = df["phase"].astype(str).str.strip()
    bad = sorted(set(df["phase"]) - set(PHASES))
    if bad:
        raise SystemExit(f"{path} has phase values {bad}. Allowed: {list(PHASES)}.")
    return df


def resolve_case_insensitive(path, strip_zeros=False):
    """Return ``path`` if it exists, otherwise the same path with case-insensitive names.

    ``strip_zeros`` also treats ``0123`` and ``123`` as the same folder name.
    """
    if path and os.path.exists(path):
        return path
    if not path:
        return None
    parts = path.strip("/").split("/")
    current = "/"
    split_idx = 0
    for i in range(len(parts), 0, -1):
        candidate = "/" + "/".join(parts[:i])
        if os.path.isdir(candidate):
            current = candidate
            split_idx = i
            break
    for part in parts[split_idx:]:
        try:
            entries = os.listdir(current)
        except (FileNotFoundError, PermissionError, NotADirectoryError):
            return None
        want = part.lower().lstrip("0") if strip_zeros else part.lower()
        match = None
        for entry in entries:
            got = entry.lower().lstrip("0") if strip_zeros else entry.lower()
            if got == want:
                match = entry
                break
        if match is None:
            return None
        current = os.path.join(current, match)
    return current


def count_files_in_folder(folder) -> int:
    if not folder or not os.path.isdir(folder):
        return 0
    try:
        return sum(1 for name in os.listdir(folder) if not name.startswith("."))
    except OSError:
        return 0
