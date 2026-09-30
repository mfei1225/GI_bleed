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


def load_cohort_subjects(path: str) -> list[str]:
    df = pd.read_csv(path, dtype=str)
    if "Subject" not in df.columns:
        raise SystemExit(f"{path} needs a Subject column.")
    subjects = []
    seen = set()
    for raw in df["Subject"]:
        sid = normalize_subject_id(raw)
        if sid and sid not in seen:
            seen.add(sid)
            subjects.append(sid)
    return subjects


def resolve_case_insensitive(path, strip_zeros=False):
    """Return the first existing path, or None.

    ``strip_zeros`` is accepted for call-site compatibility and ignored:
    folder names must match the catalog.
    """
    del strip_zeros
    if path and os.path.isdir(path):
        return path
    return None


def count_files_in_folder(folder) -> int:
    if not folder or not os.path.isdir(folder):
        return 0
    try:
        return sum(1 for name in os.listdir(folder) if not name.startswith("."))
    except OSError:
        return 0
