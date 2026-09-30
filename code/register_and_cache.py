"""Register multi-phase abdominal CT volumes to the Venous phase.

Fixed  : Venous (reference)
Moving : NonCon and Arterial
Method : Two-stage Rigid (Euler3D) -> Affine, Mattes Mutual Information, multi-resolution

Outputs a directory tree of compressed NIfTI volumes:
    <OUT_ROOT>/<subject>/<accession>/{NonCon,Arterial,Venous}.nii.gz

Also writes:
    <OUT_ROOT>/manifest.json    # subject -> accession -> phase -> path
    <OUT_ROOT>/registration_log.csv  # per-subject status

Resumable: skips any subject whose output folder already matches the current plan.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from catalog import (  # noqa: E402
    count_files_in_folder,
    load_series_csv,
    normalize_subject_id,
    resolve_case_insensitive,
)

# Filled in main() from --series_csv / --image_root. Workers inherit these via fork.
PHASES_CSV = None

IMG_ROOT = ""
DEFAULT_OUT_ROOT = "registered_nifti"

PHASE_FIXED = "Venous"
# Delayed is used only when a thick/thin group has no venous series.
PHASE_FIXED_FALLBACKS = ("Delayed",)

# Output slot layout on disk: always {Venous.nii.gz, NonCon.nii.gz, Arterial.nii.gz}.
# The reference series (venous, or delayed if that group has no venous) is written
# untransformed. Other slots are warped into that coordinate system.
# index.json records reference_phase, reference_slot, and slot_phases.
OUTPUT_SLOTS = ("Venous", "NonCon", "Arterial")
# For each slot, list the phases to try when choosing a source series (in order):
SLOT_PHASE_PREFERENCE = {
    "Venous": ("Venous", "Delayed"),
    "NonCon": ("NonCon",),
    "Arterial": ("Arterial",),
}


# ---------------------------------------------------------------------------
# Series discovery
# ---------------------------------------------------------------------------
def get_series_path(subject, accession, scan_id):
    path = resolve_case_insensitive(os.path.join(IMG_ROOT, str(subject)), strip_zeros=True)
    if not path:
        return None
    path = resolve_case_insensitive(os.path.join(path, str(accession)), strip_zeros=True)
    if not path:
        return None
    path = resolve_case_insensitive(os.path.join(path, "CT"), strip_zeros=True)
    if not path:
        return None
    path = resolve_case_insensitive(os.path.join(path, str(scan_id)), strip_zeros=True)
    return path


def _is_vnc_scan_name(scan_id) -> bool:
    """True for dual-energy virtual/material non-contrast (often absent or less preferred)."""
    s = str(scan_id).lower()
    if "vnc" in s or "vue" in s:
        return True
    if "virtual" in s and ("non" in s or "unenh" in s):
        return True
    # DE water material maps (e.g. water__pv, pv_2_5_water) — prefer true WO instead
    if re.search(r"(^|[_\s/-])water([_\s/-]|$)", s):
        return True
    return False


def _is_true_wo_scan_name(scan_id) -> bool:
    """True for true without-contrast series names (not VNC / keV VMI)."""
    s = str(scan_id).lower()
    if _is_vnc_scan_name(s) or "kev" in s:
        return False
    return bool(
        re.search(r"(^|[_\s/-])wo([_\s/-]|$)|w/o|without|non[\s_-]*contrast|noncontrast", s)
    )


def _slice_thickness_mm(folder):
    """SliceThickness from the first readable DICOM, or None."""
    import pydicom

    tried = 0
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        tried += 1
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SliceThickness"])
            th = float(getattr(ds, "SliceThickness", 0) or 0)
            if th > 0:
                return th
        except Exception:
            pass
        if tried >= 5:
            break
    return None


def _thickness_labels(thicknesses, jump=2.0):
    """Map each SliceThickness onto a thick/thin family.

    0.625 mm and 1.25 mm stay together (ratio 2). 1.25 mm and 5 mm split.
    """
    known = sorted({float(t) for t in thicknesses if t})
    if not known:
        return {}
    families = [[known[0]]]
    for th in known[1:]:
        if th / families[-1][-1] > jump:
            families.append([th])
        else:
            families[-1].append(th)
    labels = {}
    for fam in families:
        label = f"th:{fam[0]:.3g}-{fam[-1]:.3g}mm"
        for th in fam:
            labels[th] = label
    return labels


def _best_scan_for_phase(subject, accession, sub, phase, min_files=10, allowed_scans=None):
    """Return (best_scan_id, n_files) for the best usable series of <phase>.

    Requires n_files >= min_files and the folder to exist on disk. Returns (None, -1).

    For NonCon, prefer true WO folders over dual-energy VNC (VNC is frequently
    catalogued in phase.csv but never copied into MGB-img). Within a preference
    tier, pick the densest series. ``allowed_scans`` limits the choice to one
    thick/thin group.
    """
    rows = sub[sub["phase"].astype(str).str.lower() == str(phase).lower()]
    prefer_wo = str(phase).lower() == "noncon"
    allowed = None if allowed_scans is None else {str(s) for s in allowed_scans}
    best_scan, best_n, best_rank = None, -1, -1
    for _, r in rows.iterrows():
        scan_id = r["Scan"]
        if allowed is not None and str(scan_id) not in allowed:
            continue
        p = get_series_path(subject, accession, scan_id)
        n = count_files_in_folder(p) if p else 0
        if n < min_files:
            continue
        if prefer_wo:
            if _is_true_wo_scan_name(scan_id):
                rank = 2
            elif _is_vnc_scan_name(scan_id):
                rank = 0
            else:
                rank = 1
        else:
            rank = 0
        if rank > best_rank or (rank == best_rank and n > best_n):
            best_scan, best_n, best_rank = scan_id, n, rank
    return best_scan, best_n


def _usable_series(subject, accession, sub, min_files=10):
    """One record per on-disk series with at least ``min_files`` DICOMs."""
    recs = []
    for _, r in sub.iterrows():
        scan_id = r["Scan"]
        folder = get_series_path(subject, accession, scan_id)
        n = count_files_in_folder(folder) if folder else 0
        if n < min_files:
            continue
        recs.append(
            {
                "scan": scan_id,
                "phase": str(r["phase"]),
                "n": int(n),
                "th": _slice_thickness_mm(folder) if folder else None,
            }
        )
    return recs


def _count_bands(recs, jump=1.8):
    """Split series into thick vs thin by slice count when thickness tags do not.

    A new band starts when the next series has at least ``jump`` times as many
    slices as the largest series already in the current band.
    """
    ordered = sorted(recs, key=lambda r: r["n"])
    bands = []
    current = [ordered[0]]
    for rec in ordered[1:]:
        if rec["n"] >= max(r["n"] for r in current) * jump:
            bands.append(current)
            current = [rec]
        else:
            current.append(rec)
    bands.append(current)
    return bands


def _thickness_groups(recs):
    """Group series that belong to the same reconstruction (thick with thick).

    Prefer DICOM SliceThickness when more than one thickness is present.
    Otherwise split on slice count, which separates a 1 mm stack from a 5 mm stack.
    """
    if not recs:
        return []
    labels = _thickness_labels(r["th"] for r in recs)
    if len(set(labels.values())) >= 2:
        groups = {}
        unbinned = []
        for rec in recs:
            if rec["th"] is None:
                unbinned.append(rec)
                continue
            groups.setdefault(labels[float(rec["th"])], []).append(rec)
        for rec in unbinned:
            nearest = min(
                groups,
                key=lambda k: abs(
                    rec["n"] - (sum(r["n"] for r in groups[k]) / len(groups[k]))
                ),
            )
            med = sum(r["n"] for r in groups[nearest]) / len(groups[nearest])
            if med > 0 and (1 / 1.5) <= rec["n"] / med <= 1.5:
                groups[nearest].append(rec)
            else:
                groups.setdefault(f"n:{rec['n']}", []).append(rec)
        return list(groups.items())
    bands = _count_bands(recs)
    if len(bands) == 1:
        return [("slices", bands[0])]
    return [(f"slices:{i}", band) for i, band in enumerate(bands)]


def _pick_slot_scans(subject, accession, sub, reference_phase, allowed_scans=None):
    """For one accession, choose a source scan for each output slot.

    Returns: {slot: (scan_id, actual_phase)} covering all OUTPUT_SLOTS, or None if we
    can't fill every slot. The reference slot is chosen so its phase == reference_phase.
    """
    slot_assignments = {}
    for slot in OUTPUT_SLOTS:
        phase_prefs = SLOT_PHASE_PREFERENCE[slot]
        # The slot that matches the reference phase must use reference_phase itself.
        if reference_phase in phase_prefs:
            phase_prefs = (reference_phase,)
        chosen = None
        for ph in phase_prefs:
            scan_id, _n = _best_scan_for_phase(
                subject, accession, sub, ph, allowed_scans=allowed_scans
            )
            if scan_id is not None:
                chosen = (scan_id, ph)
                break
        if chosen is None:
            # For the Venous slot we also allow the reference phase as a final fallback
            # (e.g. a subject that only has Arterial+NonCon+Delayed where Delayed is
            # already in the Venous slot, or a subject labeled on Arterial with no Venous
            # or Delayed — in that case the Venous slot repeats the reference).
            if slot == "Venous":
                scan_id, _n = _best_scan_for_phase(
                    subject, accession, sub, reference_phase, allowed_scans=allowed_scans
                )
                if scan_id is not None:
                    chosen = (scan_id, reference_phase)
        if chosen is None:
            return None
        slot_assignments[slot] = chosen
    return slot_assignments


def _complete_plans_for_accession(subject, accession, sub):
    """One plan per thick/thin group that contains all three phases.

    Groups that are missing a phase are dropped, so a thin venous series is not
    paired with a thick non-contrast series. When more than one group is
    complete, the caller keeps the one with the most reference slices.
    """
    recs = _usable_series(subject, accession, sub)
    plans = []
    for key, members in _thickness_groups(recs):
        allowed = {str(m["scan"]) for m in members}
        for candidate in (PHASE_FIXED, *PHASE_FIXED_FALLBACKS):
            ref_scan, ref_n = _best_scan_for_phase(
                subject, accession, sub, candidate, allowed_scans=allowed
            )
            if ref_scan is None:
                continue
            slot_assignments = _pick_slot_scans(
                subject, accession, sub, candidate, allowed_scans=allowed
            )
            if slot_assignments is None:
                continue
            plans.append(
                {
                    "accession": str(accession),
                    "reference_phase": candidate,
                    "reference_scan": ref_scan,
                    "reference_count": int(ref_n),
                    "slot_assignments": slot_assignments,
                    "driven_by": f"heuristic:{key}",
                }
            )
            break
    return plans


def _plan_from_heuristic(subject):
    """Plan for a subject with no label map.

    Thick and thin reconstructions of the same phase stay in separate groups.
    A group is used only when it has venous (or delayed), non-contrast, and
    arterial. If both a thick group and a thin group are complete, the one
    with more reference slices is kept.
    """
    df = PHASES_CSV[PHASES_CSV["Subject"].astype(str) == str(subject)]
    if df.empty:
        return None

    best = None
    for accession, sub in df.groupby("Accession"):
        for plan in _complete_plans_for_accession(subject, accession, sub):
            if best is None or plan["reference_count"] > best["reference_count"]:
                best = plan
    return best


def plan_registration(subject):
    """Return a registration plan dict or None if the subject can't be registered."""
    return _plan_from_heuristic(normalize_subject_id(subject))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def _to_int16(img):
    """Clamp to int16 range and cast. CT HU is natively int16; float inputs get rounded."""
    clamp = sitk.ClampImageFilter()
    clamp.SetOutputPixelType(sitk.sitkInt16)
    clamp.SetLowerBound(-32768)
    clamp.SetUpperBound(32767)
    return clamp.Execute(img)


def _keep_modal_matrix(names):
    """Drop the rare off-size slice that breaks SimpleITK series reads."""
    import pydicom
    from collections import Counter

    sizes = []
    for fn in names:
        try:
            ds = pydicom.dcmread(fn, stop_before_pixels=True, specific_tags=["Rows", "Columns"])
            sizes.append((int(ds.Rows), int(ds.Columns)))
        except Exception:
            sizes.append(None)
    ok = [s for s in sizes if s is not None]
    if not ok:
        return names
    modal = Counter(ok).most_common(1)[0][0]
    kept = [fn for fn, s in zip(names, sizes) if s == modal]
    return kept if kept else names


def dicom_series_filenames(folder, preferred_series_uid=None):
    """Return DICOM paths for one SeriesInstanceUID in ``folder``.

    Prefer ``preferred_series_uid`` (the GSPS-labeled series) when present.
    Otherwise take the longest UID. A folder can hold two equal-length
    reconstructions; longest-only then picks an arbitrary one.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(folder) or [])
    names = []
    if preferred_series_uid and preferred_series_uid in series_ids:
        names = list(reader.GetGDCMSeriesFileNames(folder, preferred_series_uid))
    elif series_ids:
        names = max(
            (list(reader.GetGDCMSeriesFileNames(folder, sid)) for sid in series_ids),
            key=len,
        )
    else:
        names = list(reader.GetGDCMSeriesFileNames(folder))
    if not names:
        raise RuntimeError(f"No DICOMs in {folder}")
    return _keep_modal_matrix(names)


def read_dicom_series(folder, preferred_series_uid=None):
    reader = sitk.ImageSeriesReader()
    # A single folder can contain more than one SeriesInstanceUID (e.g. a short
    # aorta stack plus the full delayed run). The no-UID helper picks an
    # arbitrary series and can silently drop the labeled volume.
    names = dicom_series_filenames(folder, preferred_series_uid=preferred_series_uid)
    reader.SetFileNames(names)
    img = reader.Execute()
    # Siemens DE VNC / secondary-capture series are often RGB (vector uint8).
    # Registration + HU pipelines need a scalar float32 volume — collapse to grayscale.
    n_comp = int(img.GetNumberOfComponentsPerPixel())
    if n_comp > 1:
        chans = [sitk.Cast(sitk.VectorIndexSelectionCast(img, i), sitk.sitkFloat32) for i in range(n_comp)]
        acc = chans[0]
        for c in chans[1:]:
            acc = acc + c
        return sitk.Cast(acc / float(n_comp), sitk.sitkFloat32)
    return sitk.Cast(img, sitk.sitkFloat32)


def _make_registration_method(iterations, shrink, smooth, sampling_pct=0.03):
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(sampling_pct, seed=42)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel(shrink)
    r.SetSmoothingSigmasPerLevel(smooth)
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return r


def _resample(fixed, moving, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    resampler.SetTransform(transform)
    return resampler.Execute(moving)


def _bone_dice(fixed, moving, thr=200.0, z_stride=8):
    """Cheap bone-overlap score on a z-decimated grid (higher is better)."""
    a = sitk.GetArrayViewFromImage(fixed)[::z_stride].astype(np.float32, copy=False)
    b = sitk.GetArrayViewFromImage(moving)[::z_stride].astype(np.float32, copy=False)
    valid = (a > -1023.0) & (b > -1023.0)
    if not np.any(valid):
        return 0.0
    ba = a > thr
    bb = b > thr
    inter = np.count_nonzero(ba & bb & valid)
    denom = np.count_nonzero(ba & valid) + np.count_nonzero(bb & valid)
    return float(2.0 * inter / denom) if denom else 0.0


def rigid_then_affine(fixed, moving):
    """Two-stage registration: rigid (Euler3D) -> affine, both Mattes MI with multi-resolution pyramid.

    Same-study multiphase CT DICOMs already share a patient/LPS frame, so identity
    resample is usually near-correct. Geometry-centered initialization is harmful
    when FOVs differ (e.g. short Venous vs full NonCon): it shifts volume centers
    apart and MI often fails to recover. Start from identity, refine, and fall back
    to identity if refinement reduces bone overlap.

    Speed-oriented schedule:
      - rigid runs on a 4x and 2x pyramid only (no full-res): rigid converges fast and
        full-res adds cost without meaningful translation/rotation gains for CT
      - affine keeps 2x and 1x so the 12 affine parameters get a full-res refinement
      - sampling 3% (was 5%) and fewer optimizer iterations: empirically these are
        ample for inter-phase CT alignment
    The rigid transform is locked via SetMovingInitialTransform for the second stage
    (avoids CompositeTransform Jacobian issues). The final resample uses both in a
    single pass.
    """
    identity = sitk.Transform(3, sitk.sitkIdentity)
    identity_img = _resample(fixed, moving, identity)
    identity_score = _bone_dice(fixed, identity_img)

    # Stage 1: rigid from identity (NOT geometry-centered — see docstring).
    init = sitk.Euler3DTransform()
    reg = _make_registration_method(
        iterations=50, shrink=[4, 2], smooth=[4.0, 2.0], sampling_pct=0.03
    )
    reg.SetInitialTransform(init, inPlace=False)
    try:
        rigid_tx = reg.Execute(fixed, moving)
    except RuntimeError as e:
        print(f"  rigid failed ({e}); using identity resample", flush=True)
        return identity_img

    # Stage 2: affine, with rigid locked in as the moving initial transform
    affine = sitk.AffineTransform(3)
    reg2 = _make_registration_method(
        iterations=40, shrink=[2, 1], smooth=[2.0, 1.0], sampling_pct=0.03
    )
    reg2.SetMovingInitialTransform(rigid_tx)
    reg2.SetInitialTransform(affine, inPlace=False)
    try:
        affine_tx = reg2.Execute(fixed, moving)
    except RuntimeError as e:
        print(f"  affine failed ({e}); rigid-only resample", flush=True)
        return _resample(fixed, moving, rigid_tx)

    composite = sitk.CompositeTransform([rigid_tx, affine_tx])
    registered = _resample(fixed, moving, composite)
    if _bone_dice(fixed, registered) + 1e-3 < identity_score:
        return identity_img
    return registered


# ---------------------------------------------------------------------------
# Per-subject worker
# ---------------------------------------------------------------------------
def process_subject(args):
    subject, out_root, overwrite = args
    subject = normalize_subject_id(subject)
    subj_dir = Path(out_root) / subject
    try:
        plan = plan_registration(subject)
        if plan is None:
            return {"subject": subject, "status": "skip_no_reference_phase"}
        accession = plan["accession"]
        reference_phase = plan["reference_phase"]
        slot_assignments = plan["slot_assignments"]  # {slot: (scan_id, actual_phase) | None}
        # Slots we actually intend to produce on disk for this subject.
        present_slots = [s for s in OUTPUT_SLOTS if slot_assignments.get(s) is not None]
        missing_slots = [s for s in OUTPUT_SLOTS if slot_assignments.get(s) is None]
        acc_dir = subj_dir / str(accession)

        expected = {slot: acc_dir / f"{slot}.nii.gz" for slot in present_slots}

        # Cache check: output files + index must match the current plan (reference,
        # present slots, AND per-slot source scans). Old plans that duplicated the
        # labeled Arterial series into the Venous slot must re-register.
        idx_path = acc_dir.parent / "index.json"
        planned_slot_scans = {
            slot: (slot_assignments[slot][0] if slot_assignments.get(slot) else None)
            for slot in OUTPUT_SLOTS
        }
        if not overwrite and all(p.exists() for p in expected.values()) and idx_path.exists():
            try:
                existing = json.load(open(idx_path))
                existing_slot_phases = existing.get("slot_phases", {})
                cached_present = {s for s, ph in existing_slot_phases.items() if ph}
                existing_scans = existing.get("slot_scans") or {}
                scans_match = all(
                    existing_scans.get(s) == planned_slot_scans.get(s) for s in OUTPUT_SLOTS
                )
                if (
                    existing.get("reference_phase") == reference_phase
                    and existing.get("accession") == str(accession)
                    and cached_present == set(present_slots)
                    and scans_match
                ):
                    return {
                        "subject": subject,
                        "accession": accession,
                        "status": "cached",
                        "reference_phase": reference_phase,
                        "present_slots": present_slots,
                    }
            except Exception:
                pass  # fall through and re-register

        # Resolve source DICOM folders for every present slot.
        slot_folders = {}
        for slot in present_slots:
            scan_id, actual_phase = slot_assignments[slot]
            folder = get_series_path(subject, accession, scan_id)
            if not folder:
                return {
                    "subject": subject,
                    "status": f"source_folder_missing_{slot}_{actual_phase}",
                }
            slot_folders[slot] = (folder, actual_phase)

        # Identify the reference slot (its actual_phase == reference_phase).
        ref_slot = None
        for slot in present_slots:
            if slot_folders[slot][1] == reference_phase:
                ref_slot = slot
                break
        if ref_slot is None:
            return {
                "subject": subject,
                "status": f"no_slot_has_reference_phase_{reference_phase}",
            }

        fixed_folder, _ = slot_folders[ref_slot]
        preferred_uid = None
        fixed = read_dicom_series(fixed_folder, preferred_series_uid=preferred_uid)
        acc_dir.mkdir(parents=True, exist_ok=True)

        # Drop stale slot NIfTIs that the new plan no longer produces (e.g. old
        # policy wrote a duplicated Arterial into Venous.nii.gz).
        for slot in OUTPUT_SLOTS:
            if slot in expected:
                continue
            stale = acc_dir / f"{slot}.nii.gz"
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass

        # Write the reference slot untransformed.
        # gzip level 1 is ~4x faster than default (9) with only ~20% larger files.
        sitk.WriteImage(
            _to_int16(fixed), str(expected[ref_slot]), useCompression=True, compressionLevel=1
        )

        # Register every non-reference slot to the reference. If a moving slot's
        # source folder is identical to the reference (happens when the labeled
        # phase is NonCon or Arterial and that slot reuses the labeled scan),
        # just copy the reference NIfTI output to avoid an identity registration.
        import shutil  # local import keeps module start-up light
        for slot in present_slots:
            if slot == ref_slot:
                continue
            folder, actual_phase = slot_folders[slot]
            if folder == fixed_folder:
                shutil.copy2(str(expected[ref_slot]), str(expected[slot]))
                continue
            moving = read_dicom_series(folder)
            registered = rigid_then_affine(fixed, moving)
            sitk.WriteImage(
                _to_int16(registered), str(expected[slot]), useCompression=True, compressionLevel=1
            )

        ref_count = int(fixed.GetSize()[2])
        # Record None for missing slots so downstream loaders know to BLANK-fill them.
        source_folders_by_slot = {
            slot: (slot_folders[slot][0] if slot in slot_folders else None)
            for slot in OUTPUT_SLOTS
        }
        slot_phases = {
            slot: (slot_assignments[slot][1] if slot_assignments.get(slot) else None)
            for slot in OUTPUT_SLOTS
        }
        slot_scans = {
            slot: (slot_assignments[slot][0] if slot_assignments.get(slot) else None)
            for slot in OUTPUT_SLOTS
        }
        phases_out = {
            slot: (str(expected[slot]) if slot in expected else None)
            for slot in OUTPUT_SLOTS
        }

        with open(idx_path, "w") as f:
            json.dump(
                {
                    "subject": subject,
                    "accession": str(accession),
                    "phases": phases_out,
                    "source_dicom_folders": source_folders_by_slot,
                    "slot_phases": slot_phases,
                    "slot_scans": slot_scans,
                    "reference_phase": reference_phase,
                    "reference_slot": ref_slot,
                    "reference_count": int(ref_count),
                    "present_slots": present_slots,
                    "missing_slots": missing_slots,
                    "driven_by": plan["driven_by"],
                    "preferred_series_uid": preferred_uid,
                    # Back-compat field: some downstream code expects "venous_slice_count"
                    "venous_slice_count": int(ref_count),
                },
                f,
                indent=2,
            )

        return {
            "subject": subject,
            "accession": str(accession),
            "status": "ok",
            "venous_slices": int(ref_count),
            "reference_phase": reference_phase,
            "present_slots": present_slots,
        }
    except Exception as e:
        return {
            "subject": subject,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_root", required=True, help="Folder of <Subject>/<Accession>/CT/<Scan>/*.dcm")
    p.add_argument("--series_csv", required=True, help="Series catalog: Subject,Accession,Scan,phase")
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Only list subjects that would be registered")
    p.add_argument("--subjects_only", nargs="*", default=None, help="Restrict to this explicit list of subject IDs")
    p.add_argument("--limit", type=int, default=0, help="Process only first N subjects (debug)")
    args = p.parse_args()

    global IMG_ROOT, PHASES_CSV
    IMG_ROOT = os.path.abspath(args.image_root)
    PHASES_CSV = load_series_csv(args.series_csv)
    print(f"[register] image_root={IMG_ROOT}", flush=True)
    print(f"[register] series_csv={args.series_csv} rows={len(PHASES_CSV)}", flush=True)

    os.makedirs(args.out_root, exist_ok=True)
    log_path = Path(args.out_root) / "registration_log.csv"

    if args.subjects_only:
        subjects = [normalize_subject_id(s) for s in args.subjects_only]
    else:
        subjects = sorted(
            {normalize_subject_id(s) for s in PHASES_CSV["Subject"].astype(str)},
            key=lambda s: int(s) if str(s).isdigit() else str(s),
        )
        print(f"[register] every subject in series_csv: {len(subjects)}", flush=True)

    if args.limit:
        subjects = subjects[: args.limit]

    if args.dry_run:
        print(f"Would register {len(subjects)} subjects")
        for s in subjects[:20]:
            print("  ", s)
        if len(subjects) > 20:
            print(f"  ... and {len(subjects) - 20} more")
        return

    # Set SimpleITK thread count per worker to keep CPU contention low.
    # Prefer SLURM_CPUS_PER_TASK so --cpus-per-task=32 WORKERS=8 → 4 threads each.
    os.environ.setdefault("SITK_SHOW_COMMAND", "")
    _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS") or "16")
    os.environ["SITK_GLOBAL_DEFAULT_THREAD_COUNT"] = str(
        max(1, _cpus // max(1, args.workers))
    )

    # CSV log (appended). Write header only if missing.
    log_fields = ["ts", "subject", "accession", "status", "venous_slices", "error", "traceback"]
    write_header = not log_path.exists()
    log_f = open(log_path, "a", newline="")
    writer = csv.DictWriter(log_f, fieldnames=log_fields)
    if write_header:
        writer.writeheader()

    manifest = {}
    manifest_path = Path(args.out_root) / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.load(open(manifest_path))
        except Exception:
            manifest = {}

    start = time.time()
    done = 0
    ok = 0
    cached = 0
    failed = 0

    tasks = [(s, args.out_root, args.overwrite) for s in subjects]

    if args.workers <= 1:
        iterator = (process_subject(t) for t in tasks)
    else:
        ex = ProcessPoolExecutor(max_workers=args.workers)
        futures = [ex.submit(process_subject, t) for t in tasks]
        iterator = (f.result() for f in as_completed(futures))

    try:
        for res in iterator:
            done += 1
            row = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "subject": res.get("subject"),
                "accession": res.get("accession", ""),
                "status": res.get("status"),
                "venous_slices": res.get("venous_slices", ""),
                "error": res.get("error", ""),
                "traceback": (res.get("traceback", "") or "").replace("\n", " | "),
            }
            writer.writerow(row)
            log_f.flush()

            if res["status"] == "ok":
                ok += 1
                # Only record phases that were actually produced on disk for this
                # subject. Labeled subjects may produce a subset (e.g. Venous+NonCon
                # with Arterial missing) under the new registration policy.
                present = res.get("present_slots") or list(OUTPUT_SLOTS)
                manifest.setdefault(res["subject"], {})[res["accession"]] = {
                    ph: str(Path(args.out_root) / res["subject"] / res["accession"] / f"{ph}.nii.gz")
                    for ph in present
                }
            elif res["status"] == "cached":
                cached += 1
            else:
                failed += 1

            if done % 10 == 0 or done == len(tasks):
                elapsed = time.time() - start
                rate = done / max(1e-6, elapsed)
                eta = (len(tasks) - done) / max(1e-6, rate)
                print(
                    f"[{done}/{len(tasks)}] ok={ok} cached={cached} failed={failed}  "
                    f"{rate:.2f} subj/s  ETA {eta/60:.1f} min  "
                    f"(last: {res['subject']} -> {res['status']})",
                    flush=True,
                )
                json.dump(manifest, open(manifest_path, "w"), indent=2)
    finally:
        log_f.close()
        json.dump(manifest, open(manifest_path, "w"), indent=2)

    print(f"\nDone. ok={ok} cached={cached} failed={failed} total={done}")
    print(f"Manifest: {manifest_path}")
    print(f"Log:      {log_path}")


if __name__ == "__main__":
    main()
