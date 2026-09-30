"""Extract per-slice DICOMs from registered NIfTI volumes.

After register_and_cache.py has produced <subject>/<accession>/{Venous,NonCon,Arterial}.nii.gz,
this script writes one DICOM per axial slice for each phase, so the existing MONAI
LoadImaged-based data loader can consume them unchanged.

Output structure:
    /MGB-img-registered-slices/<subject>/<accession>/
        Venous/<z:04d>.dcm
        Arterial/<z:04d>.dcm
        NonCon/<z:04d>.dcm
        ipp_map.json     # z-index -> IPP axis position + slice normal + origin

Each per-slice DICOM gets a valid MONAI-readable minimal header:
    - Bits/Rows/Columns correct
    - RescaleSlope=1, RescaleIntercept=0
    - PixelData = registered int16 slice

This is sufficient for MONAI LoadImaged (which reads via pydicom/ITK).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import SimpleITK as sitk

CODE_DIR = Path(__file__).resolve().parent
sys.path.append(str(CODE_DIR))

from catalog import (  # noqa: E402
    load_series_csv,
    normalize_subject_id,
    resolve_case_insensitive,
)

DEFAULT_NIFTI_ROOT = "registered_nifti"
DEFAULT_SLICE_ROOT = "registered_slices"
IMG_ROOT = ""
PHASES_CSV = None

PHASES = ("Venous", "NonCon", "Arterial")


def _slice_normal_from_iop(iop):
    iop = np.asarray(iop, dtype=float)
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    nn = np.linalg.norm(n)
    return n / nn if nn > 0 else n


def _load_venous_source_folder(subject, accession, nifti_root):
    """Find the original Venous DICOM folder used for registration.

    1) Check index.json (written by register_and_cache.py)
    2) Fall back: match by slice count from PHASES_CSV + filesystem
    """
    idx = Path(nifti_root) / subject / "index.json"
    if idx.exists():
        try:
            with open(idx) as f:
                d = json.load(f)
            venous = d.get("source_dicom_folders", {}).get("Venous")
            if venous and os.path.isdir(venous):
                return venous
        except Exception:
            pass

    # Fallback: look up via PHASES_CSV, picking the Venous series whose file count
    # matches the NIfTI's z-size. PHASES_CSV / IMG_ROOT are set in main().

    if PHASES_CSV is None or not IMG_ROOT:
        return None
    nii = Path(nifti_root) / subject / accession / "Venous.nii.gz"
    if not nii.exists():
        return None
    img = sitk.ReadImage(str(nii))
    target_nz = int(img.GetSize()[2])

    mask = (
        (PHASES_CSV["Subject"].astype(str) == str(subject))
        & (PHASES_CSV["Accession"].astype(str) == str(accession))
        & (PHASES_CSV["phase"] == "Venous")
    )
    cand = PHASES_CSV[mask]
    for _, row in cand.iterrows():
        subj_dir = resolve_case_insensitive(os.path.join(IMG_ROOT, str(subject)), strip_zeros=True)
        if not subj_dir:
            continue
        folder = resolve_case_insensitive(
            os.path.join(subj_dir, str(accession), "CT", str(row["Scan"])), strip_zeros=True
        )
        if folder and os.path.isdir(folder):
            try:
                cnt = len([f for f in os.listdir(folder) if not f.startswith(".")])
            except Exception:
                continue
            if cnt == target_nz:
                return folder
    return None


def _axial_from_dicom_folder(folder, preferred_series_uid=None):
    """IPP-along-normal for one series, in the same file order sitk uses."""
    import pydicom
    import sys

    from register_and_cache import dicom_series_filenames  # noqa: E402

    try:
        files = dicom_series_filenames(folder, preferred_series_uid=preferred_series_uid)
    except Exception:
        return None, None
    ipps, iops = [], []
    for f in files:
        try:
            ds = pydicom.dcmread(
                f,
                stop_before_pixels=True,
                force=True,
                specific_tags=["ImagePositionPatient", "ImageOrientationPatient"],
            )
        except Exception:
            continue
        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        if ipp is None or iop is None:
            continue
        ipps.append([float(x) for x in ipp])
        iops.append([float(x) for x in iop])
    if not ipps:
        return None, None
    ipps = np.asarray(ipps, dtype=float)
    n = _slice_normal_from_iop(iops[0])
    axial = (ipps @ n).astype(float)
    return axial, n


def _slice_normal_from_dir(direction_cosines_9):
    """Given SITK GetDirection() (row-major 3x3), return the slice-normal unit vector (3rd column)."""
    d = np.array(direction_cosines_9).reshape(3, 3)
    n = d[:, 2]
    nn = np.linalg.norm(n)
    return n / nn if nn > 0 else n


def _axis_positions_from_image(img):
    """Compute per-slice IPP-along-normal positions for a SimpleITK 3D image."""
    origin = np.array(img.GetOrigin(), dtype=float)  # LPS origin of voxel (0,0,0)
    direction = img.GetDirection()
    spacing = img.GetSpacing()
    n = _slice_normal_from_dir(direction)
    nz = img.GetSize()[2]
    # For each slice z, IPP = origin + z * spacing_z * slice_normal
    positions = origin[None, :] + np.arange(nz)[:, None] * spacing[2] * n[None, :]
    axial = positions @ n  # scalar position along slice normal
    return axial.astype(float), n.astype(float)


def _build_dicom_template(
    spacing_xy, rows, cols, series_uid, series_description, patient_id
):
    """Build a reusable FileDataset with all subject/phase-level fields preset.

    Per-slice work is then limited to updating SOPInstanceUID, InstanceNumber,
    ImagePositionPatient, SliceLocation, PixelData -- roughly 5x faster than
    reconstructing a fresh Dataset for every slice.
    """
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPInstanceUID = series_uid  # per-slice value is set per write

    ds = FileDataset("", {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.PatientID = str(patient_id)
    ds.PatientName = str(patient_id)
    ds.Modality = "CT"
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = series_uid
    ds.SeriesDescription = series_description

    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.PixelSpacing = [float(spacing_xy[1]), float(spacing_xy[0])]
    ds.SliceThickness = 1.0
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleSlope = 1
    ds.RescaleIntercept = 0
    ds.RescaleType = "HU"
    ds.WindowCenter = 40
    ds.WindowWidth = 400
    # Placeholder for per-slice fields so the VR is pre-baked:
    ds.SOPInstanceUID = series_uid + ".0"
    ds.InstanceNumber = 0
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.SliceLocation = 0.0
    ds.PixelData = b"\0" * (int(rows) * int(cols) * 2)
    return ds


def _write_phase_slices(phase_dir, template, vol_int16, ipp_positions, series_uid):
    """Write all slices of one phase using the prebuilt template.

    vol_int16: np.ndarray of shape [nz, rows, cols], dtype int16 (already cast).
    ipp_positions: np.ndarray of shape [nz, 3] with ImagePositionPatient per slice.
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    # Contiguous int16 slice bytes are fastest.
    vol_int16 = np.ascontiguousarray(vol_int16, dtype=np.int16)
    nz = vol_int16.shape[0]

    # One shared template; we copy lightly per slice and overwrite 5 tags.
    # Using copy.copy (shallow) is enough because we only reassign simple attributes
    # and PixelData; we never mutate shared sub-structures.
    for z in range(nz):
        ds = copy.copy(template)
        ds.file_meta = copy.copy(template.file_meta)
        sop = f"{series_uid}.{z + 1}"
        ds.file_meta.MediaStorageSOPInstanceUID = sop
        ds.SOPInstanceUID = sop
        ds.InstanceNumber = z + 1
        ipp = ipp_positions[z]
        ds.ImagePositionPatient = [float(ipp[0]), float(ipp[1]), float(ipp[2])]
        # SliceLocation = axial-normal projection (kept consistent with ipp build)
        ds.SliceLocation = float(ipp[2])
        ds.PixelData = vol_int16[z].tobytes()
        ds.save_as(str(phase_dir / f"{z:04d}.dcm"), write_like_original=True)


def _all_slices_exist(acc_out: Path, nz: int, phases=PHASES) -> bool:
    """Quick check: every listed phase dir has all <z:04d>.dcm files. Uses os.listdir only."""
    for ph in phases:
        d = acc_out / ph
        try:
            names = os.listdir(d)
        except FileNotFoundError:
            return False
        # fast containment check
        existing = {n for n in names if n.endswith(".dcm")}
        if len(existing) < nz:
            return False
        # final exactness check on last slice
        if f"{nz - 1:04d}.dcm" not in existing:
            return False
    return True


def _get_axial_positions(subject, accession, nifti_root, ref, idx_json, force=False):
    """Return (axial_positions[nz], slice_normal[3]).

    Priority order:
      1) index.json already has axial_positions_mm + slice_normal_xyz (cache hit, zero I/O)
      2) Re-derive from the original reference DICOM folder (slow: reads N headers)
      3) Fall back to the NIfTI image geometry
    Results are cached back into index.json for the next run.
    """
    cached_pos = idx_json.get("axial_positions_mm")
    cached_norm = idx_json.get("slice_normal_xyz")
    if (
        not force
        and cached_pos
        and cached_norm
        and len(cached_pos) == ref.GetSize()[2]
    ):
        return np.asarray(cached_pos, dtype=float), np.asarray(cached_norm, dtype=float)

    nz = ref.GetSize()[2]
    axial_positions = slice_normal = None
    preferred_uid = idx_json.get("preferred_series_uid")
    ref_slot = idx_json.get("reference_slot") or "Venous"
    src_folders = idx_json.get("source_dicom_folders") or {}
    ref_folder = src_folders.get(ref_slot) or src_folders.get("Venous")
    if ref_folder and os.path.isdir(ref_folder):
        axial_positions, slice_normal = _axial_from_dicom_folder(
            ref_folder, preferred_series_uid=preferred_uid
        )
        if axial_positions is not None and len(axial_positions) != nz:
            axial_positions = None
    if axial_positions is None:
        venous_folder = _load_venous_source_folder(subject, accession, nifti_root)
        if venous_folder:
            axial_positions, slice_normal = _axial_from_dicom_folder(
                venous_folder, preferred_series_uid=preferred_uid
            )
            if axial_positions is not None and len(axial_positions) != nz:
                axial_positions = None
    if axial_positions is None:
        axial_positions, slice_normal = _axis_positions_from_image(ref)

    # Cache back into index.json so next run / concurrent extract skips the DICOM read.
    try:
        idx_json["axial_positions_mm"] = [float(v) for v in axial_positions]
        idx_json["slice_normal_xyz"] = [float(v) for v in slice_normal]
        idx_path = Path(nifti_root) / subject / "index.json"
        with open(idx_path, "w") as f:
            json.dump(idx_json, f, indent=2)
    except Exception:
        pass
    return axial_positions, slice_normal


def process_subject(args):
    subject, accession, nifti_root, slice_root, overwrite, delete_nifti, min_age_s = args
    acc_in = Path(nifti_root) / subject / accession
    acc_out = Path(slice_root) / subject / accession
    try:
        idx_path = Path(nifti_root) / subject / "index.json"
        if not idx_path.exists():
            return {"subject": subject, "accession": accession, "status": "no_index_json"}

        # Read index.json once (reused for caching axial positions)
        try:
            with open(idx_path) as f:
                idx_json = json.load(f)
        except Exception:
            idx_json = {}

        # Figure out which phases this subject actually has on disk. Under the
        # new registration policy labeled subjects may be missing NonCon and/or
        # Arterial, so we only extract slices for the phases listed in the plan.
        slot_phases = idx_json.get("slot_phases") or {}
        if slot_phases:
            present_phases = tuple(ph for ph in PHASES if slot_phases.get(ph))
        else:
            # Legacy index.json without slot_phases: assume all three.
            present_phases = PHASES
        if not present_phases:
            return {"subject": subject, "accession": accession, "status": "no_present_phases"}

        # Safety against concurrent registration writes:
        # require every present-phase NIfTI file to exist AND be older than min_age_s.
        now = time.time()
        for ph in present_phases:
            p = acc_in / f"{ph}.nii.gz"
            if not p.exists():
                return {"subject": subject, "accession": accession, "status": f"missing_{ph}"}
            if min_age_s > 0 and (now - p.stat().st_mtime) < min_age_s:
                return {"subject": subject, "accession": accession, "status": "too_fresh_retry_later"}

        # Fast-path cache check BEFORE loading NIfTIs (saves a lot of I/O on reruns)
        if not overwrite:
            cached_nz = idx_json.get("axial_positions_mm")
            if cached_nz is not None:
                nz_hint = len(cached_nz)
                if _all_slices_exist(acc_out, nz_hint, phases=present_phases):
                    return {
                        "subject": subject,
                        "accession": accession,
                        "status": "cached",
                        "nz": nz_hint,
                        "present_phases": list(present_phases),
                    }

        # Load reference first to get shape/spacing, then any other present phases.
        # "Venous" is always the reference slot on disk (even if the underlying
        # reference phase is Delayed/Arterial/NonCon -- see register_and_cache.py).
        if "Venous" not in present_phases:
            return {"subject": subject, "accession": accession, "status": "missing_reference_slot"}
        ref = sitk.ReadImage(str(acc_in / "Venous.nii.gz"), sitk.sitkInt16)
        nz = ref.GetSize()[2]
        rows = ref.GetSize()[1]
        cols = ref.GetSize()[0]
        spacing_xy = (ref.GetSpacing()[0], ref.GetSpacing()[1])
        origin_xyz = np.array(ref.GetOrigin(), dtype=float)

        if not overwrite and _all_slices_exist(acc_out, nz, phases=present_phases):
            return {
                "subject": subject,
                "accession": accession,
                "status": "cached",
                "nz": nz,
                "present_phases": list(present_phases),
            }

        # Axial positions: cached if possible, else derive + cache.
        axial_positions, slice_normal = _get_axial_positions(
            subject, accession, nifti_root, ref, idx_json, force=overwrite
        )

        # Precompute ImagePositionPatient for every slice: IPP = origin + z_pos * normal
        ipp_positions = origin_xyz[None, :] + axial_positions[:, None] * slice_normal[None, :]

        # Load each present volume and convert to contiguous int16 arrays once.
        vols_int16 = {
            "Venous": np.ascontiguousarray(sitk.GetArrayFromImage(ref), dtype=np.int16),
        }
        for ph in present_phases:
            if ph == "Venous":
                continue
            img = sitk.ReadImage(str(acc_in / f"{ph}.nii.gz"), sitk.sitkInt16)
            vols_int16[ph] = np.ascontiguousarray(sitk.GetArrayFromImage(img), dtype=np.int16)

        # Build one reusable template per present phase and write slices. We parallelize
        # across phases with a ThreadPoolExecutor -- pydicom.save_as is I/O bound so
        # threads give us real overlap without the fork overhead.
        # Truncate the per-phase series UID so series_uid + ".<z>" stays <= 64 chars
        # for any realistic nz (DICOM UI VR max length). Reserve 7 chars for ".<nz>".
        series_uids = {ph: generate_uid()[:57] for ph in present_phases}
        templates = {
            ph: _build_dicom_template(
                spacing_xy=spacing_xy,
                rows=rows,
                cols=cols,
                series_uid=series_uids[ph],
                series_description=f"{ph}_REG",
                patient_id=subject,
            )
            for ph in present_phases
        }

        def _do_phase(ph):
            _write_phase_slices(
                phase_dir=acc_out / ph,
                template=templates[ph],
                vol_int16=vols_int16[ph],
                ipp_positions=ipp_positions,
                series_uid=series_uids[ph],
            )

        with ThreadPoolExecutor(max_workers=max(1, len(present_phases))) as tpool:
            list(tpool.map(_do_phase, present_phases))

        # IPP map (per-z axis position + slice normal + origin)
        ipp_map = {
            "subject": subject,
            "accession": accession,
            "nz": int(nz),
            "spacing_mm": [float(s) for s in ref.GetSpacing()],
            "origin_xyz_lps": [float(o) for o in origin_xyz],
            "direction_9": [float(v) for v in ref.GetDirection()],
            "slice_normal_xyz": [float(v) for v in slice_normal],
            "axial_positions_mm": [float(v) for v in axial_positions],
            "present_phases": list(present_phases),
        }
        with open(acc_out / "ipp_map.json", "w") as f:
            json.dump(ipp_map, f)

        # Optionally delete source NIfTI files once per-slice DICOMs are saved.
        if delete_nifti:
            for ph in present_phases:
                p = acc_in / f"{ph}.nii.gz"
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass

        return {
            "subject": subject,
            "accession": accession,
            "status": "ok",
            "nz": int(nz),
            "present_phases": list(present_phases),
        }
    except Exception as e:
        return {
            "subject": subject,
            "accession": accession,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nifti_root", default=DEFAULT_NIFTI_ROOT)
    p.add_argument("--slice_root", default=DEFAULT_SLICE_ROOT)
    p.add_argument("--image_root", default="", help="Only needed if index.json source folders are missing.")
    p.add_argument("--series_csv", default="", help="Only needed for the same fallback.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--subjects_only", nargs="*", default=None)
    p.add_argument("--delete_nifti_after", action="store_true",
                   help="Delete a subject's NIfTI files once per-slice DICOMs are saved successfully.")
    p.add_argument("--min_age_s", type=int, default=30,
                   help="Only process NIfTIs whose mtime is older than this many seconds. "
                        "Guards against reading a file mid-write during concurrent registration.")
    args = p.parse_args()

    global IMG_ROOT, PHASES_CSV
    IMG_ROOT = os.path.abspath(args.image_root) if args.image_root else ""
    if args.series_csv:
        PHASES_CSV = load_series_csv(args.series_csv)

    # Scan the filesystem for subjects with complete registration output.
    # This is safer than trusting the top-level manifest.json because register_and_cache.py
    # writes that file only every ~10 subjects, while per-subject index.json is final.
    nifti_root = Path(args.nifti_root)
    if not nifti_root.exists():
        sys.exit(f"nifti_root does not exist: {nifti_root}")

    tasks = []
    for subj_dir in sorted(nifti_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name
        if args.subjects_only:
            want = {normalize_subject_id(s) for s in args.subjects_only}
            if normalize_subject_id(subject) not in want:
                continue
        # index.json is the last file written by register_and_cache.process_subject,
        # so its presence means all three NIfTI files are finalized.
        idx_path = subj_dir / "index.json"
        if not idx_path.exists():
            continue
        try:
            idx = json.load(open(idx_path))
            accession = idx.get("accession")
            slot_phases = idx.get("slot_phases") or {}
        except Exception:
            # fall back to first accession dir
            acc_candidates = [d.name for d in subj_dir.iterdir() if d.is_dir()]
            accession = acc_candidates[0] if acc_candidates else None
            slot_phases = {}
        if not accession:
            continue
        acc_dir = subj_dir / accession
        # Require that every phase present in the plan has its NIfTI on disk.
        # Legacy indices without slot_phases default to "all three phases required".
        required_phases = tuple(ph for ph in PHASES if slot_phases.get(ph)) or PHASES
        if not all((acc_dir / f"{ph}.nii.gz").exists() for ph in required_phases):
            continue
        tasks.append((subject, accession, args.nifti_root, args.slice_root,
                      args.overwrite, args.delete_nifti_after, args.min_age_s))

    print(f"Planning to process {len(tasks)} (subject, accession) entries with {args.workers} workers")
    print(f"  delete_nifti_after={args.delete_nifti_after}  min_age_s={args.min_age_s}")
    os.makedirs(args.slice_root, exist_ok=True)

    # Keep SimpleITK threading low per worker since DICOM writing is I/O-bound
    os.environ["SITK_GLOBAL_DEFAULT_THREAD_COUNT"] = str(max(1, 8 // max(1, args.workers)))

    start = time.time()
    done = ok = cached = failed = 0

    # Start from existing manifest so partial runs accumulate rather than overwrite
    manifest_out_path = Path(args.slice_root) / "manifest.json"
    if manifest_out_path.exists():
        try:
            with open(manifest_out_path) as f:
                out_manifest = json.load(f)
        except Exception:
            out_manifest = {}
    else:
        out_manifest = {}

    if args.workers <= 1:
        for res in (process_subject(t) for t in tasks):
            _handle(res, out_manifest)
            done += 1
            if res["status"] == "ok":
                ok += 1
            elif res["status"] == "cached":
                cached += 1
            else:
                failed += 1
            _print_progress(done, len(tasks), ok, cached, failed, start, res)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process_subject, t) for t in tasks]
            for fut in as_completed(futs):
                res = fut.result()
                _handle(res, out_manifest)
                done += 1
                if res["status"] == "ok":
                    ok += 1
                elif res["status"] == "cached":
                    cached += 1
                else:
                    failed += 1
                _print_progress(done, len(tasks), ok, cached, failed, start, res)

    json.dump(out_manifest, open(manifest_out_path, "w"), indent=2)
    print(f"\nDone. ok={ok} cached={cached} failed={failed} total={done}")
    print(f"Manifest: {manifest_out_path} ({len(out_manifest)} subjects total)")


def _handle(res, out_manifest):
    if res["status"] in ("ok", "cached"):
        present = res.get("present_phases") or list(PHASES)
        out_manifest.setdefault(res["subject"], {})[res["accession"]] = {
            ph: str(Path(DEFAULT_SLICE_ROOT) / res["subject"] / res["accession"] / ph)
            for ph in present
        }


def _print_progress(done, total, ok, cached, failed, start, res):
    if done % 5 == 0 or done == total:
        elapsed = time.time() - start
        rate = done / max(1e-6, elapsed)
        eta = (total - done) / max(1e-6, rate)
        print(
            f"[{done}/{total}] ok={ok} cached={cached} failed={failed}  "
            f"{rate:.2f} subj/s  ETA {eta/60:.1f} min  "
            f"(last: {res.get('subject')} {res.get('accession','')} -> {res['status']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
