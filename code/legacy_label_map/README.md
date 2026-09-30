# Multi-phase CT preprocessing

This folder registers a contrast-phase CT study into one coordinate system, segments the venous volume, writes one DICOM per axial slice, and optionally bakes a windowed volume cache.

Copy this whole folder. It does not need the rest of the training repository.

```bash
pip install -r requirements.txt
```

Needs Python 3.10+. Segmentation is a second install (`pip install -r requirements-seg.txt`) and wants a GPU.

## What the pipeline does

One study often has three axial series acquired at different times: non-contrast, arterial, and portal venous (a delayed series is treated as venous). They are not on the same grid. The model expects them stacked as if they were the same anatomy.

1. **Register.** Pick one series as the fixed reference and leave it untransformed. Rigidly, then affinely, register the other phases onto that grid with Mattes mutual information (SimpleITK Euler3D, then affine). Write one compressed NIfTI per phase. Pixel values stay Hounsfield units, stored as int16.
2. **Export slices.** Split each NIfTI into one DICOM per axial slice so a slice loader can read them. `RescaleSlope` is 1 and `RescaleIntercept` is 0, so the stored pixels are already HU. Every phase of a study shares the reference slice positions.
3. **Segment.** TotalSegmentator (`total` task) runs on the registered `Venous.nii.gz` only. Masks are written per organ. The bowel masks (colon, small bowel, duodenum, stomach, esophagus) become a per-slice GI flag: a slice counts as GI when any of those organs has at least 50 voxels. Patient scoring keeps GI slices and drops the rest. If a subject has no masks, every slice is kept.
4. **Cache (optional).** Soft-tissue window each slice (center 40 HU, width 400 HU, display range −160 to 240), pad to a square, resize, and save a uint8 NumPy volume. Inference reads this cache when it is present, and otherwise windows the slice DICOMs the same way on the fly.

A subject is skipped at registration unless the chosen accession can fill all three output slots (venous or delayed, non-contrast, and arterial), each with at least 10 DICOM files. If several series share a phase, non-contrast prefers a true without-contrast series over a dual-energy virtual non-contrast (`VNC`, `VUE`, water maps). Other phases prefer the series with the most slices.

Without a label map, the reference is the densest venous series across the subject's accessions. Delayed is used only when that accession has no venous series. Pass `--label_phase_map` to force a specific accession, phase, and scan folder (the series that was annotated, for example).

## Input you need to prepare

```
site/
  series.csv
  cohort.csv          # optional; limits who is processed
  images/
    <Subject>/
      <Accession>/
        CT/
          <Scan>/
            *.dcm
```

`series.csv` columns:

| Column | Meaning |
|---|---|
| `Subject` | Patient id. Leading zeros on numeric ids are stripped (`0012` and `12` are the same). |
| `Accession` | Study id. Must match the folder name. |
| `Scan` | Series folder name under `CT/`. |
| `phase` | One of `Venous`, `Delayed`, `NonCon`, `Arterial`. |

One row per series. `Delayed` is written into the venous output slot. True without-contrast is `NonCon`, not a virtual non-contrast.

Example:

```csv
Subject,Accession,Scan,phase
1001,ACC100,3_venous,Venous
1001,ACC100,1_wo,NonCon
1001,ACC100,2_arterial,Arterial
```

`cohort.csv` only needs a `Subject` column. If you omit it, every subject in `series.csv` is processed.

Optional label map (`label_phase_map.json`). Keys are subject ids. This forces the reference series and its accession:

```json
{
  "1001": {"accession": "ACC100", "phase": "Venous", "scan": "3_venous"}
}
```

`phase` is the contrast phase of that series (`Venous`, `Delayed`, `NonCon`, or `Arterial`). `scan` is the `Scan` folder name.

## How to run it on a new dataset

From this folder:

```bash
python run_preprocess.py \
  --image_root /data/site/images \
  --series_csv /data/site/series.csv \
  --cohort_csv /data/site/cohort.csv \
  --out /data/site/preprocessed \
  --size 512 \
  --workers 4
```

`--size` is the square side of the training cache (512 for the MaxViT used here, 256 is smaller). `--workers` is the number of subjects registered at once. Registration and TotalSegmentator are the slow steps. `--skip_seg` leaves segmentation out. `--seg_device cpu` runs it without a GPU.

To force annotated series to be the untransformed reference:

```bash
python run_preprocess.py \
  --image_root /data/site/images \
  --series_csv /data/site/series.csv \
  --label_phase_map /data/site/label_phase_map.json \
  --out /data/site/preprocessed
```

Each step can also be run on its own. Add `--overwrite` to replace existing outputs. Registration is otherwise resumable: a subject whose `index.json` matches the current plan is skipped. Segmentation skips a study whose organ masks are already present unless `--force` is set.

```bash
python register_and_cache.py \
  --image_root /data/site/images \
  --series_csv /data/site/series.csv \
  --cohort_csv /data/site/cohort.csv \
  --out_root /data/site/preprocessed/nifti \
  --workers 4

python segment.py \
  --nifti_root /data/site/preprocessed/nifti \
  --out_root /data/site/preprocessed/seg \
  --device gpu

python gi_occupancy.py \
  --nifti_root /data/site/preprocessed/nifti \
  --seg_root /data/site/preprocessed/seg \
  --out /data/site/preprocessed/gi_occupancy.json

python extract_registered_slices.py \
  --nifti_root /data/site/preprocessed/nifti \
  --slice_root /data/site/preprocessed/slices \
  --workers 4 \
  --min_age_s 0

python cache_volumes.py \
  --slice_root /data/site/preprocessed/slices \
  --cache_root /data/site/preprocessed/volcache \
  --size 512 \
  --workers 4
```

`--dry_run` on `register_and_cache.py` only prints the subject list. Check `nifti/registration_log.csv` for `ok`, `cached`, `skip_no_reference_phase`, or `error`.

## Output folder structure

```
<preprocessed>/
  gi_occupancy.json
  nifti/
    registration_log.csv
    manifest.json
    <Subject>/
      index.json
      <Accession>/
        Venous.nii.gz
        NonCon.nii.gz
        Arterial.nii.gz
  seg/
    <Subject>/
      <Accession>/
        totalseg_meta.json
        Venous/
          colon.nii.gz
          small_bowel.nii.gz
          duodenum.nii.gz
          stomach.nii.gz
          esophagus.nii.gz
          ... other organs in the GI-filter list ...
  slices/
    <Subject>/
      <Accession>/
        ipp_map.json
        Venous/0000.dcm
        Venous/0001.dcm
        NonCon/0000.dcm
        Arterial/0000.dcm
  volcache/
    s512/
      <Subject>/
        <Accession>/
          Venous.npy
          NonCon.npy
          Arterial.npy
```

Slice index `0000.dcm` is the first slice along the reference series (SimpleITK order), not necessarily the most superior slice.

### `nifti/<Subject>/index.json`

| Field | Meaning |
|---|---|
| `accession` | Study that was registered. |
| `reference_phase` | Contrast phase left untransformed (`Venous`, `Delayed`, `NonCon`, or `Arterial`). |
| `reference_slot` | Output file that holds that series (`Venous`, `NonCon`, or `Arterial`). Delayed lands in `Venous`. |
| `slot_phases` | Actual contrast phase written into each output file, or null if that slot was not produced. |
| `slot_scans` | Source `Scan` folder name per slot. |
| `source_dicom_folders` | Absolute path of the original DICOM folder per slot. |
| `present_slots` / `missing_slots` | Which of the three files exist. |
| `reference_count` | Number of axial slices in the reference volume. |
| `driven_by` | `heuristic` or `label_map`. |

### `slices/<Subject>/<Accession>/ipp_map.json`

Shared by all phases of that study.

| Field | Meaning |
|---|---|
| `nz` | Number of slices. File `0000.dcm` is z=0. |
| `spacing_mm` | SimpleITK spacing `(x, y, z)` in mm. |
| `origin_xyz_lps` | Origin of the reference volume. |
| `direction_9` | 3×3 direction matrix, row-major. |
| `slice_normal_xyz` | Unit normal of the axial plane. |
| `axial_positions_mm` | Length `nz`. Position of each slice along that normal, in mm. |
| `present_phases` | Phases that were exported. |

### Slice DICOM

Minimal secondary-capture header plus the registered pixels:

- `PixelData` is int16 HU.
- `RescaleSlope = 1`, `RescaleIntercept = 0`.
- `ImagePositionPatient` is the reference position for that z, copied onto every phase.
- Filename is `{z:04d}.dcm` (`0000.dcm`, `0001.dcm`, ...).

### Volume cache `.npy`

`np.load` returns `uint8` array shaped `(nz, size, size)`.

- 0 = −160 HU or lower, 255 = 240 HU or higher, linear in between.
- Slices were zero-padded to a square before resize, so the patient stays centered.
- Divide by 255 to get the `[0, 1]` image the classifier uses.
- Path `volcache/s512/<Subject>/<Accession>/<phase>.npy` is the layout the training code expects. Point `PATIENT_CLS_VOLCACHE` at the `volcache` folder.

### `gi_occupancy.json`

One object per subject. `gi_z` is the slice indexes to keep (the same index as `0000.dcm`). `z0` is 0. `n_gi` is the length of `gi_z`.

### What inference still does, and what it does not

Nothing else is written to disk before a study is scored.

At score time the loader only:

- reads the uint8 cache (or windows the slice DICOM the same way),
- stacks channels in memory from those three phases (the difference slab, or the raw phases with adjacent slices, depending on which model is loaded),
- drops slices whose z is absent from `gi_z` when that subject is in `gi_occupancy.json`.

There is no lung crop, no extra normalization, and no histogram matching. Positive-slice labels from annotation boxes are used to train and to score against ground truth. They are not an input to inference.
