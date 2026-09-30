# GI_bleed

Code for multi-phase CT GI-bleed preprocessing (registration, slice export, TotalSegmentator GI occupancy, optional windowed cache).

## Layout

```
code/                     # main preprocessing package (run from here)
  run_preprocess.py
  register_and_cache.py
  ...
  legacy_label_map/       # older variant that supports --label_phase_map
```

## Quick start

```bash
cd code
pip install -r requirements.txt
# optional GPU segmentation
pip install -r requirements-seg.txt

python run_preprocess.py \
  --image_root /path/to/images \
  --series_csv /path/to/series.csv \
  --out_root /path/to/out
```

See `code/README.md` for input layout, series.csv schema, and full pipeline details.
