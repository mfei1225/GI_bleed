# GI_bleed

## preprocessing

Multi-phase CT GI-bleed preprocessing code: registration, slice export, TotalSegmentator GI occupancy, optional windowed cache.

```bash
cd preprocessing
pip install -r requirements.txt
# optional GPU segmentation
pip install -r requirements-seg.txt

python run_preprocess.py \
  --image_root /path/to/images \
  --series_csv /path/to/series.csv \
  --out_root /path/to/out
```

See `preprocessing/README.md` for input layout, series.csv schema, and full pipeline details.
