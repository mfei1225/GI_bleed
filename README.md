# GI_bleed

## processing

Multi-phase CT GI-bleed processing code: registration, slice export, TotalSegmentator GI occupancy, optional windowed cache.

```bash
cd processing
pip install -r requirements.txt
# optional GPU segmentation
pip install -r requirements-seg.txt

python run_preprocess.py \
  --image_root /path/to/images \
  --series_csv /path/to/series.csv \
  --out_root /path/to/out
```

See `processing/README.md` for input layout, series.csv schema, and full pipeline details.
