# GI_bleed

End-to-end GI-bleed CT scoring pipeline in three steps. There are two versions, each with its own `preprocessing/`, `inference/`, and `aggregation/`:

| Folder | Slice context | Detector slice score | Rank blend |
|---|---|---|---|
| `native/` | five slices at −2, −1, 0, +1, +2 mm | maximum across folds | 0.7 MaxViT / 0.3 detector |
| `standardized/` | five slices at −2, −1, 0, +1, +2 mm | mean across folds | 0.7 MaxViT / 0.3 detector |

Both versions still segment the venous volume and write `gi_occupancy.json`. Preprocessing keeps the reference series grid and records each slice position in `ipp_map.json`. Inference reads the slices nearest to −2, −1, 0, +1, and +2 mm. In `native/`, a segmentation mask on a coarser grid is resampled onto the venous slices before the GI mask is built.

The diagram and commands below are for `native/`. `standardized/` uses the same commands, and its `infer.py` reads `standardized/inference/weights/` unless you pass `--weights`. See each folder's `README.md`.

```
raw multiphase CT
        │
        ▼
┌───────────────────┐
│  preprocessing/   │  register NonCon / Arterial / Venous, export slices,
│                   │  segment venous GI tract, write volcache + gi_occupancy.json
└─────────┬─────────┘
          │  preprocessed/
          │    slices/  volcache/s512/  gi_occupancy.json
          ▼
┌───────────────────┐
│   inference/      │  MaxViT + Faster R-CNN score every axial slice
└─────────┬─────────┘
          │  results/slices.csv  results/boxes.csv
          ▼
┌───────────────────┐
│  aggregation/     │  keep GI slices; MaxViT noisy-OR + detector mean → blend_score
└─────────┬─────────┘
          │
          ▼
     scores.csv (one row per study)
```

## Why this order

1. **preprocessing** puts all phases on one grid and marks which slices are GI tract. Downstream models assume that layout (`volcache/s512`, `slices/`, `gi_occupancy.json`).
2. **inference** only scores slices. It does not decide the patient number; MaxViT and the detector write per-slice probabilities/boxes.
3. **aggregation** applies the patient rules (GI filter, top-8 noisy-OR for MaxViT, mean of detector scores > 0.1, 0.7/0.3 rank blend). That keeps scoring reproducible and separate from the GPU step.

## Quick start

```bash
# 1) preprocess
cd native/preprocessing && pip install -r requirements.txt
python run_preprocess.py --image_root ... --series_csv ... --out /data/site/preprocessed

# 2) infer (GPU). Pass the checkpoint folder; it is not in git.
cd ../inference && pip install -r requirements.txt
python infer.py \
  --data /data/site/preprocessed \
  --weights /data/site/weights \
  --out /data/site/results

# 3) aggregate
cd ../aggregation && pip install -r requirements.txt
python aggregate.py \
  --results /data/site/results \
  --gi /data/site/preprocessed/gi_occupancy.json
```

See each folder’s `README.md` for inputs, flags, and output columns.

## Weights

Checkpoints are not in git. `native/inference/infer.py` requires `--weights`, a folder with `maxvit/` and `detector/` inside it. `standardized/inference/infer.py` uses `standardized/inference/weights/` unless you pass `--weights`.

## Report labels

`llm_labeling/` labels report text (injury and bleed site). It does not read images. See `llm_labeling/README.md`.
