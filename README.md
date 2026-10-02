# GI_bleed

End-to-end GI-bleed CT scoring pipeline in three steps:

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
          │  scores_slices.csv
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
3. **aggregation** applies the locked patient rules (GI filter, top-8 noisy-OR for MaxViT, mean of detector scores > 0.1, 0.6/0.4 rank blend). That keeps scoring reproducible and separate from the GPU step.

## Quick start

```bash
# 1) preprocess
cd preprocessing && pip install -r requirements.txt
python run_preprocess.py --image_root ... --series_csv ... --out_root /data/site/preprocessed

# 2) infer (GPU; place checkpoints in inference/weights/)
cd ../inference && pip install -r requirements.txt
python infer.py --data /data/site/preprocessed --out /data/site/scores_slices.csv

# 3) aggregate
cd ../aggregation && pip install -r requirements.txt
python aggregate.py \
  --slices /data/site/scores_slices.csv \
  --gi /data/site/preprocessed/gi_occupancy.json \
  --out /data/site/scores.csv
```

See each folder’s `README.md` for inputs, flags, and output columns.
