# GI-bleed share, 1 mm neighbors

This folder is the sendable pipeline for the 1 mm, 5-slice models. Copy the whole folder. The weights are not included.

The volumes are not resampled to 1 mm. Preprocessing keeps each study on the acquired axial grid and writes the slice position, in millimetres, to `ipp_map.json`. Inference and training pick the acquired slice nearest to −2, −1, 0, +1, and +2 mm. The ends of a volume repeat the last available slice. A missing position map falls back to neighboring slice indexes.

```
preprocessing/   register, segment, export slices, GI occupancy, optional volume cache
inference/       MaxViT and detector slice scores
aggregation/     one score per study
```

Run them in that order. Each subfolder has its own README and requirements.

## Patient score

GI slices are those where colon, small bowel, duodenum, stomach, or esophagus has at least 50 voxels. A coarser organ mask is resampled onto the venous volume before that count, so the slice index matches `0000.dcm`. A subject with no mask keeps every slice.

- MaxViT: noisy-OR of the top 8 GI slice probabilities.
- Detector: mean of the 5 highest GI slice scores. This is the rule with the highest detector AUC on the 1 mm training out-of-fold scores.
- Blend: `0.7 * rank(MaxViT) + 0.3 * rank(detector)`, then divided by the number of studies. Rank 1 is the lowest score in that file.

The earlier share still uses the mean of detector slices above 0.1. This folder does not.
