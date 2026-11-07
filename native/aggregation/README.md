# GI-bleed aggregation

This folder turns the slice scores from `inference` into one score per study. It does not run the networks.

Copy this folder. It only needs NumPy.

```bash
pip install -r requirements.txt
```

Python 3.10+.

## What it does

Inference writes one probability per axial slice. This step keeps the GI slices and turns that stack into one number per study.

`gi_occupancy.json` comes from preprocessing. A slice is GI when colon, small bowel, duodenum, stomach, or esophagus has at least 50 voxels. `gi_z` uses the same index as `0000.dcm`. A subject missing from the file keeps every slice.

The locked patient rules, applied to those GI slices, are:

**MaxViT.** Noisy-OR of the top 8 slice probabilities:

```
1 - (1-p1)(1-p2)...(1-p8)
```

A few high slices raise the score. A long stack of low slices does not. If a study has fewer than 8 slices, every slice is used.

**Detector.** Mean of the slice scores that are above 0.1. Slices at or below 0.1 are left out. If none are above 0.1, the patient score is 0.

**Blend.** Rank both patient scores inside this file. Rank 1 is the lowest score and rank n is the highest.

```
blend_rank  = 0.6 * rank(MaxViT) + 0.4 * rank(detector)
blend_score = blend_rank / number of studies
```

`blend_score` is in (0, 1]. It depends on who else is in the file, so score a cohort in one run. Adding a study changes the blend of the studies already there.

There is no yes/no cutoff in this folder. Higher means the study ranks higher in this cohort.

## Input

`scores_slices.csv` from `inference`:

| Column | Meaning |
|---|---|
| `Subject` | Patient id |
| `Accession` | Study |
| `nz` | Slices in the venous volume |
| `z` | Slice index |
| `maxvit_prob` | MaxViT extravasation probability |
| `detector_score` | Highest detector box score on that slice |

`gi_occupancy.json` from `preprocessing`:

```json
{
  "1001": {"acc": "ACC100", "nz": 320, "z0": 0, "gi_z": [40, 41], "n_gi": 2}
}
```

## How to run it

```bash
python aggregate.py \
  --slices /data/site/scores_slices.csv \
  --gi /data/site/preprocessed/gi_occupancy.json \
  --out /data/site/scores.csv
```

## Output

`scores.csv`, one row per study:

| Column | Meaning |
|---|---|
| `Subject` | Patient id |
| `Accession` | Study |
| `nz` | Slices in the venous volume, copied from the slice file |
| `n_scored` | GI slices that entered the patient scores |
| `gi_z` | Those slice indexes, same index as `0000.dcm` |
| `maxvit_max_slice` | Highest MaxViT slice probability |
| `maxvit_score` | Noisy-OR of the top 8 |
| `detector_score` | Mean of detector slice scores above 0.1 |
| `blend_rank` | 0.6 MaxViT rank + 0.4 detector rank |
| `blend_score` | `blend_rank` divided by the number of studies |
